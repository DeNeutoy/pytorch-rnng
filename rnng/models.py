import abc
from collections import OrderedDict
from typing import Collection, List, Mapping, NamedTuple, Sequence, Tuple, Union, cast
from typing import Dict  # noqa

from nltk.tree import Tree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from rnng.actions import Action, ShiftAction, ReduceAction, NonTerminalAction
from rnng.typing import Word, POSTag, NonTerminalLabel, WordId, POSId, NonTerminalId, ActionId
from rnng.utils import ItemStore
from rnng.stack_lstm import StackLSTM


def log_softmax(inputs: Variable, restrictions=None) -> Variable:
    if restrictions is None:
        return F.log_softmax(inputs)

    if restrictions.dim() != 1:
        raise RuntimeError('restrictions must be one dimensional')

    addend = Variable(
        inputs.data.new(inputs.size()).zero_().index_fill_(
            inputs.dim() - 1, restrictions, -float('inf')))
    return F.log_softmax(inputs + addend)


class StackElement(NamedTuple):
    subtree: Union[Word, Tree]
    emb: Variable
    is_open_non_terminal: bool


class IllegalActionError(Exception):
    pass


class RnnGrammar(nn.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def stack_buffer(self) -> Sequence[Union[Tree, Word]]:
        pass

    @property
    @abc.abstractmethod
    def action_history(self) -> Sequence[Action]:
        pass

    @property
    @abc.abstractmethod
    def finished(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def started(self) -> bool:
        pass

    @abc.abstractmethod
    def initialise_stacks_and_buffers(self, tagged_words: Sequence[Tuple[Word, POSTag]]) -> None:
        pass

    @abc.abstractmethod
    def push_non_terminal(self, nonterm: NonTerminalLabel) -> None:
        pass

    @abc.abstractmethod
    def reduce(self) -> None:
        pass

    @abc.abstractmethod
    def can_push_non_terminal(self) -> bool:
        pass

    @abc.abstractmethod
    def can_reduce(self) -> bool:
        pass


class DiscriminativeRnnGrammar(RnnGrammar):
    MAX_OPEN_NON_TERMINALS = 100

    def __init__(self,
                 word2id: Mapping[Word, WordId],
                 pos2id: Mapping[POSTag, POSId],
                 non_terminal2id: Mapping[NonTerminalLabel, NonTerminalId],
                 action_store: ItemStore,
                 word_dim: int = 32,
                 pos_dim: int = 12,
                 non_terminal_dim: int = 60,
                 action_dim: int = 16,
                 input_dim: int = 128,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.) -> None:

        if ShiftAction() not in action_store:
            raise ValueError('SHIFT action ID must be specified')
        if ReduceAction() not in action_store:
            raise ValueError('REDUCE action ID must be specified')

        num_words = len(word2id)
        num_pos = len(pos2id)
        num_non_terminals = len(non_terminal2id)
        num_actions = len(action_store)

        for wid in word2id.values():
            if wid < 0 or wid >= num_words:
                raise ValueError(f'word ID of {wid} is out of range')
        for pid in pos2id.values():
            if pid < 0 or pid >= num_pos:
                raise ValueError(f'POS tag ID of {pid} is out of range')
        for nid in non_terminal2id.values():
            if nid < 0 or nid >= num_non_terminals:
                raise ValueError(f'nonterminal ID of {nid} is out of range')

        super().__init__()
        self.word2id = word2id
        self.pos2id = pos2id
        self.nt2id = non_terminal2id
        self.action_store = action_store
        self.num_words = num_words
        self.num_pos = num_pos
        self.num_non_terminals = num_non_terminals
        self.num_actions = num_actions
        self.word_dim = word_dim
        self.pos_dim = pos_dim
        self.non_terminal_dim = non_terminal_dim
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Parser states
        self._stack = []  # type: List[StackElement]
        self._buffer = []  # type: List[Word]
        self._history = []  # type: List[Action]
        self._num_open_non_terminals = 0
        self._started = False

        # Parser state encoders
        self.stack_lstm = StackLSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        # can use an LSTM, but this is easier.
        self.buffer_lstm = StackLSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        # can use LSTM, but this is more efficient
        self.history_lstm = StackLSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        # Composition
        self.compose_fwd_lstm = nn.LSTM(input_dim, input_dim, num_layers=num_layers, dropout=dropout)
        self.compose_bwd_lstm = nn.LSTM(input_dim, input_dim, num_layers=num_layers, dropout=dropout)

        # Transformations
        self.word2lstm = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(word_dim + pos_dim, input_dim)),
            ('relu', nn.ReLU())
        ]))
        self.nt2lstm = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(non_terminal_dim, input_dim)),
            ('relu', nn.ReLU())
        ]))
        self.action2lstm = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(action_dim, input_dim)),
            ('relu', nn.ReLU())
        ]))
        self.fwdbwd2composed = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(2 * input_dim, input_dim)),
            ('relu', nn.ReLU())
        ]))
        self.lstms2summary = nn.Sequential(OrderedDict([  # Stack LSTMs to parser summary
            ('dropout', nn.Dropout(dropout)),
            ('linear', nn.Linear(3 * hidden_dim, hidden_dim)),
            ('relu', nn.ReLU())
        ]))
        self.summary2actions = nn.Linear(hidden_dim, num_actions)

        # Embeddings
        self.word_embs = nn.Embedding(num_words, word_dim)
        self.pos_embs = nn.Embedding(num_pos, pos_dim)
        self.nt_embs = nn.Embedding(num_non_terminals, non_terminal_dim)
        self.action_embs = nn.Embedding(num_actions, action_dim)

        # Guard parameters for stack, buffer, and action history
        self.stack_guard = nn.Parameter(torch.Tensor(input_dim))
        self.buffer_guard = nn.Parameter(torch.Tensor(input_dim))
        self.history_guard = nn.Parameter(torch.Tensor(input_dim))

        # Final embeddings
        self._word_emb = {}  # type: Dict[WordId, Variable]
        self._nt_emb = {}  # type: Variable
        self._action_emb = {}  # type: Variable

    @property
    def stack_buffer(self) -> List[Union[Tree, Word]]:
        return [x.subtree for x in self._stack]

    @property
    def input_buffer(self) -> List[Word]:
        return list(reversed(self._buffer))

    @property
    def action_history(self) -> List[Action]:
        return list(self._history)

    @property
    def finished(self) -> bool:
        return (len(self._stack) == 1
                and not self._stack[0].is_open_non_terminal
                and len(self._buffer) == 0)

    @property
    def started(self) -> bool:
        return self._started

    def initialise_stacks_and_buffers(self, tagged_words: Sequence[Tuple[Word, POSTag]]) -> None:
        if len(tagged_words) == 0:
            raise ValueError('parser cannot be started with empty sequence of words')
        for word, pos in tagged_words:
            if word not in self.word2id:
                raise ValueError(f"unknown word '{word}' encountered")
            if pos not in self.pos2id:
                raise ValueError(f"unknown POS tag '{pos}' encountered")

        self._stack = []
        self._buffer = []
        self._history = []
        self._num_open_non_terminals = 0
        self._started = False

        self.history_lstm.reset()
        self.stack_lstm.reset()
        self.buffer_lstm.reset()

        # Feed guards as inputs
        self.stack_lstm.push(self.stack_guard)
        self.buffer_lstm.push(self.buffer_guard)
        self.history_lstm.push(self.history_guard)

        # Initialize input buffer and its LSTM encoder
        words, pos_tags = tuple(zip(*tagged_words))
        self._prepare_embeddings(words, pos_tags)
        for word in reversed(words):
            self._buffer.append(word)
            assert word in self.word2id
            wid = self.word2id[word]
            assert wid in self._word_emb
            self.buffer_lstm.push(self._word_emb[wid])
        self._started = True

    def push_non_terminal(self, nonterm: NonTerminalLabel) -> None:
        if nonterm not in self.nt2id:
            raise KeyError(f"unknown nonterminal '{nonterm}' encountered")
        action = NonTerminalAction(nonterm)
        if action not in self.action_store:
            raise KeyError(f"unknown action '{action}' encountered")

        if not self.can_push_non_terminal():
            raise IllegalActionError(f"Illegal NT({nonterm}) action taken.")
        self._push_non_terminal(nonterm)
        self._append_history(action)

    def shift(self) -> None:
        if not self.can_shift():
            raise IllegalActionError("Illegal SHIFT action attempted.")
        self._shift()
        self._append_history(ShiftAction())

    def reduce(self) -> None:
        if not self.can_reduce():
            raise IllegalActionError("Illegal REDUCE action attempted.")
        self._reduce()
        self._append_history(ReduceAction())

    def forward(self):
        if not self._started:
            raise RuntimeError('parser is not started yet, please call `start` method first')

        lstm_embs = [self.stack_lstm.top, self.buffer_lstm.top, self.history_lstm.top]
        assert all(emb is not None for emb in lstm_embs)
        lstms_emb = torch.cat(lstm_embs).view(1, -1)
        parser_summary = self.lstms2summary(lstms_emb)
        illegal_action_ids = self._get_illegal_action_ids()
        if illegal_action_ids.dim() == 0:
            illegal_action_ids = None
        return log_softmax(self.summary2actions(parser_summary),
                           restrictions=illegal_action_ids).view(-1)

    def _prepare_embeddings(self, words: Collection[Word], pos_tags: Collection[POSTag]):
        assert len(words) == len(pos_tags)
        assert all(w in self.word2id for w in words)
        assert all(p in self.pos2id for p in pos_tags)

        word_ids = [self.word2id[w] for w in words]
        pos_ids = [self.pos2id[p] for p in pos_tags]
        assert all(0 <= wid < self.num_words for wid in word_ids)
        assert all(0 <= pid < self.num_pos for pid in pos_ids)
        nt_ids = list(range(self.num_non_terminals))
        action_ids = list(range(self.num_actions))

        volatile = not self.training
        word_indices = Variable(self._new(word_ids).long().view(1, -1), volatile=volatile)
        pos_indices = Variable(self._new(pos_ids).long().view(1, -1), volatile=volatile)
        non_terminal_indices = Variable(self._new(nt_ids).long().view(1, -1), volatile=volatile)
        action_indices = Variable(self._new(action_ids).long().view(1, -1), volatile=volatile)

        word_embeddings = self.word_embs(word_indices).view(-1, self.word_dim)
        pos_embeddings = self.pos_embs(pos_indices).view(-1, self.pos_dim)
        non_terminal_embeddings = self.nt_embs(non_terminal_indices).view(-1, self.non_terminal_dim)
        action_embeddings = self.action_embs(action_indices).view(-1, self.action_dim)

        final_word_embeddings = self.word2lstm(torch.cat([word_embeddings, pos_embeddings], dim=1))
        final_non_terminal_embeddings = self.nt2lstm(non_terminal_embeddings)
        final_action_embeddings = self.action2lstm(action_embeddings)

        self._word_emb = dict(zip(word_ids, final_word_embeddings))
        self._nt_emb = final_non_terminal_embeddings
        self._action_emb = final_action_embeddings

    def _append_history(self, action: Action) -> None:
        self._history.append(action)
        assert action in self.action_store
        aid = self.action_store[action]
        assert isinstance(self._action_emb, Variable)
        assert 0 <= aid < self._action_emb.size(0)
        self.history_lstm.push(self._action_emb[aid])

    def _shift(self) -> None:
        assert len(self._buffer) > 0
        assert len(self.buffer_lstm) > 0
        word = self._buffer.pop()
        self.buffer_lstm.pop()
        assert word in self.word2id
        wid = self.word2id[word]
        assert wid in self._word_emb
        self._stack.append(StackElement(word, self._word_emb[wid], False))
        self.stack_lstm.push(self._word_emb[wid])

    def _reduce(self) -> None:

        # Pop all the children of the non-terminal off
        # the stack.
        children = []
        while len(self._stack) > 0 and not self._stack[-1].is_open_non_terminal:
            children.append(self._stack.pop()[:-1])
        assert len(children) > 0
        assert len(self._stack) > 0

        children.reverse()
        child_subtrees, child_embs = zip(*children)
        open_nt = self._stack.pop()
        assert isinstance(open_nt.subtree, Tree)
        parent_subtree = cast(Tree, open_nt.subtree)
        parent_subtree.extend(child_subtrees)
        composed_embedding = self._get_composed_representation(open_nt.emb, child_embs)
        self._stack.append(StackElement(parent_subtree, composed_embedding, False))
        self._num_open_non_terminals -= 1
        assert self._num_open_non_terminals >= 0

    def _push_non_terminal(self, nonterminal: NonTerminalLabel) -> None:
        nid = self.nt2id[nonterminal]
        assert isinstance(self._nt_emb, Variable)
        assert 0 <= nid < self._nt_emb.size(0)
        self._stack.append(StackElement(Tree(nonterminal, []),
                                        self._nt_emb[nid],
                                        True))
        self.stack_lstm.push(self._nt_emb[nid])
        self._num_open_non_terminals += 1

    def _get_composed_representation(self,
                                     open_non_terminal_embedding: Variable,
                                     children_embeddings: Sequence[Variable]) -> Variable:
        """
        Given a non-terminal symbol and it's children, create a representation
        of the completed non-terminal by encoding the children using a Bi-LSTM.
        The embedding of the non-terminal symbol is pre-pended to the sequence
        before the BiLSTM is applied.
        """
        assert open_non_terminal_embedding.dim() == 1
        assert all(x.dim() == 1 for x in children_embeddings)
        assert open_non_terminal_embedding.size(0) == self.input_dim
        assert all(x.size(0) == self.input_dim for x in children_embeddings)

        fwd_input = [open_non_terminal_embedding]
        bwd_input = [open_non_terminal_embedding]
        for i in range(len(children_embeddings)):
            fwd_input.append(children_embeddings[i])
            bwd_input.append(children_embeddings[-i - 1])

        fwd_input = torch.cat(fwd_input).view(-1, 1, self.input_dim)
        bwd_input = torch.cat(bwd_input).view(-1, 1, self.input_dim)
        fwd_output, _ = self.compose_fwd_lstm(fwd_input, self._init_compose_states())
        bwd_output, _ = self.compose_bwd_lstm(bwd_input, self._init_compose_states())
        fwd_emb = F.dropout(fwd_output[-1, 0], p=self.dropout, training=self.training)
        bwd_emb = F.dropout(bwd_output[-1, 0], p=self.dropout, training=self.training)
        return self.fwdbwd2composed(torch.cat([fwd_emb, bwd_emb]).view(1, -1)).view(-1)

    def _get_illegal_action_ids(self) -> Variable:
        illegal_action_ids = [aid for aid in range(self.num_actions)
                              if not self._is_legal(aid)]
        return self._new(illegal_action_ids).long()

    def _is_legal(self, aid: ActionId) -> bool:
        assert 0 <= aid < len(self.action_store)

        return self.action_store.get_by_id(aid).is_legal_on(self)

    def _verify_action(self) -> None:
        if not self._started:
            raise RuntimeError('parser is not started yet, please call `start` method first')
        if self.finished:
            raise RuntimeError('cannot do more action when parser is finished')

    def can_push_non_terminal(self) -> bool:
        self._verify_action()
        if len(self._buffer) == 0:
            # cannot do NT(X) when input buffer is empty
            return False
        if self._num_open_non_terminals >= self.MAX_OPEN_NON_TERMINALS:
            # max number of open nonterminals is reached
            return False
        return True

    def can_shift(self) -> bool:
        self._verify_action()
        if len(self._buffer) == 0:
            # cannot SHIFT when input buffer is empty
            return False
        if self._num_open_non_terminals == 0:
            # cannot SHIFT when no open nonterminal exists
            return False
        return True

    def can_reduce(self) -> bool:
        self._verify_action()
        last_is_non_terminal = (len(self._history) > 0 and
                                isinstance(self._history[-1], NonTerminalAction))
        if last_is_non_terminal:
            # cannot REDUCE when top of stack is an open nonterminal
            return False
        if self._num_open_non_terminals < 2 and len(self._buffer) > 0:
            # cannot REDUCE because there are words not SHIFT-ed yet
            return False

        return True

    def reset_parameters(self) -> bool:
        # Stack LSTMs
        for name in ['stack', 'buffer', 'history']:
            lstm = getattr(self, f'{name}_lstm')
            for pname, pval in lstm.named_parameters():
                if pname.startswith('lstm.weight'):
                    init.orthogonal(pval)
                else:
                    assert pname.startswith('lstm.bias') or pname in ('h0', 'c0')
                    init.constant(pval, 0.)

        # Composition
        for name in ['fwd', 'bwd']:
            lstm = getattr(self, f'compose_{name}_lstm')
            for pname, pval in lstm.named_parameters():
                if pname.startswith('weight'):
                    init.orthogonal(pval)
                else:
                    assert pname.startswith('bias')
                    init.constant(pval, 0.)

        # Transformations
        gain = init.calculate_gain('relu')
        for name in ['word', 'nt', 'action']:
            layer = getattr(self, f'{name}2lstm')
            init.xavier_uniform(layer.linear.weight, gain=gain)
            init.constant(layer.linear.bias, 1.)
        init.xavier_uniform(self.fwdbwd2composed.linear.weight, gain=gain)
        init.constant(self.fwdbwd2composed.linear.bias, 1.)
        init.xavier_uniform(self.lstms2summary.linear.weight, gain=gain)
        init.constant(self.lstms2summary.linear.bias, 1.)
        init.xavier_uniform(self.summary2actions.weight)
        init.constant(self.summary2actions.bias, 0.)

        # Embeddings
        for name in ['word', 'pos', 'nt', 'action']:
            layer = getattr(self, f'{name}_embs')
            init.uniform(layer.weight, -0.01, 0.01)

        # Guards
        for name in ['stack', 'buffer', 'history']:
            guard = getattr(self, f'{name}_guard')
            init.uniform(guard, -0.01, 0.01)

    def _init_compose_states(self) -> Tuple[Variable, Variable]:
        h0 = Variable(self._new(self.num_layers, 1, self.input_dim).zero_())
        c0 = Variable(self._new(self.num_layers, 1, self.input_dim).zero_())
        return h0, c0

    def _new(self, *args, **kwargs):
        return next(self.parameters()).data.new(*args, **kwargs)
