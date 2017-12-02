import pytest
import torch
from nltk.tree import Tree
from torch.autograd import Variable

from rnng.models import (DiscRNNGrammar, EmptyStackError, StackLSTM, log_softmax,
                  IllegalActionError)
from rnng.utils import ItemStore
from rnng.actions import ShiftAction, ReduceAction, NonTerminalAction


class MockLSTM:
    def __init__(self, input_size, hidden_size, num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.index = 0
        self.retvals = [(self._get_output(), self._get_hn_cn()) for _ in range(3)]

    def __call__(self, inputs, init_states):
        retval = self.retvals[self.index]
        self.index = (self.index + 1) % len(self.retvals)
        return retval

    def _get_output(self):
        return Variable(torch.randn(1, 1, self.hidden_size))

    def _get_hn_cn(self):
        return (Variable(torch.randn(self.num_layers, 1, self.hidden_size)),
                Variable(torch.randn(self.num_layers, 1, self.hidden_size)))


class TestStackLSTM:
    input_size = 10
    hidden_size = 5
    num_layers = 3
    seq_len = 3

    def test_call(self, mocker):
        mock_lstm = MockLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        mocker.patch('rnng.models.nn.LSTM', return_value=mock_lstm, autospec=True)
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)

        assert len(lstm) == 0
        h, c = lstm(inputs[0])
        assert torch.equal(h.data, mock_lstm.retvals[0][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[0][1][1].data)
        assert len(lstm) == 1
        h, c = lstm(inputs[1])
        assert torch.equal(h.data, mock_lstm.retvals[1][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[1][1][1].data)
        assert len(lstm) == 2
        h, c = lstm(inputs[2])
        assert torch.equal(h.data, mock_lstm.retvals[2][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[2][1][1].data)
        assert len(lstm) == 3

    def test_top(self, mocker):
        mock_lstm = MockLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        mocker.patch('rnng.models.nn.LSTM', return_value=mock_lstm, autospec=True)
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)

        assert lstm.top is None
        lstm(inputs[0])
        assert torch.equal(lstm.top.data, mock_lstm.retvals[0][0].data.squeeze())
        lstm(inputs[1])
        assert torch.equal(lstm.top.data, mock_lstm.retvals[1][0].data.squeeze())
        lstm(inputs[2])
        assert torch.equal(lstm.top.data, mock_lstm.retvals[2][0].data.squeeze())

    def test_pop(self, mocker):
        mock_lstm = MockLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        mocker.patch('rnng.models.nn.LSTM', return_value=mock_lstm, autospec=True)
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        lstm(inputs[0])
        lstm(inputs[1])
        lstm(inputs[2])

        h, c = lstm.pop()
        assert torch.equal(h.data, mock_lstm.retvals[2][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[2][1][1].data)
        assert torch.equal(lstm.top.data, mock_lstm.retvals[1][0].data.squeeze())
        assert len(lstm) == 2
        h, c = lstm.pop()
        assert torch.equal(h.data, mock_lstm.retvals[1][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[1][1][1].data)
        assert torch.equal(lstm.top.data, mock_lstm.retvals[0][0].data.squeeze())
        assert len(lstm) == 1
        h, c = lstm.pop()
        assert torch.equal(h.data, mock_lstm.retvals[0][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[0][1][1].data)
        assert lstm.top is None
        assert len(lstm) == 0

    def test_pop_when_empty(self, mocker):
        mock_lstm = MockLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        mocker.patch('rnng.models.nn.LSTM', return_value=mock_lstm, autospec=True)

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        with pytest.raises(EmptyStackError):
            lstm.pop()

    def test_num_layers_too_low(self):
        with pytest.raises(ValueError):
            StackLSTM(10, 5, num_layers=0)


def test_log_softmax():
    restrictions = torch.LongTensor([0, 2])
    inputs = Variable(torch.randn(1, 5))

    outputs = log_softmax(inputs, restrictions)

    assert isinstance(outputs, Variable)
    assert outputs.size() == (1, 5)
    nonzero_indices = outputs.view(-1).exp().data.nonzero().view(-1)
    assert all(nonzero_indices.eq(torch.LongTensor([1, 3, 4])))


class TestDiscRNNGrammar:
    word2id = {'John': 0, 'loves': 1, 'Mary': 2}
    pos2id = {'NNP': 0, 'VBZ': 1}
    nt2id = {'S': 2, 'NP': 1, 'VP': 0}
    action_store = ItemStore()
    actions = [NonTerminalAction('S'), NonTerminalAction('NP'), NonTerminalAction('VP'), ShiftAction(), ReduceAction()]
    for a in actions:
        action_store.add(a)

    def test_init(self):
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        assert len(parser.stack_buffer) == 0
        assert len(parser.input_buffer) == 0
        assert len(parser.action_history) == 0
        assert not parser.finished
        assert not parser.started

    def test_init_no_shift_action(self):
        action_store = ItemStore()
        actions = [NonTerminalAction('S'), NonTerminalAction('NP'), NonTerminalAction('VP'), ReduceAction()]
        for a in actions:
            action_store.add(a)

        with pytest.raises(ValueError):
            DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, action_store)

    def test_init_no_reduce_action(self):
        action_store = ItemStore()
        actions = [NonTerminalAction('S'), NonTerminalAction('NP'), NonTerminalAction('VP'), ShiftAction()]
        for a in actions:
            action_store.add(a)

        with pytest.raises(ValueError):
            DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, action_store)

    def test_init_word_id_out_of_range(self):
        word2id = dict(self.word2id)

        word2id['John'] = len(word2id)
        with pytest.raises(ValueError):
            DiscRNNGrammar(word2id, self.pos2id, self.nt2id, self.action_store)

        word2id['John'] = -1
        with pytest.raises(ValueError):
            DiscRNNGrammar(word2id, self.pos2id, self.nt2id, self.action_store)

    def test_init_pos_id_out_of_range(self):
        pos2id = dict(self.pos2id)

        pos2id['NNP'] = len(pos2id)
        with pytest.raises(ValueError):
            DiscRNNGrammar(self.word2id, pos2id, self.nt2id, self.action_store)

        pos2id['NNP'] = -1
        with pytest.raises(ValueError):
            DiscRNNGrammar(self.word2id, pos2id, self.nt2id, self.action_store)

    def test_init_nt_id_out_of_range(self):
        nt2id = dict(self.nt2id)

        nt2id['S'] = len(nt2id)
        with pytest.raises(ValueError):
            DiscRNNGrammar(self.word2id, self.pos2id, nt2id, self.action_store)

        nt2id['S'] = -1
        with pytest.raises(ValueError):
            DiscRNNGrammar(self.word2id, self.pos2id, nt2id, self.action_store)

    def test_start(self):
        words = ['John', 'loves', 'Mary']
        pos_tags = ['NNP', 'VBZ', 'NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        parser.start(list(zip(words, pos_tags)))

        assert len(parser.stack_buffer) == 0
        assert parser.input_buffer == words
        assert len(parser.action_history) == 0
        assert not parser.finished
        assert parser.started

    def test_start_with_empty_tagged_words(self):
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        with pytest.raises(ValueError):
            parser.start([])

    def test_start_with_invalid_word_or_pos(self):
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        with pytest.raises(ValueError):
            parser.start([('Bob', 'NNP')])

        with pytest.raises(ValueError):
            parser.start([('John', 'VBD')])

    def test_do_nt_action(self):
        words = ['John', 'loves', 'Mary']
        pos_tags = ['NNP', 'VBZ', 'NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)
        parser.start(list(zip(words, pos_tags)))
        prev_input_buffer = parser.input_buffer

        parser.push_nt('S')

        assert len(parser.stack_buffer) == 1
        last = parser.stack_buffer[-1]
        assert isinstance(last, Tree)
        assert last.label() == 'S'
        assert len(last) == 0
        assert parser.input_buffer == prev_input_buffer
        assert len(parser.action_history) == 1
        assert parser.action_history[-1] == NonTerminalAction('S')
        assert not parser.finished

    def test_do_illegal_push_nt_action(self):
        words = ['John']
        pos_tags = ['NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        # Buffer is empty
        parser.start(list(zip(words, pos_tags)))
        parser.push_nt('S')
        parser.shift()
        with pytest.raises(IllegalActionError):
            parser.push_nt('NP')

        # More than 100 open nonterminals
        parser.start(list(zip(words, pos_tags)))
        for i in range(100):
            parser.push_nt('S')
        with pytest.raises(IllegalActionError):
            parser.push_nt('NP')

    def test_push_unknown_nt(self):
        words = ['John']
        pos_tags = ['NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)
        parser.start(list(zip(words, pos_tags)))

        with pytest.raises(KeyError):
            parser.push_nt('asdf')

    def test_push_known_nt_but_unknown_action(self):
        actions = [NonTerminalAction('NP'), NonTerminalAction('VP'), ShiftAction(), ReduceAction()]
        action_store = ItemStore()
        for a in actions:
            action_store.add(a)
        words = ['John']
        pos_tags = ['NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, action_store)
        parser.start(list(zip(words, pos_tags)))

        with pytest.raises(KeyError):
            parser.push_nt('S')

    def test_do_shift_action(self):
        words = ['John', 'loves', 'Mary']
        pos_tags = ['NNP', 'VBZ', 'NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)
        parser.start(list(zip(words, pos_tags)))
        parser.push_nt('S')
        parser.push_nt('NP')

        parser.shift()

        assert len(parser.stack_buffer) == 3
        last = parser.stack_buffer[-1]
        assert last == 'John'
        assert parser.input_buffer == words[1:]
        assert len(parser.action_history) == 3
        assert parser.action_history[-1] == ShiftAction()
        assert not parser.finished

    def test_do_illegal_shift_action(self):
        words = ['John']
        pos_tags = ['NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        # No open nonterminal
        parser.start(list(zip(words, pos_tags)))
        with pytest.raises(IllegalActionError):
            parser.shift()

        # Buffer is empty
        parser.start(list(zip(words, pos_tags)))
        parser.push_nt('S')
        parser.shift()
        with pytest.raises(IllegalActionError):
            parser.shift()

    def test_do_reduce_action(self):
        words = ['John', 'loves', 'Mary']
        pos_tags = ['NNP', 'VBZ', 'NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)
        parser.start(list(zip(words, pos_tags)))
        parser.push_nt('S')
        parser.push_nt('NP')
        parser.shift()
        prev_input_buffer = parser.input_buffer

        parser.reduce()

        assert len(parser.stack_buffer) == 2
        last = parser.stack_buffer[-1]
        assert isinstance(last, Tree)
        assert last.label() == 'NP'
        assert len(last) == 1
        assert last[0] == 'John'
        assert parser.input_buffer == prev_input_buffer
        assert len(parser.action_history) == 4
        assert parser.action_history[-1] == ReduceAction()
        assert not parser.finished

    def test_do_illegal_reduce_action(self):
        words = ['John', 'loves']
        pos_tags = ['NNP', 'VBZ']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        # Top of stack is an open nonterminal
        parser.start(list(zip(words, pos_tags)))
        parser.push_nt('S')
        with pytest.raises(IllegalActionError):
            parser.reduce()

        # Buffer is not empty and REDUCE will finish parsing
        parser.start(list(zip(words, pos_tags)))
        parser.push_nt('S')
        parser.shift()
        with pytest.raises(IllegalActionError):
            parser.reduce()

    def test_do_action_when_not_started(self):
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        with pytest.raises(RuntimeError):
            parser.push_nt('S')
        with pytest.raises(RuntimeError):
            parser.shift()
        with pytest.raises(RuntimeError):
            parser.reduce()

    def test_forward(self):
        words = ['John', 'loves', 'Mary']
        pos_tags = ['NNP', 'VBZ', 'NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)
        parser.start(list(zip(words, pos_tags)))
        parser.push_nt('S')
        parser.push_nt('NP')
        parser.shift()
        parser.reduce()

        action_logprobs = parser()

        assert isinstance(action_logprobs, Variable)
        assert action_logprobs.size() == (len(self.action_store),)
        sum_prob = action_logprobs.exp().sum().data[0]
        assert 0.999 <= sum_prob <= 1.001

    def test_forward_with_illegal_actions(self):
        words = ['John', 'loves', 'Mary']
        pos_tags = ['NNP', 'VBZ', 'NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)
        parser.start(list(zip(words, pos_tags)))

        action_probs = parser().exp().data

        assert action_probs[self.action_store[NonTerminalAction('S')]] > 0.
        assert action_probs[self.action_store[NonTerminalAction('NP')]] > 0.
        assert action_probs[self.action_store[NonTerminalAction('VP')]] > 0.
        assert -0.001 <= action_probs[self.action_store[ShiftAction()]] <= 0.001
        assert -0.001 <= action_probs[self.action_store[ReduceAction()]] <= 0.001

    def test_forward_when_not_started(self):
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)

        with pytest.raises(RuntimeError):
            parser()

    def test_finished(self):
        words = ['John', 'loves', 'Mary']
        pos_tags = ['NNP', 'VBZ', 'NNP']
        parser = DiscRNNGrammar(self.word2id, self.pos2id, self.nt2id, self.action_store)
        exp_parse_tree = Tree('S', [Tree('NP', ['John']),
                                    Tree('VP', ['loves', Tree('NP', ['Mary'])])])

        parser.start(list(zip(words, pos_tags)))
        parser.push_nt('S')
        parser.push_nt('NP')
        parser.shift()
        parser.reduce()
        parser.push_nt('VP')
        parser.shift()
        parser.push_nt('NP')
        parser.shift()
        parser.reduce()
        parser.reduce()
        parser.reduce()

        assert parser.finished
        parse_tree = parser.stack_buffer[-1]
        assert parse_tree == exp_parse_tree
        with pytest.raises(RuntimeError):
            parser()
        with pytest.raises(RuntimeError):
            parser.push_nt('NP')
        with pytest.raises(RuntimeError):
            parser.shift()
        with pytest.raises(RuntimeError):
            parser.reduce()
