import pytest
from nltk.tree import Tree

from rnng.oracle import DiscriminativeOracle, GenerativeOracle, OracleDataset
from rnng.utils import ItemStore
from rnng.actions import ShiftAction, ReduceAction, NonTerminalAction, GenerateAction


class TestDiscOracle:
    def test_from_parsed_sent(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = [
            NonTerminalAction('S'),
            NonTerminalAction('NP'),
            ShiftAction(),
            ReduceAction(),
            NonTerminalAction('VP'),
            ShiftAction(),
            NonTerminalAction('NP'),
            ShiftAction(),
            ReduceAction(),
            ReduceAction(),
            ReduceAction(),
        ]
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']
        expected_words = ['John', 'loves', 'Mary']

        oracle = DiscriminativeOracle.from_parsed_sentence(Tree.fromstring(s))

        assert isinstance(oracle, DiscriminativeOracle)
        assert oracle.actions == expected_actions
        assert oracle.pos_tags == expected_pos_tags
        assert oracle.words == expected_words

    def test_from_string(self):
        s = 'asdf fdsa\nNNP VBZ\nNT(S)\nSHIFT\nSHIFT\nREDUCE'

        oracle = DiscriminativeOracle.from_string(s)

        assert isinstance(oracle, DiscriminativeOracle)
        assert oracle.words == ['asdf', 'fdsa']
        assert oracle.pos_tags == ['NNP', 'VBZ']
        assert oracle.actions == [NonTerminalAction('S'),
                                  ShiftAction(),
                                  ShiftAction(),
                                  ReduceAction()]

    def test_from_string_too_short(self):
        s = 'asdf asdf\nNT(S)\nSHIFT\nSHIFT\nREDUCE'

        with pytest.raises(ValueError):
            DiscriminativeOracle.from_string(s)


class TestGenOracle:
    def test_from_parsed_sent(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = [
            NonTerminalAction('S'),
            NonTerminalAction('NP'),
            GenerateAction('John'),
            ReduceAction(),
            NonTerminalAction('VP'),
            GenerateAction('loves'),
            NonTerminalAction('NP'),
            GenerateAction('Mary'),
            ReduceAction(),
            ReduceAction(),
            ReduceAction()
        ]
        expected_words = ['John', 'loves', 'Mary']
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']

        oracle = GenerativeOracle.from_parsed_sentence(Tree.fromstring(s))

        assert isinstance(oracle, GenerativeOracle)
        assert oracle.actions == expected_actions
        assert oracle.words == expected_words
        assert oracle.pos_tags == expected_pos_tags

    def test_from_string(self):
        s = 'NNP VBZ\nNT(S)\nGEN(asdf)\nGEN(fdsa)\nREDUCE'

        oracle = GenerativeOracle.from_string(s)

        assert isinstance(oracle, GenerativeOracle)
        assert oracle.words == ['asdf', 'fdsa']
        assert oracle.pos_tags == ['NNP', 'VBZ']
        assert oracle.actions == [NonTerminalAction('S'),
                                  GenerateAction('asdf'),
                                  GenerateAction('fdsa'),
                                  ReduceAction()]

    def test_from_string_too_short(self):
        s = 'NT(S)'

        with pytest.raises(ValueError):
            GenerativeOracle.from_string(s)


class TestOracleDataset:
    bracketed_sents = [
        '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))',
        '(S (NP (NNP Mary)) (VP (VBZ hates) (NP (NNP John))))'  # poor John
    ]
    words = {'John', 'loves', 'hates', 'Mary'}
    pos_tags = {'NNP', 'VBZ'}
    nt_labels = {'S', 'NP', 'VP'}
    actions = {NonTerminalAction('S'),
               NonTerminalAction('NP'),
               NonTerminalAction('VP'),
               ShiftAction(),
               ReduceAction()}

    def test_init(self):
        oracles = [DiscriminativeOracle.from_parsed_sentence(Tree.fromstring(s))
                   for s in self.bracketed_sents]

        dataset = OracleDataset(oracles)

        assert isinstance(dataset.word_store, ItemStore)
        assert set(dataset.word_store) == self.words
        assert isinstance(dataset.pos_store, ItemStore)
        assert set(dataset.pos_store) == self.pos_tags
        assert isinstance(dataset.nt_store, ItemStore)
        assert set(dataset.nt_store) == self.nt_labels
        assert isinstance(dataset.action_store, ItemStore)
        assert set(dataset.action_store) == self.actions

    def test_getitem(self):
        oracles = [DiscriminativeOracle.from_parsed_sentence(Tree.fromstring(s))
                   for s in self.bracketed_sents]

        dataset = OracleDataset(oracles)

        assert oracles[0] is dataset[0]
        assert oracles[1] is dataset[1]

    def test_len(self):
        oracles = [DiscriminativeOracle.from_parsed_sentence(Tree.fromstring(s))
                   for s in self.bracketed_sents]

        dataset = OracleDataset(oracles)

        assert len(dataset) == len(oracles)
