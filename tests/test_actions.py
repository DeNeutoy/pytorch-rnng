import pytest

from rnng.actions import ShiftAction, ReduceAction, NonTerminalAction, GenerateAction


class TestShiftAction:
    as_str = 'SHIFT'

    def test_to_string(self):
        a = ShiftAction()
        assert str(a) == self.as_str

    def test_hash(self):
        a = ShiftAction()
        assert hash(a) == hash(self.as_str)

    def test_eq(self):
        assert ShiftAction() == ShiftAction()
        assert ShiftAction() != ReduceAction()

    def test_from_string(self):
        a = ShiftAction.from_string(self.as_str)
        assert isinstance(a, ShiftAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            ShiftAction.from_string('asdf')


class TestReduceAction:
    as_str = 'REDUCE'

    def test_to_string(self):
        a = ReduceAction()
        assert str(a) == self.as_str

    def test_hash(self):
        a = ReduceAction()
        assert hash(a) == hash(self.as_str)

    def test_eq(self):
        assert ReduceAction() == ReduceAction()
        assert ReduceAction() != ShiftAction()

    def test_from_string(self):
        a = ReduceAction.from_string(self.as_str)
        assert isinstance(a, ReduceAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            ReduceAction.from_string('asdf')


class TestNTAction:
    as_str = 'NT({label})'

    def test_to_string(self):
        label = 'NP'
        a = NonTerminalAction(label)
        assert str(a) == self.as_str.format(label=label)

    def test_hash(self):
        label = 'NP'
        a = NonTerminalAction(label)
        assert hash(a) == hash(self.as_str.format(label=label))

    def test_eq(self):
        a = NonTerminalAction('NP')
        assert a == NonTerminalAction(a.label)
        assert a != NonTerminalAction('asdf')
        assert a != ShiftAction()

    def test_from_string(self):
        label = 'NP'
        a = NonTerminalAction.from_string(self.as_str.format(label=label))
        assert isinstance(a, NonTerminalAction)
        assert a.label == label

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            NonTerminalAction.from_string('asdf')


class TestGenAction:
    as_str = 'GEN({word})'

    def test_to_string(self):
        word = 'asdf'
        a = GenerateAction(word)
        assert str(a) == self.as_str.format(word=word)

    def test_hash(self):
        word = 'asdf'
        a = GenerateAction(word)
        assert hash(a) == hash(self.as_str.format(word=word))

    def test_eq(self):
        a = GenerateAction('asdf')
        assert a == GenerateAction(a.word)
        assert a != GenerateAction('fdsa')
        assert a != ReduceAction()

    def test_from_string(self):
        word = 'asdf'
        a = GenerateAction.from_string(self.as_str.format(word=word))
        assert isinstance(a, GenerateAction)
        assert a.word == word

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            GenerateAction.from_string('asdf')
