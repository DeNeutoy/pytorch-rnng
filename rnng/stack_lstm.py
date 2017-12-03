from typing import List, Sized, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable


class EmptyStackError(Exception):
    def __init__(self):
        super().__init__('stack is already empty')


class StackLSTM(nn.Module, Sized):
    BATCH_SIZE = 1
    SEQ_LEN = 1

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.) -> None:
        if num_layers < 1:
            raise ValueError('number of layers is at least 1')

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.h0 = nn.Parameter(torch.FloatTensor(num_layers, self.BATCH_SIZE, hidden_size))
        self.c0 = nn.Parameter(torch.FloatTensor(num_layers, self.BATCH_SIZE, hidden_size))
        init_states = (self.h0, self.c0)
        self._state_history = [init_states]
        self._output_history = []  # type: List[Variable]

    def forward(self, inputs: Variable) -> Tuple[Variable, Variable]:
        # inputs: input_size
        assert self._state_history

        # Set seq_len and batch_size to 1
        inputs = inputs.view(self.SEQ_LEN, self.BATCH_SIZE, inputs.numel())
        next_outputs, next_states = self.lstm(inputs, self._state_history[-1])
        self._state_history.append(next_states)
        self._output_history.append(next_outputs)
        return next_states

    def push(self, *args, **kwargs):
        return self(*args, **kwargs)

    def pop(self) -> Tuple[Variable, Variable]:
        if len(self._state_history) > 1:
            self._output_history.pop()
            return self._state_history.pop()
        else:
            raise EmptyStackError()

    def reset(self) -> None:
        self._state_history = [(self.h0, self.c0)]
        self._output_history = []  # type: List[Variable]

    @property
    def top(self) -> Variable:
        # outputs: hidden_size
        return self._output_history[-1].squeeze() if self._output_history else None

    def __repr__(self) -> str:
        res = ('{}(input_size={input_size}, hidden_size={hidden_size}, '
               'num_layers={num_layers}, dropout={dropout})')
        return res.format(self.__class__.__name__, **self.__dict__)

    def __len__(self):
        return len(self._output_history)
