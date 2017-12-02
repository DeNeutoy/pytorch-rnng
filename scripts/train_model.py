#!/usr/bin/env python -O

import os
import sys

from argparse import ArgumentParser

from torch.utils.data.dataloader import DataLoader
from torch.optim.sgd import SGD

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from rnng.training import train_early_stopping
from rnng.models import DiscriminativeRnnGrammar
from rnng.oracle import DiscriminativeOracle
from rnng.oracle import OracleDataset


def read_oracles_from_file(oracle_class, filename):
    with open(filename) as f:
        oracles = [oracle_class.from_string(oracle_str)
                   for oracle_str in f.read().split('\n\n') if oracle_str]
    return oracles


def train_model(training_file: str):

    oracles = read_oracles_from_file(DiscriminativeOracle, training_file)
    dataset = OracleDataset(oracles)
    dataset.load()
    dataset_loader = DataLoader(dataset, collate_fn=lambda x: x[0])

    parser = DiscriminativeRnnGrammar(action_store=dataset.action_store,
                                      word2id=dataset.word_store,
                                      pos2id=dataset.pos_store,
                                      non_terminal2id=dataset.nt_store)
    optimiser = SGD(parser.parameters(), 0.1)

    train_early_stopping(dataset_loader, dataset_loader, parser, optimiser)


if __name__ == "__main__":
    argparser = ArgumentParser(description='Unkify an oracle file based on the given training oracle')
    argparser.add_argument('training_file', help='path to training oracle file')

    args = argparser.parse_args()
    train_model(args.training_file)