#!/usr/bin/env python -O

import os
import sys

from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from rnng.oracle import DiscriminativeOracle, GenerativeOracle
from rnng.corpus import Treebank

parser = ArgumentParser(description='Get oracle for a given corpus')
parser.add_argument('--corpus', help='path to corpus file')
parser.add_argument('--generative', '-g', action='store_true', default=False,
                    help='get oracle for the generative parser')
parser.add_argument('--no-lowercase', action='store_false', default=True, dest='lowercase',
                    help='do not lowercase words')
args = parser.parse_args()

# python scripts/get_oracle.py --corpus "/Users/markn/pytorch-rnng/data/penn-tree-bank/treebank/combined/.*\.mrg"

treebank = Treebank(args.corpus, lowercase=args.lowercase)
for parsed_sent in treebank.parsed_sentences():
    oracle_class = GenerativeOracle if args.generative else DiscriminativeOracle
    oracle = oracle_class.from_parsed_sentence(parsed_sent)
    print(oracle, end='\n\n')
