import argparse
from pathlib import Path

from cnmt.eval.eval import evaluate


def define_parser(parser: argparse.ArgumentParser):
    """Define command specific options.

    See `cnmt.train.train.ConstArguments`
    """
    parser.add_argument('--source-vocabulary-size', type=int, default=40000,
                        help='The number of words of source language')
    parser.add_argument('--source-word-embeddings-size', type=int, default=640,
                        help='')
    parser.add_argument('--encoder-hidden-layer-size', type=int, default=1024,
                        help='')
    parser.add_argument('--encoder-num-steps', type=int, default=1,
                        help='')
    parser.add_argument('--encoder-dropout', type=float, default=0.1,
                        help='')
    parser.add_argument('--target-vocabulary-size', type=int, default=40000,
                        help='')
    parser.add_argument('--target-word-embeddings-size', type=int, default=640,
                        help='')
    parser.add_argument('--decoder-hidden-layer-size', type=int, default=1024,
                        help='')
    parser.add_argument('--attention-hidden-layer-size', type=int,
                        default=1024, help='')
    parser.add_argument('--maxout-layer-size', type=int, default=512,
                        help='')

    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (-1 means CPU)')
    parser.add_argument('--minibatch-size', type=int, default=64,
                        help='')
    parser.add_argument('--source-vocab', type=Path, required=True,
                        help='')
    parser.add_argument('--target-vocab', type=Path, required=True,
                        help='')
    parser.add_argument('--source', type=Path, required=True,
                        help='')
    parser.add_argument('--ga-file', type=Path, required=True,
                        help='')
    parser.add_argument('--wo-file', type=Path, required=True,
                        help='')
    parser.add_argument('--ni-file', type=Path, required=True,
                        help='')
    parser.add_argument('--ga2-file', type=Path, required=True,
                        help='')
    parser.add_argument('--target', type=Path, required=True,
                        help='')
    parser.add_argument('--translation-output-file', type=Path,
                        default='output.txt', help='')
    parser.add_argument('--models', nargs='+', type=Path, required=True,
                        help='best_bleu.npz')
    parser.add_argument('--max-translation-length', type=int, default=100,
                        help='')


def run(args: argparse.Namespace):
    """Run the command."""
    evaluate(args)
