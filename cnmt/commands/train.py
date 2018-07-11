import argparse
from pathlib import Path

from cnmt.train.train import train


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
    parser.add_argument('--epoch', type=int, default=20,
                        help='')
    parser.add_argument('--source-vocab', type=Path, required=True,
                        help='')
    parser.add_argument('--target-vocab', type=Path, required=True,
                        help='')
    parser.add_argument('--training-source', type=Path, required=True,
                        help='')
    parser.add_argument('--training-ga', type=Path, required=True,
                        help='')
    parser.add_argument('--training-wo', type=Path, required=True,
                        help='')
    parser.add_argument('--training-ni', type=Path, required=True,
                        help='')
    parser.add_argument('--training-ga2', type=Path, required=True,
                        help='')
    parser.add_argument('--training-target', type=Path, required=True,
                        help='')
    parser.add_argument('--validation-source', type=Path, default=None,
                        help='')
    parser.add_argument('--validation-ga', type=Path, default=None,
                        help='')
    parser.add_argument('--validation-wo', type=Path, default=None,
                        help='')
    parser.add_argument('--validation-ni', type=Path, default=None,
                        help='')
    parser.add_argument('--validation-ga2', type=Path, default=None,
                        help='')
    parser.add_argument('--validation-target', type=Path, default=None,
                        help='')
    parser.add_argument('--loss-plot-file', type=Path, default='loss.png',
                        help='')
    parser.add_argument('--bleu-plot-file', type=Path, default='bleu.png',
                        help='')
    parser.add_argument('--resume-file', type=Path, default=None,
                        help='')
    parser.add_argument('--extension-trigger', type=int, default=100,
                        help='The number of iterations to trigger extensions')
    parser.add_argument('--max-translation-length', type=int, default=100,
                        help='')


def run(args: argparse.Namespace):
    """Run the command."""
    train(args)
