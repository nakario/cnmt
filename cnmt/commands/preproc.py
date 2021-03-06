from pathlib import Path

from cnmt.preproc.preproc import preproc


def define_parser(parser):
    """Define command specific options."""
    parser.add_argument('source', type=Path,
                        help='path of source document')
    parser.add_argument('target', type=Path,
                        help='path of target document')
    parser.add_argument('output', type=Path,
                        help='path of output directory')
    parser.add_argument('--min-source-len', type=int, default=1,
                        help='')
    parser.add_argument('--max-source-len', type=int, default=70,
                        help='')
    parser.add_argument('--min-target-len', type=int, default=1,
                        help='')
    parser.add_argument('--max-target-len', type=int, default=70,
                        help='')
    parser.add_argument('--source-dev', type=Path,
                        help='')
    parser.add_argument('--target-dev', type=Path,
                        help='')
    parser.add_argument('--source-test', type=Path,
                        help='')
    parser.add_argument('--target-test', type=Path,
                        help='')
    parser.add_argument('--skip-create-bpe', action='store_true',
                        help='')
    parser.add_argument('--skip-bpe-encode', action='store_true',
                        help='')
    parser.add_argument('--skip-make-voc', action='store_true',
                        help='')


def run(args):
    """Run the command."""
    preproc(args)
