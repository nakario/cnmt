import argparse
from logging import getLogger
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import chainer
from chainer.dataset import to_device
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import Variable
import matplotlib
from nltk.translate import bleu_score
import numpy as np
from progressbar import ProgressBar

from cnmt.misc.constants import EOS, eos
from cnmt.misc.constants import NIL, nil
from cnmt.misc.constants import PAD
from cnmt.misc.constants import REA, rea
from cnmt.misc.constants import UNK, unk
from cnmt.misc.constants import UNS, uns
from cnmt.misc.constants import WRI, wri
from cnmt.misc.functions import flen
from cnmt.misc.typing import ndarray
from cnmt.models.encdec import EncoderDecoder


logger = getLogger(__name__)
matplotlib.use('Agg')


class ConstArguments(NamedTuple):
    # Encoder-Decoder arguments
    source_vocabulary_size: int
    source_word_embeddings_size: int
    encoder_hidden_layer_size: int
    encoder_num_steps: int
    encoder_dropout: float
    target_vocabulary_size: int
    target_word_embeddings_size: int
    decoder_hidden_layer_size: int
    attention_hidden_layer_size: int
    maxout_layer_size: int

    gpu: int
    minibatch_size: int
    epoch: int
    source_vocab: Path
    target_vocab: Path
    training_source: Path
    training_ga: Path
    training_wo: Path
    training_ni: Path
    training_ga2: Path
    training_target: Path
    validation_source: Optional[Path]
    validation_ga: Optional[Path]
    validation_wo: Optional[Path]
    validation_ni: Optional[Path]
    validation_ga2: Optional[Path]
    validation_target: Optional[Path]
    loss_plot_file: Path
    bleu_plot_file: Path
    resume_file: Optional[Path]
    extension_trigger: int
    max_translation_length: int


def decode_bpe(sentence: List[str], separator: str = '._@@@') -> List[str]:
    decoded = []
    merge_to_previous = False
    for word in sentence:
        if merge_to_previous:
            decoded[-1] = decoded[-1][:-len(separator)] + word
        else:
            decoded.append(word)

        merge_to_previous = word.endswith(separator)
    return decoded


class CalculateBleu(chainer.training.Extension):
    triger = (1, 'epoch')
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self,
            validation_iter: chainer.iterators.SerialIterator,
            model: EncoderDecoder,
            converter: Callable[
                    [List[Tuple[
                        np.ndarray,
                        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray
                    ]], Optional[int]],
                    Tuple[
                        ndarray,
                        ndarray, ndarray, ndarray, ndarray,
                        ndarray
                    ]
            ],
            id2word: Dict[int, str],
            key: str,
            device: int,
            max_translation_length: int = 100
    ):
        self.iter = validation_iter
        self.model = model
        self.converter = converter
        self.id2word = id2word
        self.device = device
        self.key = key
        self.max_translation_length = max_translation_length
        self.best_bleu = 0.0

    def __call__(self, trainer):
        list_of_references: List[List[List[str]]] = []
        hypotheses: List[List[str]] = []
        self.iter.reset()
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for minibatch in self.iter:
                target_sentences: List[np.ndarray] = tuple(zip(*minibatch))[-1]
                list_of_references.extend([
                    [decode_bpe([
                        self.id2word.get(id_, unk)
                        for id_ in sentence.tolist()
                    ])] for sentence in target_sentences
                ])
                converted = self.converter(minibatch, self.device)
                source, ga, wo, ni, ga2, target = converted
                results = self.model.translate(
                    source, ga, wo, ni, ga2,
                    max_translation_length=self.max_translation_length
                )
                hypotheses.extend([
                    decode_bpe([
                        self.id2word.get(id_, unk)
                        for id_ in sentence.tolist()[:-1]
                    ]) for sentence in results
                ])
        bleu = bleu_score.corpus_bleu(
            list_of_references,
            hypotheses
        )
        chainer.report({self.key: bleu})
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            print("saving model...")
            chainer.serializers.save_npz("best_bleu.npz", self.model)
            print("saved model.")


def convert(
        minibatch: List[Tuple[
            np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray
        ]],
        device: Optional[int]
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    # Append eos to the end of sentence
    eos_ = np.array([EOS], 'i')
    (
        src_batch,
        ga_batch, wo_batch, ni_batch, ga2_batch,
        tgt_batch
    ) = zip(*minibatch)
    with chainer.no_backprop_mode():
        src_sentences = \
            [Variable(np.hstack((sentence, eos_))) for sentence in src_batch]
        ga_sentences = \
            [Variable(np.hstack((sentence, eos_))) for sentence in ga_batch]
        wo_sentences = \
            [Variable(np.hstack((sentence, eos_))) for sentence in wo_batch]
        ni_sentences = \
            [Variable(np.hstack((sentence, eos_))) for sentence in ni_batch]
        ga2_sentences = \
            [Variable(np.hstack((sentence, eos_))) for sentence in ga2_batch]
        tgt_sentences = \
            [Variable(np.hstack((sentence, eos_))) for sentence in tgt_batch]

        src_block = F.pad_sequence(src_sentences, padding=PAD).data
        ga_block = F.pad_sequence(ga_sentences, padding=PAD).data
        wo_block = F.pad_sequence(wo_sentences, padding=PAD).data
        ni_block = F.pad_sequence(ni_sentences, padding=PAD).data
        ga2_block = F.pad_sequence(ga2_sentences, padding=PAD).data
        tgt_block = F.pad_sequence(tgt_sentences, padding=PAD).data

    return (
        to_device(device, src_block),
        to_device(device, ga_block),
        to_device(device, wo_block),
        to_device(device, ni_block),
        to_device(device, ga2_block),
        to_device(device, tgt_block)
    )


def load_vocab(vocab_file: Path, size: int) -> Dict[str, int]:
    """Create a vocabulary from a file.

    The file specified by `vocab` must be contain one word per line.
    """

    assert vocab_file.exists()

    words = [unk, eos, nil, wri, rea, uns]
    with open(vocab_file) as f:
        words += [line.strip() for line in f]

    vocab = {word: index for index, word in enumerate(words) if index < size}
    assert vocab[unk] == UNK
    assert vocab[eos] == EOS
    assert vocab[nil] == NIL
    assert vocab[wri] == WRI
    assert vocab[rea] == REA
    assert vocab[uns] == UNS

    return vocab


def load_data(
        source: Path,
        ga_file: Path,
        wo_file: Path,
        ni_file: Path,
        ga2_file: Path,
        target: Path,
        source_vocab: Dict[str, int],
        target_vocab: Dict[str, int]
) -> List[Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]]:
    assert source.exists()
    assert ga_file.exists()
    assert wo_file.exists()
    assert ni_file.exists()
    assert ga2_file.exists()
    assert target.exists()

    data = []

    file_len = flen(source)
    assert file_len == flen(ga_file)
    assert file_len == flen(wo_file)
    assert file_len == flen(ni_file)
    assert file_len == flen(ga2_file)
    assert file_len == flen(target)

    logger.info(f'loading {source.absolute()} and {target.absolute()}')
    src = open(source)
    ga = open(ga_file)
    wo = open(wo_file)
    ni = open(ni_file)
    ga2 = open(ga2_file)
    tgt = open(target)

    bar = ProgressBar()
    for i, (s, g, w, n, g2, t) in bar(
            enumerate(zip(src, ga, wo, ni, ga2, tgt)),
            max_value=file_len
    ):
        s_words = s.strip().split()
        g_words = g.strip().split()
        w_words = w.strip().split()
        n_words = n.strip().split()
        g2_words = g2.strip().split()
        t_words = t.strip().split()

        s_array = \
            np.array([source_vocab.get(w, UNK) for w in s_words], 'i')
        g_array = \
            np.array([source_vocab.get(w, UNK) for w in g_words], 'i')
        w_array = \
            np.array([source_vocab.get(w, UNK) for w in w_words], 'i')
        n_array = \
            np.array([source_vocab.get(w, UNK) for w in n_words], 'i')
        g2_array = \
            np.array([source_vocab.get(w, UNK) for w in g2_words], 'i')
        t_array = \
            np.array([target_vocab.get(w, UNK) for w in t_words], 'i')
        data.append((s_array, g_array, w_array, n_array, g2_array, t_array))

    src.close()
    ga.close()
    wo.close()
    ni.close()
    ga2.close()
    tgt.close()

    return data


def train(args: argparse.Namespace):
    cargs = ConstArguments(**vars(args))
    logger.info(f'cargs: {cargs}')
    model = EncoderDecoder(
        cargs.source_vocabulary_size,
        cargs.source_word_embeddings_size,
        cargs.encoder_hidden_layer_size,
        cargs.encoder_num_steps,
        cargs.encoder_dropout,
        cargs.target_vocabulary_size,
        cargs.target_word_embeddings_size,
        cargs.decoder_hidden_layer_size,
        cargs.attention_hidden_layer_size,
        cargs.maxout_layer_size
    )

    if cargs.gpu >= 0:
        chainer.cuda.get_device_from_id(cargs.gpu).use()
        model.to_gpu(cargs.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))
    optimizer.add_hook(chainer.optimizer.WeightDecay(10e-6))

    source_vocab = load_vocab(cargs.source_vocab, cargs.source_vocabulary_size)
    target_vocab = load_vocab(cargs.target_vocab, cargs.target_vocabulary_size)

    training_data = load_data(
        cargs.training_source,
        cargs.training_ga,
        cargs.training_wo,
        cargs.training_ni,
        cargs.training_ga2,
        cargs.training_target,
        source_vocab,
        target_vocab
    )

    training_iter = chainer.iterators.SerialIterator(training_data,
                                                     cargs.minibatch_size)
    converter = convert
    updater = training.StandardUpdater(
        training_iter, optimizer, converter=converter, device=cargs.gpu)
    trainer = training.Trainer(updater, (cargs.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(
        trigger=(cargs.extension_trigger, 'iteration')
    ))
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
             'validation/main/bleu', 'elapsed_time']
        ),
        trigger=(cargs.extension_trigger, 'iteration')
    )
    trainer.extend(
        extensions.snapshot(),
        trigger=(cargs.extension_trigger * 50, 'iteration'))
    # Don't set `trigger` argument to `dump_graph`
    trainer.extend(extensions.dump_graph('main/loss'))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            trigger=(cargs.extension_trigger, 'iteration'),
            file_name=str(cargs.loss_plot_file)
        ))
        trainer.extend(extensions.PlotReport(
            ['validation/main/bleu'],
            trigger=(cargs.extension_trigger, 'iteration'),
            file_name=str(cargs.bleu_plot_file)
        ))
        trainer.extend(extensions.PlotReport(
            ['main/lambda'],
            trigger=(cargs.extension_trigger, 'iteration'),
            file_name='lambda.png'
        ))
        trainer.extend(extensions.PlotReport(
            ['main/gate'],
            trigger=(cargs.extension_trigger, 'iteration'),
            file_name='gate.png'
        ))
        trainer.extend(extensions.PlotReport(
            ['main/beta'],
            trigger=(cargs.extension_trigger, 'iteration'),
            file_name='beta.png'
        ))
        trainer.extend(extensions.PlotReport(
            ['main/max_score'],
            trigger=(cargs.extension_trigger, 'iteration'),
            file_name='max_score.png'
        ))
    else:
        logger.warning('PlotReport is not available.')

    if cargs.validation_source is not None and \
            cargs.validation_target is not None:
        validation_data = load_data(
            cargs.validation_source,
            cargs.validation_ga,
            cargs.validation_wo,
            cargs.validation_ni,
            cargs.validation_ga2,
            cargs.validation_target,
            source_vocab,
            target_vocab
        )

        v_iter1 = chainer.iterators.SerialIterator(
            validation_data,
            cargs.minibatch_size,
            repeat=False,
            shuffle=False
        )
        v_iter2 = chainer.iterators.SerialIterator(
            validation_data,
            cargs.minibatch_size,
            repeat=False,
            shuffle=False
        )

        source_word = {index: word for word, index in source_vocab.items()}
        target_word = {index: word for word, index in target_vocab.items()}

        trainer.extend(extensions.Evaluator(
            v_iter1, model, converter=converter, device=cargs.gpu
        ), trigger=(cargs.extension_trigger * 5, 'iteration'))
        trainer.extend(CalculateBleu(
            v_iter2, model, converter=converter, device=cargs.gpu,
            key='validation/main/bleu', id2word=target_word,
            max_translation_length=cargs.max_translation_length
        ), trigger=(cargs.extension_trigger * 5, 'iteration'))

        validation_size = len(validation_data)

        @chainer.training.make_extension(trigger=(200, 'iteration'))
        def translate(_):
            data = validation_data[np.random.choice(validation_size)]
            converted = converter([data], cargs.gpu)
            source, ga, wo, ni, ga2, target = converted
            result = model.translate(
                source, ga, wo, ni, ga2,
                max_translation_length=cargs.max_translation_length
            )[0].reshape((1, -1))

            source_sentence = ' '.join(decode_bpe(
                [source_word[int(word)] for word in source[0]]
            ))
            target_sentence = ' '.join(decode_bpe(
                [target_word[int(word)] for word in target[0]]
            ))
            result_sentence = ' '.join(decode_bpe(
                [target_word[int(word)] for word in result[0]]
            ))
            logger.info('# source       : ' + source_sentence)
            logger.info('# output       : ' + result_sentence)
            logger.info('# reference    : ' + target_sentence)

        trainer.extend(
            translate,
            trigger=(cargs.extension_trigger * 5, 'iteration')
        )

    trainer.extend(
        extensions.ProgressBar(
            update_interval=max(cargs.extension_trigger // 5, 1)
        )
    )

    if cargs.resume_file is not None:
        chainer.serializers.load_npz(cargs.resume_file, trainer)

    print('start training')

    trainer.run()
