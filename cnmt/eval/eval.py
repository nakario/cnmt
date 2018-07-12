import argparse
from pathlib import Path
from logging import getLogger
from typing import List
from typing import NamedTuple

import chainer
import matplotlib
from nltk.translate import bleu_score
from progressbar import ProgressBar

from cnmt.misc.constants import unk
from cnmt.models.encdec import EncoderDecoder
from cnmt.models.encdec import translate_ensemble
from cnmt.train.train import decode_bpe
from cnmt.train.train import convert
from cnmt.train.train import load_vocab
from cnmt.train.train import load_data


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
    source_vocab: Path
    target_vocab: Path
    source: Path
    ga_file: Path
    wo_file: Path
    ni_file: Path
    ga2_file: Path
    target: Path
    translation_output_file: Path
    models: List[Path]
    max_translation_length: int
    beam_width: int


def evaluate(args: argparse.Namespace):
    cargs = ConstArguments(**vars(args))
    logger.info(f'cargs: {cargs}')

    models = []
    for model_file in cargs.models:
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
            cargs.maxout_layer_size,
            dynamic_attention=True
        )

        if cargs.gpu >= 0:
            chainer.cuda.get_device_from_id(cargs.gpu).use()
            model.to_gpu(cargs.gpu)

        chainer.serializers.load_npz(model_file, model)

        models.append(model)

    source_vocab = load_vocab(cargs.source_vocab, cargs.source_vocabulary_size)
    target_vocab = load_vocab(cargs.target_vocab, cargs.target_vocabulary_size)

    converter = convert

    validation_data = load_data(
        cargs.source,
        cargs.ga_file,
        cargs.wo_file,
        cargs.ni_file,
        cargs.ga2_file,
        cargs.target,
        source_vocab,
        target_vocab
    )

    v_iter = chainer.iterators.SerialIterator(
        validation_data,
        1,
        repeat=False,
        shuffle=False
    )

    target_sentences: List[List[List[str]]]
    with open(cargs.target) as f:
        target_sentences = \
            list(map(lambda x: [x.strip().split()], f.readlines()))

    target_word = {index: word for word, index in target_vocab.items()}

    list_of_references: List[List[List[str]]] = []
    hypotheses: List[List[str]] = []
    v_iter.reset()
    print("start translation")
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        bar = ProgressBar(max_value=len(target_sentences))
        for i, minibatch in bar(enumerate(v_iter)):
            list_of_references.append(target_sentences[i])
            converted = converter(minibatch, cargs.gpu)
            source, ga, wo, ni, ga2, target = converted
            results = translate_ensemble(
                models, source, ga, wo, ni, ga2,
                translation_limit=cargs.max_translation_length,
                beam_width=cargs.beam_width
            )
            hypotheses.extend([
                decode_bpe([
                    target_word.get(id_, unk)
                    for id_ in sentence.tolist()[:-1]
                ]) for sentence in results
            ])
    print("start write file")
    assert len(list_of_references) == len(hypotheses)
    with open(cargs.translation_output_file, 'w') as output:
        for i in range(len(list_of_references)):
            output.write(f"src: {' '.join(list_of_references[i][0])}\n")
            output.write(f"out: {' '.join(hypotheses[i])}\n\n")
    print("start calc bleu")
    bleu = bleu_score.corpus_bleu(
        list_of_references,
        hypotheses
    )
    print(f"BLEU: {bleu}")
