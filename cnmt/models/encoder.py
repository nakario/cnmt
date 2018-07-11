from typing import List

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import numpy as np

from cnmt.misc.constants import PAD, NIL
from cnmt.misc.typing import ndarray


class Encoder(chainer.Chain):
    def __init__(self,
                 vocabulary_size: int,
                 word_embeddings_size: int,
                 hidden_layer_size: int,
                 num_steps: int,
                 dropout: float,
                 ignore_label: int = -1):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed_id = L.EmbedID(
                vocabulary_size,
                word_embeddings_size,
                ignore_label=ignore_label
            )
            self.linear_ga = L.Linear(
                word_embeddings_size,
                word_embeddings_size
            )
            self.linear_wo = L.Linear(
                word_embeddings_size,
                word_embeddings_size
            )
            self.linear_ni = L.Linear(
                word_embeddings_size,
                word_embeddings_size
            )
            self.linear_ga2 = L.Linear(
                word_embeddings_size,
                word_embeddings_size
            )
            self.nstep_birnn = L.NStepBiLSTM(
                num_steps,
                word_embeddings_size,
                hidden_layer_size,
                dropout
            )
        self.word_embeddings_size = word_embeddings_size
        self.output_size = hidden_layer_size * 2

    def __call__(
            self,
            source: ndarray,
            ga: ndarray,
            wo: ndarray,
            ni: ndarray,
            ga2: ndarray
    ) -> Variable:
        minibatch_size, max_sentence_size = source.shape
        assert source.shape == ga.shape == wo.shape == ni.shape == ga2.shape

        embedded_source = self.embed_id(source)
        assert embedded_source.shape == \
            (minibatch_size, max_sentence_size, self.word_embeddings_size)
        ga_mask = self.xp.stack(
            [(ga != NIL)] * self.word_embeddings_size, axis=2)
        e_ga = nested_linear(self.embed_id(ga), self.linear_ga) * ga_mask
        wo_mask = self.xp.stack(
            [(wo != NIL)] * self.word_embeddings_size, axis=2)
        e_wo = nested_linear(self.embed_id(wo), self.linear_wo) * wo_mask
        ni_mask = self.xp.stack(
            [(ni != NIL)] * self.word_embeddings_size, axis=2)
        e_ni = nested_linear(self.embed_id(ni), self.linear_ni) * ni_mask
        ga2_mask = self.xp.stack(
            [(ga2 != NIL)] * self.word_embeddings_size, axis=2)
        e_ga2 = nested_linear(self.embed_id(ga2), self.linear_ga2) * ga2_mask
        embedded_source = embedded_source + e_ga + e_wo + e_ni + e_ga2

        embedded_sentences = F.separate(embedded_source, axis=0)
        sentence_masks: List[ndarray] = \
            self.xp.vsplit(source != PAD, minibatch_size)

        masked_sentences: List[Variable] = []
        for sentence, mask in zip(embedded_sentences, sentence_masks):
            masked_sentences.append(sentence[mask.reshape((-1,))])

        encoded_sentences: List[Variable] = \
            self.nstep_birnn(None, None, masked_sentences)[-1]
        assert len(encoded_sentences) == minibatch_size

        encoded: Variable = F.pad_sequence(
            encoded_sentences,
            length=max_sentence_size,
            padding=0.0
        )
        assert encoded.shape == \
            (minibatch_size, max_sentence_size, self.output_size)

        return encoded


def nested_linear(v: Variable, l: L.Linear) -> Variable:
    last = v.shape[-1]
    rest = np.prod(v.shape[:-1])
    reshaped = F.reshape(v, (rest, last))
    transformed = l(reshaped)
    return F.reshape(transformed, v.shape[:-1] + (l.out_size,))
