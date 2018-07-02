from logging import getLogger
from typing import List

import chainer
from chainer import Variable

from cnmt.misc.typing import ndarray
from cnmt.models.encoder import Encoder
from cnmt.models.decoder import Decoder


logger = getLogger(__name__)


class EncoderDecoder(chainer.Chain):
    def __init__(self,
                 input_vocabulary_size: int,
                 input_word_embeddings_size: int,
                 encoder_hidden_layer_size: int,
                 encoder_num_steps: int,
                 encoder_dropout: float,
                 output_vocabulary_size: int,
                 output_word_embeddings_size: int,
                 decoder_hidden_layer_size: int,
                 attention_hidden_layer_size: int,
                 maxout_layer_size: int):
        super(EncoderDecoder, self).__init__()
        with self.init_scope():
            self.enc = Encoder(input_vocabulary_size,
                               input_word_embeddings_size,
                               encoder_hidden_layer_size,
                               encoder_num_steps,
                               encoder_dropout)
            self.dec = Decoder(output_vocabulary_size,
                               output_word_embeddings_size,
                               decoder_hidden_layer_size,
                               attention_hidden_layer_size,
                               encoder_hidden_layer_size * 2,
                               maxout_layer_size)

    def __call__(
            self,
            source: ndarray,
            target: ndarray
    ) -> Variable:
        # source.shape == (minibatch_size, source_max_sentence_size)
        # target.shape == (minibatch_size, target_max_sentence_size)
        encoded = self.enc(source)
        loss = self.dec(encoded, target)
        chainer.report({'loss': loss}, self)
        return loss

    def translate(
            self,
            sentences: ndarray,
            max_translation_length: int = 100
    ) -> List[ndarray]:
        # sentences.shape == (sentence_count, max_sentence_size)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            encoded = self.enc(sentences)
            translated = self.dec.translate(
                encoded,
                max_translation_length
            )
            return translated
