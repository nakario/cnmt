from typing import List
from typing import Tuple

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Parameter
from chainer import Variable
import numpy as np

from cnmt.misc.constants import EOS
from cnmt.misc.constants import PAD
from cnmt.misc.typing import ndarray
from cnmt.models.attention import AttentionModule
from cnmt.models.attention import DynamicAttentionModule


class Decoder(chainer.Chain):
    def __init__(self,
                 vocabulary_size: int,
                 word_embeddings_size: int,
                 hidden_layer_size: int,
                 attention_hidden_layer_size: int,
                 encoder_output_size: int,
                 maxout_layer_size: int,
                 maxout_pool_size: int = 2,
                 ignore_label: int = -1,
                 dynamic_attention: bool = False):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed_id = L.EmbedID(vocabulary_size,
                                      word_embeddings_size,
                                      ignore_label=ignore_label)
            self.rnn = L.StatelessLSTM(
                word_embeddings_size + encoder_output_size,
                hidden_layer_size
            )
            self.maxout = L.Maxout(word_embeddings_size +
                                   encoder_output_size +
                                   hidden_layer_size,
                                   maxout_layer_size,
                                   maxout_pool_size)
            self.linear = L.Linear(maxout_layer_size, vocabulary_size)
            if dynamic_attention:
                self.attention = DynamicAttentionModule(
                    encoder_output_size,
                    attention_hidden_layer_size,
                    hidden_layer_size,
                    word_embeddings_size
                )
            else:
                self.attention = AttentionModule(
                    encoder_output_size,
                    attention_hidden_layer_size,
                    hidden_layer_size,
                    word_embeddings_size
                )
            self.bos_state = Parameter(
                initializer=self.xp.random.randn(
                    1,
                    hidden_layer_size
                ).astype('f')
            )
        self.vocabulary_size = vocabulary_size
        self.word_embeddings_size = word_embeddings_size
        self.hidden_layer_size = hidden_layer_size
        self.encoder_output_size = encoder_output_size

    def __call__(
            self,
            encoded: Variable,
            target: ndarray
    ) -> Variable:
        minibatch_size, max_sentence_size, encoder_output_size = encoded.shape
        assert encoder_output_size == self.encoder_output_size
        assert target.shape[0] == minibatch_size

        self.setup(encoded)
        cell, state, previous_words = self.get_initial_states(minibatch_size)

        total_loss = Variable(self.xp.array(0, 'f'))
        total_predictions = 0

        for target_id in self.xp.hsplit(target, target.shape[1]):
            target_id = target_id.reshape((minibatch_size,))
            cell, state, context, concatenated = \
                self.advance_one_step(cell, state, previous_words)
            logit, state = self.compute_logit(concatenated, state, context)

            current_sentence_count = self.xp.sum(target_id != PAD)

            loss = F.softmax_cross_entropy(logit, target_id, ignore_label=PAD)
            total_loss += loss * current_sentence_count
            total_predictions += current_sentence_count

            previous_words = target_id

        return total_loss / total_predictions

    def setup(self, encoded: Variable):
        if self.bos_state.array is None:
            self.bos_state.initialize((1, self.hidden_layer_size))
        self.attention.precompute(encoded)

    def get_initial_states(
            self,
            minibatch_size: int
    ) -> Tuple[Variable, Variable, ndarray]:
        cell = Variable(
            self.xp.zeros((minibatch_size, self.hidden_layer_size), 'f')
        )
        state = F.broadcast_to(
            self.bos_state, (minibatch_size, self.hidden_layer_size)
        )
        previous_words = self.xp.full((minibatch_size,), EOS, 'i')
        return cell, state, previous_words

    def advance_one_step(
            self,
            cell: Variable,
            state: Variable,
            previous_words: ndarray
    ) -> Tuple[Variable, Variable, Variable, Variable]:
        minibatch_size = cell.shape[0]
        previous_embedding = self.embed_id(previous_words)
        context = self.attention(state, previous_embedding)
        assert context.shape == (minibatch_size, self.encoder_output_size)
        concatenated = F.concat((previous_embedding, context))
        cell, state = self.rnn(cell, state, concatenated)
        return cell, state, context, concatenated

    def compute_logit(
            self,
            concatenated: Variable,
            state: Variable,
            context: Variable
    ) -> Tuple[Variable, Variable]:
        all_concatenated = F.concat((concatenated, state))
        logit = self.linear(self.maxout(all_concatenated))
        return logit, state

    def translate(
            self,
            encoded: Variable,
            max_length: int = 100
    ) -> List[ndarray]:
        sentence_count = encoded.shape[0]

        self.setup(encoded)
        cell, state, previous_words = self.get_initial_states(sentence_count)

        result = []
        for _ in range(max_length):
            cell, state, context, concatenated = \
                self.advance_one_step(cell, state, previous_words)
            logit, state = self.compute_logit(concatenated, state, context)

            output_id = F.reshape(F.argmax(logit, axis=1), (sentence_count,))
            result.append(output_id)

            previous_words = output_id

        # Remove words after <EOS>
        outputs = F.separate(F.transpose(F.vstack(result)), axis=0)
        assert len(outputs) == sentence_count
        output_sentences = []
        for output in outputs:
            assert output.shape == (max_length,)
            indexes = np.argwhere(output.data == EOS)
            if len(indexes) > 0:
                output = output[:indexes[0, 0] + 1]
            output_sentences.append(output.data)

        return output_sentences
