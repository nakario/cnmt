from logging import getLogger
from typing import Generator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import chainer
from chainer import Variable
import chainer.functions as F
import numpy as np

from cnmt.misc.constants import EOS
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
                 maxout_layer_size: int,
                 dynamic_attention: bool = False):
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
                               maxout_layer_size,
                               dynamic_attention=dynamic_attention)

    def __call__(
            self,
            source: ndarray,
            ga: ndarray,
            wo: ndarray,
            ni: ndarray,
            ga2: ndarray,
            target: ndarray
    ) -> Variable:
        # source.shape == (minibatch_size, source_max_sentence_size)
        # source.shape == ga.shape == wo.shape == ni.shape == ga2.shape
        # target.shape == (minibatch_size, target_max_sentence_size)
        encoded = self.enc(source, ga, wo, ni, ga2)
        loss = self.dec(encoded, target)
        chainer.report({'loss': loss}, self)
        return loss

    def translate(
            self,
            sentences: ndarray,
            ga: ndarray,
            wo: ndarray,
            ni: ndarray,
            ga2: ndarray,
            max_translation_length: int = 100
    ) -> List[ndarray]:
        # sentences.shape == (sentence_count, max_sentence_size)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            encoded = self.enc(sentences, ga, wo, ni, ga2)
            translated = self.dec.translate(
                encoded,
                max_translation_length
            )
            return translated


class ModelState(NamedTuple):
    cell: Variable
    hidden: Variable


class TranslationStates(NamedTuple):
    translations: List
    scores: ndarray
    states: List[ModelState]
    words: ndarray


def compute_next_states_and_scores(
        models: List[EncoderDecoder],
        states: List[ModelState],
        words: ndarray
) -> Tuple[ndarray, List[ModelState]]:
    xp = models[0].xp
    cells, hiddens, contexts, concatenateds = zip(*[
        model.dec.advance_one_step(state.cell, state.hidden, words)
        for model, state in zip(models, states)
    ])
    logits, hiddens = zip(*[
        model.dec.compute_logit(concatenated, hidden, context)
        for model, concatenated, context, hidden
        in zip(models, concatenateds, contexts, hiddens)
    ])

    combined_scores = xp.zeros_like(logits[0].array, 'f')

    for logit in logits:
        combined_scores += xp.log(F.softmax(logit).array)
    combined_scores /= float(len(models))

    new_states = []
    for cell, hidden in zip(cells, hiddens):
        new_states.append(ModelState(cell, hidden))

    return combined_scores, new_states


def iterate_best_score(scores: ndarray, beam_width: int) -> Generator[
    Tuple[int, int, float], None, None
]:
    case_, voc_size = scores.shape

    costs_flattened: np.ndarray = chainer.cuda.to_cpu(-scores).ravel()
    best_index = np.argpartition(costs_flattened, beam_width)[:beam_width]

    which_case = np.floor_divide(best_index, voc_size)
    index_in_case = best_index % voc_size

    for i, idx in enumerate(best_index):
        case = which_case[i]
        idx_in_case = index_in_case[i]
        yield int(case), int(idx_in_case), float(costs_flattened[idx])


def update_next_lists(
        case: int,
        idx_in_case: int,
        cost: float,
        states: List[ModelState],
        translations: List[List[int]],
        finished_translations: List,
        next_states_list: List[List[ModelState]],
        next_words_list: List[int],
        next_score_list: List[float],
        next_translations_list: List[List[int]]
):
    if idx_in_case == EOS:
        finished_translations.append(
            (translations[case] + [idx_in_case], -cost)
        )
    else:
        next_states_list.append([
            ModelState(
                Variable(state.cell.array[case].reshape(1, -1)),
                Variable(state.hidden.array[case].reshape(1, -1))
            )
            for state in states
        ])
        next_words_list.append(idx_in_case)
        next_score_list.append(-cost)
        next_translations_list.append(translations[case] + [idx_in_case])


def compute_next_lists(
        states: List[ModelState],
        translations: List[List[int]],
        scores: ndarray,
        finished_translations,
        beam_width: int
) -> Tuple[List[List[ModelState]], List[int], List[float], List[List[int]]]:
    next_states_list = []
    next_words_list = []
    next_score_list = []
    next_translations_list = []

    score_iterator = iterate_best_score(scores, beam_width)

    for case, idx_in_case, cost in score_iterator:
        update_next_lists(
            case,
            idx_in_case,
            cost,
            states,
            translations,
            finished_translations,
            next_states_list,
            next_words_list,
            next_score_list,
            next_translations_list
        )

    return (
        next_states_list,
        next_words_list,
        next_score_list,
        next_translations_list
    )


def advance_one_step(
        models: List[EncoderDecoder],
        translation_states: TranslationStates,
        finished_translations: List,
        beam_width: int
) -> Optional[TranslationStates]:
    xp = models[0].xp
    translations = translation_states.translations
    scores = translation_states.scores
    states = translation_states.states
    assert len(states) > 0
    words = translation_states.words
    combined_scores, new_states = compute_next_states_and_scores(
        models, states, words
    )

    count, voc_size = combined_scores.shape
    assert count <= beam_width

    word_scores = scores[:, None] + combined_scores

    (
        next_states_list,
        next_words_list,
        next_score_list,
        next_translations_list
    ) = compute_next_lists(
        new_states,
        translations,
        word_scores,
        finished_translations,
        beam_width
    )

    if len(next_states_list) == 0:
        return None

    concatenated_next_states_list = []
    for ss in zip(*next_states_list):  # loops for len(models) times
        # concat beams
        cell = F.concat([s.cell for s in ss], axis=0)
        hidden = F.concat([s.hidden for s in ss], axis=0)
        concatenated_next_states_list.append(ModelState(cell, hidden))

    return TranslationStates(
        next_translations_list,
        xp.array(next_score_list, 'f'),
        concatenated_next_states_list,
        xp.array(next_words_list, 'i')
    )


def translate_ensemble(
        models: List[EncoderDecoder],
        source: ndarray,
        ga: ndarray,
        wo: ndarray,
        ni: ndarray,
        ga2: ndarray,
        translation_limit: int,
        beam_width: int
) -> List[ndarray]:
    xp = chainer.cuda.get_array_module(source)
    assert source.shape[0] == 1
    encodeds = [model.enc(source, ga, wo, ni, ga2) for model in models]

    states: List[ModelState] = []
    finished_translations: List[Tuple[List[int], float]] = []
    previous_words = None  # shared in all models

    for model, encoded in zip(models, encodeds):
        model.dec.setup(encoded)
        cell, hidden, previous_words = model.dec.get_initial_states(1)
        states.append(ModelState(cell, hidden))

    translation_states = TranslationStates(
        [[]], xp.array([0], 'f'), states, previous_words
    )

    for i in range(translation_limit):
        translation_states = advance_one_step(
            models,
            translation_states,
            finished_translations,
            beam_width
        )
        if translation_states is None:
            break

    if len(finished_translations) == 0:
        finished_translations.append(([], 0))

    finished_translations.sort(
        key=lambda x: x[1] / (len(x[0]) + 2),
        reverse=True
    )
    return [xp.array(finished_translations[0][0])]
