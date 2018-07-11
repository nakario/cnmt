from argparse import Namespace
from collections import Counter
from logging import getLogger
from pathlib import Path
import re
from typing import Dict
from typing import NamedTuple
from typing import Union

from progressbar import ProgressBar

from cnmt.external_libs.bpe import learn_bpe
from cnmt.external_libs.bpe import apply_bpe
from cnmt.misc.constants import GA, WO, NI, GA2
from cnmt.misc.constants import nil
from cnmt.misc.functions import flen


class ConstArguments(NamedTuple):
    source: Path
    target: Path
    output: Path
    min_source_len: int
    max_source_len: int
    min_target_len: int
    max_target_len: int
    source_dev: Path
    target_dev: Path
    source_test: Path
    target_test: Path
    skip_create_bpe: bool
    skip_bpe_encode: bool
    skip_make_voc: bool


logger = getLogger(__name__)


def get_word(s: str) -> str:
    while s.startswith("<"):
        s = s[s.find(">")+1:]
    return s.split(" ")[0]


def get_case_word(result: str, case: str) -> str:
    found = re.findall(case + r"/\w/([^/]*)/", result)
    if (not found) or found[0] == "-":
        return nil
    found = found[0].strip()
    if not found:
        return nil
    return found.split()[0]


def get_result(s: str) -> Dict[str, str]:
    result = re.findall(r"<格解析結果:([^>]*)>", s)
    if not result:
        return {GA: nil, WO: nil, NI: nil, GA2: nil}
    result = result[0]
    ga = get_case_word(result, GA)
    wo = get_case_word(result, WO)
    ni = get_case_word(result, NI)
    ga2 = get_case_word(result, GA2)
    return {GA: ga, WO: wo, NI: ni, GA2: ga2}


def preproc_anap(
        anap_file: Path,
        raw_file: Path,
        ga_file: Path,
        wo_file: Path,
        ni_file: Path,
        ga2_file: Path
):
    assert anap_file.exists()
    assert raw_file.parent.exists()
    assert ga_file.parent.exists()
    assert wo_file.parent.exists()
    assert ni_file.parent.exists()
    assert ga2_file.parent.exists()

    raw_list = []
    ga_list = []
    wo_list = []
    ni_list = []
    ga2_list = []
    file_len = flen(anap_file)
    with open(anap_file) as anap:
        raw = []
        ga = []
        wo = []
        ni = []
        ga2 = []
        result = {}
        bar = ProgressBar(max_value=file_len)
        for line in bar(anap):
            if line.startswith(("#", "*")):
                continue
            if line.startswith("+"):
                result = get_result(line.strip())
                continue
            if line.strip() == "EOS":
                raw_list.append(" ".join(raw))
                ga_list.append(" ".join(ga))
                wo_list.append(" ".join(wo))
                ni_list.append(" ".join(ni))
                ga2_list.append(" ".join(ga2))
                raw = []
                ga = []
                wo = []
                ni = []
                ga2 = []
                continue
            raw.append(get_word(line.strip()))
            ga.append(result.get(GA, nil))
            wo.append(result.get(WO, nil))
            ni.append(result.get(NI, nil))
            ga2.append(result.get(GA2, nil))
            result = {}

    with open(raw_file, "w") as f:
        f.write("\n".join(raw_list))

    with open(ga_file, "w") as f:
        f.write("\n".join(ga_list))

    with open(wo_file, "w") as f:
        f.write("\n".join(wo_list))

    with open(ni_file, "w") as f:
        f.write("\n".join(ni_list))

    with open(ga2_file, "w") as f:
        f.write("\n".join(ga2_list))


def copy_data_with_limit(
        source: Path,
        target: Path,
        ga_file: Path,
        wo_file: Path,
        ni_file: Path,
        ga2_file: Path,
        source_copy: Path,
        target_copy: Path,
        ga_copy: Path,
        wo_copy: Path,
        ni_copy: Path,
        ga2_copy: Path,
        min_source_len: int,
        max_source_len: int,
        min_target_len: int,
        max_target_len: int
):
    assert source.exists()
    assert target.exists()
    assert ga_file.exists()
    assert wo_file.exists()
    assert ni_file.exists()
    assert ga2_file.exists()
    assert source_copy.parent.exists()
    assert target_copy.parent.exists()
    assert ga_copy.parent.exists()
    assert wo_copy.parent.exists()
    assert ni_copy.parent.exists()
    assert ga2_copy.parent.exists()

    src = open(source)
    tgt = open(target)
    ga = open(ga_file)
    wo = open(wo_file)
    ni = open(ni_file)
    ga2 = open(ga2_file)

    src_c = open(source_copy, 'w')
    tgt_c = open(target_copy, 'w')
    ga_c = open(ga_copy, 'w')
    wo_c = open(wo_copy, 'w')
    ni_c = open(ni_copy, 'w')
    ga2_c = open(ga2_copy, 'w')

    for s, t, g, w, n, g2 in zip(src, tgt, ga, wo, ni, ga2):
        s_words = s.strip().split()
        t_words = t.strip().split()
        s_len = len(s_words)
        t_len = len(t_words)
        if s_len < min_source_len or max_source_len < s_len:
            continue
        if t_len < min_target_len or max_target_len < t_len:
            continue
        assert s_len == len(g.strip().split())
        assert s_len == len(w.strip().split())
        assert s_len == len(n.strip().split())
        assert s_len == len(g2.strip().split())
        src_c.write(s)
        tgt_c.write(t)
        ga_c.write(g)
        wo_c.write(w)
        ni_c.write(n)
        ga2_c.write(g2)

    src.close()
    tgt.close()
    ga.close()
    wo.close()
    ni.close()
    ga2.close()

    src_c.close()
    tgt_c.close()
    ga_c.close()
    wo_c.close()
    ni_c.close()
    ga2_c.close()


def create_bpe_file(
        document: Union[Path, str],
        bpe_file: Union[Path, str]
):
    if isinstance(document, str):
        document = Path(document)
    if isinstance(bpe_file, str):
        bpe_file = Path(bpe_file)
    assert document.exists()
    assert bpe_file.parent.exists()

    logger.info('Start learning BPE')
    with open(document) as doc, open(bpe_file, 'w') as out:
        iterable = map(lambda x: x.strip().split(), doc)
        learn_bpe.learn_bpe_from_sentence_iterable(
            iterable,
            out,
            symbols=10000,
            min_frequency=2,
            verbose=False
        )


def bpe_encode(
        document: Union[Path, str],
        compressed: Union[Path, str],
        bpe_file: Union[Path, str]
):
    if isinstance(document, str):
        document = Path(document)
    if isinstance(compressed, str):
        compressed = Path(compressed)
    if isinstance(bpe_file, str):
        bpe_file = Path(bpe_file)
    assert document.exists()
    assert compressed.parent.exists()
    assert bpe_file.exists()

    with open(bpe_file) as file:
        bpe = apply_bpe.BPE(file, separator='._@@@')
    logger.info(f'Start applying BPE to {document.absolute()}')
    with open(document) as src, open(compressed, 'w') as src_bpe:
        for line in src:
            src_bpe.write(
                ' '.join(bpe.segment_splitted(line.strip().split())) + '\n'
            )


def make_voc(
        document: Union[str, Path],
        out_file: Union[str, Path],
):
    """Create a vocabulary file."""
    if isinstance(document, str):
        document = Path(document)
    if isinstance(out_file, str):
        out_file = Path(out_file)
    assert document.exists()
    assert out_file.parent.exists()

    logger.info(f'Preprocessing {document.absolute()}')
    sentence_count = 0
    word_count = 0
    counts = Counter()
    with open(document) as doc:
        for sentence in doc:
            sentence_count += 1
            words = sentence.strip().split()
            word_count += len(words)
            for word in words:
                counts[word] += 1

    vocab = [word for (word, _) in counts.most_common()]
    with open(out_file, 'w') as out:
        for word in vocab:
            out.write(word)
            out.write('\n')

    logger.info(f'Number of sentences: {sentence_count}')
    logger.info(f'Number of words    : {word_count}')
    logger.info(f'Size of vocabulary : {len(vocab)}')


def make_config(config_file: Path):
    pass


def preproc(args: Namespace):
    cargs = ConstArguments(**vars(args))

    output = Path(cargs.output)
    if not output.exists():
        logger.warning(f'{output.absolute()} does not exist')
        output.mkdir(parents=True, exist_ok=True)
        logger.info(f'created {output.absolute()}')

    # files for parsing anap
    raw_source = output / Path('raw')
    raw_ga = output / Path('raw_ga')
    raw_wo = output / Path('raw_wo')
    raw_ni = output / Path('raw_ni')
    raw_ga2 = output / Path('raw_ga2')

    preproc_anap(cargs.source, raw_source, raw_ga, raw_wo, raw_ni, raw_ga2)

    # training dataset
    source = output / Path('source')
    target = output / Path('target')
    ga = output / Path('limited_ga')
    wo = output / Path('limited_wo')
    ni = output / Path('limited_ni')
    ga2 = output / Path('limited_ga2')

    copy_data_with_limit(
        raw_source, cargs.target, raw_ga, raw_wo, raw_ni, raw_ga2,
        source, target, ga, wo, ni, ga2,
        cargs.min_source_len, cargs.max_source_len,
        cargs.min_target_len, cargs.max_target_len
    )

    bpe_target = output / Path('bpe_dic_target')
    if not cargs.skip_create_bpe:
        create_bpe_file(target, bpe_target)
    source_compressed = source
    target_compressed = target
    if not cargs.skip_bpe_encode:
        target_compressed = output / Path('target_compressed')
        bpe_encode(target, target_compressed, bpe_target)
    if not cargs.skip_make_voc:
        make_voc(source_compressed, output / Path('source_bpe_voc'))
        make_voc(target_compressed, output / Path('target_bpe_voc'))

    # validation dataset
    if cargs.source_dev is None or cargs.target_dev is None:
        return

    raw_source_dev = output / Path('raw_dev')
    raw_ga_dev = output / Path('raw_ga_dev')
    raw_wo_dev = output / Path('raw_wo_dev')
    raw_ni_dev = output / Path('raw_ni_dev')
    raw_ga2_dev = output / Path('raw_ga2_dev')

    preproc_anap(
        cargs.source_dev,
        raw_source_dev, raw_ga_dev, raw_wo_dev, raw_ni_dev, raw_ga2_dev
    )

    source_dev = output / Path('source_dev')
    target_dev = output / Path('target_dev')
    ga_dev = output / Path('limited_ga_dev')
    wo_dev = output / Path('limited_wo_dev')
    ni_dev = output / Path('limited_ni_dev')
    ga2_dev = output / Path('limited_ga2_dev')

    copy_data_with_limit(
        raw_source_dev, cargs.target_dev,
        raw_ga_dev, raw_wo_dev, raw_ni_dev, raw_ga2_dev,
        source_dev, target_dev,
        ga_dev, wo_dev, ni_dev, ga2_dev,
        cargs.min_source_len, cargs.max_source_len,
        cargs.min_target_len, cargs.max_target_len
    )

    target_dev_compressed = output / Path('target_dev_compressed')
    if not cargs.skip_bpe_encode:
        bpe_encode(target_dev, target_dev_compressed, bpe_target)

    # test dataset
    if cargs.source_test is None or cargs.target_test is None:
        return

    raw_source_test = output / Path('raw_test')
    raw_ga_test = output / Path('raw_ga_test')
    raw_wo_test = output / Path('raw_wo_test')
    raw_ni_test = output / Path('raw_ni_test')
    raw_ga2_test = output / Path('raw_ga2_test')

    preproc_anap(
        cargs.source_test,
        raw_source_test, raw_ga_test, raw_wo_test, raw_ni_test, raw_ga2_test
    )

    source_test = output / Path('source_test')
    target_test = output / Path('target_test')
    ga_test = output / Path('limited_ga_test')
    wo_test = output / Path('limited_wo_test')
    ni_test = output / Path('limited_ni_test')
    ga2_test = output / Path('limited_ga2_test')

    copy_data_with_limit(
        raw_source_test, cargs.target_test,
        raw_ga_test, raw_wo_test, raw_ni_test, raw_ga2_test,
        source_test, target_test,
        ga_test, wo_test, ni_test, ga2_test,
        1, 1000,
        1, 1000
    )

    target_test_compressed = output / Path('target_test_compressed')
    if not cargs.skip_bpe_encode:
        bpe_encode(target_test, target_test_compressed, bpe_target)
