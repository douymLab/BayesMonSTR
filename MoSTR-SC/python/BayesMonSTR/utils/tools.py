import cProfile
import logging
import math
import os
import time
import traceback
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import wraps
from itertools import combinations, combinations_with_replacement

import numpy as np
import psutil
from Bio import Align
from rich.console import Console
from rich.logging import RichHandler
from scipy.special import logsumexp

from ..utils import FLK_BP


console = Console(width=144)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%Y-%m-%d [%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True, console=console)],
)
logger = logging.getLogger("BayesMoSTR")
logger_gp = logging.getLogger("GP")
logger_gp.setLevel(logging.ERROR)


def find_paired_val(val, pos, data):
    """Find the other value in the corresponding pair.

    Parameters
    ----------
    val : `int`
        The input value.
    pos : {1, 2}
        1 indicates `val` is the first in the pair,
        and 2 indicates `val` is the second in the pair,
    data : List of tuple
        The list of tuples data

    Returns
    -------
    `int`
        The output value.
    """
    # assert val is not None
    if pos == 1:
        for pair in data:
            if pair[0] == val:
                return pair[1]

    elif pos == 2:
        for pair in data:
            if pair[1] == val:
                return pair[0]

    else:
        raise ValueError("The `pos` should be 1 or 2.")


def extract_allele(read, region):
    """Extract the MS allele from a read given the region.

    Parameters
    ----------
    read : `pysam.AlignedSegment`
        A read.
    region : `dict`
        A record from the MS reference panel.

    Returns
    -------
    `str`
        The extracted length-based allele.
    """
    aligned_pairs = read.get_aligned_pairs()
    idx_ref_start = region["start"] - 1
    idx_ref_end = region["end"]

    # left part
    idx_query_start = find_paired_val(idx_ref_start, 2, aligned_pairs)
    if idx_query_start:
        while True:
            if find_paired_val(idx_query_start - 1, 1, aligned_pairs):
                break
            idx_query_start -= 1
    else:
        # deletion
        while True:
            idx_ref_start += 1
            idx_query_start = find_paired_val(idx_ref_start, 2, aligned_pairs)
            if idx_query_start:
                break

    # right part
    idx_query_end = find_paired_val(idx_ref_end, 2, aligned_pairs)
    if idx_query_end:
        while True:
            if find_paired_val(idx_query_end + 1, 1, aligned_pairs):
                break
            idx_query_end += 1
    else:
        # deletion
        while True:
            idx_ref_end += 1
            idx_query_end = find_paired_val(idx_ref_end, 2, aligned_pairs)
            if idx_query_end:
                break

    return len(read.query_sequence[idx_query_start:idx_query_end])


def extract_hap(r, region):
    """Extract STR and flanking haplotype anchor, [0-based)"""
    try:
        aligned_pairs = r.get_aligned_pairs()
        idx_ref_start = region["start"] - 1
        idx_ref_end = region["end"]

        # left part
        idx_query_start = find_paired_val(idx_ref_start, 2, aligned_pairs)
        if idx_query_start:
            while True:
                if find_paired_val(idx_query_start - 1, 1, aligned_pairs):
                    break
                idx_query_start -= 1
        else:
            # deletion
            while True:
                idx_ref_start += 1
                idx_query_start = find_paired_val(idx_ref_start, 2, aligned_pairs)
                if idx_query_start:
                    break

        # right part
        idx_query_end = find_paired_val(idx_ref_end, 2, aligned_pairs)
        if idx_query_end:
            while True:
                if find_paired_val(idx_query_end + 1, 1, aligned_pairs):
                    break
                idx_query_end += 1
        else:
            # deletion
            while True:
                idx_ref_end += 1
                idx_query_end = find_paired_val(idx_ref_end, 2, aligned_pairs)
                if idx_query_end:
                    break

        # return r.query_alignment_sequence[idx_query_start:idx_query_end]
        # return r.query_alignment_qualities[idx_query_start:idx_query_end]
        # if not r.query_sequence:  # DEBUG:
        #     pass
        if (
            idx_query_start < 5
            or idx_query_end + 5 > r.query_length
            or idx_query_start >= idx_query_end
            or "N" in r.query_sequence[idx_query_start - 5 : idx_query_end + 5]
        ):
            return None
        return idx_query_start - 1, idx_query_end
        return idx_query_start, idx_query_end
    except Exception:
        return None


def extract_hanging_hap(r, region):
    """Extract STR and flanking haplotype anchor, [0-based)"""
    try:
        if "N" in r.cigarstring:
            return None
        aligned_pairs = r.get_aligned_pairs()
        idx_ref_start = region["start"] - 1
        idx_ref_end = region["end"]

        # left part
        idx_query_start = find_paired_val(idx_ref_start, 2, aligned_pairs)
        if idx_query_start:
            while True:
                if find_paired_val(idx_query_start - 1, 1, aligned_pairs):
                    break
                idx_query_start -= 1
        else:
            # deletion
            while True:
                idx_ref_start += 1
                idx_query_start = find_paired_val(idx_ref_start, 2, aligned_pairs)
                if idx_query_start:
                    break

        # right part
        idx_query_end = find_paired_val(idx_ref_end, 2, aligned_pairs)
        if idx_query_end:
            while True:
                if find_paired_val(idx_query_end, 1, aligned_pairs):
                    break
                idx_query_end += 1
        else:
            # deletion
            while True:
                idx_ref_end += 1
                idx_query_end = find_paired_val(idx_ref_end, 2, aligned_pairs)
                if idx_query_end:
                    break

        if (
            idx_query_start < 0
            or idx_query_end > r.query_length
            or idx_query_start >= idx_query_end
            or "N" in r.query_sequence[idx_query_start:idx_query_end]
        ):
            return None
        return idx_query_start - 1, idx_query_end
        return idx_query_start, idx_query_end
    except Exception:
        return None


def cartesian_product(pool):
    """Generate ordered genotype pool"""
    return [(a, b) for a in pool for b in pool]


def cartesian_prod(alleles):
    """Generate unordered genotype pool"""
    return list(combinations_with_replacement(alleles, 2))


def cartesian_prod_het(alleles):
    """Generate unordered het genotype pool"""
    return list(combinations(alleles, 2))


def flatten(iterable):
    """Flatten an nested iterable."""
    for item in iterable:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


def flatten_list(iterable):
    """Flatten an nested list."""
    for item in iterable:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def map_nested(func, iterable, *args):
    """Apply the given function to an arbitrarily nested iterable.

    Parameters
    ----------
    func : callable
        The function to be applied.
    iterable : list
        A arbitrarily nested iterable.
    *args
        Additional positional arguments that pass into the function.

    Returns
    -------
    iterable
        Return the arbitrarily nested iterable.
    """
    if isinstance(iterable, tuple) or isinstance(iterable, array):
        return tuple(map_nested(func, item, *args) for item in iterable)
    elif isinstance(iterable, Iterable):
        return [map_nested(func, item, *args) for item in iterable]
    else:
        return func(iterable, *args)


def map_nested_list(func, iterable, *args):
    """Apply the given function to an arbitrarily nested list.

    Parameters
    ----------
    func : callable
        The function to be applied.
    iterable : list
        A arbitrarily nested list.
    *args
        Additional positional arguments that pass into the function.

    Returns
    -------
    iterable
        Return the arbitrarily nested list.
    """
    # if isinstance(iterable, Iterable):
    if isinstance(iterable, list):
        if iterable == []:
            return []
        return [map_nested_list(func, item, *args) for item in iterable]
    else:
        return func(iterable, *args)


def normalize_dict(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


def normalize(arr):
    """Normalize a list of values that sums to one."""
    # factor = 1 / sum(arr)
    # norm = [i * factor for i in arr]
    # return norm

    s = sum(arr)
    if s == 0:
        norm = [0] * len(arr)
    else:
        factor = 1 / s
        norm = [i * factor for i in arr]
    return norm


def normalize_log(arr):
    r"""Normalize a list of values that sums to one.

    .. math::

        [\log x, \log y, \log z] \text{ to }
        [\log (\frac{x}{x + y + z}), \log (\frac{y}{x + y + z}), \log (\frac{z}{x + y + z})]

    """
    if not len(arr):
        return []
    try:
        arr = np.array(arr)
        arr = np.where(arr == np.inf, 1e255, arr)
        arr = np.where(arr == -np.inf, -1e255, arr)
        factor = logsumexp(arr)
        return [i - factor for i in arr]
    except:
        logger.warning("%s: normalize_log", arr)
        return [np.log(1 / len(arr))] * len(arr)


def log_sum_exp(a, b):
    r"""LSE function.

    Calculate :math:`log {(x + y)}`
    given :math:`log {(x)}` and :math:`log {(y)}` as following:

    .. math::

        \log {(x + y)} = x + \log {(1 + \exp^{y - x})}

    Parameters
    ----------
    a : `float`
        Log value, log_{x}.
    b : `float`
        Log value,  log_{y}.

    Returns
    -------
    `float`
        Log of the sum, e.g. log_{x + y}.
    """
    if a > b:
        return a + np.log(1 + np.exp(b - a))
    else:
        return b + np.log(1 + np.exp(a - b))


def timing(func):
    """Time a function.

    Parameters
    ----------
    func : callable
        The function to be timed.

    Returns
    -------
    callable
        Function with timer.
    """

    @wraps(func)
    def wrapper(*args, **kw):
        t0 = time.time()
        result = func(*args, **kw)
        t1 = time.time()
        print(f"func: [{func.__name__}] args:[{args}, {kw}] took: [{(t1 - t0):2.4f}s]")
        return result

    return wrapper


def greater(read, allele, motif_size, frame):
    """Return log value."""
    is_inframe = (read - allele) % motif_size == 0
    if frame == "in" and is_inframe:
        if read > allele:
            return 0
    elif frame == "out" and not is_inframe:
        if read > allele:
            return 0
    return float("-inf")


def not_equal(read, allele, motif_size, frame):
    is_inframe = (read - allele) % motif_size == 0
    if frame == "in" and is_inframe:
        if read != allele:
            return 0
    elif frame == "out" and not is_inframe:
        if read != allele:
            return 0
    return float("-inf")


def less(read, allele, motif_size, frame):
    is_inframe = (read - allele) % motif_size == 0
    if frame == "in" and is_inframe:
        if read < allele:
            return 0
    elif frame == "out" and not is_inframe:
        if read < allele:
            return 0
    return float("-inf")


def length(read, allele, motif_size, frame):
    diff = abs(read - allele)
    if diff:
        is_inframe = (read - allele) % motif_size == 0
        if frame == "in" and is_inframe:
            return np.log(diff / motif_size)
        elif frame == "out" and not is_inframe:
            return np.log(diff - diff // motif_size)
    return float("-inf")


# def same(read, allele):
#     return 0


def sputnik(seq, repeat_units, flank_size=5):
    """Find anchors for segmentation flanking region from microsatellite sequence

    Parameters
    ----------
    seq : `str`
        Input sequence.
    flank_size : `int`
        The size of flanking region.
    repeat_units : `int`
        Microsatellite motif length.

    Returns
    -------
    tuple[`int`, `int`]
        Anchors.
    """

    res = find_repeats_target(seq, flank_size, repeat_units)

    return res[0][1] - 1, res[0][2] + 1


def find_repeats_target(seq, flank_size, repeat_units):
    # copy from https://github.com/parklab/MSIprofiler/blob/master/utils.py
    MATCH_SCORE = 1  # score for a match
    MISMATCH_SCORE = -6  # score penalty for a mismatch
    FAIL_SCORE = -1  # score value to stop searching
    MIN_SCORE = 4  # minimum score value to pass. The minimum length of MS repeats
    bases = len(seq)  # number of bases in the input sequence

    # save output as a list of lists
    out = []
    # use sets: they are much faster with 'in'[]#np.array([], dtype='int')
    exclude = set()

    for ru in [repeat_units]:
        positions_motif = range(0, ru)
        nb_positions_motif = len(positions_motif)

        # note that the flank is a range, whether python is zero-based
        not_found = True
        base = flank_size

        while base < bases - flank_size:  # and base not in exclude:
            if base in exclude:
                base += 1
                continue
            elif not_found:
                test_pos = base + ru
                current_pos = base
            else:
                current_pos = base
                not_found = True
                test_pos = test_pos + ru

            pos_in_motif = 0  # update_current_pos(ru)
            score = 0
            depth = 0

            max_observed_score = 0
            scores = []
            while (
                ((test_pos) < (bases - flank_size))
                and score > FAIL_SCORE
                and test_pos not in exclude
            ):  # XX the minus one check
                match = seq[current_pos + pos_in_motif] == seq[test_pos]

                if match:
                    test_pos += 1
                    pos_in_motif = positions_motif[
                        (pos_in_motif + 1) % nb_positions_motif
                    ]
                    score += MATCH_SCORE
                    scores.append(score)
                    depth = 0

                # no mismatch: check for N, insertions, deletions and missense
                else:
                    score += MISMATCH_SCORE
                    scores.append(score)
                    pos_in_motif = positions_motif[
                        (pos_in_motif + 1) % nb_positions_motif
                    ]

                    if score > FAIL_SCORE and depth < 5:
                        depth += 1
                        test_pos += 1

                # keep track of the best observed score
                if score > max_observed_score:
                    max_observed_score = score
            test_pos = test_pos  # - depth
            if max_observed_score >= MIN_SCORE:
                mm = scores.index(max(scores))
                mm = mm + ru
                out.append([ru, base, base + mm, seq[base : base + mm + 1]])
                not_found = False
                exclude.update(range(base, base + mm + 1))
                test_pos = base + mm
                base = test_pos

            # increment base
            base += 1
        else:
            base += 1
    return out


def segmentation(s, anchors, flank_size=5, source=False):
    """Segment a read given anchors

    Parameters
    ----------
    s : `str`
        Read.
    anchors : tuple
        Left anchor and right anchor.
    flank_size : int, optional
        Flanking sequece length, by default 5

    Returns
    -------
    Tuple of `str`
        Left flanking, microsatellite and right flanking sequence, respectively.
    """
    if not anchors:
        return None

    anchor_left, anchor_right = anchors[:2]
    res = (
        s[max(0, anchor_left + 1 - flank_size) : anchor_left + 1],
        s[anchor_left + 1 : anchor_right],
        s[anchor_right : min(len(s), anchor_right + flank_size)],
    )

    # If source information is provided (3 elements in anchors), return it with the result
    if len(anchors) > 2 and source:
        return res, anchors[2]
    return res


def desegmentation(t):
    return t[0] + t[1] + t[2]


def phred_score_q(q):
    """Convert phred-scale quality to probability"""
    return 1 - (10 ** (-q / 10))


def phred_score_p(p):
    """Convert probability to phred-scale quality"""
    return -10 * np.log10(1 - p)


def sigmoid(x):
    return 1 / (1 + np.e ^ (-x / 10))


def check_spn(r, pos, bam=None):
    # 0-based
    try:
        if r.reference_start < pos < r.reference_start + r.query_alignment_length:
            return True
        # return False
        if not r.mate_is_unmapped:
            mate = bam.mate(r)
            if (
                mate.reference_start
                < pos
                < mate.reference_start + mate.query_alignment_length
            ):
                return True
    except Exception as e:
        return False
        e.add_note(f"check spn failed: {r.reference_name} {pos}")
        traceback.print_exc()
    return False


def check_flk(r, region):
    if (
        r.reference_start <= region["start"] - FLK_BP - 1
        and r.reference_end >= region["end"] + FLK_BP
    ):
        indels = []
        reference_position = r.reference_start

        for op, length in r.cigar:
            if op == 0:
                reference_position += length
            elif op == 1:
                indels.append(reference_position)
            elif op == 2:
                indels.append(reference_position)
                reference_position += length
        mm = any(
            region["start"] - FLK_BP <= pos + 1 < region["start"]
            or region["end"] < pos + 1 <= region["end"] + FLK_BP
            for pos in indels
        )
        if not mm:
            return True
    return False


def get_seq_from_reads(
    filtered_reads_list, reads_data_anchor, best_records=None, bams=None, filters=False
):
    reads_data_seq = []
    if best_records:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            if not best_records[idx_ind]:
                reads_data_seq.append([[], []])
                continue
            ind_bulk_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                ind_bulk_s_reads_data_seq = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][0][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    anchor = reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                    seq_input = (
                        read.query_sequence
                        if len(anchor) >= 3 and anchor[2] == 0
                        else read.query_alignment_sequence
                    )
                    ms = segmentation(
                        seq_input,
                        anchor,
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_alignment_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_sequence[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][0][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_sequence[idx_hSNP],)
                        else:
                            continue
                    ind_bulk_s_reads_data_seq.append(ms + hSNP)
                ind_bulk_reads_data_seq.append(ind_bulk_s_reads_data_seq)

            ind_sc_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                ind_sc_s_reads_data_seq = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][1][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    anchor = reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                    seq_input = (
                        read.query_sequence
                        if len(anchor) >= 3 and anchor[2] == 0
                        else read.query_alignment_sequence
                    )
                    ms = segmentation(
                        seq_input,
                        anchor,
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_alignment_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_sequence[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][1][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_sequence[idx_hSNP],)
                        else:
                            continue
                    ind_sc_s_reads_data_seq.append(ms + hSNP)
                ind_sc_reads_data_seq.append(ind_sc_s_reads_data_seq)
            reads_data_seq.append([ind_bulk_reads_data_seq, ind_sc_reads_data_seq])
        return reads_data_seq
    else:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            ind_bulk_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                if filters:
                    ind_bulk_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence
                                if len(
                                    reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][0][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_sequence,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    ind_bulk_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence
                                if len(
                                    reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][0][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_sequence,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                        ]
                    )

            ind_sc_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                if filters:
                    ind_sc_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence
                                if len(
                                    reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][1][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_sequence,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    ind_sc_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence
                                if len(
                                    reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][1][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_sequence,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                        ]
                    )
            reads_data_seq.append([ind_bulk_reads_data_seq, ind_sc_reads_data_seq])
        return reads_data_seq


def get_seq_from_reads_src(
    filtered_reads_list, reads_data_anchor, best_records=None, bams=None, filters=False
):
    reads_data_seq = []
    has_coord_reads = False

    if best_records:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            if not best_records[idx_ind]:
                reads_data_seq.append([[], []])
                continue
            ind_bulk_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                ind_bulk_s_reads_data_seq = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][0][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    anchor = reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                    seq_input = (
                        read.query_sequence
                        if len(anchor) >= 3 and anchor[2] == 0
                        else read.query_alignment_sequence
                    )
                    ms = segmentation(
                        seq_input,
                        anchor,
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_alignment_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_sequence[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][0][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_sequence[idx_hSNP],)
                        else:
                            continue
                    ind_bulk_s_reads_data_seq.append(ms + hSNP)
                ind_bulk_reads_data_seq.append(ind_bulk_s_reads_data_seq)

            ind_sc_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                ind_sc_s_reads_data_seq = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][1][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    anchor = reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                    seq_input = (
                        read.query_sequence
                        if len(anchor) >= 3 and anchor[2] == 0
                        else read.query_alignment_sequence
                    )
                    ms = segmentation(
                        seq_input,
                        anchor,
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_alignment_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_sequence[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][1][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_sequence[idx_hSNP],)
                        else:
                            continue
                    ind_sc_s_reads_data_seq.append(ms + hSNP)
                ind_sc_reads_data_seq.append(ind_sc_s_reads_data_seq)
            reads_data_seq.append([ind_bulk_reads_data_seq, ind_sc_reads_data_seq])
        return reads_data_seq
    else:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            ind_bulk_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                if filters:
                    ind_bulk_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence
                                if len(
                                    reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][0][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_sequence,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    segment_counter = {}
                    seg_results = []

                    for idx_read, read in enumerate(reads):
                        if reads_data_anchor[idx_ind][0][idx_reads][idx_read]:
                            anchor = reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                            seq_input = (
                                read.query_sequence
                                if len(anchor) >= 3 and anchor[2] == 0
                                else read.query_alignment_sequence
                            )
                            seg_result = segmentation(
                                seq_input,
                                anchor,
                                source=True,
                            )

                            if seg_result:
                                res, src = seg_result
                                seg_results.append(res)

                                if src == 0:
                                    if res in segment_counter:
                                        segment_counter[res] += 1
                                    else:
                                        segment_counter[res] = 1

                                    if segment_counter[res] >= 3:
                                        has_coord_reads = True
                            else:
                                seg_results.append(None)

                    ind_bulk_reads_data_seq.append(seg_results)

            ind_sc_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                if filters:
                    ind_sc_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence
                                if len(
                                    reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][1][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_sequence,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    segment_counter = {}
                    seg_results = []

                    for idx_read, read in enumerate(reads):
                        if reads_data_anchor[idx_ind][1][idx_reads][idx_read]:
                            anchor = reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                            seq_input = (
                                read.query_sequence
                                if len(anchor) >= 3 and anchor[2] == 0
                                else read.query_alignment_sequence
                            )
                            seg_result = segmentation(
                                seq_input,
                                anchor,
                                source=True,
                            )

                            if seg_result:
                                res, src = seg_result
                                seg_results.append(res)

                                if src == 0:
                                    if res in segment_counter:
                                        segment_counter[res] += 1
                                    else:
                                        segment_counter[res] = 1

                                    if segment_counter[res] >= 3:
                                        has_coord_reads = True
                            else:
                                seg_results.append(None)

                    ind_sc_reads_data_seq.append(seg_results)

            reads_data_seq.append([ind_bulk_reads_data_seq, ind_sc_reads_data_seq])

        return reads_data_seq, has_coord_reads


def get_qual_from_reads(
    filtered_reads_list, reads_data_anchor, best_records=None, bams=None, filters=False
):
    reads_data_qual = []
    if best_records:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            if not best_records[idx_ind]:
                reads_data_qual.append([[], []])
                continue
            ind_bulk_reads_data_qual = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                ind_bulk_s_reads_data_qual = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][0][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    anchor = reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                    qual_input = (
                        read.query_qualities
                        if len(anchor) >= 3 and anchor[2] == 0
                        else read.query_alignment_qualities
                    )
                    ms = segmentation(
                        qual_input,
                        anchor,
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_alignment_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_qualities[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][0][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_qualities[idx_hSNP],)
                        else:
                            continue
                    ind_bulk_s_reads_data_qual.append(ms + hSNP)
                ind_bulk_reads_data_qual.append(ind_bulk_s_reads_data_qual)

            ind_sc_reads_data_qual = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                ind_sc_s_reads_data_qual = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][1][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    anchor = reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                    qual_input = (
                        read.query_qualities
                        if len(anchor) >= 3 and anchor[2] == 0
                        else read.query_alignment_qualities
                    )
                    ms = segmentation(
                        qual_input,
                        anchor,
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_alignment_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_qualities[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][1][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_qualities[idx_hSNP],)
                        else:
                            continue
                    ind_sc_s_reads_data_qual.append(ms + hSNP)
                ind_sc_reads_data_qual.append(ind_sc_s_reads_data_qual)
            reads_data_qual.append([ind_bulk_reads_data_qual, ind_sc_reads_data_qual])
        return map_nested(phred_score_q, reads_data_qual)
    else:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            ind_bulk_reads_data_qual = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                if filters:
                    ind_bulk_reads_data_qual.append(
                        [
                            segmentation(
                                read.query_qualities
                                if len(
                                    reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][0][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_qualities,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    ind_bulk_reads_data_qual.append(
                        [
                            segmentation(
                                read.query_qualities
                                if len(
                                    reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][0][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_qualities,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                        ]
                    )

            ind_sc_reads_data_qual = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                if filters:
                    ind_sc_reads_data_qual.append(
                        [
                            segmentation(
                                read.query_qualities
                                if len(
                                    reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][1][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_qualities,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    ind_sc_reads_data_qual.append(
                        [
                            segmentation(
                                read.query_qualities
                                if len(
                                    reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                                )
                                >= 3
                                and reads_data_anchor[idx_ind][1][idx_reads][idx_read][
                                    2
                                ]
                                == 0
                                else read.query_alignment_qualities,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                        ]
                    )
            reads_data_qual.append([ind_bulk_reads_data_qual, ind_sc_reads_data_qual])
        return map_nested(phred_score_q, reads_data_qual)


def filename_wo_ext(file_path):
    base_name = os.path.basename(file_path)
    filename_without_extension, _ = os.path.splitext(base_name)
    return filename_without_extension


def max_div_sec(dat, maximum=1e3):
    """input is log value"""
    numbers = dat[:]
    if len(numbers) >= 2:
        # inf = (float("inf"), -float("inf"))
        max_num = max(numbers)
        numbers.remove(max_num)
        sec_max_num = max(numbers)
        if max_num == sec_max_num:
            return 0
        else:
            ratio = max_num - sec_max_num
        return ratio if ratio < maximum else maximum
    else:
        return maximum


def is_valid_rec(filtered_rec):
    try:
        target_row = filtered_rec.iloc[0]
        float(target_row.get("rho_bulk_in"))
        return True
    except:
        return False


def clip_value(value):
    return max(0.0, min(value, 1.0))


def hapToLen(hapCounts, tot_allele_pool):
    lenCounts = defaultdict(int)

    for hap, value in hapCounts.items():
        lenCounts[len(tot_allele_pool[int(hap)][1])] += value

    return dict(lenCounts)



def display_nested_list_structure(nested_list, indent=0):
    for item in nested_list:
        if isinstance(item, list):
            print("    " * indent + "[")
            display_nested_list_structure(item, indent + 1)
            print("    " * indent + "]")
        else:
            pass  # Do nothing for non-list elements


def unique(l):
    return sorted(set(l))
    return sorted(list(dict.fromkeys(l).keys()))


def profile_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # stats = pstats.Stats(profiler)
        # stats.print_stats()
        profiler.dump_stats("program_curr.prof")

        return result

    return wrapper


def calc_err_ms(read, start, end):
    """
    Calculate the number of mismatches/indels outside the specified region.
    0-based
    """
    if read.infer_query_length() - end + start <= 0:
        return 0, 0
    indel_count = 0
    mismatch_count = 0
    read_pos = 0
    ref_pos = read.reference_start
    ref_seq = read.get_reference_sequence()
    read_seq = read.query_alignment_sequence

    for op, length in read.cigartuples:
        if op == 0:  # match or mismatch
            for i in range(length):
                if not (start <= ref_pos + i < end):
                    if (
                        read_seq[read_pos + i]
                        != ref_seq[ref_pos - read.reference_start + i]
                    ):
                        mismatch_count += 1
            read_pos += length
            ref_pos += length
        elif op == 1:  # insertion
            if not (start <= ref_pos < end):
                indel_count += 1
            read_pos += length
        elif op == 2:  # deletion
            if not (start <= ref_pos < end):
                indel_count += 1
            ref_pos += length
    return (
        mismatch_count / (read.infer_query_length() - end + start),
        indel_count / (read.infer_query_length() - end + start),
    )


def calc_err_flk(read, start, end):
    """
    Calculate the number of mismatches inside the specified region.
    0-based
    """
    if read.infer_query_length() - end + start <= 0:
        return 0, 0
    indel_count = 0
    mismatch_count = 0
    read_pos = 0
    ref_pos = read.reference_start
    ref_seq = read.get_reference_sequence()
    read_seq = read.query_alignment_sequence

    for op, length in read.cigartuples:
        if op == 0:
            for i in range(length):
                if start <= ref_pos + i < end:
                    if (
                        read_seq[read_pos + i]
                        != ref_seq[ref_pos - read.reference_start + i]
                    ):
                        mismatch_count += 1
            read_pos += length
            ref_pos += length
        elif op == 1:
            if start <= ref_pos < end:
                indel_count += 1
            read_pos += length
        elif op == 2:
            if start <= ref_pos < end:
                indel_count += 1
            ref_pos += length
    return mismatch_count / (end - start), indel_count / (end - start)



def remove_na(list1, list2):
    """Remove 'NA' string."""
    new_list1 = []
    new_list2 = []

    for x, y in zip(list1, list2):
        if y != "NA":
            new_list1.append(x)
            new_list2.append(y)

    return new_list1, new_list2


def get_memory_usage():
    """Get the current memory usage (MB)"""
    pid = psutil.Process()
    memory_info = pid.memory_info()
    return memory_info.rss / 1024 / 1024  # / 1024


def ms_distance(x, pos_MS):
    return abs(x.pos - pos_MS)


def score_func(s1, s2):
    assert len(s1) == len(s2)
    return sum(1 if a == b else -1 for a, b in zip(s1, s2))


def check_seg_indel(germline, mosaic):
    """refine seg in flanking to determine is_ms_indel
    germline: (5-1)flk + 5flk + (5-1)STR = 13bp
    mosaic: 5bp
    germline="FFFFTTTGGSSSS", mosaic="TTTTG"
    germline="FFFTAAAGAGASS", mosaic="AAGAA"
    germline="FFCGGATTTTTTG", mosaic="ATTTT"
    germline="FFFFAAAAASSSS", mosaic="AAAAA"
    return - del; + ins.
    """
    flk_size = len(mosaic)
    padding_size = int((len(germline) - len(mosaic)) / 2)
    highest_score = float("-inf")
    for i in range(0, flk_size):
        germline_flk = germline[i : i + flk_size]
        # print(i, germline_flk, mosaic, sep="\n")
        score = score_func(germline_flk, mosaic)
        if score > highest_score:
            highest_score = score
            best_shift = i
    for i in range(flk_size, flk_size + padding_size):
        germline_flk = germline[i : i + flk_size]
        # print(i, germline_flk, mosaic, sep="\n")
        germ_f1, germ_f2 = (
            germline_flk[: flk_size + padding_size - i],
            germline_flk[flk_size + padding_size - i :],
        )
        mo_f1, mo_f2 = mosaic[0 : -i + padding_size], mosaic[-i + padding_size :]

        score = score_func(germ_f1, mo_f1) + max(
            (
                score_func(germ_f2, mo_f2),
                (
                    score_func(germ_f1[-len(mo_f2) :], mo_f2)
                    if len(germ_f1) >= len(mo_f2)
                    else score_func(
                        germline[
                            flk_size + padding_size - len(mo_f2) : flk_size
                            + padding_size
                        ],
                        mo_f2,
                    )
                ),
            )
        )
        if score > highest_score:
            highest_score = score
            best_shift = i
    # print(best_shift - 4)
    return best_shift - padding_size


def trim_aln(g, m, g_first=4, g_end=4, m_first=4, m_end=4):
    def remove_start(s, num):
        count = 0
        i = 0
        while i < len(s) and count < num:
            if s[i] != "-":
                count += 1
            i += 1
        return i

    def remove_end(s, num):
        count = 0
        i = len(s) - 1
        while i >= 0 and count < num:
            if s[i] != "-":
                count += 1
            i -= 1
        return i

    # idx_start = max((remove_start(g, g_first), remove_start(m, m_first)))
    # idx_end = min((remove_end(g, g_end), remove_end(m, m_end)))
    idx_start = remove_start(g, g_first)
    idx_end = remove_end(g, g_end)
    # breakpoint()
    return g[idx_start : idx_end + 1], m[idx_start : idx_end + 1]


def analyze_aln(mu_j_seqs, mu_k_seqs, region, germ_coords):
    """1-based alignment analyzer"""
    # germline = "ACGTAACTAGACATGCAGAT"
    # mosaic = "ACGTAGCACCAGACGCAGAT"
    # mu_j_seqs = ("AAAA", "TACAC", "ATGTGTGTGTGTGCGTGTGTGTGTG", "CATTT", "GGGT")
    # mu_k_seqs = ("AAAA", "TACAC", "TGTGTGTGTGTGTGTGTGTGTGTG", "CATTT", "GGGT")
    germ_coords = [v for v in germ_coords if v is not None]
    if None in mu_j_seqs or None in mu_k_seqs or not germ_coords:
        return None, None, None, None
    germline = "".join(mu_j_seqs)
    mosaic = "".join(mu_k_seqs)
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0

    # alignments = pairwise2.align.globalms(germline, mosaic, 5, -4, -3, -0.1)
    alignment = aligner.align(germline, mosaic)[0]
    # print(alignment._format_unicode())  # DEBUG:
    (
        aligned_germline,
        _,
        aligned_mosaic,
        _,
    ) = alignment._format_unicode().split("\n")
    # aligned_mosaic = trim_aln(aligned_mosaic, len(mu_k_seqs[0]), len(mu_k_seqs[4]))
    aligned_germline, aligned_mosaic = trim_aln(
        aligned_germline,
        aligned_mosaic,
        len(mu_j_seqs[0]),
        len(mu_j_seqs[4]),
        len(mu_k_seqs[0]),
        len(mu_k_seqs[4]),
    )
    # aligned_germline, aligned_mosaic, score, begin, end = alignment

    position = 0
    position_mosaic = 0
    current_mutation = None
    mutation_start = None
    mutation_size = 0
    mutation_seq = ""
    mutation_start_mosaic = None

    mutations = []

    for i, (a, b) in enumerate(zip(aligned_germline, aligned_mosaic)):
        if a == "-" and b == "-":
            continue

        # Determine the current mutation type, if any
        if a == "-":
            mutation_type = "insertion"
        elif b == "-":
            mutation_type = "deletion"
        elif a != b:
            mutation_type = "mismatch"
        else:
            mutation_type = None

        # Update position for seqA
        if a != "-":
            position += 1
        if b != "-":
            position_mosaic += 1

        # If there's a change in mutation type or sequence end, store the current mutation
        if current_mutation != mutation_type:
            if current_mutation is not None:
                if current_mutation == "mismatch":
                    mutations.append(
                        (
                            mutation_start,
                            mutation_size,
                            current_mutation,
                            mutation_seq[0],
                            mutation_seq[1],
                            mutation_start_mosaic,
                        )
                    )
                else:
                    mutations.append(
                        (
                            mutation_start,
                            mutation_size,
                            current_mutation,
                            mutation_seq,
                            mutation_start_mosaic,
                        )
                    )
            mutation_start = position if mutation_type is not None else None
            mutation_start_mosaic = (
                position_mosaic if mutation_type is not None else None
            )
            mutation_size = 0
            mutation_seq = ""

        current_mutation = mutation_type
        if mutation_type is not None:
            mutation_size += 1
            if mutation_type == "insertion":
                mutation_seq += b
            elif mutation_type == "deletion":
                mutation_seq += a
            elif mutation_type == "mismatch":
                mutation_seq = a, b

    # Handle any ongoing mutation at the end
    if current_mutation is not None:
        if current_mutation == "mismatch":
            mutations.append(
                (
                    mutation_start,
                    mutation_size,
                    current_mutation,
                    mutation_seq[0],
                    mutation_seq[1],
                    mutation_start_mosaic,
                )
            )
        else:
            mutations.append(
                (
                    mutation_start,
                    mutation_size,
                    current_mutation,
                    mutation_seq,
                    mutation_start_mosaic,
                )
            )
    germline_hap = aligned_germline.replace("-", "")
    res = sorted(mutations, key=lambda x: abs(x[0] - len(germline_hap) / 2))
    if not res:
        return None, None, None, None
    info = None
    cxt = None
    refalt = None
    germ_coord = max(set(germ_coords), key=germ_coords.count)
    if res[0][2] == "mismatch":
        (
            mutation_start,
            mutation_size,
            mutation,
            mutation_seq_j,
            mutation_seq_k,
            mutation_start_mosaic,
        ) = res[0]
        info = (
            mutation_start,
            mutation_size,
            mutation,
            mutation_seq_j,
            mutation_seq_k,
            mutation_start_mosaic,
        )
        cxt = [
            germline_hap[
                max(mutation_start - 4, 0) : min(mutation_start - 1, len(germline_hap))
            ],
            germline_hap[
                max(0, mutation_start + mutation_size - 1) : min(
                    mutation_start + mutation_size + 2, len(germline_hap)
                )
            ],
        ]
        refalt = (  # 1-based
            region.get("chr"),
            germ_coord + mutation_start,
            germ_coord + mutation_start + mutation_size - 1,
            mutation_seq_j,
            mutation_seq_k,
        )
    elif res[0][2] == "insertion":
        (
            mutation_start,
            mutation_size,
            current_mutation,
            mutation_seq,
            mutation_start_mosaic,
        ) = res[0]
        info = (
            mutation_start,
            mutation_size,
            current_mutation,
            mutation_seq,
            mutation_start_mosaic,
        )
        cxt = [
            germline_hap[
                max(0, mutation_start - 3) : min(mutation_start, len(germline_hap))
            ],
            germline_hap[
                max(0, mutation_start) : min(mutation_start + 3, len(germline_hap))
            ],
        ]
        refalt = (
            region.get("chr"),
            germ_coord + mutation_start,
            germ_coord + mutation_start,
            germline_hap[mutation_start - 1],
            germline_hap[mutation_start - 1] + mutation_seq,
        )
    elif res[0][2] == "deletion":
        (
            mutation_start,
            mutation_size,
            current_mutation,
            mutation_seq,
            mutation_start_mosaic,
        ) = res[0]
        info = (
            mutation_start,
            mutation_size,
            current_mutation,
            mutation_seq,
            mutation_start_mosaic,
        )
        cxt = [
            germline_hap[
                max(0, mutation_start - 4) : min(mutation_start - 1, len(germline_hap))
            ],
            germline_hap[
                max(0, mutation_start - 1 + mutation_size) : min(
                    mutation_start + 2 + mutation_size, len(germline_hap)
                )
            ],
        ]
        refalt = (
            region.get("chr"),
            germ_coord + mutation_start - 1,
            germ_coord + mutation_start - 1 + mutation_size,
            germline_hap[mutation_start - 2] + mutation_seq,
            germline_hap[mutation_start - 2],
        )
    # breakpoint()
    return info, cxt, refalt, res


def trim_flk(s, allele_flk_size):
    if not s or not allele_flk_size:
        return None
    if len(s[0]) < allele_flk_size[0] or len(s[2]) < allele_flk_size[1]:
        return None
    l = s[0][-allele_flk_size[0] :] if allele_flk_size[0] else ""
    r = s[2][: allele_flk_size[1]]
    if len(s) == 3:
        return l, s[1], r
    else:
        return l, s[1], r, s[3]


def map_nested_filter(fn, lst, *args, **kwargs):
    if isinstance(lst, list):
        if lst == []:
            return []
        return [
            elem
            for elem in (map_nested_filter(fn, elem, *args, **kwargs) for elem in lst)
            if elem is not None
        ]
    return fn(lst, *args, **kwargs)


def ext_bp(sequence, start, end, max_padding=4):
    """
    Extracts up to 'max_padding' bases from both sides of a specified region in a DNA sequence,
    considering sequence boundaries and handling extreme cases.

    Args:
    sequence (str): The DNA sequence from which to extract flanking bases.
    start (int): The zero-based start coordinate of the region (inclusive).
    end (int): The zero-based end coordinate of the region (inclusive).
    max_padding (int): Maximum number of bases to extract from each side.

    Returns:
    tuple: A tuple containing the left and right flanking bases as strings.
    """
    if start < 0 or end >= len(sequence) or start > end:
        return None, None

    # Determine the actual number of left and right padding bases considering boundaries
    left_padding = min(max_padding, start)
    right_padding = min(max_padding, len(sequence) - end - 1)

    # Extract the flanking bases
    left_bases = sequence[start - left_padding : start] if left_padding > 0 else ""
    right_bases = (
        sequence[end + 1 : end + 1 + right_padding] if right_padding > 0 else ""
    )

    return left_bases, right_bases


def get_ext(seqs):
    filtered_seqs = [s for s in seqs if s is not None]
    if not filtered_seqs:
        return None
    max_length = max(len(s) for s in filtered_seqs)
    longest_seqs = [s for s in filtered_seqs if len(s) == max_length]
    most_common_seq = (
        Counter(longest_seqs).most_common(1)[0][0] if longest_seqs else None
    )
    return most_common_seq


def conv(x):
    if x is None:
        return 0
    if math.isinf(x):
        return 1e7 if x > 0 else -1e7
    if math.isnan(x):
        return 0
    return x


def transformv(v):
    if 0 <= v <= 0.5:
        return -2 * v + 1
    elif 0.5 < v <= 1:
        return 2 * v - 1
    return None


def transformvv(v):
    if 0 <= v <= 0.5:
        return 2 * v
    elif 0.5 < v <= 1:
        return -2 * v + 2
    return None


def filter_stutter(germline, mosaic, reads_seq, reads_qual):
    bulk_reads_seq, sc_reads_seq = reads_seq
    bulk_reads_qual, sc_reads_qual = reads_qual

    bulk_reads_seq_ = []
    bulk_reads_qual_ = []
    bulk_alt = 0
    for idx, r in enumerate(bulk_reads_seq):
        if r in (germline, mosaic):
            bulk_reads_seq_.append(r)
            bulk_reads_qual_.append(bulk_reads_qual[idx])
            if r == mosaic:
                bulk_alt += 1
    alt_list_f = [bulk_alt]
    dp_list_f = [len(bulk_reads_seq_)]

    sc_reads_seq_ = []
    sc_reads_qual_ = []
    for idx_sc, sc in enumerate(sc_reads_seq):
        sc_reads_seq__ = []
        sc_reads_qual__ = []
        sc_alt = 0
        for idx, r in enumerate(sc):
            if r in (germline, mosaic):
                sc_reads_seq__.append(r)
                sc_reads_qual__.append(sc_reads_qual[idx_sc][idx])
                if r == mosaic:
                    sc_alt += 1
        alt_list_f.append(sc_alt)
        dp_list_f.append(len(sc_reads_seq__))
        sc_reads_seq_.append(sc_reads_seq__)
        sc_reads_qual_.append(sc_reads_qual__)

    return (
        tuple(alt_list_f),
        tuple(dp_list_f),
        [bulk_reads_seq_, sc_reads_seq_],
        [bulk_reads_qual_, sc_reads_qual_],
    )


def check_spn_phasing(m):
    return (
        dict(
            sorted(
                Counter(flatten_list(m.reads_data_spn_seq)).most_common(),
                key=lambda x: (x[1], x[0]),
                reverse=True,
            )
        ),
        m.phase_probs,
        m.hSNP,
        f"{m.supp[0]['gt_mosaic']}: {m.phase_probs[0].get(m.supp[0]['gt_mosaic'])}"
        if m.supp[0]["gt_mosaic"]
        else None,
    )


def check_all_reads(m):
    return (
        dict(
            sorted(
                Counter(flatten_list(m.reads_data_seq)).most_common(),
                key=lambda x: (x[1], x[0]),
                reverse=True,
            )
        ),
    )


def get_seq_from_reads_bwa(
    filtered_reads_list, reads_data_anchor, best_records=None, bams=None, filters=False
):
    reads_data_seq = []
    if best_records:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            if not best_records[idx_ind]:
                reads_data_seq.append([[], []])
                continue
            ind_bulk_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                ind_bulk_s_reads_data_seq = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][0][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    ms = segmentation(
                        read.query_sequence,
                        reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_sequence[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][0][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_sequence[idx_hSNP],)
                        else:
                            continue
                    ind_bulk_s_reads_data_seq.append(ms + hSNP)
                ind_bulk_reads_data_seq.append(ind_bulk_s_reads_data_seq)

            ind_sc_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                ind_sc_s_reads_data_seq = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][1][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    ms = segmentation(
                        read.query_sequence,
                        reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_sequence[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][1][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_sequence[idx_hSNP],)
                        else:
                            continue
                    ind_sc_s_reads_data_seq.append(ms + hSNP)
                ind_sc_reads_data_seq.append(ind_sc_s_reads_data_seq)
            reads_data_seq.append([ind_bulk_reads_data_seq, ind_sc_reads_data_seq])
        return reads_data_seq
    else:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            ind_bulk_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                if filters:
                    ind_bulk_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    ind_bulk_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                        ]
                    )

            ind_sc_reads_data_seq = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                if filters:
                    ind_sc_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    ind_sc_reads_data_seq.append(
                        [
                            segmentation(
                                read.query_sequence,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                        ]
                    )
            reads_data_seq.append([ind_bulk_reads_data_seq, ind_sc_reads_data_seq])
        return reads_data_seq


def get_qual_from_reads_bwa(
    filtered_reads_list, reads_data_anchor, best_records=None, bams=None, filters=False
):
    reads_data_qual = []
    if best_records:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            if not best_records[idx_ind]:
                reads_data_qual.append([[], []])
                continue
            ind_bulk_reads_data_qual = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                ind_bulk_s_reads_data_qual = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][0][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    ms = segmentation(
                        read.query_qualities,
                        reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_qualities[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][0][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_qualities[idx_hSNP],)
                        else:
                            continue
                    ind_bulk_s_reads_data_qual.append(ms + hSNP)
                ind_bulk_reads_data_qual.append(ind_bulk_s_reads_data_qual)

            ind_sc_reads_data_qual = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                ind_sc_s_reads_data_qual = []
                for idx_read, read in enumerate(reads):
                    if not reads_data_anchor[idx_ind][1][idx_reads][idx_read]:
                        continue
                    if filters:
                        if (
                            read.mapping_quality < 20
                            or np.mean(read.query_alignment_qualities) < 20
                        ):
                            continue
                    # MS
                    ms = segmentation(
                        read.query_qualities,
                        reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                    )
                    # hSNP
                    if (
                        0
                        <= best_records[idx_ind].start - read.reference_start
                        < read.query_length
                    ):
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in read.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (read.query_qualities[idx_hSNP],)
                        else:
                            continue
                    else:
                        mate = bams[idx_ind][1][idx_reads].mate(read)
                        idx_hSNP = next(
                            (
                                b[0]
                                for b in mate.get_aligned_pairs()
                                if b[1] == best_records[idx_ind].start
                            ),
                            None,
                        )
                        if idx_hSNP:
                            hSNP = (mate.query_qualities[idx_hSNP],)
                        else:
                            continue
                    ind_sc_s_reads_data_qual.append(ms + hSNP)
                ind_sc_reads_data_qual.append(ind_sc_s_reads_data_qual)
            reads_data_qual.append([ind_bulk_reads_data_qual, ind_sc_reads_data_qual])
        return map_nested(phred_score_q, reads_data_qual)
    else:
        for idx_ind, bam_ind in enumerate(filtered_reads_list):
            ind_bulk_reads_data_qual = []
            for idx_reads, reads in enumerate(bam_ind[0]):
                if filters:
                    ind_bulk_reads_data_qual.append(
                        [
                            segmentation(
                                read.query_qualities,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    ind_bulk_reads_data_qual.append(
                        [
                            segmentation(
                                read.query_qualities,
                                reads_data_anchor[idx_ind][0][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][0][idx_reads][idx_read]
                        ]
                    )

            ind_sc_reads_data_qual = []
            for idx_reads, reads in enumerate(bam_ind[1]):
                if filters:
                    ind_sc_reads_data_qual.append(
                        [
                            segmentation(
                                read.query_qualities,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                            and read.mapping_quality >= 20
                            and np.mean(read.query_alignment_qualities) >= 20
                        ]
                    )
                else:
                    ind_sc_reads_data_qual.append(
                        [
                            segmentation(
                                read.query_qualities,
                                reads_data_anchor[idx_ind][1][idx_reads][idx_read],
                            )
                            for idx_read, read in enumerate(reads)
                            if reads_data_anchor[idx_ind][1][idx_reads][idx_read]
                        ]
                    )
            reads_data_qual.append([ind_bulk_reads_data_qual, ind_sc_reads_data_qual])
        return map_nested(phred_score_q, reads_data_qual)


def process_group(group):
    if group["muts"].sum() >= 1:
        first_one_index = group[group["muts"] == 1].index[0]
        group["muts"] = 0
        group.loc[first_one_index, "muts"] = 1
    return group


def count_ele(nested_list):
    count = 0
    for item in nested_list:
        if isinstance(item, list):
            count += count_ele(item)
        else:
            count += 1
    return count


# @timing
def avg_dp(filtered_reads_list):
    num_r = count_ele(filtered_reads_list)
    num_sample = 0
    if not num_r:
        return 0
    for ind in filtered_reads_list:
        num_sample += len(ind[0])
        num_sample += len(ind[1])
    return num_r / num_sample if num_sample else 0


def get_nonseg_prop(nested_list):
    none_count = 0
    non_list_count = 0

    def recursive_count(item):
        nonlocal none_count, non_list_count

        if item is None:
            none_count += 1
            non_list_count += 1
        elif isinstance(item, list):
            for sub_item in item:
                recursive_count(sub_item)
        else:
            non_list_count += 1

    recursive_count(nested_list)
    return none_count / non_list_count if non_list_count else 0


def is_tr_perf(sequence, motif_length):
    sequence_length = len(sequence)
    motif = sequence[0:motif_length]
    tr_sequence = motif * (sequence_length // motif_length)
    all_tr_sequence = tr_sequence + motif[: (sequence_length % motif_length)]
    if sequence == all_tr_sequence:
        return True
    else:
        return False
