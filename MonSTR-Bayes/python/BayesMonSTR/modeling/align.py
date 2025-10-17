# -*- coding: utf-8 -*-

import warnings
from functools import lru_cache

import numpy as np
from scipy import stats
from scipy.special import logsumexp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import numba as nb
# from numba import jit, njit, prange, types

LARGE_NUM = 10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376


@lru_cache(maxsize=512)
def align_flk(read, hap, read_qualities):
    # return 0
    delta = 10 ** (-4.5)  # gap opening
    epsilon = 0.1  # gap extension
    l1 = len(read)
    l2 = len(hap)
    if read == hap == "":
        return 0
    assert l1 == l2
    if not l2:
        return -float("inf")  # HACK: -100?
    # recurrence
    match = np.zeros([l1 + 1, l2 + 1], dtype=np.float64)
    insertion = np.zeros([l1 + 1, l2 + 1], dtype=np.float64)
    deletion = np.zeros([l1 + 1, l2 + 1], dtype=np.float64)
    match[0] = np.zeros_like(match[0])
    insertion[0] = np.zeros_like(insertion[0])
    deletion[0] = np.full_like(deletion[0], LARGE_NUM / l2)  # 2**1000 optimize
    for i in range(1, l1 + 1):  # 1-based indices
        for j in range(1, l2 + 1):
            match[i][j] = _emiss(read[i - 1], hap[j - 1], read_qualities[i - 1]) * (
                match[i - 1][j - 1] * (1 - 2 * delta)
                + insertion[i - 1][j - 1] * (1 - epsilon)
                + deletion[i - 1][j - 1] * (1 - epsilon)
            )
            insertion[i][j] = match[i - 1][j] * delta + insertion[i - 1][j] * epsilon
            deletion[i][j] = match[i][j - 1] * delta + deletion[i][j - 1] * epsilon
    # termination
    val = (insertion[l1] + match[l1]).sum() / LARGE_NUM
    if val > 0:
        return np.log(val)
    else:
        return -100  # HACK:


def _emiss(r, h, q):
    """per-base emission in the M state for function align_flk"""
    if r == h:
        return q
    else:
        return (1 - q) / 3


def no_stutter_str_alignment(
    hap, reads_str_block, reads_str_block_baseq_accuracy, insertion_rate, deletion_rate
):
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
    alignment = hap == reads_str_block
    # stutter_error_likelihood=np.log(1-insertion_rate-deletion_rate)
    if (1 - insertion_rate - deletion_rate) == 0.0:
        stutter_error_likelihood = -100  # -np.inf
    else:
        stutter_error_likelihood = np.log(1 - insertion_rate - deletion_rate)
    sequencing_error_likelihood = np.sum(
        np.log(
            np.array(
                list(
                    map(
                        lambda x, y: (1 - float(y)) / 3 if int(x) == 0 else float(y),
                        alignment,
                        reads_str_block_baseq_accuracy,
                    )
                )
            )
        )
    )
    alignment_likelihood = sequencing_error_likelihood + stutter_error_likelihood
    return alignment_likelihood


def stutter_deletion(
    hap,
    reads_str_block,
    reads_str_block_baseq_accuracy,
    deletion_rate,
    p_step,
    motif_length,
):
    deletion_size = len(hap) - len(reads_str_block)
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
    all_deletion_location = len(hap) - deletion_size + 1
    sequencing_error_likelihood_list = []
    for deletion_location in np.arange(0, all_deletion_location):
        hap_deletion = np.delete(
            hap, np.arange(deletion_location, deletion_location + deletion_size)
        )
        alignment = hap_deletion == reads_str_block
        sequencing_error_likelihood = np.sum(
            np.log(
                np.array(
                    list(
                        map(
                            lambda x, y: (1 - float(y)) / 3
                            if int(x) == 0
                            else float(y),
                            alignment,
                            reads_str_block_baseq_accuracy,
                        )
                    )
                )
            )
        )
        sequencing_error_likelihood_list.append(sequencing_error_likelihood)
    total_sequencing_error_likelihood = logsumexp(sequencing_error_likelihood_list)
    if p_step == 1.0 and (deletion_size / motif_length) > 1:
        # stutter_error_likelihood=np.log(1e-9)+np.log(deletion_rate)
        p_step = 0.999
    if deletion_rate == 0.0:
        deletion_rate = 1e-9
    if stats.geom.pmf(deletion_size / motif_length, p_step) == 0.0:
        geom_probs = stats.geom.pmf(
            np.arange(1, deletion_size / motif_length + 1, 1), p_step
        )
        geom_probs_value = geom_probs[geom_probs.nonzero()].min()
        stutter_error_likelihood = np.log(deletion_rate) + np.log(geom_probs_value)
    else:
        stutter_error_likelihood = np.log(deletion_rate) + np.log(
            stats.geom.pmf(deletion_size / motif_length, p_step)
        )
    alignment_likelihood = (
        total_sequencing_error_likelihood
        - np.log(all_deletion_location)
        + stutter_error_likelihood
    )
    return alignment_likelihood


def stutter_insertion(
    hap,
    reads_str_block,
    reads_str_block_baseq_accuracy,
    insertion_rate,
    p_step,
    motif_length,
):
    insertion_size = len(reads_str_block) - len(hap)
    if insertion_size > len(hap):
        return -len(reads_str_block)  # -np.inf
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
    all_insertion_location = len(hap) + 1
    sequencing_error_likelihood_list = []
    for insertion_location in np.arange(0, all_insertion_location):
        reads_str_block_remove_insertion = np.delete(
            reads_str_block,
            np.arange(insertion_location, insertion_location + insertion_size),
        )
        reads_str_block_remove_insertion_baseq_accuracy = np.delete(
            reads_str_block_baseq_accuracy,
            np.arange(insertion_location, insertion_location + insertion_size),
        )
        insertion_from_reads_seq = reads_str_block[
            insertion_location : insertion_location + insertion_size
        ]
        insertion_from_reads_baseq_accuracy = reads_str_block_baseq_accuracy[
            insertion_location : insertion_location + insertion_size
        ]
        reads_str_block_remove_insertion_alignment = (
            reads_str_block_remove_insertion == hap
        )
        reads_str_block_remove_insertion_sequencing_error_likelihood = np.sum(
            np.log(
                np.array(
                    list(
                        map(
                            lambda x, y: (1 - float(y)) / 3
                            if int(x) == 0
                            else float(y),
                            reads_str_block_remove_insertion_alignment,
                            reads_str_block_remove_insertion_baseq_accuracy,
                        )
                    )
                )
            )
        )
        # if insertion_location >(len(hap)-1) or (insertion_location+insertion_size) >(len(hap)):
        #     all_insertion_location-=1
        #     continue
        try:
            if insertion_location < insertion_size:
                insertion_from_hap_right_seq = hap[
                    insertion_location : insertion_location + insertion_size
                ]
                insertion_alignment = (
                    insertion_from_reads_seq == insertion_from_hap_right_seq
                )
                insertion_sequencing_error_likelihood = np.sum(
                    np.log(
                        np.array(
                            list(
                                map(
                                    lambda x, y: (1 - float(y)) / 3
                                    if int(x) == 0
                                    else float(y),
                                    insertion_alignment,
                                    insertion_from_reads_baseq_accuracy,
                                )
                            )
                        )
                    )
                )
            elif (
                insertion_location >= insertion_size
                and insertion_location <= len(hap) - insertion_size
            ):
                insertion_from_hap_left_seq = hap[
                    insertion_location - insertion_size : insertion_location
                ]
                insertion_from_hap_right_seq = hap[
                    insertion_location : insertion_location + insertion_size
                ]
                insertion_alignment_left = (
                    insertion_from_reads_seq == insertion_from_hap_left_seq
                )
                insertion_alignment_right = (
                    insertion_from_reads_seq == insertion_from_hap_right_seq
                )
                insertion_left_sequencing_error_likelihood = np.sum(
                    np.log(
                        np.array(
                            list(
                                map(
                                    lambda x, y: (1 - float(y)) / 3
                                    if int(x) == 0
                                    else float(y),
                                    insertion_alignment_left,
                                    insertion_from_reads_baseq_accuracy,
                                )
                            )
                        )
                    )
                )
                insertion_right_sequencing_error_likelihood = np.sum(
                    np.log(
                        np.array(
                            list(
                                map(
                                    lambda x, y: (1 - float(y)) / 3
                                    if int(x) == 0
                                    else float(y),
                                    insertion_alignment_right,
                                    insertion_from_reads_baseq_accuracy,
                                )
                            )
                        )
                    )
                )
                insertion_sequencing_error_likelihood_list = [
                    insertion_left_sequencing_error_likelihood,
                    insertion_right_sequencing_error_likelihood,
                ]
                insertion_sequencing_error_likelihood = logsumexp(
                    insertion_sequencing_error_likelihood_list
                ) - np.log(2)
            elif insertion_location > len(hap) - insertion_size:
                insertion_from_hap_left_seq = hap[
                    insertion_location - insertion_size : insertion_location
                ]
                insertion_alignment = (
                    insertion_from_reads_seq == insertion_from_hap_left_seq
                )
                insertion_sequencing_error_likelihood = np.sum(
                    np.log(
                        np.array(
                            list(
                                map(
                                    lambda x, y: (1 - float(y)) / 3
                                    if int(x) == 0
                                    else float(y),
                                    insertion_alignment,
                                    insertion_from_reads_baseq_accuracy,
                                )
                            )
                        )
                    )
                )
            sequencing_error_likelihood = (
                reads_str_block_remove_insertion_sequencing_error_likelihood
                + insertion_sequencing_error_likelihood
            )
            sequencing_error_likelihood_list.append(sequencing_error_likelihood)
        except:
            all_insertion_location -= 1
            continue
    total_sequencing_error_likelihood = logsumexp(sequencing_error_likelihood_list)
    if p_step == 1.0 and (insertion_size / motif_length) > 1:
        # stutter_error_likelihood=np.log(1e-9)+np.log(insertion_rate)
        p_step = 0.999
    if insertion_rate == 0.0:
        insertion_rate = 1e-9
    if stats.geom.pmf(insertion_size / motif_length, p_step) == 0.0:
        geom_probs = stats.geom.pmf(
            np.arange(1, insertion_size / motif_length + 1, 1), p_step
        )
        geom_probs_value = geom_probs[geom_probs.nonzero()].min()
        stutter_error_likelihood = np.log(insertion_rate) + np.log(geom_probs_value)
    else:
        stutter_error_likelihood = np.log(insertion_rate) + np.log(
            stats.geom.pmf(insertion_size / motif_length, p_step)
        )
    alignment_likelihood = (
        total_sequencing_error_likelihood
        - np.log(all_insertion_location)
        + stutter_error_likelihood
    )
    return alignment_likelihood


@lru_cache(maxsize=4096)
def align_ms(hap, reads_str_block, reads_str_block_baseq_accuracy, stutter_model):
    """MS-specific error model

    Parameters
    ----------
    hap : `str`
        The MS haplotype sequence.
    reads_str_block : `str`
        The MS sequence in the read.
    reads_str_block_baseq_accuracy : list
        The corrospoding base qualities.
    stutter_model : tuple[`float`, `float`, `float`, `int`]
        A trained stutter error model at the locus.
        insertion_rate, deletion_rate, p_step, motif_length.

    Returns
    -------
    `float`
        Log alignment likelihood.
    """
    insertion_rate, deletion_rate, p_step, motif_length = stutter_model
    len_read = len(reads_str_block)
    len_hap = len(hap)

    if len_hap == len_read:  # no indel
        return no_stutter_str_alignment(
            hap,
            reads_str_block,
            reads_str_block_baseq_accuracy,
            insertion_rate,
            deletion_rate,
        )

    elif len_hap > len_read:  # del
        return stutter_deletion(
            hap,
            reads_str_block,
            reads_str_block_baseq_accuracy,
            deletion_rate,
            p_step,
            motif_length,
        )

    else:  # ins
        return stutter_insertion(
            hap,
            reads_str_block,
            reads_str_block_baseq_accuracy,
            insertion_rate,
            p_step,
            motif_length,
        )


def align_ms_lrt(hap, reads_str_block, reads_str_block_baseq_accuracy, discordant_rate):
    len_read = len(reads_str_block)
    len_hap = len(hap)

    if len_hap == len_read:  # no indel
        hap = np.array(list(hap))
        reads_str_block = np.array(list(reads_str_block))
        reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
        alignment = hap == reads_str_block
        stutter_error_likelihood = np.log(1 - discordant_rate)
        sequencing_error_likelihood = np.sum(
            np.log(
                np.array(
                    list(
                        map(
                            lambda x, y: 1 - float(y) if int(x) == 0 else float(y),
                            alignment,
                            reads_str_block_baseq_accuracy,
                        )
                    )
                )
            )
        )
        alignment_likelihood = sequencing_error_likelihood + stutter_error_likelihood
        return alignment_likelihood

    elif len_hap > len_read:  # del
        deletion_size = len(hap) - len(reads_str_block)
        hap = np.array(list(hap))
        reads_str_block = np.array(list(reads_str_block))
        reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
        all_deletion_location = len(hap) - deletion_size + 1
        sequencing_error_likelihood_list = []
        for deletion_location in np.arange(0, all_deletion_location):
            hap_deletion = np.delete(
                hap, np.arange(deletion_location, deletion_location + deletion_size)
            )
            alignment = hap_deletion == reads_str_block
            sequencing_error_likelihood = np.sum(
                np.log(
                    np.array(
                        list(
                            map(
                                lambda x, y: 1 - float(y) if int(x) == 0 else float(y),
                                alignment,
                                reads_str_block_baseq_accuracy,
                            )
                        )
                    )
                )
            )
            sequencing_error_likelihood_list.append(sequencing_error_likelihood)
        total_sequencing_error_likelihood = logsumexp(sequencing_error_likelihood_list)
        stutter_error_likelihood = np.log(discordant_rate)
        alignment_likelihood = (
            total_sequencing_error_likelihood
            - np.log(all_deletion_location)
            + stutter_error_likelihood
        )
        return alignment_likelihood

    else:  # ins
        insertion_size = len(reads_str_block) - len(hap)
        if insertion_size > len(hap):
            return -len(reads_str_block)  # - np.inf
        hap = np.array(list(hap))
        reads_str_block = np.array(list(reads_str_block))
        reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
        all_insertion_location = len(hap) + 1
        sequencing_error_likelihood_list = []
        for insertion_location in np.arange(0, all_insertion_location):
            reads_str_block_remove_insertion = np.delete(
                reads_str_block,
                np.arange(insertion_location, insertion_location + insertion_size),
            )
            reads_str_block_remove_insertion_baseq_accuracy = np.delete(
                reads_str_block_baseq_accuracy,
                np.arange(insertion_location, insertion_location + insertion_size),
            )
            insertion_from_reads_seq = reads_str_block[
                insertion_location : insertion_location + insertion_size
            ]
            insertion_from_reads_baseq_accuracy = reads_str_block_baseq_accuracy[
                insertion_location : insertion_location + insertion_size
            ]
            reads_str_block_remove_insertion_alignment = (
                reads_str_block_remove_insertion == hap
            )
            reads_str_block_remove_insertion_sequencing_error_likelihood = np.sum(
                np.log(
                    np.array(
                        list(
                            map(
                                lambda x, y: 1 - float(y) if int(x) == 0 else float(y),
                                reads_str_block_remove_insertion_alignment,
                                reads_str_block_remove_insertion_baseq_accuracy,
                            )
                        )
                    )
                )
            )
            # if insertion_location >(len(hap)-1) or (insertion_location+insertion_size) >(len(hap)):
            #     all_insertion_location-=1
            #     continue
            try:
                if insertion_location < insertion_size:
                    insertion_from_hap_right_seq = hap[
                        insertion_location : insertion_location + insertion_size
                    ]
                    insertion_alignment = (
                        insertion_from_reads_seq == insertion_from_hap_right_seq
                    )
                    insertion_sequencing_error_likelihood = np.sum(
                        np.log(
                            np.array(
                                list(
                                    map(
                                        lambda x, y: (
                                            1 - float(y) if int(x) == 0 else float(y)
                                        ),
                                        insertion_alignment,
                                        insertion_from_reads_baseq_accuracy,
                                    )
                                )
                            )
                        )
                    )
                elif (
                    insertion_location >= insertion_size
                    and insertion_location <= len(hap) - insertion_size
                ):
                    insertion_from_hap_left_seq = hap[
                        insertion_location - insertion_size : insertion_location
                    ]
                    insertion_from_hap_right_seq = hap[
                        insertion_location : insertion_location + insertion_size
                    ]
                    insertion_alignment_left = (
                        insertion_from_reads_seq == insertion_from_hap_left_seq
                    )
                    insertion_alignment_right = (
                        insertion_from_reads_seq == insertion_from_hap_right_seq
                    )
                    insertion_left_sequencing_error_likelihood = np.sum(
                        np.log(
                            np.array(
                                list(
                                    map(
                                        lambda x, y: (
                                            1 - float(y) if int(x) == 0 else float(y)
                                        ),
                                        insertion_alignment_left,
                                        insertion_from_reads_baseq_accuracy,
                                    )
                                )
                            )
                        )
                    )
                    insertion_right_sequencing_error_likelihood = np.sum(
                        np.log(
                            np.array(
                                list(
                                    map(
                                        lambda x, y: (
                                            1 - float(y) if int(x) == 0 else float(y)
                                        ),
                                        insertion_alignment_right,
                                        insertion_from_reads_baseq_accuracy,
                                    )
                                )
                            )
                        )
                    )
                    insertion_sequencing_error_likelihood_list = [
                        insertion_left_sequencing_error_likelihood,
                        insertion_right_sequencing_error_likelihood,
                    ]
                    insertion_sequencing_error_likelihood = logsumexp(
                        insertion_sequencing_error_likelihood_list
                    ) - np.log(2)
                elif insertion_location > len(hap) - insertion_size:
                    insertion_from_hap_left_seq = hap[
                        insertion_location - insertion_size : insertion_location
                    ]
                    insertion_alignment = (
                        insertion_from_reads_seq == insertion_from_hap_left_seq
                    )
                    insertion_sequencing_error_likelihood = np.sum(
                        np.log(
                            np.array(
                                list(
                                    map(
                                        lambda x, y: (
                                            1 - float(y) if int(x) == 0 else float(y)
                                        ),
                                        insertion_alignment,
                                        insertion_from_reads_baseq_accuracy,
                                    )
                                )
                            )
                        )
                    )
                sequencing_error_likelihood = (
                    reads_str_block_remove_insertion_sequencing_error_likelihood
                    + insertion_sequencing_error_likelihood
                )
                sequencing_error_likelihood_list.append(sequencing_error_likelihood)
            except:
                all_insertion_location -= 1
                continue
        total_sequencing_error_likelihood = logsumexp(sequencing_error_likelihood_list)
        stutter_error_likelihood = np.log(discordant_rate)
        alignment_likelihood = (
            total_sequencing_error_likelihood
            - np.log(all_insertion_location)
            + stutter_error_likelihood
        )
        return alignment_likelihood
