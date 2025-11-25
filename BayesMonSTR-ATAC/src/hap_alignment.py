import numpy as np
from scipy.special import logsumexp
import config_params
from functools import lru_cache

HAP_ALIGNMENT = config_params.HAP_ALIGNMENT
CONSIDER_INS_BASEQ = HAP_ALIGNMENT["CONSIDER_INS_BASEQ"]  # True
MAX_LIK_ALIGNMENT = HAP_ALIGNMENT["MAX_LIK_ALIGNMENT"]  # True
IMPOSSIBLE_INS_RATE = HAP_ALIGNMENT["IMPOSSIBLE_INS_RATE"]  # np.log(10**-9)
# DEFAULT_MAX_STEP = 100
# HACK: For default steps and max step limitation for KeyErrors
# DEBUG: (1-Q) / 3
MAX_LIK_INS_BASEQ = True
 

# @lru_cache(maxsize=4096)
def no_stutter_str_alignment_add_mis_score(
    hap,
    reads_str_block,
    reads_str_block_baseq_accuracy,
    non_stutter_rate
    # in_frame_insertion_rate,
    # in_frame_deletion_rate,
    # out_frame_insertion_rate,
    # out_frame_deletion_rate,
):
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
    alignment = hap == reads_str_block
    mis_score = len(alignment) - np.sum(alignment)
    max_alignment_length = len(alignment)
    stutter_error_likelihood = non_stutter_rate
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
    alignment_likelihood = (
        sequencing_error_likelihood + stutter_error_likelihood
    )
    return alignment_likelihood, mis_score, max_alignment_length

# @lru_cache(maxsize=4096)
def stutter_deletion_add_mis_score(
    hap,
    reads_str_block,
    reads_str_block_baseq_accuracy,
    steps,
    log_stutter_model_dic,
    # deletion_rate,
    # p_step,
    # motif_length,
):
    deletion_size = len(hap) - len(reads_str_block)
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
    all_deletion_location = len(hap) - deletion_size + 1
    sequencing_error_likelihood_list = []
    alignment_list = []
    for deletion_location in np.arange(0, all_deletion_location):
        hap_deletion = np.delete(
            hap,
            np.arange(deletion_location, deletion_location + deletion_size),
        )
        alignment = hap_deletion == reads_str_block
        alignment_list.append(alignment)
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

    total_sequencing_error_likelihood = logsumexp(
        sequencing_error_likelihood_list
    )
    # if p_step == 1.0 and (deletion_size / motif_length) > 1:
    #     # stutter_error_likelihood=np.log(1e-9)+np.log(deletion_rate)
    #     p_step = 0.999
    # if p_step == 1.0:
    #     p_step = MAX_PGEOM
    # if deletion_rate == 0.0:
    #     deletion_rate = MIN_INDEL_RATE
    stutter_error_likelihood = log_stutter_model_dic.get(steps, log_stutter_model_dic[min(log_stutter_model_dic.keys())])
    # if in_or_out_frame == "in_frame":
    # stutter_error_likelihood = log_stutter_model_dic[-(deletion_size / motif_length)]
    # geom_pmf_prob = stats.geom.pmf(deletion_size / motif_length, p_step)
    # else:
    # step = deletion_size - deletion_size // motif_length
    # stutter_error_likelihood = log_stutter_model_dic[-step]
    # step = deletion_size - deletion_size // motif_length
    # geom_pmf_prob = stats.geom.pmf(step, p_step)
    # if geom_pmf_prob == 0.0:
    #     if in_or_out_frame == "in_frame":
    #         geom_probs = stats.geom.pmf(
    #             np.arange(1, deletion_size / motif_length + 1, 1), p_step
    #         )
    #     else:
    #         geom_probs = stats.geom.pmf(np.arange(1, step + 1, 1), p_step)
    #     geom_probs_value = geom_probs[geom_probs.nonzero()].min()
    #     stutter_error_likelihood = np.log(deletion_rate) + np.log(
    #         geom_probs_value
    #     )
    # else:
    #     stutter_error_likelihood = np.log(deletion_rate) + np.log(
    #         geom_pmf_prob
    #     )
    if MAX_LIK_ALIGNMENT:
        alignment_likelihood = (
            max(sequencing_error_likelihood_list) + stutter_error_likelihood
        )
    else:
        alignment_likelihood = (
            total_sequencing_error_likelihood
            - np.log(all_deletion_location)
            + stutter_error_likelihood
        )
    max_alignment_index = np.argmax(sequencing_error_likelihood_list)
    max_alignment = alignment_list[max_alignment_index]
    mis_score = len(max_alignment) - np.sum(max_alignment)
    max_alignment_length = len(max_alignment)
    return alignment_likelihood, mis_score, max_alignment_length, max_alignment_index

# @lru_cache(maxsize=4096)
def stutter_insertion_add_mis_score(
    # in_or_out_frame,
    hap,
    reads_str_block,
    reads_str_block_baseq_accuracy,
    steps,
    log_stutter_model_dic,
    # insertion_rate,
    # p_step,
    # motif_length,
):
    insertion_size = len(reads_str_block) - len(hap)
    # if CONSIDER_INS_BASEQ and insertion_size > len(hap):
    #     sequencing_error_likelihood_list = [IMPOSSIBLE_INS_RATE]
    #     total_sequencing_error_likelihood = IMPOSSIBLE_INS_RATE
    #     all_insertion_location = 1
    #     mis_score = 0
    # else:
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(
        reads_str_block_baseq_accuracy
    )
    all_insertion_location = len(hap) + 1
    sequencing_error_likelihood_list = []
    alignment_list = []
    for insertion_location in np.arange(0, all_insertion_location):
        reads_str_block_remove_insertion = np.delete(
            reads_str_block,
            np.arange(
                insertion_location, insertion_location + insertion_size
            ),
        )
        reads_str_block_remove_insertion_baseq_accuracy = np.delete(
            reads_str_block_baseq_accuracy,
            np.arange(
                insertion_location, insertion_location + insertion_size
            ),
        )
        insertion_from_reads_seq = reads_str_block[
            insertion_location : insertion_location + insertion_size
        ]
        insertion_from_reads_baseq_accuracy = (
            reads_str_block_baseq_accuracy[
                insertion_location : insertion_location + insertion_size
            ]
        )
        reads_str_block_remove_insertion_alignment = (
            reads_str_block_remove_insertion == hap
        )
        alignment_list.append(reads_str_block_remove_insertion_alignment)
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
        # if CONSIDER_INS_BASEQ and insertion_size > len(hap):
        if insertion_size > len(hap):
            insertion_sequencing_error_likelihood = IMPOSSIBLE_INS_RATE
        else:
            try:  # 存在不可能插入的位点，因为没有足够的局部重复长度来产生这样的插入，这些位点不考虑其插入的可能性
                if insertion_location < insertion_size:
                    insertion_from_hap_right_seq = hap[
                        insertion_location : insertion_location
                        + insertion_size
                    ]
                    insertion_alignment = (
                        insertion_from_reads_seq
                        == insertion_from_hap_right_seq
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
                        insertion_location
                        - insertion_size : insertion_location
                    ]
                    insertion_from_hap_right_seq = hap[
                        insertion_location : insertion_location
                        + insertion_size
                    ]
                    insertion_alignment_left = (
                        insertion_from_reads_seq == insertion_from_hap_left_seq
                    )
                    insertion_alignment_right = (
                        insertion_from_reads_seq
                        == insertion_from_hap_right_seq
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
                    if MAX_LIK_INS_BASEQ:
                        insertion_sequencing_error_likelihood = max(insertion_sequencing_error_likelihood_list)
                    else:
                        insertion_sequencing_error_likelihood = logsumexp(
                            insertion_sequencing_error_likelihood_list
                        ) - np.log(2)
                elif insertion_location > len(hap) - insertion_size:
                    insertion_from_hap_left_seq = hap[
                        insertion_location
                        - insertion_size : insertion_location
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
            except:
                all_insertion_location -= 1
                continue
        if CONSIDER_INS_BASEQ:
            sequencing_error_likelihood = (
                reads_str_block_remove_insertion_sequencing_error_likelihood
                + insertion_sequencing_error_likelihood
            )
        else:
            sequencing_error_likelihood = reads_str_block_remove_insertion_sequencing_error_likelihood
        sequencing_error_likelihood_list.append(
            sequencing_error_likelihood
        )
    max_alignment_index = np.argmax(sequencing_error_likelihood_list)
    max_alignment = alignment_list[max_alignment_index]
    mis_score = len(max_alignment) - np.sum(max_alignment)
    max_alignment_length = len(max_alignment)
    total_sequencing_error_likelihood = logsumexp(
        sequencing_error_likelihood_list
    )
    stutter_error_likelihood = log_stutter_model_dic.get(steps, log_stutter_model_dic[max(log_stutter_model_dic.keys())])
    # if p_step == 1.0:
    #     # if p_step == 1.0 and (insertion_size / motif_length) > 1:
    #     #     # stutter_error_likelihood=np.log(1e-9)+np.log(insertion_rate)
    #     p_step = MAX_PGEOM
    # if insertion_rate == 0.0:
    #     insertion_rate = MIN_INDEL_RATE
    # if in_or_out_frame == "in_frame":
    #     geom_pmf_prob = stats.geom.pmf(insertion_size / motif_length, p_step)
    # else:
    #     step = insertion_size - insertion_size // motif_length
    #     geom_pmf_prob = stats.geom.pmf(step, p_step)
    # if geom_pmf_prob == 0.0:
    #     if in_or_out_frame == "in_frame":
    #         geom_probs = stats.geom.pmf(
    #             np.arange(1, insertion_size / motif_length + 1, 1), p_step
    #         )
    #     else:
    #         geom_probs = stats.geom.pmf(np.arange(1, step + 1, 1), p_step)
    #     geom_probs_value = geom_probs[geom_probs.nonzero()].min()
    #     stutter_error_likelihood = np.log(insertion_rate) + np.log(
    #         geom_probs_value
    #     )
    # else:
    #     stutter_error_likelihood = np.log(insertion_rate) + np.log(
    #         geom_pmf_prob
    #     )
    if MAX_LIK_ALIGNMENT:
        alignment_likelihood = (
            max(sequencing_error_likelihood_list) + stutter_error_likelihood
        )
    else:
        alignment_likelihood = (
            total_sequencing_error_likelihood
            - np.log(all_insertion_location)
            + stutter_error_likelihood
        )
    return alignment_likelihood, mis_score, max_alignment_length, max_alignment_index


# @lru_cache(maxsize=4096)
def no_stutter_str_alignment(
    hap,
    reads_str_block,
    reads_str_block_baseq_accuracy,
    non_stutter_rate
    # in_frame_insertion_rate,
    # in_frame_deletion_rate,
    # out_frame_insertion_rate,
    # out_frame_deletion_rate,
):
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
    alignment = hap == reads_str_block
    stutter_error_likelihood = non_stutter_rate
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
    alignment_likelihood = (
        sequencing_error_likelihood + stutter_error_likelihood
    )
    return alignment_likelihood

# @lru_cache(maxsize=4096)
def stutter_deletion(
    hap,
    reads_str_block,
    reads_str_block_baseq_accuracy,
    steps,
    log_stutter_model_dic,
    # deletion_rate,
    # p_step,
    # motif_length,
):
    deletion_size = len(hap) - len(reads_str_block)
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
    all_deletion_location = len(hap) - deletion_size + 1
    sequencing_error_likelihood_list = []
    for deletion_location in np.arange(0, all_deletion_location):
        hap_deletion = np.delete(
            hap,
            np.arange(deletion_location, deletion_location + deletion_size),
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

    total_sequencing_error_likelihood = logsumexp(
        sequencing_error_likelihood_list
    )
    # if p_step == 1.0 and (deletion_size / motif_length) > 1:
    #     # stutter_error_likelihood=np.log(1e-9)+np.log(deletion_rate)
    #     p_step = 0.999
    # if p_step == 1.0:
    #     p_step = MAX_PGEOM
    # if deletion_rate == 0.0:
    #     deletion_rate = MIN_INDEL_RATE
    stutter_error_likelihood = log_stutter_model_dic.get(steps, log_stutter_model_dic[min(log_stutter_model_dic.keys())])
    # if in_or_out_frame == "in_frame":
    # stutter_error_likelihood = log_stutter_model_dic[-(deletion_size / motif_length)]
    # geom_pmf_prob = stats.geom.pmf(deletion_size / motif_length, p_step)
    # else:
    # step = deletion_size - deletion_size // motif_length
    # stutter_error_likelihood = log_stutter_model_dic[-step]
    # step = deletion_size - deletion_size // motif_length
    # geom_pmf_prob = stats.geom.pmf(step, p_step)
    # if geom_pmf_prob == 0.0:
    #     if in_or_out_frame == "in_frame":
    #         geom_probs = stats.geom.pmf(
    #             np.arange(1, deletion_size / motif_length + 1, 1), p_step
    #         )
    #     else:
    #         geom_probs = stats.geom.pmf(np.arange(1, step + 1, 1), p_step)
    #     geom_probs_value = geom_probs[geom_probs.nonzero()].min()
    #     stutter_error_likelihood = np.log(deletion_rate) + np.log(
    #         geom_probs_value
    #     )
    # else:
    #     stutter_error_likelihood = np.log(deletion_rate) + np.log(
    #         geom_pmf_prob
    #     )
    if MAX_LIK_ALIGNMENT:
        alignment_likelihood = (
            max(sequencing_error_likelihood_list) + stutter_error_likelihood
        )
    else:
        alignment_likelihood = (
            total_sequencing_error_likelihood
            - np.log(all_deletion_location)
            + stutter_error_likelihood
        )
    return alignment_likelihood


# @lru_cache(maxsize=4096)
def stutter_insertion(
    # in_or_out_frame,
    hap,
    reads_str_block,
    reads_str_block_baseq_accuracy,
    steps,
    log_stutter_model_dic,
    # insertion_rate,
    # p_step,
    # motif_length,
):
    insertion_size = len(reads_str_block) - len(hap)
    # if CONSIDER_INS_BASEQ and insertion_size > len(hap):
    #     sequencing_error_likelihood_list = [IMPOSSIBLE_INS_RATE]
    #     total_sequencing_error_likelihood = IMPOSSIBLE_INS_RATE
    #     all_insertion_location = 1
    # else:
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(
        reads_str_block_baseq_accuracy
    )
    all_insertion_location = len(hap) + 1
    sequencing_error_likelihood_list = []
    for insertion_location in np.arange(0, all_insertion_location):
        reads_str_block_remove_insertion = np.delete(
            reads_str_block,
            np.arange(
                insertion_location, insertion_location + insertion_size
            ),
        )
        reads_str_block_remove_insertion_baseq_accuracy = np.delete(
            reads_str_block_baseq_accuracy,
            np.arange(
                insertion_location, insertion_location + insertion_size
            ),
        )
        insertion_from_reads_seq = reads_str_block[
            insertion_location : insertion_location + insertion_size
        ]
        insertion_from_reads_baseq_accuracy = (
            reads_str_block_baseq_accuracy[
                insertion_location : insertion_location + insertion_size
            ]
        )
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
        if insertion_size > len(hap):
            insertion_sequencing_error_likelihood = IMPOSSIBLE_INS_RATE
        else:
            try:  # 存在不可能插入的位点，因为没有足够的局部重复长度来产生这样的插入，这些位点不考虑其插入的可能性
                if insertion_location < insertion_size:
                    insertion_from_hap_right_seq = hap[
                        insertion_location : insertion_location
                        + insertion_size
                    ]
                    insertion_alignment = (
                        insertion_from_reads_seq
                        == insertion_from_hap_right_seq
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
                        insertion_location
                        - insertion_size : insertion_location
                    ]
                    insertion_from_hap_right_seq = hap[
                        insertion_location : insertion_location
                        + insertion_size
                    ]
                    insertion_alignment_left = (
                        insertion_from_reads_seq == insertion_from_hap_left_seq
                    )
                    insertion_alignment_right = (
                        insertion_from_reads_seq
                        == insertion_from_hap_right_seq
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
                    if MAX_LIK_INS_BASEQ:
                        insertion_sequencing_error_likelihood = max(insertion_sequencing_error_likelihood_list)
                    else:
                        insertion_sequencing_error_likelihood = logsumexp(
                            insertion_sequencing_error_likelihood_list
                        ) - np.log(2)
                elif insertion_location > len(hap) - insertion_size:
                    insertion_from_hap_left_seq = hap[
                        insertion_location
                        - insertion_size : insertion_location
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
            except:
                all_insertion_location -= 1
                continue
        if CONSIDER_INS_BASEQ:
            sequencing_error_likelihood = (
                reads_str_block_remove_insertion_sequencing_error_likelihood
                + insertion_sequencing_error_likelihood
            )
        else:
            sequencing_error_likelihood = reads_str_block_remove_insertion_sequencing_error_likelihood
        sequencing_error_likelihood_list.append(
            sequencing_error_likelihood
        )
    total_sequencing_error_likelihood = logsumexp(
        sequencing_error_likelihood_list
    )
    stutter_error_likelihood = log_stutter_model_dic.get(steps, log_stutter_model_dic[max(log_stutter_model_dic.keys())])
    # if p_step == 1.0:
    #     # if p_step == 1.0 and (insertion_size / motif_length) > 1:
    #     #     # stutter_error_likelihood=np.log(1e-9)+np.log(insertion_rate)
    #     p_step = MAX_PGEOM
    # if insertion_rate == 0.0:
    #     insertion_rate = MIN_INDEL_RATE
    # if in_or_out_frame == "in_frame":
    #     geom_pmf_prob = stats.geom.pmf(insertion_size / motif_length, p_step)
    # else:
    #     step = insertion_size - insertion_size // motif_length
    #     geom_pmf_prob = stats.geom.pmf(step, p_step)
    # if geom_pmf_prob == 0.0:
    #     if in_or_out_frame == "in_frame":
    #         geom_probs = stats.geom.pmf(
    #             np.arange(1, insertion_size / motif_length + 1, 1), p_step
    #         )
    #     else:
    #         geom_probs = stats.geom.pmf(np.arange(1, step + 1, 1), p_step)
    #     geom_probs_value = geom_probs[geom_probs.nonzero()].min()
    #     stutter_error_likelihood = np.log(insertion_rate) + np.log(
    #         geom_probs_value
    #     )
    # else:
    #     stutter_error_likelihood = np.log(insertion_rate) + np.log(
    #         geom_pmf_prob
    #     )
    if MAX_LIK_ALIGNMENT:
        alignment_likelihood = (
            max(sequencing_error_likelihood_list) + stutter_error_likelihood
        )
    else:
        alignment_likelihood = (
            total_sequencing_error_likelihood
            - np.log(all_insertion_location)
            + stutter_error_likelihood
        )
    return alignment_likelihood


## BELOW is from WENXUAN FAN LIKE GATK pairhmm.pdf ##
## BUT MAY EXIST SOME ISSUES THROUGH TESTING ##
import numpy as np

# from numba import jit, njit, prange, types
from functools import lru_cache

# from scipy import stats
from scipy.special import logsumexp


LARGE_NUM = 10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376

# ? https://docs.python.org/3/library/functools.html#functools.lru_cache ?#
# Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can save time when an expensive or I/O bound function is periodically called with the same arguments.

# The cache is threadsafe so that the wrapped function can be used in multiple threads. This means that the underlying data structure will remain coherent during concurrent updates.

# It is possible for the wrapped function to be called more than once if another thread makes an additional call before the initial call has been completed and cached.

# Since a dictionary is used to cache results, the positional and keyword arguments to the function must be hashable.


# Distinct argument patterns may be considered to be distinct calls with separate cache entries. For example, f(a=1, b=2) and f(b=2, a=1) differ in their keyword argument order and may have two separate cache entries.
# That the i = 0 rows of M and I are initialized to zero corresponds to starting at an imaginary position one base before the read start in a deletion state4. The factor of 1/|H| corresponds to a flat prior on which j an alignment starts at. This is important for a local alignment because we don’t want to penalize reads that start in the middle of the haplotype. The curious factor of 21020 is a huge number to prevent underflow – all multiplications are by numbers less than 1, so we needn’t worry about overflow – which is a much more efficient approach than performing the computation in log space. The omission of D from the returned value recognizes the fact that a terminal deletion is meaningless.
# Finally, we note a shortcut that the GATK exploits: when two consecutive haplotypes of the same length5 agree up to the kth position, the first k columns of M, D and I are recycled and the inner loop is over k < j ≤ H.
# 4 Since TDI = 0 this means that an alignment may not start with an insertion, though it may begin with one. This limitation is unnecessary and could be fixed by initializing the i = 0 row of I to a non-zero value as well.
# 5 The condition on the same length could easily be relaxed simply by accounting for the constant 1/|H| initialization.
# GATK Pair HMM probabilistic realignment in HaplotypeCaller and Mutect pair_hmm.pdf
@lru_cache(maxsize=512)
def align_flk(read, hap, read_qualities):
    # DEBUG:
    # return 0
    delta = 10 ** (-4.5)  # gap opening
    epsilon = 0.1  # gap extension
    l1 = len(read)
    l2 = len(hap)
    if not l2:
        return -float("inf")  # HACK: -100?
    # recurrence
    match = np.zeros([l1 + 1, l2 + 1], dtype=np.float64)
    insertion = np.zeros([l1 + 1, l2 + 1], dtype=np.float64)
    deletion = np.zeros([l1 + 1, l2 + 1], dtype=np.float64)
    match[0] = np.zeros_like(
        match[0]
    )  # ? np.zeros_like 函数对于需要创建一个与给定数组形状相同的全零数组非常方便。 ?#
    insertion[0] = np.zeros_like(insertion[0])
    deletion[0] = np.full_like(
        deletion[0], LARGE_NUM / l2
    )  # 2**1000 optimize #???????? why so big value ?????????#
    for i in range(1, l1 + 1):  # 1-based indices
        for j in range(1, l2 + 1):
            match[i][j] = _emiss(
                read[i - 1], hap[j - 1], read_qualities[i - 1]
            ) * (
                match[i - 1][j - 1] * (1 - 2 * delta)
                + insertion[i - 1][j - 1] * (1 - epsilon)
                + deletion[i - 1][j - 1] * (1 - epsilon)
            )  # ??????? correct transition ????????#
            insertion[i][j] = (
                match[i - 1][j] * delta + insertion[i - 1][j] * epsilon
            )  # ??????? correct transition ????????#
            deletion[i][j] = (
                match[i][j - 1] * delta + deletion[i][j - 1] * epsilon
            )  # ??????? correct transition ????????#
    # termination
    val = (insertion[l1] + match[l1]).sum() / LARGE_NUM
    if val > 0:
        return np.log(val)
    else:
        return -100  # HACK:


def _emiss(r, h, q):  # ? calculate base accuracy for match base and allele ?#
    """per-base emission in the M state for function align_flk"""
    if r == h:
        return q
    else:
        return (1 - q) / 3


## TEST ABOUT FLANKING ALIGNMENT ##

# np.exp(align_flk("ATCG","ATCG",tuple([0.99,0.99,0.99,0.99])))
# 0.21610081389892516

# np.exp(align_flk("ATGG","ATCG",tuple([0.99,0.99,0.99,0.99])))
# 0.0007284035433426598

# 0.21610081389892516/0.0007284035433426598 = 296.67732381865386
# 0.99/(0.01/3) = 297.0

## ABOVE is from WENXUAN FAN LIKE GATK pairhmm.pdf ##

## TODO
## segmentation methods 2, based on coordinate and flanking indel ##
## hap alignment methods from DINDEL ##
