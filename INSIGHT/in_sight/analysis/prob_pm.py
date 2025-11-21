import numpy as np
from numpy import log
from scipy import stats
from scipy.special import logsumexp

# run by:
# python prob.py

def read_allele_prob(read, allele, str_unit_length, stutter_dict, read_qualities, pro_type=None, hsnp=None) -> float:
    """
    read is []
    allele is []
    read_qualities is []
    pro_type is "bulk", "scc", "mda", "pta" or "sc"
    """
    assert pro_type in ["bulk", "scc", "mda", "pta", "sc"]
    # print(f"read: {read}")
    
    # print(f"stutter_dict: {stutter_dict}")
    # print(f"type: {pro_type}")
    current_stutter = stutter_dict[pro_type]
    # print(f"current_stutter: {current_stutter}")
    
    if (len(read[1]) - len(allele[1])) % str_unit_length == 0:
        prob = (
            align_flk(read[0], allele[0], read_qualities[0])
            + align_flk(read[2], allele[2], read_qualities[2])
            + align_ms(
                allele[1],
                read[1],
                read_qualities[1],
                (
                    current_stutter[f"up_{pro_type}_in"],
                    current_stutter[f"down_{pro_type}_in"],
                    current_stutter[f"rho_{pro_type}_in"],
                    str_unit_length,
                ),
            )
            + (log(_emiss(read[3], hsnp, read_qualities[3])) if hsnp else 0)
        )
    else:
        prob = (
            align_flk(read[0], allele[0], read_qualities[0])
            + align_flk(read[2], allele[2], read_qualities[2])
            + align_ms(
                allele[1],
                read[1],
                read_qualities[1],
                (
                    current_stutter[f"up_{pro_type}_out"],
                    current_stutter[f"down_{pro_type}_out"],
                    current_stutter[f"rho_{pro_type}_out"],
                    str_unit_length,
                ),
            )
            + (log(_emiss(read[3], hsnp, read_qualities[3])) if hsnp else 0)
        )
    return prob


def _emiss(r, h, q):
    """per-base emission in the M state for function align_flk"""
    if r == h:
        return q
    else:
        return (1 - q) / 3


LARGE_NUM = 10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376


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


def no_stutter_str_alignment(
    hap, reads_str_block, reads_str_block_baseq_accuracy, insertion_rate, deletion_rate
):
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    reads_str_block_baseq_accuracy = np.array(reads_str_block_baseq_accuracy)
    alignment = hap == reads_str_block
    # stutter_error_likelihood=np.log(1-insertion_rate-deletion_rate)
    if (1 - insertion_rate - deletion_rate) == 0.0:
        stutter_error_likelihood = -np.inf
    else:
        stutter_error_likelihood = np.log(1 - insertion_rate - deletion_rate)
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
        return -np.inf
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
    
def phred_score_q(q):
    """Convert phred-scale quality to probability"""
    return 1 - (10 ** (-q / 10))
    
def classify_allelic_reads(str_unit_length, stutter, read, mu_g_allele, mu_m_allele, ref_g_allele, read_qualities, pro_type, is_heterozygous=False, include_debug_info=False):
    """
    Classifies reads as more likely originating from allele_1, allele_2, or indeterminate.

    Parameters:
    - str_unit_length: length of the STR unit.
    - stutter: Dictionary with stutter noise parameters.
    - read: List of reads to be analyzed.
    - mu_g_allele: List of reads representing the mu_g allele.
    - mu_m_allele: List of reads representing the mu_m allele.
    - ref_g_allele: List of reads representing the reference allele.
    - read_qualities: List of lists with quality scores for each base in each read.
    - pro_type: "bulk", "scc", "mda", or "pta"
    - is_heterozygous: Boolean indicating if the sample is heterozygous.
    - include_debug_info: Boolean indicating whether to include debug information in the result.

    Returns:
    - Dictionary containing classification results and optionally debug information.
    """
    converted_qualities = [[phred_score_q(q) for q in sublist] for sublist in read_qualities]
    
    try:
        read_probs = [[phred_score_q(q) for q in qualities] for qualities in read_qualities]
        
        if mu_g_allele is None:
            prob_to_mu_g = -np.inf  
        else:
            prob_to_mu_g = read_allele_prob(read, mu_g_allele, str_unit_length, stutter, read_probs, pro_type=pro_type)
        if mu_m_allele is None:
            prob_to_mu_m = -np.inf
        else:
            prob_to_mu_m = read_allele_prob(read, mu_m_allele, str_unit_length, stutter, read_probs, pro_type=pro_type)

        def create_result(allele_source, probs):
            res = {'allele_source': allele_source, 'probs': probs}
            if include_debug_info:
                res['debug_info'] = {
                    'read': read,
                    'mu_g_allele': mu_g_allele,
                    'mu_m_allele': mu_m_allele,
                    'str_unit_length': str_unit_length,
                    'stutter': stutter,
                    'read_probabilities': read_probs,
                    'read_qualities': converted_qualities
                }
            return res

        if not is_heterozygous:
            if prob_to_mu_g >= prob_to_mu_m:
                return create_result(0, [prob_to_mu_g, prob_to_mu_m])
            elif prob_to_mu_m > prob_to_mu_g:
                return create_result(1, [prob_to_mu_g, prob_to_mu_m])
            else:
                return create_result(-1, [prob_to_mu_g, prob_to_mu_m])
        else:
            prob_to_ref_g = read_allele_prob(read, ref_g_allele, str_unit_length, stutter, read_probs, pro_type=pro_type)
            probs = [prob_to_mu_g, prob_to_mu_m, prob_to_ref_g]
            
            if max(probs) != prob_to_mu_m:
                return create_result(0, probs)
            elif max(probs) == prob_to_mu_m:
                return create_result(1, probs)
            else:
                return create_result(-1, probs)

    except Exception as e:
        print(f"Error occurred: {e}")
        return create_result(-1, [])
    
# test
# region = {  # need change
#     "chr": "4",
#     "start": 112806445,
#     "end": 112806461,
#     "motif_len": 5,
#     "len": 3.4,
#     "name": "Human_STR_1072453",
#     "motif": "AAAAT",
# }

# stutter = {  # need change
#     "rho_bulk_in": 0.999,
#     "up_bulk_in": 0.0001,
#     "down_bulk_in": 0.0001,
#     "rho_sc_in": 0.999,
#     "up_sc_in": 0.0001,
#     "down_sc_in": 0.0001,
#     "rho_bulk_out": 0.999,
#     "up_bulk_out": 0.0001,
#     "down_bulk_out": 0.0001,
#     "rho_sc_out": 0.999,
#     "up_sc_out": 0.0001,
#     "down_sc_out": 0.0001,
# }

# read = ["TGTTG", "TTTTTTTTTTTTT", "AATTT"]  # need change
# allele_1 = ["TGTTG", "TTTTTTTTTTTTT", "AATTT"]  # need change
# allele_2 = ["TGTTG", "TTTTTTTTTTTTT", "ATTTG"]  # need change
# read_qualities = [[0.99] * 5, [0.99] * 13, [0.99] * 5]  # need change

# classify_allelic_reads(region=region,stutter=stutter,read=read,allele_1=allele_1,allele_2=allele_2,read_qualities=read_qualities)

# Usage
# region = {  # need change
#     "chr": "4",
#     "start": 112806445,
#     "end": 112806461,
#     "motif_len": 5,
#     "len": 3.4,
#     "name": "Human_STR_1072453",
#     "motif": "AAAAT",
# }

# stutter = {  # need change
#     "rho_bulk_in": 0.999,
#     "up_bulk_in": 0.0001,
#     "down_bulk_in": 0.0001,
#     "rho_sc_in": 0.999,
#     "up_sc_in": 0.0001,
#     "down_sc_in": 0.0001,
#     "rho_bulk_out": 0.999,
#     "up_bulk_out": 0.0001,
#     "down_bulk_out": 0.0001,
#     "rho_sc_out": 0.999,
#     "up_sc_out": 0.0001,
#     "down_sc_out": 0.0001,
# }

# if __name__ == "__main__":
#     read = ["TGTTG", "TTTTTTTTTTTTT", "AATTT"]  # need change
#     allele_1 = ["TGTTG", "TTTTTTTTTTTTT", "AATTT"]  # need change
#     allele_2 = ["TGTTG", "TTTTTTTTTTTTT", "ATTTG"]  # need change
#     read_qualities = [[0.99] * 5, [0.99] * 13, [0.99] * 5]  # need change

#     if read_allele_prob(read, allele_1, read_qualities, type="sc") > read_allele_prob(
#         read, allele_2, read_qualities, type="sc"
#     ):
#         print("allele_1")
#     else:
#         print("allele_2")
