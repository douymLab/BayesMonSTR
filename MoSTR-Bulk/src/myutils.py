import logging
import fcntl
import pysam
import itertools
import numpy as np
from scipy.stats import geom
import hap_alignment

# import sys
# sys.path.append("../configs")
import config_params

# from ..configs import config_params
import collections

EM_PARAMS = config_params.EM_PARAMS
MIN_STUTTER_INS = EM_PARAMS["init_params"]["MIN_STUTTER_INS"]  # 0.000001
MIN_STUTTER_DEL = EM_PARAMS["init_params"]["MIN_STUTTER_DEL"]  # 0.000001
MAX_SINGLE_STEP_PROB = EM_PARAMS["init_params"][
    "MAX_SINGLE_STEP_PROB"
]  # 0.999
MOSAIC_FRACTION_ESTIMATION_PARAMS = (
    config_params.MOSAIC_FRACTION_ESTIMATION_PARAMS
)
LIKELIHOOD_MODE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "likelihood_mode"
]  # "seq-based"
PHASING_PARAMS = config_params.PHASING_PARAMS
MIN_PER_READ_LIKELIHOOD = PHASING_PARAMS["MIN_PER_READ_LIKELIHOOD"]  # 1e-10
MIN_NO_STUTTER_RATE = EM_PARAMS["init_params"]["MIN_NO_STUTTER_RATE"]  # 0.0001
MIN_SINGLE_STEP_PROB = EM_PARAMS["init_params"][
    "MIN_SINGLE_STEP_PROB"
]  # 0.001
MAX_STUTTER_INS = EM_PARAMS["init_params"]["MAX_STUTTER_INS"]  # 0.9
MAX_STUTTER_DEL = EM_PARAMS["init_params"]["MAX_STUTTER_DEL"]  # 0.9
DONT_ASSIGNMENT_FOR_IDENTICAL_PROBS = True
ALLELE_EXTRACT = config_params.ALLELE_EXTRACT
PADDING_BPS = ALLELE_EXTRACT["PADDING_BPS"]  # 5 # 10 # revise_padding
DEBUG = False
STR_LENGTH_CUTOFF = 9
MOTIF_LENGTH_CUTOFF = 4
STRAND_DEPTH_CUTOFF = 5
ACCURACY_FRACTION_CUTOFF = 0.5
SOURCE_CHECK_TARGET_DEPTH_CUTOFF = 3
ALLOW_NUMPY_MIN_VALUE = np.finfo(float).tiny  # 1e-300

def baseq_2_error_rate(baseq) -> float:
    """Convert base quality to error rate.

    Args:
    ----
        baseq ([float]): [phred-scaled base quality]

    Returns:
    -------
        [float]: [error rate]
    """
    return np.power(10, baseq / (-10))


def acquire_lock(file):
    try:
        fcntl.flock(file, fcntl.LOCK_EX)
        return True
    except BlockingIOError:
        return False


def release_lock(file):
    fcntl.flock(file, fcntl.LOCK_UN)


class LockedFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        # 在写入日志前获取锁
        fd = self.stream.fileno()
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            super().emit(record)
        finally:
            # 确保释放锁
            fcntl.flock(fd, fcntl.LOCK_UN)


def check_reference_fasta_name(fasta):
    with open(fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                fasta_chromosome_name = line.strip().split()[0][1:]
                break
    return fasta_chromosome_name


def check_reference_bed_name(bed):
    with pysam.TabixFile(bed) as tbx:
        for record in tbx.fetch():
            bed_chromosome_name = record.split("\t")[
                0
            ]  # Split by tab and take the first column
            break  # This will exit after fetching the first record
    return bed_chromosome_name


def get_all_gt_unordered_combinations_with_replacement(
    allele_index_list, ploidy
):
    return list(
        itertools.combinations_with_replacement(allele_index_list, ploidy)
    )  # return [tuple, tuple]


def get_all_gt_ordered_permutations_with_replacement(
    allele_index_list, ploidy
):
    return list(
        itertools.product(allele_index_list, repeat=ploidy)
    )  # return [tuple, tuple]


def get_mutated_to_mutant_allele_permutations_without_replacement(
    allele_index_list, ploidy
):
    return list(
        itertools.permutations(allele_index_list, ploidy)
    )  # return [tuple, tuple]


def stutter_model(
    stutter_single_step_prob,
    stutter_ins_prob,
    stutter_del_prob,
    max_stutter_steps,
):
    # XXX Avoid python imprecision float calculation to produce the value > 1 or < 0
    # HACK: A more optimal solution for stutter_single_step_prob = 1 is necessary
    # stutter_single_step_prob = min(
    #     [stutter_single_step_prob, MAX_SINGLE_STEP_PROB]
    # )
    # stutter_ins_prob = max([stutter_ins_prob, MIN_STUTTER_INS])
    # stutter_del_prob = max([stutter_del_prob, MIN_STUTTER_DEL])
    max_stutter_steps = max(
        [max_stutter_steps, 1]
    )  # Prevent errors when invoking the homopolymer outframe stutter model function.
    # XXX Avoid key error because stutter model don‘t consider other samples’ alleles when initialized

    # stutter_single_step_prob = max(
    #     [stutter_single_step_prob, MIN_SINGLE_STEP_PROB]
    # )
    # stutter_ins_prob = min([stutter_ins_prob, MAX_STUTTER_INS])
    # stutter_del_prob = min([stutter_del_prob, MAX_STUTTER_DEL])
    steps = np.arange(1, max_stutter_steps + 1, 1)
    step_probs = geom.pmf(steps, stutter_single_step_prob)
    if step_probs.min() == 0.0:
        step_probs[step_probs == 0.0] = step_probs[step_probs.nonzero()].min()
    stutter_sizes = list(map(lambda x: -x, steps[::-1])) + [0] + list(steps)
    no_stutter_rate = max(
        [
            1.0 - stutter_del_prob - stutter_ins_prob,
            MIN_NO_STUTTER_RATE,
        ]
    )
    stutter_probs = (
        list(np.log(stutter_del_prob + ALLOW_NUMPY_MIN_VALUE) + np.log(step_probs[::-1]))
        + [np.log(no_stutter_rate)]
        + list(np.log(stutter_ins_prob + ALLOW_NUMPY_MIN_VALUE) + np.log(step_probs))
    )  # HACK: avoid log(0) through plus a small value DEBUG DONE
    log_stutter_model_dic = dict(zip(stutter_sizes, stutter_probs))
    return log_stutter_model_dic


def cal_observed_haplotype_depth_dict_phase(
    phase_read_list, alleles_order, nearby_snp
):
    ## Max Lik assignment or Posterior add up or observed allele number
    ## 分母选择: all alleles within specific range or all observed alleles or selected mosaic alleles
    hap_depth_list = []
    phase_hap_depth = dict(collections.Counter(phase_read_list))
    for snp in nearby_snp:
        for str in alleles_order:
            hap_depth_list.append(phase_hap_depth.get((str, snp), 0))
    return hap_depth_list


def approximation_flanking_mismatch_check(mis_loc, allele_length):
    if ((mis_loc in list(range(0, PADDING_BPS))) or (mis_loc in list(range(allele_length - PADDING_BPS, allele_length)))):
        return True
    else:
        return False


def cal_within_mosaic_gts_stutter_rate(
    inframe_stutter_model_dict,
    outframe_stutter_model_dict,
    motif_length,
    snv_error_rate,
    neabysnp_list,
    source_hap_length,
    stutter_hap1_length,
    stutter_hap2_length,
    stutter_hap3_length,
):
    str_bp_diff_allele_1 = stutter_hap1_length - source_hap_length
    str_bp_diff_allele_2 = stutter_hap2_length - source_hap_length
    str_bp_diff_allele_3 = stutter_hap3_length - source_hap_length
    snv_log_lik_allele1 = (
        np.log(1 - snv_error_rate)
        if neabysnp_list[0] == neabysnp_list[1]
        else np.log(snv_error_rate/3)
    )
    snv_log_lik_allele2 = (
        np.log(1 - snv_error_rate)
        if neabysnp_list[0] == neabysnp_list[2]
        else np.log(snv_error_rate/3)
    )
    snv_log_lik_allele3 = (
        np.log(1 - snv_error_rate)
        if neabysnp_list[0] == neabysnp_list[3]
        else np.log(snv_error_rate/3)
    )
    log_stutter_probs_allele1 = cal_stutter_probs_given_allele_len_diff(
        str_bp_diff_allele_1,
        inframe_stutter_model_dict,
        outframe_stutter_model_dict,
        motif_length,
    )
    log_stutter_probs_allele2 = cal_stutter_probs_given_allele_len_diff(
        str_bp_diff_allele_2,
        inframe_stutter_model_dict,
        outframe_stutter_model_dict,
        motif_length,
    )
    log_stutter_probs_allele3 = cal_stutter_probs_given_allele_len_diff(
        str_bp_diff_allele_3,
        inframe_stutter_model_dict,
        outframe_stutter_model_dict,
        motif_length,
    )
    log_snv_str_probs_allele1 = snv_log_lik_allele1 + log_stutter_probs_allele1
    log_snv_str_probs_allele2 = snv_log_lik_allele2 + log_stutter_probs_allele2
    log_snv_str_probs_allele3 = snv_log_lik_allele3 + log_stutter_probs_allele3
    return (
        log_snv_str_probs_allele1,
        log_snv_str_probs_allele2,
        log_snv_str_probs_allele3,
    )


def cal_concordant_rate(
    inframe_stutter_model_dict,
    outframe_stutter_model_dict,
    mosaic_gts_alleles,
    allele_balance,
    nearby_snp,
    snv_error_rate,
    mosaic_fraction,
    motif_length,
):
    stutter_error_rate = (
        np.exp(inframe_stutter_model_dict[0])
        + np.exp(outframe_stutter_model_dict[0])
        - 1
    )
    non_error_rate = (1 - stutter_error_rate) * (1 - snv_error_rate)
    neabysnp_list = [
        nearby_snp[0],
        nearby_snp[1],
        nearby_snp[0],
        nearby_snp[1],
    ]
    if LIKELIHOOD_MODE == "length-based":
        source_hap_length = mosaic_gts_alleles[0][0]
        stutter_hap1_length = mosaic_gts_alleles[0][1]
        stutter_hap2_length = mosaic_gts_alleles[1][0]
        stutter_hap3_length = mosaic_gts_alleles[1][1]
    else:
        source_hap_length = len(mosaic_gts_alleles[0][0][1])
        stutter_hap1_length = len(mosaic_gts_alleles[0][1][1])
        stutter_hap2_length = len(mosaic_gts_alleles[1][0][1])
        stutter_hap3_length = len(mosaic_gts_alleles[1][1][1])
    if DEBUG:
        print(mosaic_gts_alleles)
        print(source_hap_length)
        print(stutter_hap1_length)
        print(stutter_hap2_length)
        print(stutter_hap3_length)
    (
        log_snv_str_probs_allele1,
        log_snv_str_probs_allele2,
        log_snv_str_probs_allele3,
    ) = cal_within_mosaic_gts_stutter_rate(
        inframe_stutter_model_dict,
        outframe_stutter_model_dict,
        motif_length,
        snv_error_rate,
        neabysnp_list,
        source_hap_length,
        stutter_hap1_length,
        stutter_hap2_length,
        stutter_hap3_length,
    )
    concordant_rate_1 = (
        non_error_rate
        + np.exp(log_snv_str_probs_allele1)
        + np.exp(log_snv_str_probs_allele2)
        + np.exp(log_snv_str_probs_allele3)
    )
    neabysnp_list = [
        nearby_snp[1],
        nearby_snp[0],
        nearby_snp[0],
        nearby_snp[1],
    ]
    if LIKELIHOOD_MODE == "length-based":
        source_hap_length = mosaic_gts_alleles[0][1]
        stutter_hap1_length = mosaic_gts_alleles[0][0]
        stutter_hap2_length = mosaic_gts_alleles[1][0]
        stutter_hap3_length = mosaic_gts_alleles[1][1]
    else:
        source_hap_length = len(mosaic_gts_alleles[0][1][1])
        stutter_hap1_length = len(mosaic_gts_alleles[0][0][1])
        stutter_hap2_length = len(mosaic_gts_alleles[1][0][1])
        stutter_hap3_length = len(mosaic_gts_alleles[1][1][1])
    (
        log_snv_str_probs_allele1,
        log_snv_str_probs_allele2,
        log_snv_str_probs_allele3,
    ) = cal_within_mosaic_gts_stutter_rate(
        inframe_stutter_model_dict,
        outframe_stutter_model_dict,
        motif_length,
        snv_error_rate,
        neabysnp_list,
        source_hap_length,
        stutter_hap1_length,
        stutter_hap2_length,
        stutter_hap3_length,
    )
    concordant_rate_2 = (
        non_error_rate
        + np.exp(log_snv_str_probs_allele1)
        + np.exp(log_snv_str_probs_allele2)
        + np.exp(log_snv_str_probs_allele3)
    )
    neabysnp_list = [
        nearby_snp[0],
        nearby_snp[0],
        nearby_snp[1],
        nearby_snp[1],
    ]
    if LIKELIHOOD_MODE == "length-based":
        source_hap_length = mosaic_gts_alleles[1][0]
        stutter_hap1_length = mosaic_gts_alleles[0][0]
        stutter_hap2_length = mosaic_gts_alleles[0][1]
        stutter_hap3_length = mosaic_gts_alleles[1][1]
    else:
        source_hap_length = len(mosaic_gts_alleles[1][0][1])
        stutter_hap1_length = len(mosaic_gts_alleles[0][0][1])
        stutter_hap2_length = len(mosaic_gts_alleles[0][1][1])
        stutter_hap3_length = len(mosaic_gts_alleles[1][1][1])
    (
        log_snv_str_probs_allele1,
        log_snv_str_probs_allele2,
        log_snv_str_probs_allele3,
    ) = cal_within_mosaic_gts_stutter_rate(
        inframe_stutter_model_dict,
        outframe_stutter_model_dict,
        motif_length,
        snv_error_rate,
        neabysnp_list,
        source_hap_length,
        stutter_hap1_length,
        stutter_hap2_length,
        stutter_hap3_length,
    )
    concordant_rate_3 = (
        non_error_rate
        + np.exp(log_snv_str_probs_allele1)
        + np.exp(log_snv_str_probs_allele2)
        + np.exp(log_snv_str_probs_allele3)
    )
    neabysnp_list = [
        nearby_snp[1],
        nearby_snp[0],
        nearby_snp[1],
        nearby_snp[0],
    ]
    if LIKELIHOOD_MODE == "length-based":
        source_hap_length = mosaic_gts_alleles[1][1]
        stutter_hap1_length = mosaic_gts_alleles[0][0]
        stutter_hap2_length = mosaic_gts_alleles[0][1]
        stutter_hap3_length = mosaic_gts_alleles[1][0]
    else:
        source_hap_length = len(mosaic_gts_alleles[1][1][1])
        stutter_hap1_length = len(mosaic_gts_alleles[0][0][1])
        stutter_hap2_length = len(mosaic_gts_alleles[0][1][1])
        stutter_hap3_length = len(mosaic_gts_alleles[1][0][1])
    (
        log_snv_str_probs_allele1,
        log_snv_str_probs_allele2,
        log_snv_str_probs_allele3,
    ) = cal_within_mosaic_gts_stutter_rate(
        inframe_stutter_model_dict,
        outframe_stutter_model_dict,
        motif_length,
        snv_error_rate,
        neabysnp_list,
        source_hap_length,
        stutter_hap1_length,
        stutter_hap2_length,
        stutter_hap3_length,
    )
    concordant_rate_4 = (
        non_error_rate
        + np.exp(log_snv_str_probs_allele1)
        + np.exp(log_snv_str_probs_allele2)
        + np.exp(log_snv_str_probs_allele3)
    )
    concordant_rate = (1 - mosaic_fraction) * (
        allele_balance * concordant_rate_1
        + (1 - allele_balance) * concordant_rate_2
    ) + mosaic_fraction * (
        allele_balance * concordant_rate_3
        + (1 - allele_balance) * concordant_rate_4
    )
    return min(
        1, concordant_rate
    )  # HACK: exist issues for hap-based lik calculation and the value may exceed 1 for concordant_rate


def is_tandem_repeated_sequence(sequence, motif_length):
    sequence_length = len(sequence)
    motif = sequence[0:motif_length]
    tr_sequence = motif * (sequence_length // motif_length)
    all_tr_sequence = tr_sequence + motif[: (sequence_length % motif_length)]
    if sequence == all_tr_sequence:
        return True
    else:
        return False


def cal_mle_haplotype_depth_dict_phase(
    all_reads_all_alleles_linking_log_lik_array,
):
    max_mle_counts = np.sum(
        all_reads_all_alleles_linking_log_lik_array
        == np.amax(
            all_reads_all_alleles_linking_log_lik_array, axis=1, keepdims=True
        ),
        axis=1,
    )
    haplotype_depth_array = np.where(
        all_reads_all_alleles_linking_log_lik_array
        == np.amax(
            all_reads_all_alleles_linking_log_lik_array, axis=1, keepdims=True
        ),
        1 / max_mle_counts.reshape(-1, 1),
        0,
    )
    hap_depth_list = np.around(np.sum(haplotype_depth_array, axis=0), 0)
    return hap_depth_list


def cal_stutter_probs_given_allele_len_diff(
    bp_diff,
    inframe_stutter_model_dict,
    outframe_stutter_model_dict,
    motif_length,
):
    if DEBUG:
        print(inframe_stutter_model_dict)
    if bp_diff == 0.0:
        log_stutter_probs = np.log(
            np.exp(inframe_stutter_model_dict[0])
            + np.exp(outframe_stutter_model_dict[0])
            - 1
        )
    else:
        if DEBUG:
            print(abs(bp_diff), motif_length)
        quotient, remainder = divmod(abs(bp_diff), motif_length)
        if remainder == 0.0:
            log_stutter_probs = inframe_stutter_model_dict[
                int((bp_diff) / motif_length)
            ]
        else:
            if bp_diff > 0.0:
                bp_diff_2_continuous_value = bp_diff - quotient
            else:
                bp_diff_2_continuous_value = bp_diff + quotient
            log_stutter_probs = outframe_stutter_model_dict[
                bp_diff_2_continuous_value
            ]
    return log_stutter_probs


def cal_mle_haplotype_depth_dict_unphase(
    all_reads_all_alleles_log_lik_array,
):
    max_mle_counts = np.sum(
        all_reads_all_alleles_log_lik_array
        == np.amax(
            all_reads_all_alleles_log_lik_array,
            axis=1,
            keepdims=True,
        ),
        axis=1,
    )
    haplotype_depth_array = np.where(
        all_reads_all_alleles_log_lik_array
        == np.amax(
            all_reads_all_alleles_log_lik_array,
            axis=1,
            keepdims=True,
        ),
        1 / max_mle_counts.reshape(-1, 1),
        0,
    )
    if DONT_ASSIGNMENT_FOR_IDENTICAL_PROBS:
        haplotype_depth_array_remove_identical_read_assign = np.where(
            haplotype_depth_array > 0.999, 1, 0
        )
        unknown_assign_num = np.sum(
            np.sum(haplotype_depth_array_remove_identical_read_assign, axis=1)
            == 0
        )
        hap_depth_list = np.around(
            np.sum(haplotype_depth_array_remove_identical_read_assign, axis=0),
            0,
        )
    else:
        unknown_assign_num = 0
        hap_depth_list = np.around(np.sum(haplotype_depth_array, axis=0), 0)
    return haplotype_depth_array, hap_depth_list, unknown_assign_num


def cal_undirectional_value(value):
    if value >= 0.5:
        return value
    else:
        return 1 - value


def cal_edit_distance_based_on_mismatch(seq1, seq2):
    seq1_array = np.array(list(seq1))
    seq2_array = np.array(list(seq2))
    alignment = seq1_array == seq2_array
    edit_distance = len(alignment) - np.sum(alignment)
    mis_location = np.where(alignment == False)[0].tolist()
    return edit_distance, mis_location


def cal_mle_haplotype_depth_dict_given_mosaicgt_phase(
    all_reads_all_alleles_linking_log_lik_array,
    mosaic_gts_alleles_linking_index,
):
    # germ_gt_allele_index = list(mosaic_gts_alleles_linking_index[0])
    # mutation_gt_allele_index = list(mosaic_gts_alleles_linking_index[1])
    # allele_index = germ_gt_allele_index + mutation_gt_allele_index
    all_reads_all_mosaicgt_alleles_linking_log_lik_array = (
        all_reads_all_alleles_linking_log_lik_array[
            :, mosaic_gts_alleles_linking_index
        ]
    )
    max_mle_counts = np.sum(
        all_reads_all_mosaicgt_alleles_linking_log_lik_array
        == np.amax(
            all_reads_all_mosaicgt_alleles_linking_log_lik_array,
            axis=1,
            keepdims=True,
        ),
        axis=1,
    )
    haplotype_depth_array = np.where(
        all_reads_all_mosaicgt_alleles_linking_log_lik_array
        == np.amax(
            all_reads_all_mosaicgt_alleles_linking_log_lik_array,
            axis=1,
            keepdims=True,
        ),
        1 / max_mle_counts.reshape(-1, 1),
        0,
    )
    if DONT_ASSIGNMENT_FOR_IDENTICAL_PROBS:
        haplotype_depth_array_remove_identical_read_assign = np.where(
            haplotype_depth_array > 0.999, 1, 0
        )
        unknown_assign_num = np.sum(
            np.sum(haplotype_depth_array_remove_identical_read_assign, axis=1)
            == 0
        )
        hap_depth_list = np.around(
            np.sum(haplotype_depth_array_remove_identical_read_assign, axis=0),
            0,
        )
    else:
        unknown_assign_num = 0
        hap_depth_list = np.around(np.sum(haplotype_depth_array, axis=0), 0)
    return hap_depth_list, unknown_assign_num
    # return allele_index, hap_depth_list


def cal_per_allele_lik(
    read_snv_hap,
    read_str_hap,
    allele_snv_hap,
    allele_str_hap,
    read_snv_hap_accuracy,
    discordant_rate,
):
    if read_snv_hap == allele_snv_hap and read_str_hap == allele_str_hap:
        allele_lik = read_snv_hap_accuracy * (1 - discordant_rate)
    elif read_snv_hap == allele_snv_hap and read_str_hap != allele_str_hap:
        allele_lik = read_snv_hap_accuracy * discordant_rate
    elif read_snv_hap != allele_snv_hap and read_str_hap == allele_str_hap:
        allele_lik = (1 - read_snv_hap_accuracy) / 3 * discordant_rate
    elif read_snv_hap != allele_snv_hap and read_str_hap != allele_str_hap:
        allele_lik = (1 - read_snv_hap_accuracy) / 3 * (1 - discordant_rate)
    return max([allele_lik, MIN_PER_READ_LIKELIHOOD])


def cal_mosaic_likelihood_using_discordant_rate(
    discordant_rate,
    max_mosaic_gt_alleles,
    nearby_snp,
    mosaic_fraction,
    allele_balance,
    reads_list,
    reads_accuracy_list,
):
    allele1_str_hap = max_mosaic_gt_alleles[0][0]
    allele1_snv_hap = nearby_snp[0]
    allele2_str_hap = max_mosaic_gt_alleles[0][1]
    allele2_snv_hap = nearby_snp[1]
    allele3_str_hap = max_mosaic_gt_alleles[1][0]
    allele3_snv_hap = nearby_snp[0]
    allele4_str_hap = max_mosaic_gt_alleles[1][1]
    allele4_snv_hap = nearby_snp[1]
    per_reads_log_lik_list = []
    for read_linking_seq, read_linking_accuracy in zip(
        reads_list, reads_accuracy_list
    ):
        read_str_hap = read_linking_seq[0]
        read_snv_hap = read_linking_seq[1]
        read_str_hap_accuracy = read_linking_accuracy[0]
        read_snv_hap_accuracy = read_linking_accuracy[1]
        allele1_lik = cal_per_allele_lik(
            read_snv_hap,
            read_str_hap,
            allele1_snv_hap,
            allele1_str_hap,
            read_snv_hap_accuracy,
            discordant_rate,
        )
        allele2_lik = cal_per_allele_lik(
            read_snv_hap,
            read_str_hap,
            allele2_snv_hap,
            allele2_str_hap,
            read_snv_hap_accuracy,
            discordant_rate,
        )
        allele3_lik = cal_per_allele_lik(
            read_snv_hap,
            read_str_hap,
            allele3_snv_hap,
            allele3_str_hap,
            read_snv_hap_accuracy,
            discordant_rate,
        )
        allele4_lik = cal_per_allele_lik(
            read_snv_hap,
            read_str_hap,
            allele4_snv_hap,
            allele4_str_hap,
            read_snv_hap_accuracy,
            discordant_rate,
        )
        per_reads_log_lik = np.log(
            (allele_balance * allele1_lik + (1 - allele_balance) * allele2_lik)
            * (1 - mosaic_fraction)
            + (
                allele_balance * allele3_lik
                + (1 - allele_balance) * allele4_lik
            )
            * mosaic_fraction
        )
        per_reads_log_lik_list.append(per_reads_log_lik)
    mosaic_log_likelihood = np.sum(per_reads_log_lik_list)
    return mosaic_log_likelihood


def check_denominator(denominator, numerator):
    if denominator == 0:
        return "NA"
    else:
        return numerator / denominator


# Referred to extract features #
def cal_per_read_log_likelihood_given_allele_seq_based(
    allele_seq,
    read_seq,
    read_accuracy,
    log_no_stutter_lik,
    motif_length,
    inframe_stutter_model_dict,
    outframe_stutter_model_dict,
    mutation_site_index_based_on_hap_zero_based_list,
):
    # XXX:(-4)%3 = 2
    bp_diff = len(read_seq) - len(allele_seq)
    indel_size = abs(bp_diff)
    if bp_diff == 0.0:
        (
            log_str_align_prob,
            mis_score,
            max_alignment_length,
        ) = hap_alignment.no_stutter_str_alignment_add_mis_score(
            allele_seq,
            read_seq,
            read_accuracy,
            log_no_stutter_lik,
        )
        type = "mismatch"
        max_alignment_index = "NA"
    else:
        quotient, remainder = divmod(abs(bp_diff), motif_length)
        if remainder == 0.0:
            steps = int((bp_diff) / motif_length)
            if bp_diff > 0:
                (
                    log_str_align_prob,
                    mis_score,
                    max_alignment_length,
                    max_alignment_index,
                ) = hap_alignment.stutter_insertion_add_mis_score(
                    allele_seq,
                    read_seq,
                    read_accuracy,
                    steps,
                    inframe_stutter_model_dict,
                )
                type = "insertion"
            else:
                (
                    log_str_align_prob,
                    mis_score,
                    max_alignment_length,
                    max_alignment_index,
                ) = hap_alignment.stutter_deletion_add_mis_score(
                    allele_seq,
                    read_seq,
                    read_accuracy,
                    steps,
                    inframe_stutter_model_dict,
                )
                type = "deletion"
            # BUG: unfiltered reads with a big stutter size and exceed the stutter model range
            # stutter_error_likelihood = log_stutter_model_dic[steps]
            # KeyError: -12
        else:
            if bp_diff > 0:
                steps = bp_diff - quotient
                (
                    log_str_align_prob,
                    mis_score,
                    max_alignment_length,
                    max_alignment_index,
                ) = hap_alignment.stutter_insertion_add_mis_score(
                    allele_seq,
                    read_seq,
                    read_accuracy,
                    steps,
                    outframe_stutter_model_dict,
                )
                type = "insertion"
            else:
                steps = bp_diff + quotient
                (
                    log_str_align_prob,
                    mis_score,
                    max_alignment_length,
                    max_alignment_index,
                ) = hap_alignment.stutter_deletion_add_mis_score(
                    allele_seq,
                    read_seq,
                    read_accuracy,
                    steps,
                    outframe_stutter_model_dict,
                )
                type = "deletion"
    if mutation_site_index_based_on_hap_zero_based_list != []:
        base_accuracy_list = find_mis_baseq_site_in_read(
            type,
            mutation_site_index_based_on_hap_zero_based_list,
            max_alignment_index,
            indel_size,
            read_accuracy,
        )
    else:
        base_accuracy_list = []
    return (
        log_str_align_prob,
        mis_score,
        max_alignment_length,
        base_accuracy_list,
    )


def find_mis_baseq_site_in_read(
    type,
    mutation_site_index_based_on_hap_zero_based_list,
    max_alignment_index,
    indel_size,
    read_accuracy,
):
    base_accuracy_list = []
    for (
        mutation_site_index_based_on_hap_zero_based
    ) in mutation_site_index_based_on_hap_zero_based_list:
        if type == "mismatch":
            if mutation_site_index_based_on_hap_zero_based < len(
                read_accuracy
            ):
                base_accuracy = read_accuracy[
                    mutation_site_index_based_on_hap_zero_based
                ]
            else:
                base_accuracy = "NA"
        elif type == "insertion":
            if (
                max_alignment_index
                <= mutation_site_index_based_on_hap_zero_based
            ):
                if (
                    mutation_site_index_based_on_hap_zero_based + indel_size
                ) < len(read_accuracy):
                    base_accuracy = read_accuracy[
                        mutation_site_index_based_on_hap_zero_based
                        + indel_size
                    ]
                else:
                    base_accuracy = "NA"
            else:
                if mutation_site_index_based_on_hap_zero_based < len(
                    read_accuracy
                ):
                    base_accuracy = read_accuracy[
                        mutation_site_index_based_on_hap_zero_based
                    ]
                else:
                    base_accuracy = "NA"
        elif type == "deletion":
            if (
                max_alignment_index
                <= mutation_site_index_based_on_hap_zero_based
                < (max_alignment_index + indel_size)
            ):
                base_accuracy = "NA"
            elif (
                mutation_site_index_based_on_hap_zero_based
                < max_alignment_index
            ):
                if mutation_site_index_based_on_hap_zero_based < len(
                    read_accuracy
                ):
                    base_accuracy = read_accuracy[
                        mutation_site_index_based_on_hap_zero_based
                    ]
                else:
                    base_accuracy = "NA"
            elif mutation_site_index_based_on_hap_zero_based >= (
                max_alignment_index + indel_size
            ):
                if (
                    mutation_site_index_based_on_hap_zero_based - indel_size
                ) < len(read_accuracy):
                    base_accuracy = read_accuracy[
                        mutation_site_index_based_on_hap_zero_based
                        - indel_size
                    ]
                else:
                    base_accuracy = "NA"
        if base_accuracy != "NA":
            base_accuracy_list.append(base_accuracy)
    return base_accuracy_list


def post_homopolymer_motif_check(motif, total_length):
    motif_length = len(motif)
    if motif_length == 1:
        if total_length >= STR_LENGTH_CUTOFF:  # HACK 降低门槛到最小的 STR length from HipSTR Panel 9
            return True
        else:
            return False
    elif motif_length >= MOTIF_LENGTH_CUTOFF:  # HACK 因为即使这样降低门槛还是有很多，所以保持这个选项，后面检测是否变成 uninterruption 进一步减少一部分的错误位点 4
        if total_length >= STR_LENGTH_CUTOFF:  # HACK 降低门槛到最小的 STR length from HipSTR Panel 9
            # if (len(set(motif[1:])) == 1) or (len(set(motif[:-1])) == 1):
            if ((len(set(motif)) <= 2) and (1 in collections.Counter(list(motif)).values())):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def approximation_flanking_first_bp_check(mis_loc, allele_length):
    if (mis_loc == (PADDING_BPS - 1)) or (
        mis_loc == (allele_length - PADDING_BPS)
    ):
        return True
    else:
        return False


def high_depth_for_strand_bias_check(depth):
    if depth >= STRAND_DEPTH_CUTOFF:
        return True
    else:
        return False


def check_low_accuracy_fraction_check(accuracy_list, base_low_accuracy_cutoff):
    low_accuracy_fraction = np.sum(
        np.array(accuracy_list) <= base_low_accuracy_cutoff
    ) / len(accuracy_list)
    if low_accuracy_fraction > ACCURACY_FRACTION_CUTOFF:
        return True
    else:
        return False


def source_remove_checked(
    used_allele_list, removed_allele, allele_source_dict, allele_mle_depth_list,
    allele_obs_depth_list,
    allele_source_edit_dis_dict
):
    final_used_allele_list = used_allele_list.copy()
    used_index_list = []
    for index, i in enumerate(used_allele_list):
        source_allele = allele_source_dict.get(i, "")
        edit_dis = allele_source_edit_dis_dict.get(i, 0)
        if source_allele in removed_allele:
            if edit_dis == 1:
                final_used_allele_list.remove(i)
            else:
                if ((
                    allele_mle_depth_list[index]
                    <= SOURCE_CHECK_TARGET_DEPTH_CUTOFF
                ) or (
                    allele_obs_depth_list[index]
                    <= SOURCE_CHECK_TARGET_DEPTH_CUTOFF
                )):
                    final_used_allele_list.remove(i)
                else:
                    used_index_list.append(index)
        else:
            used_index_list.append(index)
    return final_used_allele_list, used_index_list


# HACK:
# homhet 2 3 4 hom1
# 这里 hom1 当成 unphase 处理
# def ym_hap_count_homhet(
#     all_phasing_hap_count_dict, scale_factor
# ):  # , variant_type):
#     # if variant_type == "mismatch":
#     #     scale_factor = scale_factor
#     # else:
#     #     scale_factor = scale_factor/2
#     # 从字典中获取各个基因型的计数
#     allele1_h1 = all_phasing_hap_count_dict.get("allele1_h1", 0)
#     allele1_h2 = all_phasing_hap_count_dict.get("allele1_h2", 0)
#     allele2_h1 = all_phasing_hap_count_dict.get("allele2_h1", 0)
#     allele2_h2 = all_phasing_hap_count_dict.get("allele2_h2", 0)

#     # 设置总计数阈值
#     total_count = allele1_h1 + allele1_h2 + allele2_h1 + allele2_h2

#     if total_count >= PHASING_DEPTH_CUTOFF:
#         # 根据变体类型进行不同的处理
#         allele1_depth = allele1_h1 + allele1_h2
#         allele2_depth = allele2_h1 + allele2_h2
#         if ((allele1_depth >= ALLELE_DEPTH_CUTOFF) and (allele2_depth >= ALLELE_DEPTH_CUTOFF)):
#             het_perfect_phase_p1 = (
#                 allele1_h1 > allele1_h2 * scale_factor
#             ) and (allele2_h2 > allele2_h1 * scale_factor)
#             het_perfect_phase_p2 = (
#                 allele1_h2 > allele1_h1 * scale_factor
#             ) and (allele2_h1 > allele2_h2 * scale_factor)
#             het_allele_balance_p1 = (
#                 allele1_h1 < allele2_h2 * scale_factor
#             ) and (allele1_h1 > allele2_h2 / scale_factor)
#             het_allele_balance_p2 = (
#                 allele1_h2 < allele2_h1 * scale_factor
#             ) and (allele1_h2 > allele2_h1 / scale_factor)
#             het_pass_depth_p1 = (allele1_h1 >= ALLELE_DEPTH_CUTOFF) and (
#                 allele2_h2 >= ALLELE_DEPTH_CUTOFF
#             )
#             het_pass_depth_p2 = (allele1_h2 >= ALLELE_DEPTH_CUTOFF) and (
#                 allele2_h1 >= ALLELE_DEPTH_CUTOFF
#             )
#             allele1_balance = (allele1_h1 < allele1_h2 * scale_factor) and (
#                 allele1_h1 > allele1_h2 / scale_factor
#             )
#             allele2_balance = (allele2_h1 < allele2_h2 * scale_factor) and (
#                 allele2_h1 > allele2_h2 / scale_factor
#             )
#             allele1_diff = (allele1_h1 > allele1_h2 * scale_factor) or (
#                 allele1_h1 < allele1_h2 / scale_factor
#             )
#             allele2_diff = (allele2_h1 > allele2_h2 * scale_factor) or (
#                 allele2_h1 < allele2_h2 / scale_factor
#             )
#             # enough_allele1 = ((allele1_h1 + allele1_h2) >= ALLELE_DEPTH_CUTOFF)
#             # enough_allele2 = ((allele2_h1 + allele2_h2) >= ALLELE_DEPTH_CUTOFF)
#             h1_allele2_mosaic_balance = (
#                 (allele1_h1 + allele2_h1) > (allele1_h2 / scale_factor)
#             ) and ((allele1_h1 + allele2_h1) < (allele1_h2 * scale_factor))
#             h1_allele1_mosaic_balance = (
#                 (allele1_h1 + allele2_h1) > (allele2_h2 / scale_factor)
#             ) and ((allele1_h1 + allele2_h1) < (allele2_h2 * scale_factor))
#             h2_allele2_mosaic_balance = (
#                 (allele1_h2 + allele2_h2) > (allele1_h1 / scale_factor)
#             ) and ((allele1_h2 + allele2_h2) < (allele1_h1 * scale_factor))
#             h2_allele1_mosaic_balance = (
#                 (allele1_h2 + allele2_h2) > (allele2_h1 / scale_factor)
#             ) and ((allele1_h2 + allele2_h2) < (allele2_h1 * scale_factor))
#             if het_perfect_phase_p1 and het_allele_balance_p1:
#                 if het_pass_depth_p1:
#                     return "phase", 2, "Pass"
#                 else:
#                     return "phase", 2, "NoEnoughAlleleDepth"
#             elif het_perfect_phase_p2 and het_allele_balance_p2:
#                 if het_pass_depth_p2:
#                     return "phase", 2, "Pass"
#                 else:
#                     return "phase", 2, "NoEnoughAlleleDepth"
#             elif allele1_balance and allele2_diff:
#                 if allele2_h1 > allele2_h2:
#                     if h1_allele2_mosaic_balance and (
#                         (allele2_h1 >= ALLELE_DEPTH_CUTOFF)
#                         and (allele1_depth >= ALLELE_DEPTH_CUTOFF)
#                     ):
#                         return "phase", 3, "Pass"
#                     elif h1_allele2_mosaic_balance:
#                         return "phase", 3, "NoEnoughAlleleDepth"
#                     elif (allele2_h1 >= ALLELE_DEPTH_CUTOFF) and (
#                         allele1_depth >= ALLELE_DEPTH_CUTOFF
#                     ):
#                         return "phase", 3, "UnBalanceH1H2"
#                     else:
#                         return (
#                             "phase",
#                             3,
#                             "NoEnoughAlleleDepthAndUnBalanceH1H2",
#                         )
#                 elif allele2_h2 > allele2_h1:
#                     if h2_allele2_mosaic_balance and (
#                         (allele2_h2 >= ALLELE_DEPTH_CUTOFF)
#                         and (allele1_depth >= ALLELE_DEPTH_CUTOFF)
#                     ):
#                         return "phase", 3, "Pass"
#                     elif h2_allele2_mosaic_balance:
#                         return "phase", 3, "NoEnoughAlleleDepth"
#                     elif (allele2_h2 >= ALLELE_DEPTH_CUTOFF) and (
#                         allele1_depth >= ALLELE_DEPTH_CUTOFF
#                     ):
#                         return "phase", 3, "UnBalanceH1H2"
#                     else:
#                         return (
#                             "phase",
#                             3,
#                             "NoEnoughAlleleDepthAndUnBalanceH1H2",
#                         )
#             elif allele2_balance and allele1_diff:
#                 if allele1_h1 > allele1_h2:
#                     if h1_allele1_mosaic_balance and (
#                         (allele1_h1 >= ALLELE_DEPTH_CUTOFF)
#                         and (allele2_depth >= ALLELE_DEPTH_CUTOFF)
#                     ):
#                         return "phase", 3, "Pass"
#                     elif h1_allele1_mosaic_balance:
#                         return "phase", 3, "NoEnoughAlleleDepth"
#                     elif (allele1_h1 >= ALLELE_DEPTH_CUTOFF) and (
#                         allele2_depth >= ALLELE_DEPTH_CUTOFF
#                     ):
#                         return "phase", 3, "UnBalanceH1H2"
#                     else:
#                         return (
#                             "phase",
#                             3,
#                             "NoEnoughAlleleDepthAndUnBalanceH1H2",
#                         )
#                 elif allele1_h2 > allele1_h1:
#                     if h2_allele1_mosaic_balance and (
#                         (allele1_h2 >= ALLELE_DEPTH_CUTOFF)
#                         and (allele2_depth >= ALLELE_DEPTH_CUTOFF)
#                     ):
#                         return "phase", 3, "Pass"
#                     elif h2_allele1_mosaic_balance:
#                         return "phase", 3, "NoEnoughAlleleDepth"
#                     elif (allele1_h2 >= ALLELE_DEPTH_CUTOFF) and (
#                         allele2_depth >= ALLELE_DEPTH_CUTOFF
#                     ):
#                         return "phase", 3, "UnBalanceH1H2"
#                     else:
#                         return (
#                             "phase",
#                             3,
#                             "NoEnoughAlleleDepthAndUnBalanceH1H2",
#                         )
#             elif (
#                 allele1_h1 > 0
#                 and allele1_h2 > 0
#                 and allele2_h1 > 0
#                 and allele2_h2 > 0
#             ):
#                 return "phase", 4, "Pass"
#             else:
#                 return "unphase", "NA", "NoOptionForJudge"
#         else:
#             return "unphase", "NA", "NoEnoughAlleleDepth"
#     else:
#         return "unphase", "NA", "LowSpanningDepth"


# 判断杂合和不平衡
def check_balance_and_depth(major, minor, mosaic_balance):
    if major > minor:
        if mosaic_balance:
            return "phase", 3, "Pass"
        else:
            return "phase", 3, "UnBalanceH1H2"
    return None


PHASING_DEPTH_CUTOFF = 10
ALLELE_DEPTH_CUTOFF = 2


# HACK:
# homhet 2 3 4 hom1
# 这里 hom1 当成 unphase 处理
def ym_hap_count_homhet_gpt(all_phasing_hap_count_dict, scale_factor):
    # 从字典中获取各个基因型的计数
    allele1_h1 = all_phasing_hap_count_dict.get("allele1_h1", 0)
    allele1_h2 = all_phasing_hap_count_dict.get("allele1_h2", 0)
    allele2_h1 = all_phasing_hap_count_dict.get("allele2_h1", 0)
    allele2_h2 = all_phasing_hap_count_dict.get("allele2_h2", 0)

    # 设置总计数阈值
    total_count = allele1_h1 + allele1_h2 + allele2_h1 + allele2_h2

    if total_count < PHASING_DEPTH_CUTOFF:
        return "unphase", "NA", "LowSpanningDepth", "NA"

    # 根据变体类型进行不同的处理
    allele1_depth = allele1_h1 + allele1_h2
    allele2_depth = allele2_h1 + allele2_h2
    h1_depth = allele1_h1 + allele2_h1
    h2_depth = allele1_h2 + allele2_h2
    if (allele1_depth < ALLELE_DEPTH_CUTOFF) or (
        allele2_depth < ALLELE_DEPTH_CUTOFF
    ):
        return "unphase", "NA", "NoEnoughSTRAlleleDepth", "NA"
    if (h1_depth < ALLELE_DEPTH_CUTOFF) or (h2_depth < ALLELE_DEPTH_CUTOFF):
        return "unphase", "NA", "NoEnoughSNPAlleleDepth", "NA"
    # if (h1_depth < h2_depth * scale_factor) and (h1_depth > h2_depth / scale_factor):
    #     return "unphase", "NA", "UnBalanceH1H2"
    # 计算各种平衡和相位
    het_perfect_phase_p1 = (allele1_h1 > allele1_h2 * scale_factor) and (
        allele2_h2 > allele2_h1 * scale_factor
    )
    het_perfect_phase_p2 = (allele1_h2 > allele1_h1 * scale_factor) and (
        allele2_h1 > allele2_h2 * scale_factor
    )
    het_allele_balance_p1 = (allele1_h1 < allele2_h2 * scale_factor) and (
        allele1_h1 > allele2_h2 / scale_factor
    )
    het_allele_balance_p2 = (allele1_h2 < allele2_h1 * scale_factor) and (
        allele1_h2 > allele2_h1 / scale_factor
    )
    # het_pass_depth_p1 = (allele1_h1 >= ALLELE_DEPTH_CUTOFF) and (
    #     allele2_h2 >= ALLELE_DEPTH_CUTOFF
    # )
    # het_pass_depth_p2 = (allele1_h2 >= ALLELE_DEPTH_CUTOFF) and (
    #     allele2_h1 >= ALLELE_DEPTH_CUTOFF
    # )
    # 判断完美相位
    if het_perfect_phase_p1:
        if het_allele_balance_p1:
            return ("phase", 2, "Pass", "allele1h1_allele2h2")
        else:
            return ("phase", 2, "UnBalanceH1H2", "allele1h1_allele2h2")
    elif het_perfect_phase_p2:
        if het_allele_balance_p2:
            return ("phase", 2, "Pass", "allele1h2_allele2h1")
        else:
            return ("phase", 2, "UnBalanceH1H2", "allele1h2_allele2h1")
    # if het_perfect_phase_p1 and het_allele_balance_p1:
    #     return (
    #         "phase",
    #         2,
    #         "Pass" if het_pass_depth_p1 else "NoEnoughAlleleDepth",
    #     )
    # elif het_perfect_phase_p2 and het_allele_balance_p2:
    #     return (
    #         "phase",
    #         2,
    #         "Pass" if het_pass_depth_p2 else "NoEnoughAlleleDepth",
    #     )

    allele1_balance = (allele1_h1 < allele1_h2 * scale_factor) and (
        allele1_h1 > allele1_h2 / scale_factor
    )
    allele2_balance = (allele2_h1 < allele2_h2 * scale_factor) and (
        allele2_h1 > allele2_h2 / scale_factor
    )
    h1_allele2_mosaic_balance = (
        allele1_h1 + allele2_h1 > allele1_h2 / scale_factor
    ) and (allele1_h1 + allele2_h1 < allele1_h2 * scale_factor)
    h2_allele2_mosaic_balance = (
        allele1_h2 + allele2_h2 > allele1_h1 / scale_factor
    ) and (allele1_h2 + allele2_h2 < allele1_h1 * scale_factor)
    h1_allele1_mosaic_balance = (
        allele1_h1 + allele2_h1 > allele2_h2 / scale_factor
    ) and (allele1_h1 + allele2_h1 < allele2_h2 * scale_factor)
    h2_allele1_mosaic_balance = (
        allele1_h2 + allele2_h2 > allele2_h1 / scale_factor
    ) and (allele1_h2 + allele2_h2 < allele2_h1 * scale_factor)
    allele1_diff = (allele1_h1 > allele1_h2 * scale_factor) or (
        allele1_h1 < allele1_h2 / scale_factor
    )
    allele2_diff = (allele2_h1 > allele2_h2 * scale_factor) or (
        allele2_h1 < allele2_h2 / scale_factor
    )
    if allele1_balance and allele2_diff:
        if (allele1_h1 < ALLELE_DEPTH_CUTOFF) or (
            allele1_h2 < ALLELE_DEPTH_CUTOFF
        ):
            # Allele2 不用管，因为 diff 了 并且 allele2 depth > 1,所以 allele2_h1 或者  allele2_h1 一定是大于 1 的
            if (allele1_h1 > allele1_h2) and (allele2_h2 > allele2_h1):
                return "phase", 2, "WithExtrahSNPInGerm", "allele1h1_allele2h2"
            elif (allele1_h2 > allele1_h1) and (allele2_h1 > allele2_h2):
                return "phase", 2, "WithExtrahSNPInGerm", "allele1h2_allele2h1"
            else:
                return "phase", 2, "ADOandAMPorMosaic", "NA"
        else:
            result = check_balance_and_depth(
                allele2_h1,
                allele2_h2,
                # (
                #     allele2_h1 >= ALLELE_DEPTH_CUTOFF
                #     and allele1_depth >= ALLELE_DEPTH_CUTOFF
                # ),
                h1_allele2_mosaic_balance,
            )
            if result:
                return result + ("allele1germ_allele2h1",)

            result = check_balance_and_depth(
                allele2_h2,
                allele2_h1,
                # (
                #     allele2_h2 >= ALLELE_DEPTH_CUTOFF
                #     and allele1_depth >= ALLELE_DEPTH_CUTOFF
                # ),
                h2_allele2_mosaic_balance,
            )
            if result:
                return result + ("allele1germ_allele2h2",)
    if allele2_balance and allele1_diff:
        if (allele2_h1 < ALLELE_DEPTH_CUTOFF) or (
            allele2_h2 < ALLELE_DEPTH_CUTOFF
        ):
            # Allele1 不用管，因为 diff 了 并且 allele1 depth > 1,所以 allele1_h1 或者  allele1_h1 一定是大于 1 的
            if (allele2_h1 > allele2_h2) and (allele1_h2 > allele1_h1):
                return "phase", 2, "WithExtrahSNPInGerm", "allele2h1_allele1h2"
            elif (allele2_h2 > allele2_h1) and (allele1_h1 > allele1_h2):
                return "phase", 2, "WithExtrahSNPInGerm", "allele2h2_allele1h1"
            else:
                return "phase", 2, "ADOandAMPorMosaic", "NA"
        else:
            result = check_balance_and_depth(
                allele1_h1,
                allele1_h2,
                # (
                #     allele1_h1 >= ALLELE_DEPTH_CUTOFF
                #     and allele2_depth >= ALLELE_DEPTH_CUTOFF
                # ),
                h1_allele1_mosaic_balance,
            )
            if result:
                return result + ("allele2germ_allele1h1",)

            result = check_balance_and_depth(
                allele1_h2,
                allele1_h1,
                # (
                #     allele1_h2 >= ALLELE_DEPTH_CUTOFF
                #     and allele2_depth >= ALLELE_DEPTH_CUTOFF
                # ),
                h2_allele1_mosaic_balance,
            )
            if result:
                return result + ("allele2germ_allele1h2",)

    if allele1_h1 > 0 and allele1_h2 > 0 and allele2_h1 > 0 and allele2_h2 > 0:
        if (allele1_h1 > allele1_h2) and (allele2_h2 > allele2_h1):
            return "phase", 4, "Pass", "allele1h1_allele2h2"
        elif (allele1_h2 > allele1_h1) and (allele2_h1 > allele2_h2):
            return "phase", 4, "Pass", "allele1h2_allele2h1"
        else:
            return "phase", 4, "Pass", "NA"
    return "unphase", "NA", "NoOptionForJudge", "NA"


# XXX:
# hap=3 有 六种 情况
# hap=2 有 六种 情况
# hap=3 有 非常多种情况
# hap 2 3 4 and unphase
# het2het hap3 and homhet hap3 也在这个目录 ?
# 这里就不考虑 homhet hap3 了，因为 uphase 已经是 hethet 了，phasable reads 比较少的话 homhet hap3 也不可信了啊
# 那 hap 2 考虑不考虑呢, hap 2 目前也先不考虑了哈，但是还是标记一下吧，所以只考虑 2 3 >3(标记为4)
# 一个 allele phasable reads 比较少的话我是当成 unphase 还是变成 het 或者 homhet 呢
# 这里我倾向于认为是 unphase 了
# hethet 不判断 hap2 ，只判断 hap3 和 hap4 以及 unphase 哈


# XXX: 注意: 纯合变杂合 特定的 allele 只有一条时 会认为 hap3 为 hap2，但杂合变杂合时，特定的 allele 只有一条时，不认为 hap4 为 hap3, 因为这边 homhet 和 hethet 的目的都是为了卡出更真的真位点
# XXX: 注意 phase 时为了卡出更真的真位点和更假的假位点时，使用的门槛和指标应该是不一样的，所以这边为了卡更真的真位点，homhet 和 hethet 虽然使用了不同的策略，但是最终的目的都是为了使 phase 卡出的真位点更真
def ym_hap_count_hethet(
    all_phasing_hap_count_dict, scale_factor
):  # , variant_type):
    allele1_h1 = all_phasing_hap_count_dict.get("allele1_h1", 0)
    allele1_h2 = all_phasing_hap_count_dict.get("allele1_h2", 0)
    allele2_h1 = all_phasing_hap_count_dict.get("allele2_h1", 0)
    allele2_h2 = all_phasing_hap_count_dict.get("allele2_h2", 0)
    allele3_h1 = all_phasing_hap_count_dict.get("allele3_h1", 0)
    allele3_h2 = all_phasing_hap_count_dict.get("allele3_h2", 0)
    total_count = (
        allele1_h1
        + allele1_h2
        + allele2_h1
        + allele2_h2
        + allele3_h1
        + allele3_h2
    )
    if total_count < PHASING_DEPTH_CUTOFF:
        return "unphase", "NA", "LowSpanningDepth", "NA"
    allele1_depth = allele1_h1 + allele1_h2
    allele2_depth = allele2_h1 + allele2_h2
    allele3_depth = allele3_h1 + allele3_h2
    if (
        (allele1_depth < ALLELE_DEPTH_CUTOFF)
        or (allele2_depth < ALLELE_DEPTH_CUTOFF)
        or (allele3_depth < ALLELE_DEPTH_CUTOFF)
    ):
        return "unphase", "NA", "NoEnoughSTRAlleleDepth", "NA"
    h1_depth = allele1_h1 + allele2_h1 + allele3_h1
    h2_depth = allele1_h2 + allele2_h2 + allele3_h2
    if (h1_depth < ALLELE_DEPTH_CUTOFF) or (h2_depth < ALLELE_DEPTH_CUTOFF):
        return "unphase", "NA", "NoEnoughSNPAlleleDepth", "NA"
    het_perfect_phase_p1_allele12 = (
        allele1_h1 > allele1_h2 * scale_factor
    ) and (allele2_h2 > allele2_h1 * scale_factor)
    het_perfect_phase_p2_allele12 = (
        allele1_h2 > allele1_h1 * scale_factor
    ) and (allele2_h1 > allele2_h2 * scale_factor)
    het_perfect_phase_p1_allele13 = (
        allele1_h1 > allele1_h2 * scale_factor
    ) and (allele3_h2 > allele3_h1 * scale_factor)
    het_perfect_phase_p2_allele13 = (
        allele1_h2 > allele1_h1 * scale_factor
    ) and (allele3_h1 > allele3_h2 * scale_factor)
    het_perfect_phase_p1_allele23 = (
        allele2_h1 > allele2_h2 * scale_factor
    ) and (allele3_h2 > allele3_h1 * scale_factor)
    het_perfect_phase_p2_allele23 = (
        allele2_h2 > allele2_h1 * scale_factor
    ) and (allele3_h1 > allele3_h2 * scale_factor)
    # het_allele_balance_p1_allele12 = (allele1_h1 < allele2_h2 * scale_factor) and (allele1_h1 > allele2_h2 / scale_factor)
    # het_allele_balance_p2_allele12 = (allele1_h2 < allele2_h1 * scale_factor) and (allele1_h2 > allele2_h1 / scale_factor)
    # het_allele_balance_p1_allele13 = (allele1_h1 < allele3_h2 * scale_factor) and (allele1_h1 > allele3_h2 / scale_factor)
    # het_allele_balance_p2_allele13 = (allele1_h2 < allele3_h1 * scale_factor) and (allele1_h2 > allele3_h1 / scale_factor)
    # het_allele_balance_p1_allele23 = (allele2_h1 < allele3_h2 * scale_factor) and (allele2_h1 > allele3_h2 / scale_factor)
    # het_allele_balance_p2_allele23 = (allele2_h2 < allele3_h1 * scale_factor) and (allele2_h2 > allele3_h1 / scale_factor)
    allele1_p1_difference = allele1_h1 > allele1_h2 * scale_factor
    allele1_p2_difference = allele1_h1 < allele1_h2 / scale_factor
    allele2_p1_difference = allele2_h1 > allele2_h2 * scale_factor
    allele2_p2_difference = allele2_h1 < allele2_h2 / scale_factor
    allele3_p1_difference = allele3_h1 > allele3_h2 * scale_factor
    allele3_p2_difference = allele3_h1 < allele3_h2 / scale_factor
    allele1_no_diff = (allele1_h1 < allele1_h2 * scale_factor) and (
        allele1_h1 > allele1_h2 / scale_factor
    )
    allele2_no_diff = (allele2_h1 < allele2_h2 * scale_factor) and (
        allele2_h1 > allele2_h2 / scale_factor
    )
    allele3_no_diff = (allele3_h1 < allele3_h2 * scale_factor) and (
        allele3_h1 > allele3_h2 / scale_factor
    )
    if het_perfect_phase_p1_allele12 and allele3_p1_difference:
        if (allele1_h1 + allele3_h1) > allele2_h2 / scale_factor and (
            allele1_h1 + allele3_h1
        ) < allele2_h2 * scale_factor:
            return "phase", 3, "Pass", "allele2h2_allele1h1_allele3h1"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele2h2_allele1h1_allele3h1"
    if het_perfect_phase_p1_allele12 and allele3_p2_difference:
        if (allele2_h2 + allele3_h2) > allele1_h1 / scale_factor and (
            allele2_h2 + allele3_h2
        ) < allele1_h1 * scale_factor:
            return "phase", 3, "Pass", "allele1h1_allele2h2_allele3h2"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele1h1_allele2h2_allele3h2"
    if het_perfect_phase_p2_allele12 and allele3_p1_difference:
        if (allele2_h1 + allele3_h1) > allele1_h2 / scale_factor and (
            allele2_h1 + allele3_h1
        ) < allele1_h2 * scale_factor:
            return "phase", 3, "Pass", "allele1h2_allele2h1_allele3h1"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele1h2_allele2h1_allele3h1"
    if het_perfect_phase_p2_allele12 and allele3_p2_difference:
        if (allele1_h2 + allele3_h2) > allele2_h1 / scale_factor and (
            allele1_h2 + allele3_h2
        ) < allele2_h1 * scale_factor:
            return "phase", 3, "Pass", "allele2h1_allele1h2_allele3h2"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele2h1_allele1h2_allele3h2"
    if het_perfect_phase_p1_allele13 and allele2_p1_difference:
        if (allele1_h1 + allele2_h1) > allele3_h2 / scale_factor and (
            allele1_h1 + allele2_h1
        ) < allele3_h2 * scale_factor:
            return "phase", 3, "Pass", "allele3h2_allele1h1_allele2h1"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele3h2_allele1h1_allele2h1"
    if het_perfect_phase_p1_allele13 and allele2_p2_difference:
        if (allele3_h2 + allele2_h2) > allele1_h1 / scale_factor and (
            allele3_h2 + allele2_h2
        ) < allele1_h1 * scale_factor:
            return "phase", 3, "Pass", "allele1h1_allele3h2_allele2h2"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele1h1_allele3h2_allele2h2"
    if het_perfect_phase_p2_allele13 and allele2_p1_difference:
        if (allele3_h1 + allele2_h1) > allele1_h2 / scale_factor and (
            allele3_h1 + allele2_h1
        ) < allele1_h2 * scale_factor:
            return "phase", 3, "Pass", "allele1h2_allele3h1_allele2h1"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele1h2_allele3h1_allele2h1"
    if het_perfect_phase_p2_allele13 and allele2_p2_difference:
        if (allele1_h2 + allele2_h2) > allele3_h1 / scale_factor and (
            allele1_h2 + allele2_h2
        ) < allele3_h1 * scale_factor:
            return "phase", 3, "Pass", "allele3h1_allele1h2_allele2h2"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele3h1_allele1h2_allele2h2"
    if het_perfect_phase_p1_allele23 and allele1_p1_difference:
        if (allele2_h1 + allele1_h1) > allele3_h2 / scale_factor and (
            allele2_h1 + allele1_h1
        ) < allele3_h2 * scale_factor:
            return "phase", 3, "Pass", "allele3h2_allele2h1_allele1h1"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele3h2_allele2h1_allele1h1"
    if het_perfect_phase_p1_allele23 and allele1_p2_difference:
        if (allele3_h2 + allele1_h2) > allele2_h1 / scale_factor and (
            allele3_h2 + allele1_h2
        ) < allele2_h1 * scale_factor:
            return "phase", 3, "Pass", "allele2h1_allele3h2_allele1h2"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele2h1_allele3h2_allele1h2"
    if het_perfect_phase_p2_allele23 and allele1_p1_difference:
        if (allele3_h1 + allele1_h1) > allele2_h2 / scale_factor and (
            allele3_h1 + allele1_h1
        ) < allele2_h2 * scale_factor:
            return "phase", 3, "Pass", "allele2h2_allele3h1_allele1h1"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele2h2_allele3h1_allele1h1"
    if het_perfect_phase_p2_allele23 and allele1_p2_difference:
        if (allele2_h2 + allele1_h2) > allele3_h1 / scale_factor and (
            allele2_h2 + allele1_h2
        ) < allele3_h1 * scale_factor:
            return "phase", 3, "Pass", "allele3h1_allele2h2_allele1h2"
        else:
            return "phase", 3, "UnBalanceH1H2", "allele3h1_allele2h2_allele1h2"
    if allele1_no_diff or allele2_no_diff or allele3_no_diff:
        if (
            (allele1_h1 > allele1_h2)
            and (allele2_h1 > allele2_h2)
            and (allele3_h2 > allele3_h1)
        ):
            return "phase", 4, "Pass", "allele3h2_allele1h1_allele2h1"
        elif (
            (allele1_h1 < allele1_h2)
            and (allele2_h1 < allele2_h2)
            and (allele3_h2 < allele3_h1)
        ):
            return "phase", 4, "Pass", "allele3h1_allele1h2_allele2h2"
        elif (
            (allele1_h1 > allele1_h2)
            and (allele2_h1 < allele2_h2)
            and (allele3_h2 > allele3_h1)
        ):
            return "phase", 4, "Pass", "allele1h1_allele2h2_allele3h2"
        elif (
            (allele1_h1 < allele1_h2)
            and (allele2_h1 > allele2_h2)
            and (allele3_h2 < allele3_h1)
        ):
            return "phase", 4, "Pass", "allele1h2_allele2h1_allele3h1"
        elif (
            (allele1_h1 < allele1_h2)
            and (allele2_h1 > allele2_h2)
            and (allele3_h2 > allele3_h1)
        ):
            return "phase", 4, "Pass", "allele2h1_allele1h2_allele3h2"
        elif (
            (allele1_h1 > allele1_h2)
            and (allele2_h1 < allele2_h2)
            and (allele3_h2 < allele3_h1)
        ):
            return "phase", 4, "Pass", "allele2h2_allele1h1_allele3h1"
        else:
            return "phase", 4, "Pass", "NA"
    return "unphase", "NA", "NoOptionForJudge", "NA"


def avoid_zero_divide_error(dividend, divisor):
    if divisor == 0:
        return "NA"
    else:
        return dividend/divisor


## 当 allele 深度较低的时候，这些 allele 容易被过滤掉，所以再次调整门槛 ##
## HACK: 先进行修改，depth = 2 时候 mismatches 如果有一条 low baseQ，则不会被过滤 ##
## 修改: 从 >= 改为 > ##
## 增加过滤
## 如果 source allele 被去掉了, 则后面的 allele 也跟着去掉嘛？source allele 被去掉了以后，降低对 target 的可信度或者直接去掉 target(这里选择直接去掉 target) ##
## OK! ##
