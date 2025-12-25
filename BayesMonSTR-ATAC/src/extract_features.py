import logging
import pysam
import config_params
import myutils
import hmmsegger
import coordinate_segger
import extract_allele
import pyBigWig
import numpy as np
import adjust_vcf
import collections
import scipy
import scipy.stats  # HACK: need import scipy.stats otherwise AttributeError: module 'scipy' has no attribute 'stats'
import hap_alignment
import sys
import argparse
import pandas as pd
import os
import textwrap
import time
import logger_config
from multiprocessing import Pool, Manager
from tqdm import tqdm
import traceback
import math
from scipy.stats import median_abs_deviation
import pdb  # debug pdb.set_trace()

__VERSION__ = "0.0.1"
READ_FILTER = True
PADDING_BPS = 5  # 10 # revise_padding
OVERALL_FEATURES_READ_FILTER = False
OVERALL_SPANNING_FILTER = True
SPANNING_FILTER = True
READS_SOFT_CLIPPING_CUTOFF = 10  # From MosaicFocast  # XXX: For short reads, long reads need consider again
READS_HARD_CLIPPING_CUTOFF = 10  # From MosaicFocast  # XXX: For short reads, long reads need consider again
DEFAULT_FIRST_PAD_BASE = "A"
AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH = 5
AF_BASED_SNP_FEATURES_SORT = True
DIRECTIONAL_STATISTICS = (
    True  # 统计量和两组数据有方向无方向的问题，很重要，涉及到特征匹配和机器学习特征训练，对值正负的敏感性，以及不同数据间相同特征的匹配问题
)
DIRECTIONAL_STR_DEPTH_PERCENTILE = True
DEPTH_POINT = 30
PHRED_P_VALUE = False
PHASING_STRATEDY = "READ_NAME"
PILEUP_STEPPER = "nofilter"
NEARBY_SNP_FILTER = False
SINGLETHREAD = True
DEBUG = True
USE_DEFAULT_MAX_STEP = True  # HACK Temp for GIAB USE_DEFAULT_MAX_STEP = True
DEFAULT_MAX_STEP = 100
FLANKING_CONTEXT_BPS = 25
BASEQ_OR_ACCURACY = (  # HACK: Use BaseQ Not Accuracy Because Q20 and Q30 and Q10 are more meaningful than 0.99 and 0.999 and 0.9
    "BASEQ"
)
LOW_BASEQ_LOW_CUTOFF = 13
LOW_BASEQ_HIGH_CUTOFF = 20  # 更好的指标，所以被用
FLANKING_BPS_FOR_FEATURE_CUTOFF = 1
RECURRENT_MOTIF_JUDGE = 6
LOW_MAPQ_CUTOFF = 30
LOW_OVERALL_LOW_BASEQ_CUTOFF = 20
HIGH_MISMATCHES_CUTOFF = 0.035
SHORT_FLANKING_BPS_CUTOFF = 5
BED_MAPPABILITY = True
GIAB = False
GIAB_STUTTER = False
# class SampleLociFeatures:
#     def __init__(self, sample_locus_params,pysam_bed_row):
#         self.pysam_bed_row = pysam_bed_row
#         self.sample_locus_params = sample_locus_params
#         self.vcf_file = sample_locus_params["vcf_file"]
#         self.sample_list = sample_locus_params["sample_list"]

#     def _extract_features(self):
#         self.features['num_loci'] = len(self.loci)
#         self.features['num_genes'] = len(set([locus.gene for locus in self.loci]))
#         self.features['num_exons'] = len(set([locus.exon for locus in self.loci]))
#         self

# Some features need normalize use read length or read number
# 和 visualization 一样
# XXX: read 的 assignment 是 single STR allele locus 的 assign, 而不是使用 hSNP 同时 assignment
# XXX: hSNP 只是用来标记，而不是用来 assign
HARD_FILTER_COLUMNS = [
    "chrom",
    "STRSTART",
    "STREND",
    "MOTIF_length",
    "PERIOD",
    "str_id",
    "MOTIF",
    "GT",
    "MGT",
    "MP",
    "CMP",
    "variant_type",
    "frame",
    "mosaic_fraction",
    "perfect_ref_str",
    "depth",
    "filtered_depth",
    "filtered_depth_category",
    "EAF",
    "observed_mosaic_allele_vaf_single_locus",
    "NSTUTTER",
    "DSTUTTER",
    "NFSTUTTER",
    "DFSTUTTER",
    "HVAF",
    "PS",
    "PP",
    "PEAF",
    "MLEPEAF",
    "MBP",
    "PHA",
    "PPP",
    "PODR",
    "PODHR",
    "PMAP",
    "PMLEEH",
    "PMLEEHR",
    "PMLEDR",
    "PFAH",
    "MAPLR",
    "MAPLRTP",
    "MAPLRS",
    "MAPLRTPS",
    "POMDR",
    "PSOMDR",
    "PGOMDR",
    "PSMMDR",
    "PGMMDR",
    "PMMDR",
    "NHSNPN",
    "NHSNPINDELN",
    "mle_mosaic_allele_vaf_two_loci",
    "observed_mosaic_allele_vaf_two_loci",
    "GIOAF",
    "GIMLEAF",
    "GIUAF",
    "PRHN",
    "PMB",
    "POB",
    "PGHAF",
    "PMHAF",
    "PH3AF",
    "PGD",
    "PSD",
    "PMD",
    "GI",
    "Filter",
    "AAD",
    "sample_name",
    "GTMGT",
    "uncallable_num",
    "uncallable_frac",
    "recurrent_num",
    "recurrent_fraction",
    "sample_num",
    "germ_popAF",
    "source_popAF",
    "mosaic_popAF",
    "muttype_hf",
    "germ_allele",
    "source_allele",
    "mosaic_allele",
    "germ_allele_seq",
    "source_allele_seq",
    "mosaic_allele_seq",
    "external_germ_popAF",
    "external_source_popAF",
    "external_mosaic_popAF",
    "allele_length_dp",
    "ref_allele_length_padding_flanking",
    "ALLELE_BARCODE_UMI_hf",
    "median_genotyping_dp"
]


def accuracy_to_phred(accuracy: float) -> float:
    """
    Transforms base accuracy (probability) to baseQ Phred score.

    Parameters:
    accuracy (float): The base accuracy as a probability (0 to 1).

    Returns:
    float: The corresponding baseQ Phred score.
    """
    if accuracy == "NA":
        return "NA"
    # Ensure the accuracy is within the valid range
    if not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be between 0 and 1.")

    # Convert accuracy probability to probability of error
    probability_of_error = 1 - accuracy

    # Calculate the Phred score
    if probability_of_error == 0:
        # Return a very high Phred score if there is no error (implies very high accuracy)
        return 40

    phred_score = int(round(-10 * math.log10(probability_of_error), 0))

    return phred_score


def unpaired_nonparams_ranksum_test(group1, group2):
    result = scipy.stats.ranksums(
        group1,
        group2,
        alternative="two-sided",
        # axis=0, # HACK: diff version compibilities
        # nan_policy="omit", # HACK: diff version compibilities
        # keepdims=False,  # HACK: diff version compibilities
    )
    statistic = result.statistic
    pvalue = result.pvalue
    return statistic, pvalue


def unpaired_nonparams_ranksum_test_side(group1, group2, side):
    result = scipy.stats.ranksums(
        group1,
        group2,
        alternative=side,
        # axis=0, # HACK: diff version compibilities
        # nan_policy="omit", # HACK: diff version compibilities
        # keepdims=False,  # HACK: diff version compibilities
    )
    statistic = result.statistic
    pvalue = result.pvalue
    return statistic, pvalue


def paired_nonparams_wilcoxon_test(group1, group2):
    result = scipy.stats.wilcoxon(
        group1,
        group2,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )
    statistic = result.statistic
    pvalue = result.pvalue
    return statistic, pvalue


def paired_nonparams_wilcoxon_test_side(group1, group2, side):
    result = scipy.stats.wilcoxon(
        group1,
        group2,
        zero_method="wilcox",
        correction=False,
        alternative=side,
        method="auto",
    )
    statistic = result.statistic
    pvalue = result.pvalue
    return statistic, pvalue


def fisher_exact_test(
    group1_columna, group1_columnb, group2_columna, group2_columnb
):
    nan_exist = np.any(
        np.isnan(
            [group1_columna, group1_columnb, group2_columna, group2_columnb]
        )
    )
    if nan_exist:
        return np.nan, np.nan
    oddsratio, pvalue = scipy.stats.fisher_exact(
        [[group1_columna, group1_columnb], [group2_columna, group2_columnb]],
        alternative="two-sided",
    )
    return oddsratio, pvalue


def fisher_exact_test_side(
    group1_columna, group1_columnb, group2_columna, group2_columnb, side
):
    nan_exist = np.any(
        np.isnan(
            [group1_columna, group1_columnb, group2_columna, group2_columnb]
        )
    )
    if nan_exist:
        return np.nan, np.nan
    oddsratio, pvalue = scipy.stats.fisher_exact(
        [[group1_columna, group1_columnb], [group2_columna, group2_columnb]],
        alternative=side,
    )
    return oddsratio, pvalue


def phred_scale_score_2_rate(x):
    return 10 ** (-x / 10)


def rate_2_phred_scale(x):
    return -10 * np.log10(x)


def percentile_rank(arr, value):
    # 将特定值插入数组
    if value not in arr:
        arr.append(value)

    # 排序数组
    sorted_arr = sorted(arr)

    # 找到特定值的第一个位置
    first_position = sorted_arr.index(value)

    # 找到特定值的最后一个位置
    last_position = len(sorted_arr) - 1 - sorted_arr[::-1].index(value)

    # 计算第一个位置的百分位
    first_percentile = (first_position + 1) / len(sorted_arr)

    # 计算最后一个位置的百分位
    last_percentile = (last_position + 1) / len(sorted_arr)

    final_percentile = (first_percentile + last_percentile) / 2

    return final_percentile


def ks_test_goodness_of_fit(
    rvs,
    args,
    cdf="randint",
    N=20,
    alternative="two-sided",
    method="auto",
    axis=0,
    nan_policy="propagate",
    keepdims=False,
):
    # The one-sample test compares the underlying distribution F(x) of a sample against a given distribution G(x). The two-sample test compares the underlying distributions of two independent samples. Both tests are valid only for continuous distributions.
    # If array_like, it should be a 1-D array of observations of random variables, and the two-sample test is performed (and rvs must be array_like). If a callable, that callable is used to calculate the cdf. If a string, it should be the name of a distribution in scipy.stats, which will be used as the cdf function.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html#scipy.stats.randint
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    D, p_value = scipy.stats.kstest(
        rvs=rvs,
        cdf=cdf,
        args=args,
        N=N,
        alternative=alternative,
        # method=method,
        # axis=axis,
        # nan_policy=nan_policy,
        # keepdims=keepdims,
    )
    return D, p_value


def chi_square_goodness_of_fit(depth, region_size, observed_frequency):
    # DEBUG: RuntimeWarning: invalid value encountered in true_divide  Done
    # DEBUG: RuntimeWarning: divide by zero encountered in true_divide Done
    # 如果 region_size 或 depth 为 0，返回 NaN，避免除零错误
    if region_size <= 0 or depth == 0:
        return np.nan, np.nan

    # 确保 region_size 不超过 observed_frequency 长度
    region_size = min(region_size, len(observed_frequency))

    # 计算期望频率，确保不包含 0 或 NaN
    expected_freq = np.full(region_size, depth / region_size, dtype=float)

    # 如果 observed_frequency 长度比 region_size 大，填充小值 (避免 0)
    if len(observed_frequency) > region_size:
        extra_values = np.full(
            len(observed_frequency) - region_size, 1e-100, dtype=float
        )
        expected_freq = np.append(expected_freq, extra_values)

    # 避免 expected_freq 为 0 或 NaN
    expected_freq[np.isnan(expected_freq) | (expected_freq <= 0)] = 1e-100

    # 处理 observed_frequency，避免负数和 NaN
    observed_frequency = np.array(observed_frequency, dtype=float)
    observed_frequency[
        np.isnan(observed_frequency) | (observed_frequency < 0)
    ] = 0
    # 计算卡方检验
    chi2_stat, p_value = scipy.stats.chisquare(
        observed_frequency, expected_freq
    )

    return chi2_stat, p_value


# def chi_square_goodness_of_fit(depth, region_size, observed_frequency):
#     # Once we know the expected frequencies, we must check for two assumptions.
#     # We must ensure that all expected frequencies are one or greater.
#     # At most, 20 % of the expected frequencies should be less than 5.
#     if region_size == 0:
#         return np.nan, np.nan
#     expected_freq = [depth / region_size] * region_size
#     expected_freq = expected_freq + (len(observed_frequency) - region_size) * [
#         0
#     ]
#     chi2_stat, p_value = scipy.stats.chisquare(
#         observed_frequency, expected_freq
#     )
#     return chi2_stat, p_value


def get_stutter_params(
    stutter_errors_bed, chrom, start, end, STR_id, motif_length, STR_length
):
    if GIAB_STUTTER:
        stutter_model_params = {}
        stutter_loci_df = stutter_errors_bed[
            (stutter_errors_bed["MOTIF_LEN"] == int(motif_length))
            & (stutter_errors_bed["left_border"] <= STR_length)
            & (stutter_errors_bed["right_border"] > STR_length)
        ]
        stutter_model_params["inframe_single_step_prob"] = float(
            stutter_loci_df["inframe_single_step_prob"]
        )
        stutter_model_params["inframe_ins_prob"] = float(
            stutter_loci_df["inframe_ins_prob"]
        )
        stutter_model_params["inframe_del_prob"] = float(
            stutter_loci_df["inframe_del_prob"]
        )
        stutter_model_params["outframe_single_step_prob"] = float(
            stutter_loci_df["outframe_single_step_prob"]
        )
        stutter_model_params["outframe_ins_prob"] = float(
            stutter_loci_df["outframe_ins_prob"]
        )
        stutter_model_params["outframe_del_prob"] = float(
            stutter_loci_df["outframe_del_prob"]
        )
        stutter_model_state = "ObsStutterModel"
        inframe_max_stutter_steps = DEFAULT_MAX_STEP
        outframe_max_stutter_steps = DEFAULT_MAX_STEP
        inframe_stutter_model_dict = myutils.stutter_model(
            stutter_model_params["inframe_single_step_prob"],
            stutter_model_params["inframe_ins_prob"],
            stutter_model_params["inframe_del_prob"],
            inframe_max_stutter_steps,
        )
        outframe_stutter_model_dict = myutils.stutter_model(
            stutter_model_params["outframe_single_step_prob"],
            stutter_model_params["outframe_ins_prob"],
            stutter_model_params["outframe_del_prob"],
            outframe_max_stutter_steps,
        )
        log_no_stutter_lik = np.log(
            np.exp(inframe_stutter_model_dict[0])
            + np.exp(outframe_stutter_model_dict[0])
            - 1
        )
    else:
        stutter_model_params = {}
        stutter_model_state = "NoAvaliableStutterModel"
        inframe_stutter_model_dict = {}
        outframe_stutter_model_dict = {}
        log_no_stutter_lik = 0
        for pysam_bed_row in stutter_errors_bed.fetch(chrom, start, end):
            row_list = pysam_bed_row.split("\t")
            if GIAB:
                if row_list[5] == STR_id:
                    stutter_model_params["inframe_single_step_prob"] = float(
                        row_list[76]
                    )
                    stutter_model_params["inframe_ins_prob"] = float(
                        row_list[74]
                    )
                    stutter_model_params["inframe_del_prob"] = float(
                        row_list[75]
                    )
                    stutter_model_params["outframe_single_step_prob"] = float(
                        row_list[79]
                    )
                    stutter_model_params["outframe_ins_prob"] = float(
                        row_list[77]
                    )
                    stutter_model_params["outframe_del_prob"] = float(
                        row_list[78]
                    )
                    stutter_model_state = "ObsStutterModel"
                    if (
                        USE_DEFAULT_MAX_STEP
                    ):  # HACK: When use for mix samples or no stutter estimate samples,please use this parameter instead
                        inframe_max_stutter_steps = DEFAULT_MAX_STEP
                        outframe_max_stutter_steps = DEFAULT_MAX_STEP
                    else:
                        inframe_max_stutter_steps = int(float(row_list[-1]))
                        outframe_max_stutter_steps = int(float(row_list[-1]))
                    inframe_stutter_model_dict = myutils.stutter_model(
                        stutter_model_params["inframe_single_step_prob"],
                        stutter_model_params["inframe_ins_prob"],
                        stutter_model_params["inframe_del_prob"],
                        inframe_max_stutter_steps,
                    )
                    outframe_stutter_model_dict = myutils.stutter_model(
                        stutter_model_params["outframe_single_step_prob"],
                        stutter_model_params["outframe_ins_prob"],
                        stutter_model_params["outframe_del_prob"],
                        outframe_max_stutter_steps,
                    )
                    log_no_stutter_lik = np.log(
                        np.exp(inframe_stutter_model_dict[0])
                        + np.exp(outframe_stutter_model_dict[0])
                        - 1
                    )
                    break
            else:
                if row_list[5] == STR_id:
                    stutter_model_params["inframe_single_step_prob"] = float(
                        row_list[24]
                    )
                    stutter_model_params["inframe_ins_prob"] = float(
                        row_list[22]
                    )
                    stutter_model_params["inframe_del_prob"] = float(
                        row_list[23]
                    )
                    stutter_model_params["outframe_single_step_prob"] = float(
                        row_list[27]
                    )
                    stutter_model_params["outframe_ins_prob"] = float(
                        row_list[25]
                    )
                    stutter_model_params["outframe_del_prob"] = float(
                        row_list[26]
                    )
                    if USE_DEFAULT_MAX_STEP:
                        inframe_max_stutter_steps = DEFAULT_MAX_STEP
                        outframe_max_stutter_steps = DEFAULT_MAX_STEP
                    else:
                        inframe_max_stutter_steps = int(float(row_list[-1]))
                        outframe_max_stutter_steps = int(float(row_list[-1]))
                    inframe_stutter_model_dict = myutils.stutter_model(
                        stutter_model_params["inframe_single_step_prob"],
                        stutter_model_params["inframe_ins_prob"],
                        stutter_model_params["inframe_del_prob"],
                        inframe_max_stutter_steps,
                    )
                    outframe_stutter_model_dict = myutils.stutter_model(
                        stutter_model_params["outframe_single_step_prob"],
                        stutter_model_params["outframe_ins_prob"],
                        stutter_model_params["outframe_del_prob"],
                        outframe_max_stutter_steps,
                    )
                    stutter_model_state = str(row_list[15])
                    log_no_stutter_lik = np.log(
                        np.exp(inframe_stutter_model_dict[0])
                        + np.exp(outframe_stutter_model_dict[0])
                        - 1
                    )
                    break
    return (
        stutter_model_state,
        stutter_model_params,
        inframe_stutter_model_dict,
        outframe_stutter_model_dict,
        log_no_stutter_lik,
    )


def get_var_record(pysam_vcf, chrom, start, end, STR_id):
    records = list(
        pysam_vcf.fetch(
            contig=chrom,
            start=start - PADDING_BPS,
            stop=end + PADDING_BPS,
        )  # which are 0-based, half-open
    )
    var = None
    for var in records:
        if var.id == STR_id:
            all_alleles = var.alleles
            var_record = var
            break
        else:
            continue
    if var is None:
        return None, None
    # print(all_alleles)
    # print(var_record)
    return all_alleles, var_record


def get_mappability_record(bw, chrom, start, end):
    # Extract signal data for the specified region
    # print(chrom)
    # print(start)
    # print(end)
    # try:
    #     signal = bw.values(chrom, start, end, numpy=True)
    # except ValueError:
    try:
        signal = bw.values(
            "chr" + str(chrom), start, end, numpy=True
        )  # HACK need chr in mappability file
    except (
        RuntimeError
    ):  # bw.chroms("chr10") 133797422, but 10:133867543 fails because RuntimeError: Invalid interval bounds!
        signal = ["NA"]

    # Close the BigWig file
    # bw.close()

    return signal


def get_phased_mosaic_GT_allele(var_record, sample_name):
    sample_var = var_record.samples[sample_name]
    phased_germline_GT = sample_var["PGT"][0].split("|")
    phased_mosaic_GT = sample_var["PMGT"][0].split("|")
    mosaic_index = sample_var["MGT"][0].split("/")
    germ_index = [str(sample_var["GT"][0]), str(sample_var["GT"][1])]
    nsnp = sample_var["NSNP"][0].split("|")
    if len(set(phased_germline_GT)) == 1:
        germ_snp1 = nsnp[0]
        germ_snp2 = nsnp[1]
        if mosaic_index[1] == phased_mosaic_GT[1]:
            mosaic_snp = nsnp[1]
        elif mosaic_index[1] == phased_mosaic_GT[0]:
            mosaic_snp = nsnp[0]
    elif len(set(phased_germline_GT)) == 2:
        if len(set(phased_mosaic_GT)) == 1:
            germ_snp1 = nsnp[0]
            germ_snp2 = nsnp[1]
            if phased_germline_GT[1] == germ_index[1]:
                mosaic_snp = nsnp[1]
            elif phased_germline_GT[0] == germ_index[1]:
                mosaic_snp = nsnp[0]
        else:
            if (
                len(
                    set(
                        [
                            int(phased_germline_GT[0]),
                            int(phased_germline_GT[1]),
                        ]
                    )
                    & set([int(phased_mosaic_GT[0]), int(phased_mosaic_GT[1])])
                )
                == 2
            ):
                if phased_germline_GT[0] == germ_index[0]:
                    germ_snp1 = nsnp[0]
                    germ_snp2 = nsnp[1]
                    mosaic_snp = nsnp[1]
                elif phased_germline_GT[1] == germ_index[0]:
                    germ_snp1 = nsnp[1]
                    germ_snp2 = nsnp[0]
                    mosaic_snp = nsnp[0]
            else:
                if phased_germline_GT[0] == germ_index[0]:
                    germ_snp1 = nsnp[0]
                    germ_snp2 = nsnp[1]
                    mosaic_snp = nsnp[1]
                elif phased_germline_GT[1] == germ_index[0]:
                    germ_snp1 = nsnp[1]
                    germ_snp2 = nsnp[0]
                    mosaic_snp = nsnp[0]
    return germ_snp1, germ_snp2, mosaic_snp


def get_mut_type_and_mosaic_allele(var_record, sample_name):
    sample_var = var_record.samples[sample_name]
    germ_index = sample_var["GT"]
    mosaic_index = sample_var["MGT"][0].split("/")
    mosaic_gt_observed_af = sample_var["EAF"]
    if len(set(germ_index)) == 1:
        if len(set(mosaic_index)) == 1:
            germ_allele = int(germ_index[0])
            source_allele = int(germ_index[0])
            mut_allele = int(mosaic_index[0])
            muttype = "germlinehom"
        else:
            germ_allele = int(germ_index[0])
            source_allele = int(germ_index[0])
            mut_allele = int(mosaic_index[1])
            muttype = "homhet"
    else:
        if len(set(mosaic_index)) == 1:
            germ_allele = int(germ_index[0])
            source_allele = int(germ_index[0])
            mut_allele = int(germ_index[1])
            muttype = "homhet"
        else:
            if (
                len(
                    set([int(germ_index[0]), int(germ_index[1])])
                    & set([int(mosaic_index[0]), int(mosaic_index[1])])
                )
                == 2
            ):
                if float(mosaic_gt_observed_af[0]) >= float(
                    mosaic_gt_observed_af[1]
                ):  # HACK: for germline het or clonal het, the germ allele is the one with higher mosaic allele frequency !
                    germ_allele = int(germ_index[0])
                    source_allele = int(germ_index[0])
                    mut_allele = int(germ_index[1])
                else:
                    germ_allele = int(germ_index[1])
                    source_allele = int(germ_index[1])
                    mut_allele = int(germ_index[0])
                muttype = "germlinehet"
            else:
                germ_allele = int(germ_index[0])
                source_allele = int(germ_index[1])
                mut_allele = int(mosaic_index[1])
                muttype = "hethet"
    return muttype, germ_allele, source_allele, mut_allele


# HACK: read-level features nearby snp features 不过滤 reads, stepper="nofilter", ignore_overlaps=True
# TODO: if PE reads both overlap and spanning with STR, i don't cosider this status here, and use both reads for genotyping and mosaic callings STR
def get_all_read_nearbysnp_assignment(pysam_bam, chrom, pos):
    # nearby SNP PS pos is 0-based in my vcf files
    all_reads_nearbysnp_dict = {}
    all_reads_nearbysnp_list_dict = {}
    # A region can either be specified by reference, start and end. start and end denote 0-based, half-open intervals.
    if PHASING_STRATEDY == "READ_NAME":
        for pileupcolumn in pysam_bam.pileup(
            chrom,
            pos,
            pos + 1,
            stepper=PILEUP_STEPPER,
            ignore_overlaps=True,
            truncate=True,
        ):
            # truncate (bool) – By default, the samtools pileup engine outputs all reads overlapping a region. If truncate is True and a region is given, only columns in the exact region specified are returned.
            # max_depth (int) – Maximum read depth permitted. The default limit is ‘8000’.
            # stepper (string) –
            # The stepper controls how the iterator advances. Possible options for the stepper are
            # pileup(self, contig=None, start=None, stop=None, region=None, reference=None, end=None, **kwargs)
            # stepper (string) – The stepper controls how the iterator advances. Possible options for the stepper are
            # all skip reads in which any of the following flags are set: BAM_FUNMAP, BAM_FSECONDARY, BAM_FQCFAIL, BAM_FDUP
            # nofilter uses every single read turning off any filtering.
            # samtools same filter and read processing as in samtools pileup.
            # flag_filter (int) – ignore reads where any of the bits in the flag are set.
            # ignore_orphans
            # ignore_overlaps
            # min_base_quality
            # min_mapping_quality
            # flag_require
            # adjust_capq_threshold
            # compute_baq
            # perform a pileup within a region. The region is specified by contig, start and stop (using 0-based indexing).
            if pileupcolumn.reference_pos == pos:
                # the position in the reference sequence (0-based).
                for pileupread in pileupcolumn.pileups:
                    if (not pileupread.is_del) and (not pileupread.is_refskip):
                        # ) and (
                        #     pileupread.alignment.query_name
                        #     in spanning_reads_name
                        # ):
                        # query position is None if is_del or is_refskip is set.
                        if NEARBY_SNP_FILTER:
                            if (
                                pileupread.alignment.reference_start
                                + AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH
                                <= pos
                                < pileupread.alignment.reference_end
                                - AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH
                            ):
                                if (
                                    all_reads_nearbysnp_dict.get(
                                        pileupread.alignment.query_name, None
                                    )
                                    == None
                                ):
                                    pass
                                else:
                                    continue
                            else:
                                continue
                        else:
                            pass
                        # TODO: process nearby SNP located in overlap regions of PE reads
                        snp_seq = pileupread.alignment.query_sequence.upper()[
                            pileupread.query_position
                        ]
                        snp_baseq = pileupread.alignment.query_qualities[
                            pileupread.query_position
                        ]
                        snp_seq_accuracy = 1 - myutils.baseq_2_error_rate(
                            snp_baseq
                        )
                        all_reads_nearbysnp_dict.setdefault(
                            pileupread.alignment.query_name, []
                        ).append(snp_seq.upper())
                        all_reads_nearbysnp_dict[
                            pileupread.alignment.query_name
                        ].append(snp_baseq)
                        strand = 1 - int(pileupread.alignment.is_reverse)
                        mapQ = pileupread.alignment.mapq
                        all_reads_nearbysnp_dict[
                            pileupread.alignment.query_name
                        ].append(str(strand))
                        all_reads_nearbysnp_dict[
                            pileupread.alignment.query_name
                        ].append(mapQ)
                        all_reads_nearbysnp_list_dict.setdefault(
                            "snp_seq", []
                        ).append(snp_seq)
                        all_reads_nearbysnp_list_dict.setdefault(
                            "snp_baseq", []
                        ).append(int(snp_baseq))
                        all_reads_nearbysnp_list_dict.setdefault(
                            "strand", []
                        ).append(int(strand))
                        all_reads_nearbysnp_list_dict.setdefault(
                            "mapQ", []
                        ).append(int(mapQ))
                        all_reads_nearbysnp_list_dict.setdefault(
                            "read_name", []
                        ).append(pileupread.alignment.query_name)
        return all_reads_nearbysnp_dict, all_reads_nearbysnp_list_dict
    else:
        return all_reads_nearbysnp_dict, all_reads_nearbysnp_list_dict
        pass
        # for read in pysam_bam.fetch(chrom, pos, pos+1):
        #     if read.query_name in spanning_reads_name:
        #         reads_snp_seq[var_loci_name][read.query_name] =


def get_chrom_depth(chrom, pysam_ref, pysam_bam):
    contig_length = pysam_ref.get_reference_length(chrom)
    average_window_length = int(contig_length / (DEPTH_POINT + 1))
    depth_list = []
    for pos in range(
        average_window_length, contig_length, average_window_length
    ):
        pos_depth = np.array(
            pysam_bam.count_coverage(chrom, pos, pos + 1, quality_threshold=0)
        ).sum(axis=0)[0]
        depth_list.append(pos_depth)
    return depth_list


def get_sample_reads_nearby_snp(
    chrom,
    var,
    sample_name,
    pysam_alignment,
    pysam_ref,
    raw_wgs_mean_depth,
    sample_median_depth,
    sample_wgs_point_depth_list,
    sample_chrom_median_point_depth,
    sample_chrom_point_depth_list,
):
    # hSNP 因为受到过滤，所以理论上都有深度，snp1 和 snp2 也都有深度 #
    # af1, af2, af3, depth, overall_mapQ,Overall_baseQ,Overall_strand,and ref vs alt
    # TODO: Read1/Read2 ??? No use, because no evident evidence for it, OK right!
    hSNP_features = {}
    phase_pos_zero_based = int(var.samples[sample_name]["PS"])
    nsnp = var.samples[sample_name]["NSNP"][0].split("|")
    nearby_snp_base1 = nsnp[0].upper()
    nearby_snp_base2 = nsnp[1].upper()
    # pysam.FastaFile fetch(self, reference=None, start=None, end=None, region=None)
    # A region can either be specified by reference, start and end. start and end denote 0-based, half-open intervals.
    # Note that region strings are 1-based, while start and end denote an interval in python coordinates.
    ref_snp_base = pysam_ref.fetch(
        chrom, phase_pos_zero_based, phase_pos_zero_based + 1
    ).upper()
    (
        all_reads_nearbysnp_dict,
        all_reads_nearbysnp_list_dict,
    ) = get_all_read_nearbysnp_assignment(
        pysam_alignment,
        chrom,
        phase_pos_zero_based,
    )
    # count_coverage(self, contig, start=None, stop=None, region=None, quality_threshold=15, read_callback='all', reference=None, end=None)
    # start (int) – start of the genomic region (0-based inclusive). If not given, count from the start of the chromosome.
    # stop (int) – end of the genomic region (0-based exclusive). If not given, count to the end of the chromosome.
    # raw_depth_list = get_chrom_depth(chrom, pysam_ref, pysam_alignment)
    raw_depth_list = sample_chrom_point_depth_list
    unfiltered_raw_depth = np.array(
        pysam_alignment.count_coverage(
            chrom,
            phase_pos_zero_based,
            phase_pos_zero_based + 1,
            quality_threshold=0,
        )
    ).sum(axis=0)[0]
    # median_raw_depth = np.median(raw_depth_list)
    median_raw_depth = sample_chrom_median_point_depth
    dp_diff = unfiltered_raw_depth - median_raw_depth
    dp_diff_fraction = dp_diff / median_raw_depth
    final_chrom_point_depth_percentile = percentile_rank(
        raw_depth_list, unfiltered_raw_depth
    )
    final_wgs_point_depth_percentile = percentile_rank(
        sample_wgs_point_depth_list, unfiltered_raw_depth
    )
    depth = len(all_reads_nearbysnp_list_dict["snp_seq"])
    all_reads_nearbysnp_snp_seq_dict = collections.Counter(
        all_reads_nearbysnp_list_dict["snp_seq"]
    )
    snp1_af = all_reads_nearbysnp_snp_seq_dict[nearby_snp_base1] / depth
    snp2_af = all_reads_nearbysnp_snp_seq_dict[nearby_snp_base2] / depth
    nearby_hsnp_depth_ratio = unfiltered_raw_depth / raw_wgs_mean_depth
    nearby_hsnp_chrom_point_median_depth_ratio = (
        unfiltered_raw_depth / median_raw_depth
    )
    nearby_hsnp_wgs_point_median_depth_ratio = (
        unfiltered_raw_depth / sample_median_depth
    )
    if AF_BASED_SNP_FEATURES_SORT:
        if snp1_af >= snp2_af:
            pass
        else:
            nearby_snp_base1, nearby_snp_base2 = (
                nearby_snp_base2,
                nearby_snp_base1,
            )
            snp1_af, snp2_af = snp2_af, snp1_af
    else:
        pass
    nearby_snp_base1_is_ref = nearby_snp_base1 == ref_snp_base
    nearby_snp_base2_is_ref = nearby_snp_base2 == ref_snp_base
    all_allele_log_lik_sorted_dict = sorted(
        all_reads_nearbysnp_snp_seq_dict.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    snp3_af = 0
    nearby_snp_base3_is_ref = False
    for snp_seq, snp_seq_count in all_allele_log_lik_sorted_dict:
        if snp_seq != nearby_snp_base1 and snp_seq != nearby_snp_base2:
            snp3_af = snp_seq_count / depth
            nearby_snp_base3_is_ref = snp_seq == ref_snp_base
            break
    overall_mean_mapQ = np.mean(all_reads_nearbysnp_list_dict["mapQ"])
    overall_mean_baseQ = np.mean(all_reads_nearbysnp_list_dict["snp_baseq"])
    overall_positive_strand_fraction = np.sum(
        all_reads_nearbysnp_list_dict["strand"]
    ) / len(all_reads_nearbysnp_list_dict["strand"])
    snp1_index = np.where(
        np.array(all_reads_nearbysnp_list_dict["snp_seq"]) == nearby_snp_base1
    )[0]
    snp2_index = np.where(
        np.array(all_reads_nearbysnp_list_dict["snp_seq"]) == nearby_snp_base2
    )[0]
    snp1_baseq = np.array(all_reads_nearbysnp_list_dict["snp_baseq"])[
        snp1_index
    ]
    snp2_baseq = np.array(all_reads_nearbysnp_list_dict["snp_baseq"])[
        snp2_index
    ]
    snp1_mapQ = np.array(all_reads_nearbysnp_list_dict["mapQ"])[snp1_index]
    snp2_mapQ = np.array(all_reads_nearbysnp_list_dict["mapQ"])[snp2_index]
    snp1_positive_strand = np.sum(
        np.array(all_reads_nearbysnp_list_dict["strand"])[snp1_index]
    )
    snp1_negative_strand = (
        len(np.array(all_reads_nearbysnp_list_dict["strand"])[snp1_index])
        - snp1_positive_strand
    )
    snp2_positive_strand = np.sum(
        np.array(all_reads_nearbysnp_list_dict["strand"])[snp2_index]
    )
    snp2_negative_strand = (
        len(np.array(all_reads_nearbysnp_list_dict["strand"])[snp2_index])
        - snp2_positive_strand
    )
    mapQ_diff = np.mean(snp1_mapQ) - np.mean(snp2_mapQ)
    mapQ_statistic, mapQ_pvalue = unpaired_nonparams_ranksum_test(
        snp1_mapQ, snp2_mapQ
    )  # 双端
    baseQ_diff = np.mean(snp1_baseq) - np.mean(snp2_baseq)
    baseQ_statistic, baseQ_pvalue = unpaired_nonparams_ranksum_test(
        snp1_baseq, snp2_baseq
    )  # 双端
    strand_oddsratio, strand_pvalue = fisher_exact_test(
        snp1_positive_strand,
        snp1_negative_strand,
        snp2_positive_strand,
        snp2_negative_strand,
    )  # 双端
    if strand_oddsratio > 1:
        if strand_oddsratio == np.inf:
            strand_oddsratio = 0
        else:
            strand_oddsratio = 1 / strand_oddsratio
    else:
        pass
    #  fisher_exact_test(20,20,10,0)
    #  (0.0, 0.003433373825779097)
    snp1_positive_strand_fraction = snp1_positive_strand / (
        snp1_positive_strand + snp1_negative_strand
    )
    snp2_positive_strand_fraction = snp2_positive_strand / (
        snp2_positive_strand + snp2_negative_strand
    )
    # if snp1_strand_fraction > 0.5:
    #     pass
    # else:
    #     snp1_strand_fraction = 1 - snp1_strand_fraction
    # if snp2_strand_fraction > 0.5:
    #     pass
    # else:
    #     snp2_strand_fraction = 1 - snp2_strand_fraction
    strand_fraction_diff = (
        snp1_positive_strand_fraction - snp2_positive_strand_fraction
    )
    if DIRECTIONAL_STATISTICS:
        overall_strand_fraction = overall_positive_strand_fraction
    else:
        strand_fraction_diff = abs(strand_fraction_diff)
        dp_diff = abs(dp_diff)
        dp_diff_fraction = abs(dp_diff_fraction)
        if overall_positive_strand_fraction > 0.5:
            overall_strand_fraction = overall_positive_strand_fraction
        else:
            overall_strand_fraction = 1 - overall_positive_strand_fraction
        if final_chrom_point_depth_percentile > 0.5:
            final_chrom_point_depth_percentile = (
                final_chrom_point_depth_percentile
            )
        else:
            final_chrom_point_depth_percentile = (
                1 - final_chrom_point_depth_percentile
            )
        if final_wgs_point_depth_percentile > 0.5:
            final_wgs_point_depth_percentile = final_wgs_point_depth_percentile
        else:
            final_wgs_point_depth_percentile = (
                1 - final_wgs_point_depth_percentile
            )
        mapQ_diff = abs(mapQ_diff)
        mapQ_statistic = abs(mapQ_statistic)
        baseQ_diff = abs(baseQ_diff)
        baseQ_statistic = abs(baseQ_statistic)
    if PHRED_P_VALUE:
        baseQ_pvalue = rate_2_phred_scale(baseQ_pvalue)
        mapQ_pvalue = rate_2_phred_scale(mapQ_pvalue)
        strand_pvalue = rate_2_phred_scale(strand_pvalue)
    hSNP_features["snp1_af"] = snp1_af
    hSNP_features["snp2_af"] = snp2_af
    hSNP_features["snp3_af"] = snp3_af
    hSNP_features["nearby_snp_base1_is_ref"] = nearby_snp_base1_is_ref
    hSNP_features["nearby_snp_base2_is_ref"] = nearby_snp_base2_is_ref
    hSNP_features["nearby_snp_base3_is_ref"] = nearby_snp_base3_is_ref
    hSNP_features["snp_overall_mean_mapQ"] = overall_mean_mapQ
    hSNP_features["snp_overall_mean_baseQ"] = overall_mean_baseQ
    hSNP_features["snp_overall_strand_fraction"] = overall_strand_fraction
    hSNP_features["dp_diff"] = dp_diff
    hSNP_features["dp_diff_fraction"] = dp_diff_fraction
    hSNP_features[
        "final_chrom_point_depth_percentile"
    ] = final_chrom_point_depth_percentile
    hSNP_features[
        "final_wgs_point_depth_percentile"
    ] = final_wgs_point_depth_percentile
    hSNP_features["nearby_hsnp_depth_ratio"] = nearby_hsnp_depth_ratio
    hSNP_features[
        "nearby_hsnp_chrom_point_median_depth_ratio"
    ] = nearby_hsnp_chrom_point_median_depth_ratio
    hSNP_features[
        "nearby_hsnp_wgs_point_median_depth_ratio"
    ] = nearby_hsnp_wgs_point_median_depth_ratio
    hSNP_features["mapQ_diff"] = mapQ_diff
    hSNP_features["mapQ_statistic"] = mapQ_statistic
    hSNP_features["mapQ_pvalue"] = mapQ_pvalue
    hSNP_features["baseQ_diff"] = baseQ_diff
    hSNP_features["baseQ_statistic"] = baseQ_statistic
    hSNP_features["baseQ_pvalue"] = baseQ_pvalue
    hSNP_features["strand_oddsratio"] = strand_oddsratio
    hSNP_features["strand_pvalue"] = strand_pvalue
    hSNP_features["strand_fraction_diff"] = strand_fraction_diff
    hSNP_features[
        "snp1_positive_strand_fraction"
    ] = snp1_positive_strand_fraction
    hSNP_features[
        "snp2_positive_strand_fraction"
    ] = snp2_positive_strand_fraction
    return (
        hSNP_features,
        all_reads_nearbysnp_dict,
        all_reads_nearbysnp_list_dict,
    )


def hard_filter_according_used_reads_features():
    # (0) 画 features 的分布来决定 hard filter 的指标 OK # TODO: remove hard_filter_according_used_reads_features
    # (1) strand bias
    # (2) mapQ bias
    # (3) overall_mean_mapQ
    # (4) overall_mean_baseQ
    # (5) remain features
    pass
    # spanning nearby snp and others features #
    # TODO: in the data analysis #


def cal_non_features_fraction_diff(feature, germ_index, mut_index):
    if len(feature) == 0:
        non_features = np.nan
        overall_non_fraction = np.nan
    else:
        non_features = 1 - np.array(feature)
        overall_non_fraction = np.sum(non_features) / len(non_features)
    germ_feature = np.array(feature)[germ_index]
    mut_feature = np.array(feature)[mut_index]
    germ_non_feature = 1 - germ_feature
    mut_non_feature = 1 - mut_feature
    if len(germ_non_feature) > 0:
        germ_non_feature_fraction = np.sum(germ_non_feature) / len(
            germ_non_feature
        )
    else:
        germ_non_feature_fraction = np.nan
    if (
        len(mut_non_feature) > 0
    ):  # BUG:RuntimeWarning: invalid value encountered in long_scalars
        mut_non_feature_fraction = np.sum(mut_non_feature) / len(
            mut_non_feature
        )
    else:
        mut_non_feature_fraction = np.nan
    if np.isnan(mut_non_feature_fraction) or np.isnan(
        germ_non_feature_fraction
    ):
        germ_mut_diff_non_feature_fraction = np.nan
    else:
        germ_mut_diff_non_feature_fraction = (
            germ_non_feature_fraction - mut_non_feature_fraction
        )
    return (
        overall_non_fraction,
        germ_non_feature_fraction,
        mut_non_feature_fraction,
        germ_mut_diff_non_feature_fraction,
    )


def get_have_bps_for_check(left_flanking_bps_num, right_flanking_bps_num):
    k = 0
    have_bps_index = []
    for left, right in zip(left_flanking_bps_num, right_flanking_bps_num):
        if (left > FLANKING_BPS_FOR_FEATURE_CUTOFF) and (
            right > FLANKING_BPS_FOR_FEATURE_CUTOFF
        ):
            have_bps_index.append(k)
        k += 1
    return have_bps_index


# XXX: source mut ok!
# HACK: ADD New Features left flanking bps and right flanking bps to avoid mapping fluctuation and ambiguous mapping
# TODO: mean left and right flanking bps number
# mean and fraction,and normalized by allele length and read length and mutant dp (can be adjust after extract features) DONE
# 和 mutant dp 和 total dp 和 mutant vaf 相关 DONE
# 我们算10个参数吧: 起始坐标数量 终止坐标数量 平均 和 比例 和 sd 和 mad ok inlist DONE
# TODO: ADD Features:  DP ok inlist  mutant dp ok inlist and VAF related ok inlist genotyping DP ok inlist and raw DP ok inlist allele length ok inlist read length ok inlist and INDEL difference fraction ok inlist and DP ratio e.tal ok inlist
# HACK: DEBUG: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray. Done
def get_left_and_right_flanking_mis_and_baseq_features(
    left_flanking_baseq,
    right_flanking_baseq,
    left_flanking_sbs_num,
    right_flanking_sbs_num,
    left_flanking_bps_num,
    right_flanking_bps_num,
    mut_index,
):
    left_flanking_baseq_mut_feature = np.array(
        left_flanking_baseq, dtype="object"
    )[
        mut_index
    ]  # DEBUG: VisibleDeprecationWarning DONE
    right_flanking_baseq_mut_feature = np.array(
        right_flanking_baseq, dtype="object"
    )[
        mut_index
    ]  # HACK: DEBUG: VisibleDeprecationWarning: because array('B', [34, 34, 31, 37, 37, 37, 37, 37, 39, 39]) left_flanking_baseq with different shape,so use dtype="object", DEBUG DONE
    left_flanking_sbs_num_mut_feature = np.array(left_flanking_sbs_num)[
        mut_index
    ]
    right_flanking_sbs_num_mut_feature = np.array(right_flanking_sbs_num)[
        mut_index
    ]
    left_flanking_bps_num_mut_feature = np.array(left_flanking_bps_num)[
        mut_index
    ]
    right_flanking_bps_num_mut_feature = np.array(right_flanking_bps_num)[
        mut_index
    ]
    if len(left_flanking_bps_num_mut_feature) > 0:
        left_flanking_bps_num_mut_feature_sd = np.std(
            left_flanking_bps_num_mut_feature, ddof=1
        )
        left_flanking_bps_num_mut_feature_mad = median_abs_deviation(
            left_flanking_bps_num_mut_feature, scale=1
        )
        left_flanking_bps_num_mut_feature_mean = np.mean(
            left_flanking_bps_num_mut_feature
        )
        left_flanking_bps_num_mut_feature_less_five_fraction = np.sum(
            left_flanking_bps_num_mut_feature < SHORT_FLANKING_BPS_CUTOFF
        ) / len(left_flanking_bps_num_mut_feature)
    else:
        left_flanking_bps_num_mut_feature_sd = np.nan
        left_flanking_bps_num_mut_feature_mad = np.nan
        left_flanking_bps_num_mut_feature_mean = np.nan
        left_flanking_bps_num_mut_feature_less_five_fraction = np.nan
    if len(right_flanking_bps_num_mut_feature) > 0:
        right_flanking_bps_num_mut_feature_sd = np.std(
            right_flanking_bps_num_mut_feature, ddof=1
        )
        right_flanking_bps_num_mut_feature_mad = median_abs_deviation(
            right_flanking_bps_num_mut_feature, scale=1
        )
        right_flanking_bps_num_mut_feature_mean = np.mean(
            right_flanking_bps_num_mut_feature
        )
        right_flanking_bps_num_mut_feature_less_five_fraction = np.sum(
            right_flanking_bps_num_mut_feature < SHORT_FLANKING_BPS_CUTOFF
        ) / len(right_flanking_bps_num_mut_feature)
    else:
        right_flanking_bps_num_mut_feature_sd = np.nan
        right_flanking_bps_num_mut_feature_mad = np.nan
        right_flanking_bps_num_mut_feature_mean = np.nan
        right_flanking_bps_num_mut_feature_less_five_fraction = np.nan
    used_read_index = get_have_bps_for_check(
        left_flanking_bps_num_mut_feature, right_flanking_bps_num_mut_feature
    )
    # pdb.set_trace()
    if used_read_index != []:
        left_flanking_baseq_mut_feature_used = left_flanking_baseq_mut_feature[
            used_read_index
        ]
        right_flanking_baseq_mut_feature_used = (
            right_flanking_baseq_mut_feature[used_read_index]
        )
        left_flanking_sbs_num_mut_feature_used = (
            left_flanking_sbs_num_mut_feature[used_read_index]
        )
        right_flanking_sbs_num_mut_feature_used = (
            right_flanking_sbs_num_mut_feature[used_read_index]
        )
        left_flanking_bps_num_mut_feature_used = (
            left_flanking_bps_num_mut_feature[used_read_index]
        )
        right_flanking_bps_num_mut_feature_used = (
            right_flanking_bps_num_mut_feature[used_read_index]
        )
        length_norm_left_flanking_sbs_num_mut_feature_used = (
            left_flanking_sbs_num_mut_feature_used
            / left_flanking_bps_num_mut_feature_used
        )
        length_norm_right_flanking_sbs_num_mut_feature_used = (
            right_flanking_sbs_num_mut_feature_used
            / right_flanking_bps_num_mut_feature_used
        )
        # left_flanking_baseq_mut_feature_mean = (
        #     left_flanking_baseq_mut_feature_used.mean(axis=1)
        # )
        left_flanking_baseq_mut_feature_mean = np.array(
            [
                np.array(arr).mean()
                for arr in left_flanking_baseq_mut_feature_used
            ]
        )
        # right_flanking_baseq_mut_feature_mean = (
        #     right_flanking_baseq_mut_feature_used.mean(axis=1)
        # )
        right_flanking_baseq_mut_feature_mean = np.array(
            [
                np.array(arr).mean()
                for arr in right_flanking_baseq_mut_feature_used
            ]
        )
        left_flanking_baseq_mean = np.mean(
            left_flanking_baseq_mut_feature_mean
        )
        right_flanking_baseq_mean = np.mean(
            right_flanking_baseq_mut_feature_mean
        )
        abs_diff_baseq_mean = abs(
            left_flanking_baseq_mean - right_flanking_baseq_mean
        )
        baseq_mean_statistic, baseq_mean_p = unpaired_nonparams_ranksum_test(
            left_flanking_baseq_mut_feature_mean,
            right_flanking_baseq_mut_feature_mean,
        )
        left_flanking_sbs_num_mean = np.mean(
            length_norm_left_flanking_sbs_num_mut_feature_used
        )
        right_flanking_sbs_num_mean = np.mean(
            length_norm_right_flanking_sbs_num_mut_feature_used
        )
        abs_diff_sbs_num_mean = abs(
            left_flanking_sbs_num_mean - right_flanking_sbs_num_mean
        )
        (
            sbs_num_mean_statistic,
            sbs_num_mean_p,
        ) = unpaired_nonparams_ranksum_test(
            length_norm_left_flanking_sbs_num_mut_feature_used,
            length_norm_right_flanking_sbs_num_mut_feature_used,
        )
        return (
            left_flanking_baseq_mean,
            right_flanking_baseq_mean,
            abs_diff_baseq_mean,
            baseq_mean_statistic,
            baseq_mean_p,
            left_flanking_sbs_num_mean,
            right_flanking_sbs_num_mean,
            abs_diff_sbs_num_mean,
            sbs_num_mean_statistic,
            sbs_num_mean_p,
            left_flanking_bps_num_mut_feature_mean,
            right_flanking_bps_num_mut_feature_mean,
            left_flanking_bps_num_mut_feature_less_five_fraction,
            right_flanking_bps_num_mut_feature_less_five_fraction,
            left_flanking_bps_num_mut_feature_sd,
            left_flanking_bps_num_mut_feature_mad,
            right_flanking_bps_num_mut_feature_sd,
            right_flanking_bps_num_mut_feature_mad,
        )
    else:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            left_flanking_bps_num_mut_feature_mean,
            right_flanking_bps_num_mut_feature_mean,
            left_flanking_bps_num_mut_feature_less_five_fraction,
            right_flanking_bps_num_mut_feature_less_five_fraction,
            left_flanking_bps_num_mut_feature_sd,
            right_flanking_bps_num_mut_feature_sd,
            left_flanking_bps_num_mut_feature_mad,
            right_flanking_bps_num_mut_feature_mad,
        )


def get_all_motifs(motif):
    d_motif = motif + motif
    all_motifs = []
    motif_length = len(motif)
    for i in range(len(motif)):
        all_motifs.append(d_motif[i : i + motif_length])
    return all_motifs


def judge_recurrent_motif(mut_seq, mis_loc, motif):
    motif_length = len(motif)
    if motif_length == 1:
        for i in mis_loc:
            if mut_seq[i] == motif:
                left_border = max(0, i - RECURRENT_MOTIF_JUDGE + 1)
                right_border = min(
                    i + 1, len(mut_seq) - RECURRENT_MOTIF_JUDGE + 1
                )
                for j in range(left_border, right_border):
                    if len(set(mut_seq[j : j + RECURRENT_MOTIF_JUDGE])) == 1:
                        return True
    else:
        all_motifs = get_all_motifs(motif)
        for i in mis_loc:
            left_border = max(0, i - RECURRENT_MOTIF_JUDGE + 1)
            right_border = min(i + 1, len(mut_seq) - RECURRENT_MOTIF_JUDGE + 1)
            for j in range(left_border, right_border):
                if len(set(mut_seq[j : j + RECURRENT_MOTIF_JUDGE])) == 1:
                    return True
            left_border2 = max(0, i - motif_length + 1)
            right_border2 = min(i + 1, len(mut_seq) - motif_length + 1)
            for k in range(left_border2, right_border2):
                if mut_seq[k : k + motif_length] in all_motifs:
                    return True
    return False


# def cal_overall_features_fraction(overall_features):
#     overall_feature_fraction = np.sum(overall_features)/len(overall_features)
#     return overall_feature_fraction

# def cal_overall_non_features_fraction(overall_features):
#     overall_non_feature = 1 - overall_features
#     overall_non_feature_fraction = np.sum(overall_non_feature)/len(overall_non_feature)
#     return overall_non_feature_fraction


def cal_features_fraction_diff(feature, germ_index, mut_index):
    if len(feature) > 0:
        overall_feature_fraction = np.sum(feature) / len(feature)
    else:
        overall_feature_fraction = np.nan
    germ_feature = np.array(feature)[germ_index]
    mut_feature = np.array(feature)[mut_index]
    if len(germ_feature) > 0:
        germ_feature_fraction = np.sum(germ_feature) / len(germ_feature)
    else:
        germ_feature_fraction = np.nan
    if (
        len(mut_feature) > 0
    ):  # BUG: RuntimeWarning: invalid value encountered in long_scalars
        mut_feature_fraction = np.sum(mut_feature) / len(mut_feature)
    else:
        mut_feature_fraction = np.nan
    if np.isnan(germ_feature_fraction) or np.isnan(mut_feature_fraction):
        germ_mut_diff_feature_fraction = np.nan
    else:
        germ_mut_diff_feature_fraction = (
            germ_feature_fraction - mut_feature_fraction
        )
    return (
        overall_feature_fraction,
        germ_feature_fraction,
        mut_feature_fraction,
        germ_mut_diff_feature_fraction,
    )


def cal_feature_mean_and_cutoff_fraction(
    feature, germ_index, mut_index, cutoff
):
    # 这边处理缺失值为 np.nan
    # 注意 np.nan == np.nan 返回 false
    # np.isnan(np.nan) 返回 true
    # None = None 返回 true
    feature = np.array(feature)
    feature_rm_NA = feature[np.where([i != "NA" for i in feature])[0]].astype(
        float
    )  # XXX: HAS DEBUG THROUGH ADDING [0] because 2D array here
    if list(feature_rm_NA) == []:
        overall_mean_feature = np.nan
    else:
        overall_mean_feature = np.mean(feature_rm_NA)
    germ_feature = np.array(feature)[germ_index]
    mut_feature = np.array(feature)[mut_index]
    germ_feature = germ_feature[
        np.where([i != "NA" for i in germ_feature])[0]
    ].astype(
        float
    )  # XXX: string 2 float
    mut_feature = mut_feature[
        np.where([i != "NA" for i in mut_feature])[0]
    ].astype(
        float
    )  # XXX array == "str"，如果 array 和 "str" 是同一种数据类型，则逐一与 "str" 扮演 element comparison，否则则只比较 array 是否等于 "str"
    # BUG:
    # >>> np.array([1,2,3])!="NA"
    # True
    # >>> np.array([1,2,3])!=1
    # array([False,  True,  True])
    # TODO:
    # np.mean([])
    # /Users/lid/opt/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3464: RuntimeWarning: Mean of empty slice.
    # return _methods._mean(a, axis=axis, dtype=dtype,
    # /Users/lid/opt/anaconda3/lib/python3.8/site-packages/numpy/core/_methods.py:192: RuntimeWarning: invalid value encountered in scalar divide
    # ret = ret.dtype.type(ret / rcount)
    # nan
    if list(germ_feature) == []:
        germ_feature_mean = np.nan
        germ_feature_more_fraction = np.nan
    else:
        germ_feature_mean = np.mean(germ_feature)
        germ_feature_more_fraction = sum(
            np.array(germ_feature) > cutoff
        ) / len(germ_feature)
    if list(mut_feature) == []:
        mut_feature_mean = np.nan
        mut_feature_more_fraction = np.nan
    else:
        mut_feature_mean = np.mean(mut_feature)
        mut_feature_more_fraction = sum(np.array(mut_feature) > cutoff) / len(
            mut_feature
        )
    return (
        germ_feature_mean,
        germ_feature_more_fraction,
        mut_feature_mean,
        mut_feature_more_fraction,
    )


def cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
    feature, germ_index, mut_index, side
):
    # 这边处理缺失值为 np.nan
    # 注意 np.nan == np.nan 返回 false
    # np.isnan(np.nan) 返回 true
    # None = None 返回 true
    feature = np.array(feature)
    feature_rm_NA = feature[np.where([i != "NA" for i in feature])[0]].astype(
        float
    )  # XXX: HAS DEBUG THROUGH ADDING [0] because 2D array here
    if list(feature_rm_NA) == []:
        overall_mean_feature = np.nan
    else:
        overall_mean_feature = np.mean(feature_rm_NA)
    germ_feature = np.array(feature)[germ_index]
    mut_feature = np.array(feature)[mut_index]
    germ_feature = germ_feature[
        np.where([i != "NA" for i in germ_feature])[0]
    ].astype(
        float
    )  # XXX: string 2 float
    mut_feature = mut_feature[
        np.where([i != "NA" for i in mut_feature])[0]
    ].astype(
        float
    )  # XXX array == "str"，如果 array 和 "str" 是同一种数据类型，则逐一与 "str" 扮演 element comparison，否则则只比较 array 是否等于 "str"
    # BUG:
    # >>> np.array([1,2,3])!="NA"
    # True
    # >>> np.array([1,2,3])!=1
    # array([False,  True,  True])
    # TODO:
    # np.mean([])
    # /Users/lid/opt/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3464: RuntimeWarning: Mean of empty slice.
    # return _methods._mean(a, axis=axis, dtype=dtype,
    # /Users/lid/opt/anaconda3/lib/python3.8/site-packages/numpy/core/_methods.py:192: RuntimeWarning: invalid value encountered in scalar divide
    # ret = ret.dtype.type(ret / rcount)
    # nan
    if list(germ_feature) == []:
        germ_feature_mean = np.nan
    else:
        germ_feature_mean = np.mean(germ_feature)
    if list(mut_feature) == []:
        mut_feature_mean = np.nan
    else:
        mut_feature_mean = np.mean(mut_feature)
    if np.isnan(germ_feature_mean) or np.isnan(mut_feature_mean):
        germ_mut_diff_feature_mean = np.nan
        germ_mut_feature_statistic = np.nan
        germ_mut_feature_pvalue = np.nan
    else:
        germ_mut_diff_feature_mean = germ_feature_mean - mut_feature_mean
        (
            germ_mut_feature_statistic,
            germ_mut_feature_pvalue,
        ) = unpaired_nonparams_ranksum_test_side(
            germ_feature, mut_feature, side
        )
    return (
        overall_mean_feature,
        germ_mut_diff_feature_mean,
        germ_mut_feature_statistic,
        germ_mut_feature_pvalue,
    )


def cal_undirection_features_fisher_test(feature, germ_index, mut_index, side):
    if len(feature) > 0:
        overall_feature_fraction = np.sum(feature) / len(feature)
    else:
        overall_feature_fraction = np.nan
    if overall_feature_fraction > 0.5:
        pass
    else:
        overall_feature_fraction = 1 - overall_feature_fraction
    germ_feature = np.array(feature)[germ_index]
    mut_feature = np.array(feature)[mut_index]
    if len(germ_feature) > 0:
        germ_feature_one_count = np.sum(germ_feature)
        germ_feature_zero_count = len(germ_feature) - germ_feature_one_count
        germ_feature_one_fraction = germ_feature_one_count / len(germ_feature)
        if germ_feature_one_fraction > 0.5:
            germ_feature_fraction = germ_feature_one_fraction
        else:
            germ_feature_fraction = 1 - germ_feature_one_fraction
    else:
        germ_feature_one_count = np.nan
        germ_feature_zero_count = np.nan
        germ_feature_one_fraction = np.nan
        germ_feature_fraction = np.nan
    if (
        len(mut_feature) > 0
    ):  # BUG: RuntimeWarning: invalid value encountered in long_scalars
        mut_feature_one_count = np.sum(mut_feature)
        mut_feature_zero_count = len(mut_feature) - mut_feature_one_count
        mut_feature_one_fraction = mut_feature_one_count / len(mut_feature)
        if mut_feature_one_fraction > 0.5:
            mut_feature_fraction = mut_feature_one_fraction
        else:
            mut_feature_fraction = 1 - mut_feature_one_fraction
    else:
        mut_feature_one_count = np.nan
        mut_feature_zero_count = np.nan
        mut_feature_one_fraction = np.nan
        mut_feature_fraction = np.nan
    if np.isnan(germ_feature_one_fraction) or np.isnan(
        mut_feature_one_fraction
    ):
        germ_mut_abs_diff_feature_fraction = np.nan
        oddsratio, pvalue = np.nan, np.nan
    else:
        germ_mut_abs_diff_feature_fraction = abs(
            germ_feature_one_fraction - mut_feature_one_fraction
        )
        oddsratio, pvalue = fisher_exact_test_side(
            germ_feature_one_count,
            germ_feature_zero_count,
            mut_feature_one_count,
            mut_feature_zero_count,
            side,
        )
        if oddsratio > 1:
            if oddsratio == np.inf:
                oddsratio = 0
            else:
                oddsratio = 1 / oddsratio
        else:
            pass
    return (
        overall_feature_fraction,
        germ_feature_fraction,
        mut_feature_fraction,
        germ_mut_abs_diff_feature_fraction,
        oddsratio,
        pvalue,
    )


def cal_noise_type_mean_diff_need_counter(noise_type, germ_index, mut_index):
    noise_type = np.array(noise_type)
    noise_type_rm_NA = noise_type[np.where(noise_type != "NA")[0]]
    noise_type_dict = collections.Counter(noise_type)
    if len(noise_type_rm_NA) > 0:
        overall_indel = (
            noise_type_dict.get("deletion", 0)
            + noise_type_dict.get("insertion", 0)
        ) / len(noise_type_rm_NA)
        overall_mis = noise_type_dict.get("mismatch", 0) / len(
            noise_type_rm_NA
        )
        overall_errors = overall_indel + overall_mis
    else:
        overall_indel = np.nan
        overall_mis = np.nan
        overall_errors = np.nan
    germ_noise_type = noise_type[germ_index]
    mut_noise_type = noise_type[mut_index]
    germ_noise_type_rm_NA = germ_noise_type[
        np.where(germ_noise_type != "NA")[0]
    ]
    mut_noise_type_rm_NA = mut_noise_type[np.where(mut_noise_type != "NA")[0]]
    germ_noise_type_dict = collections.Counter(germ_noise_type)
    mut_noise_type_dict = collections.Counter(mut_noise_type)
    if len(germ_noise_type_rm_NA) > 0:
        germ_indel = (
            germ_noise_type_dict.get("deletion", 0)
            + germ_noise_type_dict.get("insertion", 0)
        ) / len(germ_noise_type_rm_NA)
        germ_mis = germ_noise_type_dict.get("mismatch", 0) / len(
            germ_noise_type_rm_NA
        )
        germ_errors = germ_indel + germ_mis
    else:
        germ_indel = np.nan
        germ_mis = np.nan
        germ_errors = np.nan
    if len(mut_noise_type_rm_NA) > 0:
        mut_indel = (
            mut_noise_type_dict.get("deletion", 0)
            + mut_noise_type_dict.get("insertion", 0)
        ) / len(mut_noise_type_rm_NA)
        mut_mis = mut_noise_type_dict.get("mismatch", 0) / len(
            mut_noise_type_rm_NA
        )
        mut_errors = mut_indel + mut_mis
    else:  # BUG: ZeroDivisionError: division by zero
        mut_indel = np.nan
        mut_mis = np.nan
        mut_errors = np.nan
    # print(germ_noise_type)
    # print(mut_noise_type)
    # print(germ_indel)
    # print(mut_indel)
    # print(germ_mis)
    # print(mut_mis)
    # print(germ_errors)
    # print(mut_errors)
    if np.isnan(mut_indel) or np.isnan(germ_indel):
        germ_mut_diff_indel = np.nan
        germ_mut_diff_mis = np.nan
        germ_mut_diff_errors = np.nan
    else:
        germ_mut_diff_indel = germ_indel - mut_indel
        germ_mut_diff_mis = germ_mis - mut_mis
        germ_mut_diff_errors = germ_errors - mut_errors
    return (
        overall_indel,
        overall_mis,
        overall_errors,
        germ_mut_diff_indel,
        germ_mut_diff_mis,
        germ_mut_diff_errors,
        germ_indel,
        germ_mis,
        germ_errors,
        mut_indel,
        mut_mis,
        mut_errors,
        len(germ_noise_type_rm_NA),
        len(mut_noise_type_rm_NA),
    )


# def get_reference_mut_location(
#     all_reads_noise_type,
#     all_reads_features,
#     mis_location,
#     mut_read_index,
#     alleles_mut_type
# ):
#     mut_noise_type = all_reads_noise_type[mut_read_index]
#     mut_reads_features = all_reads_features[mut_read_index]
#     exact_match_read_feature = list(mut_reads_features[np.where(mut_noise_type == "no_noise")[0]])
#     zero_based_left_mis = min(mis_location)
#     zero_based_right_mis = max(mis_location)
#     if exact_match_read_feature == []:
#         return np.nan
#     else:
#         exact_match_read = exact_match_read_feature[0]
#         queryindex_refpos_dict = {i[0]: [i[1], i[2]] for i in exact_match_read.aligned_pairs}
#         str_padding_bps_start_query_index_include = exact_match_read.str_padding_bps_start_query_index_include
#         if alleles_mut_type in ["SNV", "continuous_MNV", "noncontinuous_MNV"]:
#             left_query_index = str_padding_bps_start_query_index_include + zero_based_left_mis
#             right_query_index = str_padding_bps_start_query_index_include + zero_based_right_mis
#             left_ref_pos =
#         elif alleles_mut_type == :

#         else:
#             return np.nan


def allele_comparison_statistic_test_for_str_loci_features(
    sample_feature_list_dict, germ_read_index, mut_read_index
):
    read_level_features_str_loci_dict = {}
    (
        overall_mean_mapQ,
        germ_mut_diff_mapQ,
        germ_mut_mapQ_statistic,
        germ_mut_mapQ_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_mapQ"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端, germ 可能来自 mismapping

    (
        germ_mapQ_mean,
        germ_mapQ_more_fraction,
        mut_mapQ_mean,
        mut_mapQ_more_fraction,
    ) = cal_feature_mean_and_cutoff_fraction(
        sample_feature_list_dict["overall_mapQ"],
        germ_read_index,
        mut_read_index,
        LOW_MAPQ_CUTOFF,
    )

    (
        overall_mean_mean_baseQ,
        germ_mut_diff_mean_baseQ,
        germ_mut_mean_baseQ_statistic,
        germ_mut_mean_baseQ_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_baseq_read"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端因为 germ het 区别不了 germ 和 mut,单端，假设 germ 更真, "greater"

    (
        germ_overall_baseQ_mean,
        germ_overall_baseQ_more_fraction,
        mut_overall_baseQ_mean,
        mut_overall_baseQ_more_fraction,
    ) = cal_feature_mean_and_cutoff_fraction(
        sample_feature_list_dict["overall_baseq_read"],
        germ_read_index,
        mut_read_index,
        LOW_OVERALL_LOW_BASEQ_CUTOFF,
    )

    (
        overall_mean_str_mis_num,
        germ_mut_diff_mean_str_mis_num,
        germ_mut_str_mis_num_statistic,
        germ_mut_str_mis_num_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["str_mis_num"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端, germ 可能来自 mismapping

    (
        overall_mean_str_mis_fraction,
        germ_mut_diff_mean_str_mis_fraction,
        germ_mut_str_mis_fraction_statistic,
        germ_mut_str_mis_fraction_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["str_mis_fraction"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端, germ 可能来自 mismapping problem

    (
        overall_mean_indel_size,
        germ_mut_diff_mean_indel_size,
        germ_mut_indel_size_statistic,
        germ_mut_indel_size_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["indel_size"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端, germ 可能来自 mismapping problem

    (
        overall_mean_indel_size_per_motif_length,
        germ_mut_diff_mean_indel_size_per_motif_length,
        germ_mut_indel_size_per_motif_length_statistic,
        germ_mut_indel_size_per_motif_length_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["indel_size_per_motif_length"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端, germ 可能来自 mismapping problem

    (
        overall_mean_flanking_sbs_num_per_bp,
        germ_mut_diff_mean_flanking_sbs_num_per_bp,
        germ_mut_flanking_sbs_num_per_bp_statistic,
        germ_mut_flanking_sbs_num_per_bp_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_read_flanking_sbs_num_per_bp"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端, germ 可能来自 mismapping problem

    (
        germ_mean_flanking_sbs_num_per_bp,
        germ_flanking_sbs_num_per_bp_more_fraction,
        mut_mean_flanking_sbs_num_per_bp,
        mut_flanking_sbs_num_per_bp_more_fraction,
    ) = cal_feature_mean_and_cutoff_fraction(
        sample_feature_list_dict["overall_read_flanking_sbs_num_per_bp"],
        germ_read_index,
        mut_read_index,
        HIGH_MISMATCHES_CUTOFF,
    )

    (
        overall_mean_flanking_indel_num_per_bp,
        germ_mut_diff_mean_flanking_indel_num_per_bp,
        germ_mut_flanking_indel_num_per_bp_statistic,
        germ_mut_flanking_indel_num_per_bp_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_read_flanking_indel_num_per_bp"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端, germ 可能来自 mismapping problem

    (
        overall_mean_flanking_bps_num,
        germ_mut_diff_mean_flanking_bps_num,
        germ_mut_flanking_bps_num_statistic,
        germ_mut_flanking_bps_num_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_flanking_bps_num"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # spanning 的程度 ，不管

    (
        overall_mean_read_pcr_cycle_percentage_sum,
        germ_mut_diff_mean_read_pcr_cycle_percentage_sum,
        germ_mut_read_pcr_cycle_percentage_sum_statistic,
        germ_mut_read_pcr_cycle_percentage_sum_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_read_pcr_cycle_percentage_sum"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # pcr cycle 的区间 ，不管

    (
        overall_mean_read_pcr_start_num,
        germ_mut_diff_mean_read_pcr_start_num,
        germ_mut_read_pcr_start_num_statistic,
        germ_mut_read_pcr_start_num_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_read_str_padding_pcr_start"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # pcr cycle start ，不管

    (
        overall_mean_read_pcr_cycle_sum,
        germ_mut_diff_mean_read_pcr_cycle_sum,
        germ_mut_read_pcr_cycle_sum_statistic,
        germ_mut_read_pcr_cycle_sum_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_read_pcr_cycle_sum"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # pcr cycle num 的区间 ，不管

    (
        overall_mean_temp_length,
        germ_mut_diff_mean_temp_length,
        germ_mut_temp_length_statistic,
        germ_mut_temp_length_pvalue,
    ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
        sample_feature_list_dict["overall_template_length"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )

    # feature = np.array(sample_feature_list_dict["overall_read_pcr_cycle_sum"])
    # germ_index = germ_read_index
    # mut_index = mut_read_index
    # feature_rm_NA = feature[np.where([i != "NA" for i in feature])[0]].astype(
    #     float
    # )  # XXX: HAS DEBUG THROUGH ADDING [0] because 2D array here
    # overall_mean_feature = np.mean(feature_rm_NA)
    # germ_feature = np.array(feature)[germ_index]
    # mut_feature = np.array(feature)[mut_index]
    # germ_feature = germ_feature[
    #     np.where([i != "NA" for i in germ_feature])[0]
    # ].astype(
    #     float
    # )  # XXX: string 2 float
    # mut_feature = mut_feature[
    #     np.where([i != "NA" for i in mut_feature])[0]
    # ].astype(
    #     float
    # )  # XXX array == "str"，如果 array 和 "str" 是同一种数据类型，则逐一与 "str" 扮演 element comparison，否则则只比较 array 是否等于 "str"
    # # BUG:
    # # >>> np.array([1,2,3])!="NA"
    # # True
    # # >>> np.array([1,2,3])!=1
    # # array([False,  True,  True])
    # # TODO:
    # # np.mean([])
    # # /Users/lid/opt/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3464: RuntimeWarning: Mean of empty slice.
    # # return _methods._mean(a, axis=axis, dtype=dtype,
    # # /Users/lid/opt/anaconda3/lib/python3.8/site-packages/numpy/core/_methods.py:192: RuntimeWarning: invalid value encountered in scalar divide
    # # ret = ret.dtype.type(ret / rcount)
    # # nan
    # germ_feature_mean = np.mean(germ_feature)
    # mut_feature_mean = np.mean(mut_feature)
    # germ_mut_diff_feature_mean = germ_feature_mean - mut_feature_mean
    # (
    #     germ_mut_feature_statistic,
    #     germ_mut_feature_pvalue,
    # ) = unpaired_nonparams_ranksum_test(germ_feature, mut_feature)

    # print(germ_feature, mut_feature)
    # print(germ_feature_mean, mut_feature_mean)
    # print(germ_mut_feature_statistic,germ_mut_feature_pvalue)
    (
        overall_non_second_fraction,
        germ_non_second_fraction,
        mut_non_second_fraction,
        germ_mut_diff_non_second_fraction,
    ) = cal_non_features_fraction_diff(
        sample_feature_list_dict["overall_secondary_read"],
        germ_read_index,
        mut_read_index,
    )
    # feature = sample_feature_list_dict["overall_secondary_read"]
    # germ_index = germ_read_index
    # mut_index = mut_read_index
    # non_features = 1 - np.array(sample_feature_list_dict["overall_secondary_read"])
    # overall_non_fraction = np.sum(non_features) / len(non_features)
    # germ_feature = np.array(feature)[germ_index]
    # mut_feature = np.array(feature)[mut_index]
    # germ_non_feature = 1 - germ_feature
    # mut_non_feature = 1 - mut_feature
    # germ_non_feature_fraction = np.sum(germ_non_feature) / len(
    #     germ_non_feature
    # )
    # if (
    #     len(mut_non_feature) > 0
    # ):  # BUG:RuntimeWarning: invalid value encountered in long_scalars
    #     mut_non_feature_fraction = np.sum(mut_non_feature) / len(
    #         mut_non_feature
    #     )
    # else:
    #     mut_non_feature_fraction = 0
    # germ_mut_diff_non_feature_fraction = (
    #     germ_non_feature_fraction - mut_non_feature_fraction
    # )
    # print(non_features)
    # print(overall_non_fraction)
    # print(germ_non_feature)
    # print(mut_non_feature)
    # print(germ_mut_diff_non_feature_fraction)
    (
        overall_non_supplementary_fraction,
        germ_non_supplementary_fraction,
        mut_non_supplementary_fraction,
        germ_mut_diff_non_supplementary_fraction,
    ) = cal_non_features_fraction_diff(
        sample_feature_list_dict["overall_supplementary_read"],
        germ_read_index,
        mut_read_index,
    )

    (
        overall_non_duplicate_fraction,
        germ_non_duplicate_fraction,
        mut_non_duplicate_fraction,
        germ_mut_diff_non_duplicate_fraction,
    ) = cal_non_features_fraction_diff(
        sample_feature_list_dict["overall_duplicate_read"],
        germ_read_index,
        mut_read_index,
    )

    (
        overall_non_unmapped_fraction,
        germ_non_unmapped_fraction,
        mut_non_unmapped_fraction,
        germ_mut_diff_non_unmapped_fraction,
    ) = cal_non_features_fraction_diff(
        sample_feature_list_dict["overall_unmapped_read"],
        germ_read_index,
        mut_read_index,
    )
    (
        overall_non_qcfail_fraction,
        germ_non_qcfail_fraction,
        mut_non_qcfail_fraction,
        germ_mut_diff_non_qcfail_fraction,
    ) = cal_non_features_fraction_diff(
        sample_feature_list_dict["overall_qcfail_read"],
        germ_read_index,
        mut_read_index,
    )

    (
        overall_non_clipped_fraction,
        germ_non_clipped_fraction,
        mut_non_clipped_fraction,
        germ_mut_diff_non_clipped_fraction,
    ) = cal_non_features_fraction_diff(
        sample_feature_list_dict["overall_clipped_read"],
        germ_read_index,
        mut_read_index,
    )

    (
        overall_non_flanking_indel_fraction,
        germ_non_flanking_indel_fraction,
        mut_non_flanking_indel_fraction,
        germ_mut_diff_non_flanking_indel_fraction,
    ) = cal_non_features_fraction_diff(
        sample_feature_list_dict["overall_flanking_indel"],
        germ_read_index,
        mut_read_index,
    )

    (
        overall_non_spanning_str_flanking_indel_fraction,
        germ_non_spanning_str_flanking_indel_fraction,
        mut_non_spanning_str_flanking_indel_fraction,
        germ_mut_diff_spanning_str_flanking_indel_fraction,
    ) = cal_non_features_fraction_diff(
        sample_feature_list_dict["overall_spanning_str_flanking_indel"],
        germ_read_index,
        mut_read_index,
    )

    (
        overall_strand_fraction,
        germ_strand_fraction,
        mut_strand_fraction,
        germ_mut_abs_diff_strand_fraction,
        strand_oddsratio,
        strand_pvalue,
    ) = cal_undirection_features_fisher_test(
        sample_feature_list_dict["overall_strand"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端

    (
        overall_orientation_fraction,
        germ_orientation_fraction,
        mut_orientation_fraction,
        germ_mut_abs_diff_orientation_fraction,
        orientation_oddsratio,
        orientation_pvalue,
    ) = cal_undirection_features_fisher_test(
        sample_feature_list_dict["overall_read1"],
        germ_read_index,
        mut_read_index,
        "two-sided",
    )  # 双端

    (
        overall_proper_pair_fraction,
        germ_proper_pair_fraction,
        mut_proper_pair_fraction,
        germ_mut_diff_proper_pair,
    ) = cal_features_fraction_diff(
        sample_feature_list_dict["overall_proper_pair"],
        germ_read_index,
        mut_read_index,
    )
    # TODO 单端和双端修改
    # TODO Fisher-Exact Test For Germ Errors and Mut Errors
    (
        overall_indel,
        overall_mis,
        overall_errors,
        germ_mut_diff_indel,
        germ_mut_diff_mis,
        germ_mut_diff_errors,
        germ_indel,
        germ_mis,
        germ_errors,
        mut_indel,
        mut_mis,
        mut_errors,
        germ_count,
        mut_count,
    ) = cal_noise_type_mean_diff_need_counter(
        sample_feature_list_dict["noise_type"], germ_read_index, mut_read_index
    )
    nan_exist = np.any(
        np.isnan([germ_count, germ_errors, mut_count, mut_errors])
    )
    if nan_exist:
        exact_match_odds_ratio, exact_match_p_value = np.nan, np.nan
    else:
        exact_match_odds_ratio, exact_match_p_value = fisher_exact_test_side(
            germ_count - germ_count * germ_errors,
            germ_count * germ_errors,
            mut_count - mut_count * mut_errors,
            mut_count * mut_errors,
            "two-sided",
        )
    # source and mutant left mean baseq right mean baseq left vs right mean base
    # source and mutant left mean misatch right mean mismatch left vs right mean mismatch(长度标准化需要)
    # 假如长度为零，去掉这对
    (
        left_flanking_baseq_mean,
        right_flanking_baseq_mean,
        flanking_abs_diff_baseq_mean,
        flanking_baseq_mean_statistic,
        flanking_baseq_mean_p,
        left_flanking_sbs_num_mean,
        right_flanking_sbs_num_mean,
        flanking_abs_diff_sbs_num_mean,
        flanking_sbs_num_mean_statistic,
        flanking_sbs_num_mean_p,
        left_flanking_bps_num_mut_feature_mean,
        right_flanking_bps_num_mut_feature_mean,
        left_flanking_bps_num_mut_feature_less_five_fraction,
        right_flanking_bps_num_mut_feature_less_five_fraction,
        left_flanking_bps_num_mut_feature_sd,
        right_flanking_bps_num_mut_feature_sd,
        left_flanking_bps_num_mut_feature_mad,
        right_flanking_bps_num_mut_feature_mad,
    ) = get_left_and_right_flanking_mis_and_baseq_features(
        sample_feature_list_dict["overall_left_flanking_baseq"],
        sample_feature_list_dict["overall_right_flanking_baseq"],
        sample_feature_list_dict["overall_left_flanking_sbs_num"],
        sample_feature_list_dict["overall_right_flanking_sbs_num"],
        sample_feature_list_dict["overall_left_flanking_bps_num"],
        sample_feature_list_dict["overall_right_flanking_bps_num"],
        mut_read_index,
    )
    # 没必要用这个
    # reference_mis_loc = get_reference_mut_location(
    #     sample_feature_list_dict["noise_type"],
    #     sample_feature_list_dict["all_reads"],
    #     sample_feature_list_dict["mis_location"],
    #     mut_read_index,
    #     sample_feature_list_dict["alleles_mut_type"],
    # )
    # print(sample_feature_list_dict["noise_type"])
    # print(len(sample_feature_list_dict["noise_type"]))
    # print(sample_feature_list_dict["noise_type"].count("mismatch"))
    # print(sample_feature_list_dict["noise_type"].count("insertion"))
    # print(sample_feature_list_dict["noise_type"].count("deletion"))
    read_level_features_str_loci_dict["overall_mean_mapQ"] = overall_mean_mapQ
    read_level_features_str_loci_dict[
        "germ_mut_diff_mapQ"
    ] = germ_mut_diff_mapQ
    read_level_features_str_loci_dict[
        "germ_mut_mapQ_statistic"
    ] = germ_mut_mapQ_statistic
    read_level_features_str_loci_dict[
        "germ_mut_mapQ_pvalue"
    ] = germ_mut_mapQ_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_mean_baseQ"
    ] = overall_mean_mean_baseQ
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_baseQ"
    ] = germ_mut_diff_mean_baseQ
    read_level_features_str_loci_dict[
        "germ_mut_mean_baseQ_statistic"
    ] = germ_mut_mean_baseQ_statistic
    read_level_features_str_loci_dict[
        "germ_mut_mean_baseQ_pvalue"
    ] = germ_mut_mean_baseQ_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_str_mis_num"
    ] = overall_mean_str_mis_num
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_str_mis_num"
    ] = germ_mut_diff_mean_str_mis_num
    read_level_features_str_loci_dict[
        "germ_mut_str_mis_num_statistic"
    ] = germ_mut_str_mis_num_statistic
    read_level_features_str_loci_dict[
        "germ_mut_str_mis_num_pvalue"
    ] = germ_mut_str_mis_num_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_str_mis_fraction"
    ] = overall_mean_str_mis_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_str_mis_fraction"
    ] = germ_mut_diff_mean_str_mis_fraction
    read_level_features_str_loci_dict[
        "germ_mut_str_mis_fraction_statistic"
    ] = germ_mut_str_mis_fraction_statistic
    read_level_features_str_loci_dict[
        "germ_mut_str_mis_fraction_pvalue"
    ] = germ_mut_str_mis_fraction_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_indel_size"
    ] = overall_mean_indel_size
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_indel_size"
    ] = germ_mut_diff_mean_indel_size
    read_level_features_str_loci_dict[
        "germ_mut_indel_size_statistic"
    ] = germ_mut_indel_size_statistic
    read_level_features_str_loci_dict[
        "germ_mut_indel_size_pvalue"
    ] = germ_mut_indel_size_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_indel_size_per_motif_length"
    ] = overall_mean_indel_size_per_motif_length
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_indel_size_per_motif_length"
    ] = germ_mut_diff_mean_indel_size_per_motif_length
    read_level_features_str_loci_dict[
        "germ_mut_indel_size_per_motif_length_statistic"
    ] = germ_mut_indel_size_per_motif_length_statistic
    read_level_features_str_loci_dict[
        "germ_mut_indel_size_per_motif_length_pvalue"
    ] = germ_mut_indel_size_per_motif_length_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_flanking_sbs_num_per_bp"
    ] = overall_mean_flanking_sbs_num_per_bp
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_flanking_sbs_num_per_bp"
    ] = germ_mut_diff_mean_flanking_sbs_num_per_bp
    read_level_features_str_loci_dict[
        "germ_mut_flanking_sbs_num_per_bp_statistic"
    ] = germ_mut_flanking_sbs_num_per_bp_statistic
    read_level_features_str_loci_dict[
        "germ_mut_flanking_sbs_num_per_bp_pvalue"
    ] = germ_mut_flanking_sbs_num_per_bp_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_flanking_indel_num_per_bp"
    ] = overall_mean_flanking_indel_num_per_bp
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_flanking_indel_num_per_bp"
    ] = germ_mut_diff_mean_flanking_indel_num_per_bp
    read_level_features_str_loci_dict[
        "germ_mut_flanking_indel_num_per_bp_statistic"
    ] = germ_mut_flanking_indel_num_per_bp_statistic
    read_level_features_str_loci_dict[
        "germ_mut_flanking_indel_num_per_bp_pvalue"
    ] = germ_mut_flanking_indel_num_per_bp_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_flanking_bps_num"
    ] = overall_mean_flanking_bps_num
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_flanking_bps_num"
    ] = germ_mut_diff_mean_flanking_bps_num
    read_level_features_str_loci_dict[
        "germ_mut_flanking_bps_num_statistic"
    ] = germ_mut_flanking_bps_num_statistic
    read_level_features_str_loci_dict[
        "germ_mut_flanking_bps_num_pvalue"
    ] = germ_mut_flanking_bps_num_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_read_pcr_cycle_percentage_sum"
    ] = overall_mean_read_pcr_cycle_percentage_sum
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_read_pcr_cycle_percentage_sum"
    ] = germ_mut_diff_mean_read_pcr_cycle_percentage_sum
    read_level_features_str_loci_dict[
        "germ_mut_read_pcr_cycle_percentage_sum_statistic"
    ] = germ_mut_read_pcr_cycle_percentage_sum_statistic
    read_level_features_str_loci_dict[
        "germ_mut_read_pcr_cycle_percentage_sum_pvalue"
    ] = germ_mut_read_pcr_cycle_percentage_sum_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_read_pcr_start_num"
    ] = overall_mean_read_pcr_start_num
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_read_pcr_start_num"
    ] = germ_mut_diff_mean_read_pcr_start_num
    read_level_features_str_loci_dict[
        "germ_mut_read_pcr_start_num_statistic"
    ] = germ_mut_read_pcr_start_num_statistic
    read_level_features_str_loci_dict[
        "germ_mut_read_pcr_start_num_pvalue"
    ] = germ_mut_read_pcr_start_num_pvalue
    read_level_features_str_loci_dict[
        "overall_mean_read_pcr_cycle_sum"
    ] = overall_mean_read_pcr_cycle_sum
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_read_pcr_cycle_sum"
    ] = germ_mut_diff_mean_read_pcr_cycle_sum
    read_level_features_str_loci_dict[
        "germ_mut_read_pcr_cycle_sum_statistic"
    ] = germ_mut_read_pcr_cycle_sum_statistic
    read_level_features_str_loci_dict[
        "germ_mut_read_pcr_cycle_sum_pvalue"
    ] = germ_mut_read_pcr_cycle_sum_pvalue
    read_level_features_str_loci_dict[
        "overall_non_second_fraction"
    ] = overall_non_second_fraction
    read_level_features_str_loci_dict[
        "germ_non_second_fraction"
    ] = germ_non_second_fraction
    read_level_features_str_loci_dict[
        "mut_non_second_fraction"
    ] = mut_non_second_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_non_second_fraction"
    ] = germ_mut_diff_non_second_fraction
    read_level_features_str_loci_dict[
        "overall_non_supplementary_fraction"
    ] = overall_non_supplementary_fraction
    read_level_features_str_loci_dict[
        "germ_non_supplementary_fraction"
    ] = germ_non_supplementary_fraction
    read_level_features_str_loci_dict[
        "mut_non_supplementary_fraction"
    ] = mut_non_supplementary_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_non_supplementary_fraction"
    ] = germ_mut_diff_non_supplementary_fraction
    read_level_features_str_loci_dict[
        "overall_non_duplicate_fraction"
    ] = overall_non_duplicate_fraction
    read_level_features_str_loci_dict[
        "germ_non_duplicate_fraction"
    ] = germ_non_duplicate_fraction
    read_level_features_str_loci_dict[
        "mut_non_duplicate_fraction"
    ] = mut_non_duplicate_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_non_duplicate_fraction"
    ] = germ_mut_diff_non_duplicate_fraction
    read_level_features_str_loci_dict[
        "overall_non_unmapped_fraction"
    ] = overall_non_unmapped_fraction
    read_level_features_str_loci_dict[
        "germ_non_unmapped_fraction"
    ] = germ_non_unmapped_fraction
    read_level_features_str_loci_dict[
        "mut_non_unmapped_fraction"
    ] = mut_non_unmapped_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_non_unmapped_fraction"
    ] = germ_mut_diff_non_unmapped_fraction
    read_level_features_str_loci_dict[
        "overall_non_qcfail_fraction"
    ] = overall_non_qcfail_fraction
    read_level_features_str_loci_dict[
        "germ_non_qcfail_fraction"
    ] = germ_non_qcfail_fraction
    read_level_features_str_loci_dict[
        "mut_non_qcfail_fraction"
    ] = mut_non_qcfail_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_non_qcfail_fraction"
    ] = germ_mut_diff_non_qcfail_fraction
    read_level_features_str_loci_dict[
        "overall_non_clipped_fraction"
    ] = overall_non_clipped_fraction
    read_level_features_str_loci_dict[
        "germ_non_clipped_fraction"
    ] = germ_non_clipped_fraction
    read_level_features_str_loci_dict[
        "mut_non_clipped_fraction"
    ] = mut_non_clipped_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_non_clipped_fraction"
    ] = germ_mut_diff_non_clipped_fraction
    read_level_features_str_loci_dict[
        "overall_non_flanking_indel_fraction"
    ] = overall_non_flanking_indel_fraction
    read_level_features_str_loci_dict[
        "germ_non_flanking_indel_fraction"
    ] = germ_non_flanking_indel_fraction
    read_level_features_str_loci_dict[
        "mut_non_flanking_indel_fraction"
    ] = mut_non_flanking_indel_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_non_flanking_indel_fraction"
    ] = germ_mut_diff_non_flanking_indel_fraction
    read_level_features_str_loci_dict[
        "overall_non_spanning_str_flanking_indel_fraction"
    ] = overall_non_spanning_str_flanking_indel_fraction
    read_level_features_str_loci_dict[
        "germ_non_spanning_str_flanking_indel_fraction"
    ] = germ_non_spanning_str_flanking_indel_fraction
    read_level_features_str_loci_dict[
        "mut_non_spanning_str_flanking_indel_fraction"
    ] = mut_non_spanning_str_flanking_indel_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_spanning_str_flanking_indel_fraction"
    ] = germ_mut_diff_spanning_str_flanking_indel_fraction
    read_level_features_str_loci_dict[
        "overall_strand_fraction"
    ] = overall_strand_fraction
    read_level_features_str_loci_dict[
        "germ_strand_fraction"
    ] = germ_strand_fraction
    read_level_features_str_loci_dict[
        "mut_strand_fraction"
    ] = mut_strand_fraction
    read_level_features_str_loci_dict[
        "germ_mut_abs_diff_strand_fraction"
    ] = germ_mut_abs_diff_strand_fraction
    read_level_features_str_loci_dict[
        "germ_mut_strand_oddsratio"
    ] = strand_oddsratio
    read_level_features_str_loci_dict["germ_mut_strand_pvalue"] = strand_pvalue
    read_level_features_str_loci_dict[
        "overall_orientation_fraction"
    ] = overall_orientation_fraction
    read_level_features_str_loci_dict[
        "germ_orientation_fraction"
    ] = germ_orientation_fraction
    read_level_features_str_loci_dict[
        "mut_orientation_fraction"
    ] = mut_orientation_fraction
    read_level_features_str_loci_dict[
        "germ_mut_abs_diff_orientation_fraction"
    ] = germ_mut_abs_diff_orientation_fraction
    read_level_features_str_loci_dict[
        "germ_mut_orientation_oddsratio"
    ] = orientation_oddsratio
    read_level_features_str_loci_dict[
        "germ_mut_orientation_pvalue"
    ] = orientation_pvalue
    read_level_features_str_loci_dict[
        "overall_proper_pair_fraction"
    ] = overall_proper_pair_fraction
    read_level_features_str_loci_dict[
        "germ_proper_pair_fraction"
    ] = germ_proper_pair_fraction
    read_level_features_str_loci_dict[
        "mut_proper_pair_fraction"
    ] = mut_proper_pair_fraction
    read_level_features_str_loci_dict[
        "germ_mut_diff_proper_pair"
    ] = germ_mut_diff_proper_pair
    read_level_features_str_loci_dict["overall_indel"] = overall_indel
    read_level_features_str_loci_dict["overall_mis"] = overall_mis
    read_level_features_str_loci_dict["overall_errors"] = overall_errors
    read_level_features_str_loci_dict[
        "germ_mut_diff_indel"
    ] = germ_mut_diff_indel
    read_level_features_str_loci_dict["germ_mut_diff_mis"] = germ_mut_diff_mis
    read_level_features_str_loci_dict[
        "germ_mut_diff_errors"
    ] = germ_mut_diff_errors
    read_level_features_str_loci_dict["germ_indel"] = germ_indel
    read_level_features_str_loci_dict["germ_mis"] = germ_mis
    read_level_features_str_loci_dict["germ_errors"] = germ_errors
    read_level_features_str_loci_dict["mut_indel"] = mut_indel
    read_level_features_str_loci_dict["mut_mis"] = mut_mis
    read_level_features_str_loci_dict["mut_errors"] = mut_errors
    read_level_features_str_loci_dict[
        "exact_match_odds_ratio"
    ] = exact_match_odds_ratio
    read_level_features_str_loci_dict[
        "exact_match_p_value"
    ] = exact_match_p_value
    read_level_features_str_loci_dict[
        "left_flanking_baseq_mean"
    ] = left_flanking_baseq_mean
    read_level_features_str_loci_dict[
        "right_flanking_baseq_mean"
    ] = right_flanking_baseq_mean
    read_level_features_str_loci_dict[
        "flanking_abs_diff_baseq_mean"
    ] = flanking_abs_diff_baseq_mean
    read_level_features_str_loci_dict[
        "flanking_baseq_mean_statistic"
    ] = flanking_baseq_mean_statistic
    read_level_features_str_loci_dict[
        "flanking_baseq_mean_p"
    ] = flanking_baseq_mean_p
    read_level_features_str_loci_dict[
        "left_flanking_sbs_num_mean"
    ] = left_flanking_sbs_num_mean
    read_level_features_str_loci_dict[
        "right_flanking_sbs_num_mean"
    ] = right_flanking_sbs_num_mean
    read_level_features_str_loci_dict[
        "flanking_abs_diff_sbs_num_mean"
    ] = flanking_abs_diff_sbs_num_mean
    read_level_features_str_loci_dict[
        "flanking_sbs_num_mean_statistic"
    ] = flanking_sbs_num_mean_statistic
    read_level_features_str_loci_dict[
        "flanking_sbs_num_mean_p"
    ] = flanking_sbs_num_mean_p
    read_level_features_str_loci_dict[
        "germ_overall_baseQ_mean"
    ] = germ_overall_baseQ_mean
    read_level_features_str_loci_dict[
        "germ_overall_baseQ_more_fraction"
    ] = germ_overall_baseQ_more_fraction
    read_level_features_str_loci_dict["germ_mapQ_mean"] = germ_mapQ_mean
    read_level_features_str_loci_dict[
        "germ_mapQ_more_fraction"
    ] = germ_mapQ_more_fraction
    read_level_features_str_loci_dict[
        "mut_overall_baseQ_mean"
    ] = mut_overall_baseQ_mean
    read_level_features_str_loci_dict[
        "mut_overall_baseQ_more_fraction"
    ] = mut_overall_baseQ_more_fraction
    read_level_features_str_loci_dict["mut_mapQ_mean"] = mut_mapQ_mean
    read_level_features_str_loci_dict[
        "mut_mapQ_more_fraction"
    ] = mut_mapQ_more_fraction
    read_level_features_str_loci_dict[
        "overall_mean_temp_length"
    ] = overall_mean_temp_length
    read_level_features_str_loci_dict[
        "germ_mut_diff_mean_temp_length"
    ] = germ_mut_diff_mean_temp_length
    read_level_features_str_loci_dict[
        "germ_mut_temp_length_statistic"
    ] = germ_mut_temp_length_statistic
    read_level_features_str_loci_dict[
        "germ_mut_temp_length_pvalue"
    ] = germ_mut_temp_length_pvalue
    read_level_features_str_loci_dict[
        "left_flanking_bps_num_mut_feature_mean"
    ] = left_flanking_bps_num_mut_feature_mean
    read_level_features_str_loci_dict[
        "right_flanking_bps_num_mut_feature_mean"
    ] = right_flanking_bps_num_mut_feature_mean
    read_level_features_str_loci_dict[
        "left_flanking_bps_num_mut_feature_less_five_fraction"
    ] = left_flanking_bps_num_mut_feature_less_five_fraction
    read_level_features_str_loci_dict[
        "right_flanking_bps_num_mut_feature_less_five_fraction"
    ] = right_flanking_bps_num_mut_feature_less_five_fraction
    read_level_features_str_loci_dict[
        "left_flanking_bps_num_mut_feature_sd"
    ] = left_flanking_bps_num_mut_feature_sd
    read_level_features_str_loci_dict[
        "right_flanking_bps_num_mut_feature_sd"
    ] = right_flanking_bps_num_mut_feature_sd
    read_level_features_str_loci_dict[
        "left_flanking_bps_num_mut_feature_mad"
    ] = left_flanking_bps_num_mut_feature_mad
    read_level_features_str_loci_dict[
        "right_flanking_bps_num_mut_feature_mad"
    ] = right_flanking_bps_num_mut_feature_mad
    read_level_features_str_loci_dict[
        "germ_mean_flanking_sbs_num_per_bp"
    ] = germ_mean_flanking_sbs_num_per_bp
    read_level_features_str_loci_dict[
        "germ_flanking_sbs_num_per_bp_more_fraction"
    ] = germ_flanking_sbs_num_per_bp_more_fraction
    read_level_features_str_loci_dict[
        "mut_mean_flanking_sbs_num_per_bp"
    ] = mut_mean_flanking_sbs_num_per_bp
    read_level_features_str_loci_dict[
        "mut_flanking_sbs_num_per_bp_more_fraction"
    ] = mut_flanking_sbs_num_per_bp_more_fraction
    return read_level_features_str_loci_dict


def transform_germ_mut_2_germ_source_feature_dict(germ_source_feature_dict):
    germ_source_feature_update_key_dict = {}
    for key, value in germ_source_feature_dict.items():
        update_key = key + "_germ_source"
        germ_source_feature_update_key_dict[update_key] = value
    return germ_source_feature_update_key_dict


def cal_allele_left_start_diff(
    obs_start, str_padding_start, read_length, allele_length
):
    mean_discrete_uniform_dis = (
        str_padding_start + (str_padding_start - (read_length - allele_length))
    ) / 2
    if len(obs_start) > 0:
        mean_obs_start = np.mean(obs_start)
        allele_left_start_diff = mean_obs_start - mean_discrete_uniform_dis
    else:
        mean_obs_start = np.nan
        allele_left_start_diff = np.nan
    return allele_left_start_diff


def cal_allele_right_end_diff(
    obs_end_exclude, str_padding_end_exclude, read_length, allele_length
):
    mean_discrete_uniform_dis = (
        str_padding_end_exclude
        + (str_padding_end_exclude + (read_length - allele_length))
    ) / 2
    if len(obs_end_exclude) > 0:
        mean_obs_end_exclude = np.mean(obs_end_exclude)
        allele_right_end_diff = (
            mean_obs_end_exclude - mean_discrete_uniform_dis
        )
    else:
        mean_obs_end_exclude = np.nan
        allele_right_end_diff = np.nan
    return allele_right_end_diff


def cal_allele_shape_based_features(
    feature_start_list,
    feature_end_list,
    str_padding_start,
    str_padding_end_exclude,
    read_length,
    allele_length,
    allele_index,
):
    allele_start_feature = np.array(feature_start_list)[allele_index]
    allele_start_feature_rm_NA = allele_start_feature[
        np.where([i != "NA" for i in allele_start_feature])[0]
    ].astype(
        int
    )  # XXX: string 2 float
    allele_end_feature = np.array(feature_end_list)[allele_index]
    allele_end_feature_rm_NA = allele_end_feature[
        np.where([i != "NA" for i in allele_end_feature])[0]
    ].astype(
        int
    )  # XXX: string 2 float
    unique_start_coordinate_number = len(set(allele_start_feature_rm_NA))
    unique_end_coordinate_number = len(set(allele_end_feature_rm_NA))
    allele_left_start_diff = cal_allele_left_start_diff(
        allele_start_feature_rm_NA,
        str_padding_start,
        read_length,
        allele_length,
    )
    allele_right_end_diff = cal_allele_right_end_diff(
        allele_end_feature_rm_NA,
        str_padding_end_exclude,
        read_length,
        allele_length,
    )
    # 2 ValueError: zero-size array to reduction operation maximum which has no identity DEBUG，理论上不会报错啊，奇怪了
    if list(allele_start_feature_rm_NA) == []:
        allele_ks_start_stat = np.nan
        allele_ks_start_p_value = np.nan
    else:
        (
            allele_ks_start_stat,
            allele_ks_start_p_value,
        ) = ks_test_goodness_of_fit(
            rvs=allele_start_feature_rm_NA,
            args=(
                (str_padding_start - (read_length - allele_length)),
                str_padding_start,
            ),
        )
    if list(allele_end_feature_rm_NA) == []:
        allele_ks_end_stat = np.nan
        allele_ks_end_p_value = np.nan
    else:
        allele_ks_end_stat, allele_ks_end_p_value = ks_test_goodness_of_fit(
            rvs=allele_end_feature_rm_NA,
            args=(
                str_padding_end_exclude,
                str_padding_end_exclude + (read_length - allele_length),
            ),
        )
    region_size = int(read_length - allele_length + 1)
    observed_start_frequency_dict = dict(
        collections.Counter(allele_start_feature_rm_NA)
    )
    observed_start_frequency = [
        observed_start_frequency_dict.get(i, 0)
        for i in range(
            int(str_padding_start - (read_length - allele_length)),
            int(str_padding_start + 1),
        )
    ]
    observed_end_frequency_dict = dict(
        collections.Counter(allele_end_feature_rm_NA)
    )
    observed_end_frequency = [
        observed_end_frequency_dict.get(i, 0)
        for i in range(
            int(str_padding_end_exclude),
            int(str_padding_end_exclude + (read_length - allele_length) + 1),
        )
    ]
    for extra_start in list(
        set(list(observed_start_frequency_dict.keys()))
        - set(
            list(
                range(
                    int(str_padding_start - (read_length - allele_length)),
                    int(str_padding_start + 1),
                )
            )
        )
    ):
        observed_start_frequency.append(
            observed_start_frequency_dict[extra_start]
        )
    for extra_end in list(
        set(list(observed_end_frequency_dict.keys()))
        - set(
            list(
                range(
                    int(str_padding_end_exclude),
                    int(
                        str_padding_end_exclude
                        + (read_length - allele_length)
                        + 1
                    ),
                )
            )
        )
    ):
        observed_end_frequency.append(observed_end_frequency_dict[extra_end])
    # 2 ValueError: zero-size array to reduction operation maximum which has no identity DEBUG
    # /home/douyanmeiLab/wangweixiang/anaconda3/lib/python3.9/site-packages/scipy/stats/stats.py:6707: RuntimeWarning: divide by zero encountered in true_divide DEBUG
    # terms = (f_obs_float - f_exp)**2 / f_exp DEBUG
    if len(allele_start_feature_rm_NA) == 0:
        allele_chi_square_start_stat = np.nan
        allele_chi_square_start_p_value = np.nan
    else:
        (
            allele_chi_square_start_stat,
            allele_chi_square_start_p_value,
        ) = chi_square_goodness_of_fit(
            len(allele_start_feature_rm_NA),
            region_size,
            observed_start_frequency,
        )
    if len(allele_end_feature_rm_NA) == 0:
        allele_chi_square_end_stat = np.nan
        allele_chi_square_end_p_value = np.nan
    else:
        (
            allele_chi_square_end_stat,
            allele_chi_square_end_p_value,
        ) = chi_square_goodness_of_fit(
            len(allele_end_feature_rm_NA), region_size, observed_end_frequency
        )
    return (
        allele_left_start_diff,
        allele_right_end_diff,
        allele_ks_start_stat,
        allele_ks_start_p_value,
        allele_ks_end_stat,
        allele_ks_end_p_value,
        allele_chi_square_start_stat,
        allele_chi_square_start_p_value,
        allele_chi_square_end_stat,
        allele_chi_square_end_p_value,
        unique_start_coordinate_number,
        unique_end_coordinate_number,
    )


def get_hSNP_number_for_cluster_SNP_loci_exclude():
    pass


# diff or fold change or 统计量 or pvalue or fraction or raw count or phred scale or percentile rank or odds ratio or log likelihood or log or probability or lik or or phred scale or percentile
def statistic_test_for_str_loci_features(
    sample_feature_list_dict,
    muttype,
    str_padding_start,
    str_padding_end_exclude,
):
    str_features_dict = {}
    used_read_index = np.where(
        np.array(sample_feature_list_dict["overall_feature_use_read"]) == 1
    )[0]
    germ_read_index = np.where(
        np.array(sample_feature_list_dict["read_hap_assignment"]) == "germ"
    )[0]
    source_read_index = np.where(
        np.array(sample_feature_list_dict["read_hap_assignment"]) == "source"
    )[0]
    mut_read_index = np.where(
        np.array(sample_feature_list_dict["read_hap_assignment"]) == "mut"
    )[0]
    unknown_hap_read_index = np.where(
        np.array(sample_feature_list_dict["read_hap_assignment"]) == "NA"
    )[0]
    unknown_hap_read_num = len(unknown_hap_read_index)
    unknown_hap_read_fraction = unknown_hap_read_num / len(
        sample_feature_list_dict["read_hap_assignment"]
    )
    germ_read_num = len(germ_read_index)
    source_read_num = len(source_read_index)
    mut_read_num = len(mut_read_index)
    germ_read_fraction = germ_read_num / len(
        sample_feature_list_dict["read_hap_assignment"]
    )
    source_read_fraction = source_read_num / len(
        sample_feature_list_dict["read_hap_assignment"]
    )
    mut_read_fraction = mut_read_num / len(
        sample_feature_list_dict["read_hap_assignment"]
    )
    read_length = np.median(sample_feature_list_dict["overall_read_length"])
    germ_str_padding_length = sample_feature_list_dict["seq0_length"]
    mut_str_padding_length = sample_feature_list_dict["seq2_length"]
    source_str_padding_length = sample_feature_list_dict["seq1_length"]
    germ_mut_str_padding_length_diff = (
        sample_feature_list_dict["seq0_length"]
        - sample_feature_list_dict["seq2_length"]
    )
    source_mut_str_padding_length_diff = (
        sample_feature_list_dict["seq1_length"]
        - sample_feature_list_dict["seq2_length"]
    )
    germ_source_str_padding_length_diff = (
        sample_feature_list_dict["seq0_length"]
        - sample_feature_list_dict["seq1_length"]
    )
    str_features_dict["unknown_hap_read_fraction"] = unknown_hap_read_fraction
    str_features_dict["germ_read_fraction"] = germ_read_fraction
    str_features_dict["source_read_fraction"] = source_read_fraction
    str_features_dict["mut_read_fraction"] = mut_read_fraction
    str_features_dict["read_length"] = read_length
    str_features_dict["germ_str_padding_length"] = germ_str_padding_length
    str_features_dict["mut_str_padding_length"] = mut_str_padding_length
    str_features_dict["source_str_padding_length"] = source_str_padding_length
    str_features_dict[
        "germ_mut_str_padding_length_diff"
    ] = germ_mut_str_padding_length_diff
    str_features_dict[
        "source_mut_str_padding_length_diff"
    ] = source_mut_str_padding_length_diff
    str_features_dict[
        "germ_source_str_padding_length_diff"
    ] = germ_source_str_padding_length_diff
    germ_mut_feature_dict = (
        allele_comparison_statistic_test_for_str_loci_features(
            sample_feature_list_dict, germ_read_index, mut_read_index
        )
    )
    (
        germ_allele_left_start_diff,
        germ_allele_right_end_diff,
        germ_allele_ks_start_stat,
        germ_allele_ks_start_p_value,
        germ_allele_ks_end_stat,
        germ_allele_ks_end_p_value,
        germ_allele_chi_square_start_stat,
        germ_allele_chi_square_start_p_value,
        germ_allele_chi_square_end_stat,
        germ_allele_chi_square_end_p_value,
        germ_allele_unique_start_coordinate_number,
        germ_allele_unique_end_coordinate_number,
    ) = cal_allele_shape_based_features(
        sample_feature_list_dict["overall_reference_start"],
        sample_feature_list_dict["overall_reference_end"],
        str_padding_start,
        str_padding_end_exclude,
        read_length,
        germ_str_padding_length,
        germ_read_index,
    )
    (
        mut_allele_left_start_diff,
        mut_allele_right_end_diff,
        mut_allele_ks_start_stat,
        mut_allele_ks_start_p_value,
        mut_allele_ks_end_stat,
        mut_allele_ks_end_p_value,
        mut_allele_chi_square_start_stat,
        mut_allele_chi_square_start_p_value,
        mut_allele_chi_square_end_stat,
        mut_allele_chi_square_end_p_value,
        mut_allele_unique_start_coordinate_number,
        mut_allele_unique_end_coordinate_number,
    ) = cal_allele_shape_based_features(
        sample_feature_list_dict["overall_reference_start"],
        sample_feature_list_dict["overall_reference_end"],
        str_padding_start,
        str_padding_end_exclude,
        read_length,
        mut_str_padding_length,
        mut_read_index,
    )
    # other hap assign features ok !
    if muttype == "hethet":
        germ_source_dict = (
            allele_comparison_statistic_test_for_str_loci_features(
                sample_feature_list_dict, germ_read_index, source_read_index
            )
        )
        germ_source_feature_dict = (
            transform_germ_mut_2_germ_source_feature_dict(germ_source_dict)
        )
        (
            overall_mean_base_accuracy,
            source_mut_diff_base_accuracy,
            source_mut_base_accuracy_statistic,
            source_mut_base_accuracy_pvalue,
        ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
            sample_feature_list_dict["mis_base_accuracy"],
            source_read_index,
            mut_read_index,
            "two-sided",
        )  # mut and source 可能反了对于 hethet
        (
            source_mean_baseq,
            source_fraction_baseq_more_cutoff,
            mut_mean_baseq,
            mut_fraction_baseq_more_cutoff,
        ) = cal_feature_mean_and_cutoff_fraction(
            sample_feature_list_dict["mis_base_accuracy"],
            source_read_index,
            mut_read_index,
            LOW_BASEQ_HIGH_CUTOFF,
        )
        (
            source_allele_left_start_diff,
            source_allele_right_end_diff,
            source_allele_ks_start_stat,
            source_allele_ks_start_p_value,
            source_allele_ks_end_stat,
            source_allele_ks_end_p_value,
            source_allele_chi_square_start_stat,
            source_allele_chi_square_start_p_value,
            source_allele_chi_square_end_stat,
            source_allele_chi_square_end_p_value,
            source_allele_unique_start_coordinate_number,
            source_allele_unique_end_coordinate_number,
        ) = cal_allele_shape_based_features(
            sample_feature_list_dict["overall_reference_start"],
            sample_feature_list_dict["overall_reference_end"],
            str_padding_start,
            str_padding_end_exclude,
            read_length,
            source_str_padding_length,
            source_read_index,
        )
    else:
        (
            source_allele_left_start_diff,
            source_allele_right_end_diff,
            source_allele_ks_start_stat,
            source_allele_ks_start_p_value,
            source_allele_ks_end_stat,
            source_allele_ks_end_p_value,
            source_allele_chi_square_start_stat,
            source_allele_chi_square_start_p_value,
            source_allele_chi_square_end_stat,
            source_allele_chi_square_end_p_value,
            source_allele_unique_start_coordinate_number,
            source_allele_unique_end_coordinate_number,
        ) = (
            germ_allele_left_start_diff,
            germ_allele_right_end_diff,
            germ_allele_ks_start_stat,
            germ_allele_ks_start_p_value,
            germ_allele_ks_end_stat,
            germ_allele_ks_end_p_value,
            germ_allele_chi_square_start_stat,
            germ_allele_chi_square_start_p_value,
            germ_allele_chi_square_end_stat,
            germ_allele_chi_square_end_p_value,
            germ_allele_unique_start_coordinate_number,
            germ_allele_unique_end_coordinate_number,
        )
        germ_source_feature_dict = {}
        if muttype == "germlinehom":
            (
                overall_mean_base_accuracy,
                source_mut_diff_base_accuracy,
                source_mut_base_accuracy_statistic,
                source_mut_base_accuracy_pvalue,
            ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
                sample_feature_list_dict["mis_base_accuracy"],
                germ_read_index,
                germ_read_index,
                "two-sided",
            )  # 双端因为 germ het 区别不了germ 和 mut,单端，假设 germ 更真, "greater"
            (
                source_mean_baseq,
                source_fraction_baseq_more_cutoff,
                mut_mean_baseq,
                mut_fraction_baseq_more_cutoff,
            ) = cal_feature_mean_and_cutoff_fraction(
                sample_feature_list_dict["mis_base_accuracy"],
                germ_read_index,
                germ_read_index,
                LOW_BASEQ_HIGH_CUTOFF,
            )
        else:
            (
                overall_mean_base_accuracy,
                source_mut_diff_base_accuracy,
                source_mut_base_accuracy_statistic,
                source_mut_base_accuracy_pvalue,
            ) = cal_feature_mean_diff_and_unpaired_nonparams_ranksum(
                sample_feature_list_dict["mis_base_accuracy"],
                germ_read_index,
                mut_read_index,
                "two-sided",
            )  # 双端因为 germ het 区别不了germ 和 mut,单端，假设 germ 更真, "greater"
            (
                source_mean_baseq,
                source_fraction_baseq_more_cutoff,
                mut_mean_baseq,
                mut_fraction_baseq_more_cutoff,
            ) = cal_feature_mean_and_cutoff_fraction(
                sample_feature_list_dict["mis_base_accuracy"],
                germ_read_index,
                mut_read_index,
                LOW_BASEQ_HIGH_CUTOFF,
            )
    # feature = np.array(sample_feature_list_dict["mis_base_accuracy"])
    # germ_index = source_read_index
    # mut_index = mut_read_index
    # feature_rm_NA = feature[np.where([i != "NA" for i in feature])[0]].astype(
    #     float
    # )  # XXX: HAS DEBUG THROUGH ADDING [0] because 2D array here
    # overall_mean_feature = np.mean(feature_rm_NA)
    # germ_feature = np.array(feature)[germ_index]
    # mut_feature = np.array(feature)[mut_index]
    # germ_feature = germ_feature[
    #     np.where([i != "NA" for i in germ_feature])[0]
    # ].astype(
    #     float
    # )  # XXX: string 2 float
    # mut_feature = mut_feature[
    #     np.where([i != "NA" for i in mut_feature])[0]
    # ].astype(
    #     float
    # )  # XXX array == "str"，如果 array 和 "str" 是同一种数据类型，则逐一与 "str" 扮演 element comparison，否则则只比较 array 是否等于 "str"
    # # BUG:
    # # >>> np.array([1,2,3])!="NA"
    # # True
    # # >>> np.array([1,2,3])!=1
    # # array([False,  True,  True])
    # # TODO:
    # # np.mean([])
    # # /Users/lid/opt/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3464: RuntimeWarning: Mean of empty slice.
    # # return _methods._mean(a, axis=axis, dtype=dtype,
    # # /Users/lid/opt/anaconda3/lib/python3.8/site-packages/numpy/core/_methods.py:192: RuntimeWarning: invalid value encountered in scalar divide
    # # ret = ret.dtype.type(ret / rcount)
    # # nan
    # germ_feature_mean = np.mean(germ_feature)
    # mut_feature_mean = np.mean(mut_feature)
    # germ_mut_diff_feature_mean = germ_feature_mean - mut_feature_mean
    # (
    #     germ_mut_feature_statistic,
    #     germ_mut_feature_pvalue,
    # ) = unpaired_nonparams_ranksum_test(germ_feature, mut_feature)

    # print(germ_feature, mut_feature)
    # print(germ_feature_mean, mut_feature_mean)
    # print(germ_mut_feature_statistic,germ_mut_feature_pvalue)
    str_features_dict[
        "overall_mean_base_accuracy"
    ] = overall_mean_base_accuracy
    str_features_dict[
        "source_mut_diff_base_accuracy"
    ] = source_mut_diff_base_accuracy
    str_features_dict[
        "source_mut_base_accuracy_statistic"
    ] = source_mut_base_accuracy_statistic
    str_features_dict[
        "source_mut_base_accuracy_pvalue"
    ] = source_mut_base_accuracy_pvalue
    str_features_dict[
        "germ_allele_left_start_diff"
    ] = germ_allele_left_start_diff
    str_features_dict[
        "germ_allele_right_end_diff"
    ] = germ_allele_right_end_diff
    str_features_dict["germ_allele_ks_start_stat"] = germ_allele_ks_start_stat
    str_features_dict[
        "germ_allele_ks_start_p_value"
    ] = germ_allele_ks_start_p_value
    str_features_dict["germ_allele_ks_end_stat"] = germ_allele_ks_end_stat
    str_features_dict[
        "germ_allele_ks_end_p_value"
    ] = germ_allele_ks_end_p_value
    str_features_dict[
        "germ_allele_chi_square_start_stat"
    ] = germ_allele_chi_square_start_stat
    str_features_dict[
        "germ_allele_chi_square_start_p_value"
    ] = germ_allele_chi_square_start_p_value
    str_features_dict[
        "germ_allele_chi_square_end_stat"
    ] = germ_allele_chi_square_end_stat
    str_features_dict[
        "germ_allele_chi_square_end_p_value"
    ] = germ_allele_chi_square_end_p_value
    str_features_dict[
        "mut_allele_left_start_diff"
    ] = mut_allele_left_start_diff
    str_features_dict["mut_allele_right_end_diff"] = mut_allele_right_end_diff
    str_features_dict["mut_allele_ks_start_stat"] = mut_allele_ks_start_stat
    str_features_dict[
        "mut_allele_ks_start_p_value"
    ] = mut_allele_ks_start_p_value
    str_features_dict["mut_allele_ks_end_stat"] = mut_allele_ks_end_stat
    str_features_dict["mut_allele_ks_end_p_value"] = mut_allele_ks_end_p_value
    str_features_dict[
        "mut_allele_chi_square_start_stat"
    ] = mut_allele_chi_square_start_stat
    str_features_dict[
        "mut_allele_chi_square_start_p_value"
    ] = mut_allele_chi_square_start_p_value
    str_features_dict[
        "mut_allele_chi_square_end_stat"
    ] = mut_allele_chi_square_end_stat
    str_features_dict[
        "mut_allele_chi_square_end_p_value"
    ] = mut_allele_chi_square_end_p_value
    str_features_dict[
        "source_allele_left_start_diff"
    ] = source_allele_left_start_diff
    str_features_dict[
        "source_allele_right_end_diff"
    ] = source_allele_right_end_diff
    str_features_dict[
        "source_allele_ks_start_stat"
    ] = source_allele_ks_start_stat
    str_features_dict[
        "source_allele_ks_start_p_value"
    ] = source_allele_ks_start_p_value
    str_features_dict["source_allele_ks_end_stat"] = source_allele_ks_end_stat
    str_features_dict[
        "source_allele_ks_end_p_value"
    ] = source_allele_ks_end_p_value
    str_features_dict[
        "source_allele_chi_square_start_stat"
    ] = source_allele_chi_square_start_stat
    str_features_dict[
        "source_allele_chi_square_start_p_value"
    ] = source_allele_chi_square_start_p_value
    str_features_dict[
        "source_allele_chi_square_end_stat"
    ] = source_allele_chi_square_end_stat
    str_features_dict[
        "source_allele_chi_square_end_p_value"
    ] = source_allele_chi_square_end_p_value
    str_features_dict["used_read_num_in_feature"] = sample_feature_list_dict[
        "used_read_num_in_feature"
    ]
    str_features_dict[
        "used_read_num_in_genotyping"
    ] = sample_feature_list_dict["used_read_num_in_genotyping"]
    str_features_dict["sample_str_used_depth"] = sample_feature_list_dict[
        "sample_str_used_depth"
    ]
    str_features_dict["sample_wgs_used_depth"] = sample_feature_list_dict[
        "sample_wgs_used_depth"
    ]
    str_features_dict["depth_norm_factor"] = sample_feature_list_dict[
        "depth_norm_factor"
    ]
    str_features_dict["expected_str_used_depth"] = sample_feature_list_dict[
        "expected_str_used_depth"
    ]
    str_features_dict[
        "expected_str_used_depth_ratio"
    ] = sample_feature_list_dict["expected_str_used_depth_ratio"]
    str_features_dict[
        "raw_wgs_expected_depth_ratio"
    ] = sample_feature_list_dict["raw_wgs_expected_depth_ratio"]
    str_features_dict[
        "nearby_raw_expected_depth_ratio"
    ] = sample_feature_list_dict["nearby_raw_expected_depth_ratio"]
    str_features_dict[
        "raw_wgs_point_expected_depth_ratio"
    ] = sample_feature_list_dict["raw_wgs_point_expected_depth_ratio"]
    str_features_dict[
        "final_padding_str_chrom_depth_percentile"
    ] = sample_feature_list_dict["final_padding_str_chrom_depth_percentile"]
    str_features_dict["source_mean_baseq"] = source_mean_baseq
    str_features_dict[
        "source_fraction_baseq_more_cutoff"
    ] = source_fraction_baseq_more_cutoff
    str_features_dict["mut_mean_baseq"] = mut_mean_baseq
    str_features_dict[
        "mut_fraction_baseq_more_cutoff"
    ] = mut_fraction_baseq_more_cutoff
    str_features_dict[
        "germ_allele_unique_start_coordinate_number"
    ] = germ_allele_unique_start_coordinate_number
    str_features_dict[
        "germ_allele_unique_end_coordinate_number"
    ] = germ_allele_unique_end_coordinate_number
    str_features_dict[
        "mut_allele_unique_start_coordinate_number"
    ] = mut_allele_unique_start_coordinate_number
    str_features_dict[
        "mut_allele_unique_end_coordinate_number"
    ] = mut_allele_unique_end_coordinate_number
    str_features_dict[
        "source_allele_unique_start_coordinate_number"
    ] = source_allele_unique_start_coordinate_number
    str_features_dict[
        "source_allele_unique_end_coordinate_number"
    ] = source_allele_unique_end_coordinate_number

    return str_features_dict, germ_mut_feature_dict, germ_source_feature_dict


def other_phasing_related_features(
    all_reads_nearbysnp_dict,
    muttype,
    sample_feature_list_dict,
    germ_snp1,
    germ_snp2,
    mosaic_snp,
):
    # Two loci assign: in mosaic calling
    # Single locus assign: here use
    other_phasing_related_features_dict = {}
    nearby_snp_seq_list = []
    for query_name in sample_feature_list_dict["overall_query_name"]:
        nearbysnp_list = all_reads_nearbysnp_dict.get(query_name, [])
        if len(nearbysnp_list) > 0:
            nearby_snp_seq_list.append(nearbysnp_list[0])
        else:
            nearby_snp_seq_list.append("NA")
    germ_read_index = np.where(
        np.array(sample_feature_list_dict["read_hap_assignment"]) == "germ"
    )[0]
    source_read_index = np.where(
        np.array(sample_feature_list_dict["read_hap_assignment"]) == "source"
    )[0]
    mut_read_index = np.where(
        np.array(sample_feature_list_dict["read_hap_assignment"]) == "mut"
    )[0]
    nearby_snp_seq_list_rm_NA = np.array(nearby_snp_seq_list)[
        np.where(np.array(nearby_snp_seq_list) != "NA")[0]
    ]
    phasing_fraction = len(nearby_snp_seq_list_rm_NA) / len(
        nearby_snp_seq_list
    )
    germ_nearby_snp_seq = np.array(nearby_snp_seq_list)[germ_read_index]
    mut_nearby_snp_seq = np.array(nearby_snp_seq_list)[mut_read_index]
    source_nearby_snp_seq = np.array(nearby_snp_seq_list)[source_read_index]
    germ_nearby_snp_seq_rm_NA = germ_nearby_snp_seq[
        np.where(germ_nearby_snp_seq != "NA")[0]
    ]
    mut_nearby_snp_seq_rm_NA = mut_nearby_snp_seq[
        np.where(mut_nearby_snp_seq != "NA")[0]
    ]
    if len(source_nearby_snp_seq) > 0:  # XXX: != np.array([])
        source_nearby_snp_seq_rm_NA = source_nearby_snp_seq[
            np.where(source_nearby_snp_seq != "NA")[0]
        ]
    else:
        source_nearby_snp_seq_rm_NA = []  # XXX: For homhet, source don't exist
    if muttype == "homhet":
        germ_phasing_fraction = len(germ_nearby_snp_seq_rm_NA) / len(
            germ_nearby_snp_seq
        )
        source_phasing_fraction = germ_phasing_fraction
        mut_phasing_fraction = len(mut_nearby_snp_seq_rm_NA) / len(
            mut_nearby_snp_seq
        )
        germ_hsnp_fraction = np.sum(
            germ_nearby_snp_seq_rm_NA == germ_snp1
        ) / len(germ_nearby_snp_seq_rm_NA)
        source_hsnp_fraction = np.sum(
            germ_nearby_snp_seq_rm_NA == germ_snp2
        ) / len(germ_nearby_snp_seq_rm_NA)
        germ_dis_fraction = 1 - germ_hsnp_fraction - source_hsnp_fraction
        source_dis_fraction = germ_dis_fraction
        if len(mut_nearby_snp_seq_rm_NA) == 0:
            mut_hsnp_fraction = np.nan
            mut_dis_fraction = (
                np.nan
            )  # BUG: RuntimeWarning: invalid value encountered in long_scalars
            all_dis_fraction = 1 - (
                (
                    np.sum(germ_nearby_snp_seq_rm_NA == germ_snp1)
                    + np.sum(germ_nearby_snp_seq_rm_NA == germ_snp2)
                )
                / (
                    len(germ_nearby_snp_seq_rm_NA)
                    + len(mut_nearby_snp_seq_rm_NA)
                )
            )
        else:
            mut_hsnp_fraction = np.sum(
                mut_nearby_snp_seq_rm_NA == mosaic_snp
            ) / len(mut_nearby_snp_seq_rm_NA)
            mut_dis_fraction = 1 - mut_hsnp_fraction
            all_dis_fraction = 1 - (
                (
                    np.sum(germ_nearby_snp_seq_rm_NA == germ_snp1)
                    + np.sum(germ_nearby_snp_seq_rm_NA == germ_snp2)
                    + np.sum(mut_nearby_snp_seq_rm_NA == mosaic_snp)
                )
                / (
                    len(germ_nearby_snp_seq_rm_NA)
                    + len(mut_nearby_snp_seq_rm_NA)
                )
            )
    elif muttype == "hethet":
        germ_phasing_fraction = len(germ_nearby_snp_seq_rm_NA) / len(
            germ_nearby_snp_seq
        )
        mut_phasing_fraction = len(mut_nearby_snp_seq_rm_NA) / len(
            mut_nearby_snp_seq
        )
        source_phasing_fraction = len(source_nearby_snp_seq_rm_NA) / len(
            source_nearby_snp_seq
        )
        if len(germ_nearby_snp_seq_rm_NA) > 0:
            germ_hsnp_fraction = np.sum(
                germ_nearby_snp_seq_rm_NA == germ_snp1
            ) / len(germ_nearby_snp_seq_rm_NA)
            germ_dis_fraction = 1 - germ_hsnp_fraction
        else:
            germ_hsnp_fraction = np.nan
            germ_dis_fraction = np.nan
        if len(mut_nearby_snp_seq_rm_NA) > 0:
            mut_hsnp_fraction = np.sum(
                mut_nearby_snp_seq_rm_NA == mosaic_snp
            ) / len(mut_nearby_snp_seq_rm_NA)
            mut_dis_fraction = 1 - mut_hsnp_fraction
        else:
            mut_hsnp_fraction = np.nan
            mut_dis_fraction = np.nan
        if len(source_nearby_snp_seq_rm_NA) > 0:
            source_hsnp_fraction = np.sum(
                source_nearby_snp_seq_rm_NA == germ_snp2
            ) / len(source_nearby_snp_seq_rm_NA)
            source_dis_fraction = 1 - source_hsnp_fraction
        else:
            source_hsnp_fraction = np.nan
            source_dis_fraction = np.nan

        all_dis_fraction = 1 - (
            (
                np.sum(germ_nearby_snp_seq_rm_NA == germ_snp1)
                + np.sum(source_nearby_snp_seq_rm_NA == germ_snp2)
                + np.sum(mut_nearby_snp_seq_rm_NA == mosaic_snp)
            )
            / (
                len(germ_nearby_snp_seq_rm_NA)
                + len(source_nearby_snp_seq_rm_NA)
                + len(mut_nearby_snp_seq_rm_NA)
            )
        )
        # all_dis_fraction = (
        #     germ_dis_fraction + mut_dis_fraction + source_dis_fraction
        # )
    elif muttype == "germlinehet":
        # XXX: For clonal mutation with impurity clonal, these features have some issues, Need DEBUG When use these features in Tumor/Clonal samples # TODO
        germ_phasing_fraction = len(germ_nearby_snp_seq_rm_NA) / len(
            germ_nearby_snp_seq
        )
        source_phasing_fraction = germ_phasing_fraction
        mut_phasing_fraction = len(mut_nearby_snp_seq_rm_NA) / len(
            mut_nearby_snp_seq
        )
        germ_hsnp_fraction = np.sum(
            germ_nearby_snp_seq_rm_NA == germ_snp1
        ) / len(germ_nearby_snp_seq_rm_NA)
        source_hsnp_fraction = germ_hsnp_fraction
        germ_dis_fraction = 1 - germ_hsnp_fraction
        source_dis_fraction = germ_dis_fraction
        if len(mut_nearby_snp_seq_rm_NA) == 0:
            mut_hsnp_fraction = 0
            mut_dis_fraction = 0  # BUG: RuntimeWarning: invalid value encountered in long_scalars
            all_dis_fraction = 1 - (
                (
                    np.sum(germ_nearby_snp_seq_rm_NA == germ_snp1)
                    + np.sum(mut_nearby_snp_seq_rm_NA == mosaic_snp)
                )
                / (
                    len(germ_nearby_snp_seq_rm_NA)
                    + len(mut_nearby_snp_seq_rm_NA)
                )
            )
        else:
            mut_hsnp_fraction = np.sum(
                mut_nearby_snp_seq_rm_NA == mosaic_snp
            ) / len(mut_nearby_snp_seq_rm_NA)
            mut_dis_fraction = 1 - mut_hsnp_fraction
            all_dis_fraction = 1 - (
                (
                    np.sum(germ_nearby_snp_seq_rm_NA == germ_snp1)
                    + np.sum(mut_nearby_snp_seq_rm_NA == mosaic_snp)
                )
                / (
                    len(germ_nearby_snp_seq_rm_NA)
                    + len(mut_nearby_snp_seq_rm_NA)
                )
            )
        # all_dis_fraction = germ_dis_fraction + mut_dis_fraction
    elif muttype == "germlinehom":
        # UnboundLocalError: local variable 'germ_phasing_fraction' referenced before assignment
        germ_phasing_fraction = len(germ_nearby_snp_seq_rm_NA) / len(
            germ_nearby_snp_seq
        )
        source_phasing_fraction = germ_phasing_fraction
        mut_phasing_fraction = germ_phasing_fraction
        germ_hsnp_fraction = np.sum(
            germ_nearby_snp_seq_rm_NA == germ_snp1
        ) / len(germ_nearby_snp_seq_rm_NA)
        source_hsnp_fraction = np.sum(
            germ_nearby_snp_seq_rm_NA == germ_snp2
        ) / len(germ_nearby_snp_seq_rm_NA)
        germ_dis_fraction = 1 - germ_hsnp_fraction - source_hsnp_fraction
        source_dis_fraction = germ_dis_fraction
        mut_hsnp_fraction = 0
        mut_dis_fraction = 0
        all_dis_fraction = germ_dis_fraction
        # if len(mut_nearby_snp_seq_rm_NA) == 0:
        #     mut_hsnp_fraction = 0
        #     mut_dis_fraction = 0  # BUG: RuntimeWarning: invalid value encountered in long_scalars
        # else:
        #     mut_hsnp_fraction = np.sum(
        #         mut_nearby_snp_seq_rm_NA == mosaic_snp
        #     ) / len(mut_nearby_snp_seq_rm_NA)
        #     mut_dis_fraction = 1 - mut_hsnp_fraction
        # all_dis_fraction = germ_dis_fraction + mut_dis_fraction
    other_phasing_related_features_dict["phasing_fraction"] = phasing_fraction
    other_phasing_related_features_dict[
        "germ_phasing_fraction"
    ] = germ_phasing_fraction
    other_phasing_related_features_dict[
        "mut_phasing_fraction"
    ] = mut_phasing_fraction
    other_phasing_related_features_dict[
        "source_phasing_fraction"
    ] = source_phasing_fraction
    other_phasing_related_features_dict[
        "germ_hsnp_fraction"
    ] = germ_hsnp_fraction
    other_phasing_related_features_dict[
        "germ_dis_fraction"
    ] = germ_dis_fraction
    other_phasing_related_features_dict[
        "mut_hsnp_fraction"
    ] = mut_hsnp_fraction
    other_phasing_related_features_dict["mut_dis_fraction"] = mut_dis_fraction
    other_phasing_related_features_dict[
        "source_hsnp_fraction"
    ] = source_hsnp_fraction
    other_phasing_related_features_dict[
        "source_dis_fraction"
    ] = source_dis_fraction
    other_phasing_related_features_dict["all_dis_fraction"] = all_dis_fraction
    return other_phasing_related_features_dict


def get_features_from_vcf(vcf_record, sample_name):
    vcf_based_features = {}
    sample_var = vcf_record.samples[sample_name]
    chrom = vcf_record.chrom
    STRSTART = vcf_record.info["STRSTART"]
    STREND = vcf_record.info["STREND"]
    MOTIF = vcf_record.info["MOTIF"]
    MOTIF_length = len(MOTIF)
    PERIOD = vcf_record.info["PERIOD"]
    str_id = vcf_record.info["STRID"]
    vcf_based_features["chrom"] = chrom
    vcf_based_features["STRSTART"] = STRSTART
    vcf_based_features["STREND"] = STREND
    vcf_based_features["MOTIF_length"] = MOTIF_length
    vcf_based_features["PERIOD"] = PERIOD
    vcf_based_features["str_id"] = str_id
    vcf_based_features["MOTIF"] = MOTIF
    if sample_var["GT"][0] == None:
        GT = "./."
    else:
        GT = str(sample_var["GT"][0]) + "/" + str(sample_var["GT"][1])
    if sample_var["MGT"][0] == ".":
        MGT = "./."
    else:
        MGT = (
            sample_var["MGT"][0].split("/")[0]
            + "/"
            + sample_var["MGT"][0].split("/")[1]
        )
    vcf_based_features["GT"] = GT
    vcf_based_features["MGT"] = MGT
    MP = sample_var["MP"]
    CMP = sample_var["CMP"]
    variant_type = sample_var["MUTP"]
    frame = sample_var["FRAME"]
    mosaic_fraction = sample_var["MF"]
    # HACK: A temp solution for previous version mosaic calling
    try:
        MF_hom2het_het2het = sample_var["RMF"]
    except:
        MF_hom2het_het2het = "NA"
    perfect_ref_str = sample_var["PSTR"][0]
    depth = sample_var["DP"]
    filtered_depth = sample_var["FDP"]
    filtered_depth_category = sample_var["FDPC"]
    mosaic_gt_observed_af = sample_var["EAF"]
    NSTUTTER = sample_var["NSTUTTER"]
    DSTUTTER = sample_var["DSTUTTER"]
    NFSTUTTER = sample_var["NFSTUTTER"]
    DFSTUTTER = sample_var["DFSTUTTER"]
    HVAF = sample_var["HVAF"][0]
    if HVAF == ".":
        HVAF = "NA"
    elif float(HVAF) <= 0.5:
        HVAF = 1 - float(HVAF)
    else:
        HVAF = float(HVAF)
    NALLELES_PER_STR_BP = int(sample_var["NALLELES"]) / (
        int(STREND) - int(STRSTART) + PADDING_BPS * 2
    )
    MLEUAF = sample_var["MLEUAF"]
    MLEPUAF = sample_var["MLEPUAF"]
    EMLEAF = sample_var["EMLEAF"]
    PS = sample_var["PS"]
    PP = sample_var["PP"]
    PEAF = sample_var["PEAF"]
    MBP = sample_var["MBP"]
    PHA = sample_var["PHA"]
    PPP = sample_var["PPP"]
    PODR = sample_var["PODR"]
    PODHR = sample_var["PODHR"]
    PMAP = sample_var["PMAP"]
    PMLEEH = sample_var["PMLEEH"]
    PMLEEHR = sample_var["PMLEEHR"]
    PMLEDR = sample_var["PMLEDR"]
    PFAH = sample_var["PFAH"]
    MAPLR = sample_var["MAPLR"]
    MAPLRTP = sample_var["MAPLRTP"]
    MAPLRS = sample_var["MAPLRS"]
    MAPLRTPS = sample_var["MAPLRTPS"]
    GIOAF = sample_var["GIOAF"]
    GIMLEAF = sample_var["GIMLEAF"]
    GIUAF = sample_var["GIUAF"]
    PRHN = sample_var["PRHN"]
    MLEPEAF = sample_var["MLEPEAF"]
    PMB = sample_var["PMB"]
    POB = sample_var["POB"]
    PGHAF = sample_var["PGHAF"]
    PMHAF = sample_var["PMHAF"]
    PH3AF = sample_var["PH3AF"]
    PGD = sample_var["PGD"]
    PSD = sample_var["PSD"]
    PMD = sample_var["PMD"]
    POMDR = sample_var["POMDR"]
    PSOMDR = sample_var["PSOMDR"]
    PGOMDR = sample_var["PGOMDR"]
    PSMMDR = sample_var["PSMMDR"]
    PGMMDR = sample_var["PGMMDR"]
    PMMDR = sample_var["PMMDR"]
    NHSNPN = sample_var["NHSNPN"]
    NHSNPINDELN = sample_var["NHSNPINDELN"]
    GQ = sample_var["GQ"]
    if GQ == ".":
        GQ_OR = "NA"
        GQ_PVAL = "NA"
    else:
        GQ_OR, GQ_PVAL = GQ[0], GQ[1]
    GG = sample_var["GG"]
    if GG == ".":
        GG_OR = "NA"
        GG_PVAL = "NA"
    else:
        GG_OR, GG_PVAL = GG[0], GG[1]
    GI = sample_var["GI"]
    GIP = sample_var["GIP"]
    GIQ = sample_var["GIQ"]
    GIQP = sample_var["GIQP"]
    AMP = sample_var["AMP"]
    ALLELE_BARCODE_UMI = sample_var["ALLELE_BARCODE_UMI"][0]
    # print(ALLELE_BARCODE_UMI)
    ALLELE_BARCODE_MOSAIC = sample_var["ALLELE_BARCODE_MOSAIC"][0]
    # 4 个 VAF：观察和 MLE，phase 和 unphase #
    phased_germline_GT = sample_var["PGT"][0].split("|")
    phased_mosaic_GT = sample_var["PMGT"][0].split("|")
    mosaic_index = sample_var["MGT"][0].split("/")
    germ_index = list(sample_var["GT"])
    if mosaic_gt_observed_af[0] == ".":
        observed_mosaic_allele_vaf_single_locus = "NA"
    else:
        if len(set(germ_index)) == 2 and len(set(mosaic_index)) == 1:
            observed_mosaic_allele_vaf_single_locus = mosaic_gt_observed_af[1]
        else:
            observed_mosaic_allele_vaf_single_locus = mosaic_gt_observed_af[3]
    if phased_mosaic_GT[0] != ".":
        phase_all_alleles = [
            int(phased_germline_GT[0]),
            int(phased_germline_GT[1]),
            int(phased_mosaic_GT[0]),
            int(phased_mosaic_GT[1]),
        ]
        unphase_all_alleles = [
            int(germ_index[0]),
            int(germ_index[1]),
            int(mosaic_index[0]),
            int(mosaic_index[1]),
        ]
        MLEPEAD = sample_var["MLEPEAD"]
        PDP = sample_var["PDP"]
        (
            mle_mosaic_allele_vaf_two_loci,
            observed_mosaic_allele_vaf_two_loci,
        ) = get_mle_and_observed_mosaic_allele_frequency_two_loci(
            MLEPEAD, PDP, phase_all_alleles, unphase_all_alleles, PEAF
        )
    else:
        mle_mosaic_allele_vaf_two_loci = "NA"
        observed_mosaic_allele_vaf_two_loci = "NA"
        PDP = "NA"
        phase_all_alleles = [
            "NA",
            "NA",
            "NA",
            "NA",
        ]
    vcf_based_features["MP"] = MP
    vcf_based_features["CMP"] = CMP
    vcf_based_features["variant_type"] = variant_type
    vcf_based_features["frame"] = frame
    vcf_based_features["mosaic_fraction"] = mosaic_fraction
    vcf_based_features["MF_hom2het_het2het"] = MF_hom2het_het2het
    vcf_based_features["perfect_ref_str"] = perfect_ref_str
    vcf_based_features["depth"] = depth
    vcf_based_features["filtered_depth"] = filtered_depth
    if filtered_depth_category == ".":
        vcf_based_features["SegmentConditionFail"] = "NA"
        vcf_based_features["SegmentResultFail"] = "NA"
        vcf_based_features["ReadN"] = "NA"
        vcf_based_features["library_issue"] = "NA"
        vcf_based_features["map_issue"] = "NA"
        vcf_based_features["spanning_issue"] = "NA"
        vcf_based_features["unmap_issue"] = "NA"
        vcf_based_features["duplicate_same"] = "NA"
        vcf_based_features["duplicate_diff"] = "NA"
        vcf_based_features["read_pair_same"] = "NA"
        vcf_based_features["read_pair_diff"] = "NA"
        vcf_based_features["duplicate_loose_same"] = "NA"
        vcf_based_features["duplicate_loose_diff"] = "NA"
        vcf_based_features["duplicate_complex_1"] = "NA"
        vcf_based_features["duplicate_complex_2"] = "NA"
        vcf_based_features["duplicate_complex_3"] = "NA"
        vcf_based_features["duplicate_complex_4"] = "NA"
    else:
        vcf_based_features["SegmentConditionFail"] = filtered_depth_category[0]
        vcf_based_features["SegmentResultFail"] = filtered_depth_category[1]
        vcf_based_features["ReadN"] = filtered_depth_category[2]
        vcf_based_features["library_issue"] = filtered_depth_category[3]
        vcf_based_features["map_issue"] = filtered_depth_category[4]
        vcf_based_features["spanning_issue"] = filtered_depth_category[5]
        vcf_based_features["unmap_issue"] = filtered_depth_category[6]
        vcf_based_features["duplicate_same"] = filtered_depth_category[7]
        vcf_based_features["duplicate_diff"] = filtered_depth_category[8]
        vcf_based_features["read_pair_same"] = filtered_depth_category[9]
        vcf_based_features["read_pair_diff"] = filtered_depth_category[10]
        vcf_based_features["duplicate_loose_same"] = filtered_depth_category[11]
        vcf_based_features["duplicate_loose_diff"] = filtered_depth_category[12]
        vcf_based_features["duplicate_complex_1"] = filtered_depth_category[13]
        vcf_based_features["duplicate_complex_2"] = filtered_depth_category[14]
        vcf_based_features["duplicate_complex_3"] = filtered_depth_category[15]
        vcf_based_features["duplicate_complex_4"] = filtered_depth_category[16]
    if mosaic_gt_observed_af[0] == ".":
        vcf_based_features["EAF"] = "NA,NA,NA,NA"
    else:
        vcf_based_features["EAF"] = (
            str(mosaic_gt_observed_af[0])
            + ","
            + str(mosaic_gt_observed_af[1])
            + ","
            + str(mosaic_gt_observed_af[2])
            + ","
            + str(mosaic_gt_observed_af[3])
        )
    vcf_based_features[
        "observed_mosaic_allele_vaf_single_locus"
    ] = observed_mosaic_allele_vaf_single_locus
    vcf_based_features["NSTUTTER"] = NSTUTTER
    vcf_based_features["DSTUTTER"] = DSTUTTER
    vcf_based_features["NFSTUTTER"] = NFSTUTTER
    vcf_based_features["DFSTUTTER"] = DFSTUTTER
    vcf_based_features["HVAF"] = HVAF
    vcf_based_features["NALLELES_PER_STR_BP"] = NALLELES_PER_STR_BP
    vcf_based_features["MLEUAF"] = MLEUAF
    vcf_based_features["MLEPUAF"] = MLEPUAF
    vcf_based_features["PS"] = PS
    vcf_based_features["PP"] = PP
    if PEAF[0] == ".":
        vcf_based_features["PEAF"] = "NA,NA,NA,NA"
    else:
        vcf_based_features["PEAF"] = (
            str(PEAF[0])
            + ","
            + str(PEAF[1])
            + ","
            + str(PEAF[2])
            + ","
            + str(PEAF[3])
        )
    if MLEPEAF == ".":
        vcf_based_features["MLEPEAF"] = "NA,NA,NA,NA"
    else:
        vcf_based_features["MLEPEAF"] = (
            str(MLEPEAF[0])
            + ","
            + str(MLEPEAF[1])
            + ","
            + str(MLEPEAF[2])
            + ","
            + str(MLEPEAF[3])
        )
    if EMLEAF == ".":
        vcf_based_features["EMLEAF"] = "NA,NA,NA,NA"
    else:
        vcf_based_features["EMLEAF"] = (
            str(EMLEAF[0])
            + ","
            + str(EMLEAF[1])
            + ","
            + str(EMLEAF[2])
            + ","
            + str(EMLEAF[3])
        )
    vcf_based_features["MBP"] = MBP
    vcf_based_features["PHA"] = PHA
    vcf_based_features["PPP"] = PPP
    vcf_based_features["PODR"] = PODR
    vcf_based_features["PODHR"] = PODHR
    vcf_based_features["PMAP"] = PMAP
    vcf_based_features["PMLEEH"] = PMLEEH
    vcf_based_features["PMLEEHR"] = PMLEEHR
    vcf_based_features["PMLEDR"] = PMLEDR
    vcf_based_features["PFAH"] = PFAH
    vcf_based_features["MAPLR"] = MAPLR
    vcf_based_features["MAPLRTP"] = MAPLRTP
    vcf_based_features["MAPLRS"] = MAPLRS
    vcf_based_features["MAPLRTPS"] = MAPLRTPS
    vcf_based_features["POMDR"] = POMDR
    vcf_based_features["PSOMDR"] = PSOMDR
    vcf_based_features["PGOMDR"] = PGOMDR
    vcf_based_features["PSMMDR"] = PSMMDR
    vcf_based_features["PGMMDR"] = PGMMDR
    vcf_based_features["PMMDR"] = PMMDR
    vcf_based_features["NHSNPN"] = NHSNPN
    vcf_based_features["NHSNPINDELN"] = NHSNPINDELN
    vcf_based_features[
        "mle_mosaic_allele_vaf_two_loci"
    ] = mle_mosaic_allele_vaf_two_loci
    vcf_based_features[
        "observed_mosaic_allele_vaf_two_loci"
    ] = observed_mosaic_allele_vaf_two_loci
    if GIOAF == ".":
        vcf_based_features["GIOAF0"] = "NA"
        vcf_based_features["GIOAF1"] = "NA"
    else:
        vcf_based_features["GIOAF0"] = GIOAF[0]
        vcf_based_features["GIOAF1"] = GIOAF[1]
    if GIMLEAF == ".":
        vcf_based_features["GIMLEAF0"] = "NA"
        vcf_based_features["GIMLEAF1"] = "NA"
    else:
        vcf_based_features["GIMLEAF0"] = GIMLEAF[0]
        vcf_based_features["GIMLEAF1"] = GIMLEAF[1]
    if GI[0] == ".":
        vcf_based_features["GI"] = "./."
    else:
        vcf_based_features["GI"] = GI[0]
    vcf_based_features["GIUAF"] = GIUAF
    vcf_based_features["PRHN"] = PRHN
    vcf_based_features["PMB"] = PMB
    vcf_based_features["POB"] = POB
    vcf_based_features["PGHAF"] = PGHAF
    vcf_based_features["PMHAF"] = PMHAF
    vcf_based_features["PH3AF"] = PH3AF
    vcf_based_features["PGD"] = PGD
    vcf_based_features["PSD"] = PSD
    vcf_based_features["PMD"] = PMD
    vcf_based_features["GQ_OR"] = GQ_OR
    vcf_based_features["GQ_PVAL"] = GQ_PVAL
    vcf_based_features["GG_OR"] = GG_OR
    vcf_based_features["GG_PVAL"] = GG_PVAL
    # vcf_based_features["GI"] = GI
    vcf_based_features["GIP"] = GIP
    vcf_based_features["GIQ"] = GIQ
    vcf_based_features["GIQP"] = GIQP
    vcf_based_features["AMP"] = AMP
    vcf_based_features["ALLSINGLE"] = sample_var["ALLSINGLE"]
    vcf_based_features["HPRHN"] = sample_var["HPRHN"]
    vcf_based_features["obs_phase_state"] = sample_var["obs_phase_state"]
    vcf_based_features["obs_hap_state"] = sample_var["obs_hap_state"]
    vcf_based_features["obs_hap_count"] = sample_var["obs_hap_count"]
    vcf_based_features["mle_phase_state"] = sample_var["mle_phase_state"]
    vcf_based_features["mle_hap_state"] = sample_var["mle_hap_state"]
    vcf_based_features["mle_hap_count"] = sample_var["mle_hap_count"]
    vcf_based_features["NORMGERM1"] = sample_var["NORMGERM1"]
    vcf_based_features["NORMGERM2"] = sample_var["NORMGERM2"]
    vcf_based_features["NORMMOSAIC"] = sample_var["NORMMOSAIC"]
    vcf_based_features["NORMSECOND"] = sample_var["NORMSECOND"]
    vcf_based_features["SECISGERM"] = sample_var["SECISGERM"]
    vcf_based_features["obs_mut_order"] = sample_var["obs_mut_order"]
    vcf_based_features["obs_depth_string"] = sample_var["obs_depth_string"]
    vcf_based_features["mle_mut_order"] = sample_var["mle_mut_order"]
    vcf_based_features["mle_depth_string"] = sample_var["mle_depth_string"]
    vcf_based_features["PDP"] = PDP
    vcf_based_features["PGT0"] = phase_all_alleles[0]
    vcf_based_features["PGT1"] = phase_all_alleles[1]
    vcf_based_features["PMGT0"] = phase_all_alleles[2]
    vcf_based_features["PMGT1"] = phase_all_alleles[3]
    vcf_based_features["ALLELE_BARCODE_UMI"] = ALLELE_BARCODE_UMI
    vcf_based_features["ALLELE_BARCODE_MOSAIC"] = ALLELE_BARCODE_MOSAIC

    return vcf_based_features


def get_mle_mosaic_allele_frequency_single_locus(sample_feature_list_dict):
    used_read_index = np.where(
        np.array(sample_feature_list_dict["overall_feature_use_read"]) == 1
    )[0]
    mut_read_index = np.where(
        np.array(sample_feature_list_dict["read_hap_assignment"]) == "mut"
    )[0]
    used_mut_dp = len(np.intersect1d(used_read_index, mut_read_index))
    mle_mosaic_allele_vaf_single_locus = used_mut_dp / len(used_read_index)
    return mle_mosaic_allele_vaf_single_locus


def get_mle_and_observed_mosaic_allele_frequency_two_loci(
    MLEPEAD, PDP, phase_all_alleles, unphase_all_alleles, PEAF
):
    MLEPEAD = [int(float(i)) for i in list(MLEPEAD)]
    if phase_all_alleles == unphase_all_alleles:
        if phase_all_alleles[0] == phase_all_alleles[1]:
            mle_mosaic_allele_vaf_two_loci = int(MLEPEAD[3]) / int(PDP)
            observed_mosaic_allele_vaf_two_loci = PEAF[3]
        else:
            if phase_all_alleles[2] == phase_all_alleles[3]:
                mle_mosaic_allele_vaf_two_loci = int(MLEPEAD[1]) / int(PDP)
                observed_mosaic_allele_vaf_two_loci = PEAF[1]
            else:
                mle_mosaic_allele_vaf_two_loci = int(MLEPEAD[3]) / int(PDP)
                observed_mosaic_allele_vaf_two_loci = PEAF[3]
    else:
        if phase_all_alleles[0] == phase_all_alleles[1]:
            mle_mosaic_allele_vaf_two_loci = int(MLEPEAD[2]) / int(PDP)
            observed_mosaic_allele_vaf_two_loci = PEAF[2]
        else:
            if phase_all_alleles[2] == phase_all_alleles[3]:
                mle_mosaic_allele_vaf_two_loci = int(MLEPEAD[0]) / int(PDP)
                observed_mosaic_allele_vaf_two_loci = PEAF[0]
            else:
                mle_mosaic_allele_vaf_two_loci = int(MLEPEAD[2]) / int(PDP)
                observed_mosaic_allele_vaf_two_loci = PEAF[2]
    return mle_mosaic_allele_vaf_two_loci, observed_mosaic_allele_vaf_two_loci


def safe_convert_to_int(value):
    try:
        # 首先转换为浮点数
        float_value = float(value)
        # 然后转换为整数
        int_value = int(float_value)
        return int_value
    except ValueError:
        print(f"无法将 '{value}' 转换为整数")
        return None


def get_average_mappability_score(mappability_score_array):
    if "NA" in list(mappability_score_array):
        average_mappability_score = "NA"
    else:
        mappability_score_array = mappability_score_array[
            ~np.isnan(mappability_score_array)
        ]
        if len(mappability_score_array) == 0:
            average_mappability_score = "NA"
        else:
            average_mappability_score = sum(mappability_score_array) / len(
                mappability_score_array
            )
    return average_mappability_score


def main_per_locus_feature_extract(parse_params, pysam_bed_row):
    pysam_bed_row_list = pysam_bed_row.split("\t")
    if len(pysam_bed_row_list) == len(HARD_FILTER_COLUMNS):
        pysam_bed_row_dict = dict(zip(HARD_FILTER_COLUMNS, pysam_bed_row_list))
    else:
        pysam_bed_row_dict = {}
    pysam_bed_row_string = ",".join(pysam_bed_row_list)
    fasta_file = parse_params["fasta_file"]
    pysam_fasta = pysam.FastaFile(fasta_file)
    bam_name = parse_params["bam_name"]
    add_chr_character = parse_params["add_chr_character"]
    remove_chr_character = parse_params["remove_chr_character"]
    bam_files = parse_params["bam_files"]
    wgs_mean_depth = parse_params["wgs_mean_depth"]
    str_used_mean_depth = parse_params["str_used_mean_depth"]
    all_samples_median_depth_dict = parse_params[
        "all_samples_median_depth_dict"
    ]
    all_samples_point_depth_dict = parse_params["all_samples_point_depth_dict"]
    all_samples_all_chroms_point_depth_dict = parse_params[
        "all_samples_all_chroms_point_depth_dict"
    ]
    wgs_mean_depth_dict = {}
    str_used_mean_depth_dict = {}
    pysam_bam = []
    for idx, bname in enumerate(bam_name):
        wgs_mean_depth_dict[bname] = wgs_mean_depth[idx]
        str_used_mean_depth_dict[bname] = str_used_mean_depth[idx]
    for bam in bam_files:
        if "bam" in bam:
            bamfile = pysam.AlignmentFile(bam, "rb")
        else:
            bamfile = pysam.AlignmentFile(
                bam, "rc", reference_filename=fasta_file
            )
        pysam_bam.append(bamfile)
    sex = parse_params["sex"]
    stutter_file = parse_params["stutter_file"]
    mappability_file = parse_params["mappability_file"]
    # HACK: mappability k24 and k100 are tried in the same time
    if BED_MAPPABILITY:
        py_mappability = pysam.TabixFile(mappability_file)
    else:
        mappability_prefix = "/".join(mappability_file.split("/")[:-1])
        mappability_k24_file = mappability_prefix + "/k24.umap.wg.bw"
        mappability_k100_file = mappability_prefix + "/k100.umap.wg.bw"
        py_bw_k24 = pyBigWig.open(mappability_k24_file)
        py_bw_k100 = pyBigWig.open(mappability_k100_file)
    results_vcf_file = parse_params["results_vcf_file"]
    output_features_file = parse_params["output_features_file"]
    pysam_vcf = pysam.VariantFile(results_vcf_file)
    if GIAB_STUTTER:
        stutter_errors_bed = pd.read_csv(
            stutter_file, sep=",", header=0, index_col=0
        )
    else:
        stutter_errors_bed = pysam.TabixFile(stutter_file)
    fail_file = parse_params["fail_file"]
    log_level = getattr(logging, parse_params["loglevel"].upper(), None)
    logfile = parse_params["log_file"]
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {parse_params['loglevel']}")
    logger_features = logging.getLogger("FeatureExtract")
    logger_features.setLevel(log_level)
    if parse_params["log_to_file"]:
        logger_features_file_handler = myutils.LockedFileHandler(logfile)
    else:
        logger_features_file_handler = logging.StreamHandler()
    # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger_features_file_handler.setFormatter(formatter)
    if not logger_features.handlers:  # 避免重复添加handler
        logger_features.addHandler(logger_features_file_handler)
    # XXX try:
    if config_params.ALLELE_EXTRACT["SegMethods"] == "HMM":
        seg_locus = hmmsegger.HMMSeggerLocus(
            hmmsegger.hmm_seg_init, pysam_bed_row, pysam_fasta
        )
    else:
        seg_locus = coordinate_segger.CoordiSeggerLocus(
            pysam_bed_row, pysam_fasta
        )
    if seg_locus.is_usable:
        if add_chr_character:
            chrom = "chr" + seg_locus.chrom
        elif remove_chr_character:
            chrom = seg_locus.chrom.replace("chr", "")
        else:
            chrom = seg_locus.chrom
        all_alleles, var_record = get_var_record(
            pysam_vcf, chrom, seg_locus.start, seg_locus.end, seg_locus.STR_id
        )
        try:
            all_samples_hSNP_INDEL_num = var_record.info["ALL_HVAR_NUM"]
        except KeyError:
            all_samples_hSNP_INDEL_num = "NA"
        if BED_MAPPABILITY:
            average_mappability_score_k24 = "NA"
            average_mappability_score_k100 = "NA"
            for mappbility_item in py_mappability.fetch(
                chrom, seg_locus.start, seg_locus.end
            ):
                mappbility_item_list = mappbility_item.split("\t")
                if mappbility_item_list[3] == seg_locus.STR_id:
                    average_mappability_score_k24 = float(
                        mappbility_item_list[4]
                    )
                    average_mappability_score_k100 = float(
                        mappbility_item_list[5]
                    )
                    break
        else:
            mappability_score_array_k24 = get_mappability_record(
                py_bw_k24, chrom, seg_locus.start, seg_locus.end
            )
            mappability_score_array_k100 = get_mappability_record(
                py_bw_k100, chrom, seg_locus.start, seg_locus.end
            )
            average_mappability_score_k24 = get_average_mappability_score(
                mappability_score_array_k24
            )
            average_mappability_score_k100 = get_average_mappability_score(
                mappability_score_array_k100
            )
            py_bw_k24.close()
            py_bw_k100.close()
        (
            stutter_model_state,
            stutter_model_params,
            inframe_stutter_model_dict,
            outframe_stutter_model_dict,
            log_no_stutter_lik,
        ) = get_stutter_params(
            stutter_errors_bed,
            chrom,
            seg_locus.start,
            seg_locus.end,
            seg_locus.STR_id,
            seg_locus.motif_length,
            seg_locus.STR_length,
        )
        all_samples_all_features_dict = {}
        per_loci_level_features_dict = {
            "average_mappability_score_k24": average_mappability_score_k24
        }
        per_loci_level_features_dict[
            "average_mappability_score_k100"
        ] = average_mappability_score_k100
        per_loci_level_features_dict = {
            **per_loci_level_features_dict,
            **stutter_model_params,
        }
        per_loci_level_features_dict[
            "all_samples_hSNP_INDEL_num"
        ] = all_samples_hSNP_INDEL_num
        left_flanking_context_seq = pysam_fasta.fetch(
            chrom,
            seg_locus.start - FLANKING_CONTEXT_BPS,
            seg_locus.start,
        ).upper()
        right_flanking_context_seq = pysam_fasta.fetch(
            chrom,
            seg_locus.end,
            seg_locus.end + FLANKING_CONTEXT_BPS,
        ).upper()
        per_loci_level_features_dict["left_flanking_context_entropy_A"] = (
            left_flanking_context_seq.count("A") / FLANKING_CONTEXT_BPS
        )
        per_loci_level_features_dict["left_flanking_context_entropy_T"] = (
            left_flanking_context_seq.count("T") / FLANKING_CONTEXT_BPS
        )
        per_loci_level_features_dict["left_flanking_context_entropy_C"] = (
            left_flanking_context_seq.count("C") / FLANKING_CONTEXT_BPS
        )
        per_loci_level_features_dict["left_flanking_context_entropy_G"] = (
            left_flanking_context_seq.count("G") / FLANKING_CONTEXT_BPS
        )
        per_loci_level_features_dict["right_flanking_context_entropy_A"] = (
            right_flanking_context_seq.count("A") / FLANKING_CONTEXT_BPS
        )
        per_loci_level_features_dict["right_flanking_context_entropy_T"] = (
            right_flanking_context_seq.count("T") / FLANKING_CONTEXT_BPS
        )
        per_loci_level_features_dict["right_flanking_context_entropy_C"] = (
            right_flanking_context_seq.count("C") / FLANKING_CONTEXT_BPS
        )
        per_loci_level_features_dict["right_flanking_context_entropy_G"] = (
            right_flanking_context_seq.count("G") / FLANKING_CONTEXT_BPS
        )
        per_loci_level_features_dict["flanking_context_GC_content"] = (
            per_loci_level_features_dict["left_flanking_context_entropy_C"]
            + per_loci_level_features_dict["left_flanking_context_entropy_G"]
            + per_loci_level_features_dict["right_flanking_context_entropy_C"]
            + per_loci_level_features_dict["right_flanking_context_entropy_G"]
        ) / (2)
        STR_flanking_context_seq = pysam_fasta.fetch(
            chrom,
            seg_locus.start - FLANKING_CONTEXT_BPS,
            seg_locus.end + FLANKING_CONTEXT_BPS,
        ).upper()
        STR_flanking_context_GC_content = (
            STR_flanking_context_seq.count("C")
            + STR_flanking_context_seq.count("G")
        ) / len(STR_flanking_context_seq)
        per_loci_level_features_dict[
            "STR_flanking_context_GC_content"
        ] = STR_flanking_context_GC_content
        per_loci_level_features_dict["motif_length"] = seg_locus.motif_length
        if PADDING_BPS > 0:
            refseq_left_padding_1bp = pysam_fasta.fetch(
                chrom,
                seg_locus.start - 1 - PADDING_BPS,
                seg_locus.start - PADDING_BPS,
            )
        else:
            refseq_left_padding_1bp = pysam_fasta.fetch(
                chrom, seg_locus.start - 1, seg_locus.start
            )
        # TODO: get flanking sequence entropy
        # homhet and hethet, phase and unphase
        for bamname, bamfile in zip(bam_name, pysam_bam):
            if GIAB:
                bamname_tmp = bam_files[0].split("/")[-1].split(".")[0]
                sample_median_depth = all_samples_median_depth_dict[
                    bamname_tmp
                ]
                sample_wgs_point_depth_list = all_samples_point_depth_dict[
                    bamname_tmp
                ]
                sample_all_chroms_point_depth_dict = (
                    all_samples_all_chroms_point_depth_dict[bamname_tmp]
                )
            else:
                sample_median_depth = all_samples_median_depth_dict[bamname]
                sample_wgs_point_depth_list = all_samples_point_depth_dict[
                    bamname
                ]
                sample_all_chroms_point_depth_dict = (
                    all_samples_all_chroms_point_depth_dict[bamname]
                )
            sample_chrom_point_depth_list = sample_all_chroms_point_depth_dict[
                chrom
            ]
            sample_chrom_median_point_depth = np.median(
                sample_chrom_point_depth_list
            )
            (
                muttype,
                germ_allele,
                source_allele,
                mut_allele,
            ) = get_mut_type_and_mosaic_allele(var_record, bamname)
            # print(muttype)

            if muttype == "homhet":
                allele_seq = [
                    all_alleles[germ_allele],
                    all_alleles[mut_allele],
                ]
                (
                    alleles_mut_type,
                    mutation_site_seq1,
                    mutation_site_seq2,
                    source_seq,
                    target_seq,
                ) = adjust_vcf.STR_specific_alignment(
                    all_alleles[germ_allele],
                    all_alleles[mut_allele],
                    refseq_left_padding_1bp,
                )
                (
                    mutation_site_type,
                    mutation_location_type,
                    refseq_seq_reference,
                    germ_allele_seq_to_refseq_alternative,
                    source_allele_seq_to_refseq_alternative,
                    mutant_allele_seq_to_refseq_alternative,
                    refseq_left_index,
                    refseq_right_index,
                ) = adjust_vcf.alignment_based_on_global_alignment(
                    all_alleles[0],
                    allele_seq[0],
                    allele_seq[0],
                    allele_seq[1],
                    refseq_left_padding_1bp,
                    PADDING_BPS,
                )
                seq0_length = len(all_alleles[germ_allele])
                seq1_length = len(all_alleles[germ_allele])
                seq2_length = len(all_alleles[mut_allele])
                if len(all_alleles[germ_allele]) == len(
                    all_alleles[mut_allele]
                ):
                    (
                        _,
                        mis_location,
                    ) = myutils.cal_edit_distance_based_on_mismatch(
                        all_alleles[germ_allele], all_alleles[mut_allele]
                    )
                else:
                    mis_location = []
            elif muttype == "hethet":
                allele_seq = [
                    all_alleles[germ_allele],
                    all_alleles[source_allele],
                    all_alleles[mut_allele],
                ]
                (
                    alleles_mut_type,
                    mutation_site_seq1,
                    mutation_site_seq2,
                    source_seq,
                    target_seq,
                ) = adjust_vcf.STR_specific_alignment(
                    all_alleles[source_allele],
                    all_alleles[mut_allele],
                    refseq_left_padding_1bp,
                )
                (
                    mutation_site_type,
                    mutation_location_type,
                    refseq_seq_reference,
                    germ_allele_seq_to_refseq_alternative,
                    source_allele_seq_to_refseq_alternative,
                    mutant_allele_seq_to_refseq_alternative,
                    refseq_left_index,
                    refseq_right_index,
                ) = adjust_vcf.alignment_based_on_global_alignment(
                    all_alleles[0],
                    allele_seq[0],
                    allele_seq[1],
                    allele_seq[2],
                    refseq_left_padding_1bp,
                    PADDING_BPS,
                )
                seq0_length = len(all_alleles[germ_allele])
                seq1_length = len(all_alleles[source_allele])
                seq2_length = len(all_alleles[mut_allele])
                if len(all_alleles[source_allele]) == len(
                    all_alleles[mut_allele]
                ):
                    (
                        _,
                        mis_location,
                    ) = myutils.cal_edit_distance_based_on_mismatch(
                        all_alleles[source_allele], all_alleles[mut_allele]
                    )
                else:
                    mis_location = []
            elif muttype == "germlinehet":
                allele_seq = [
                    all_alleles[germ_allele],
                    all_alleles[mut_allele],
                ]
                (
                    alleles_mut_type,
                    mutation_site_seq1,
                    mutation_site_seq2,
                    source_seq,
                    target_seq,
                ) = adjust_vcf.STR_specific_alignment(
                    all_alleles[germ_allele],
                    all_alleles[mut_allele],
                    refseq_left_padding_1bp,
                )
                (
                    mutation_site_type,
                    mutation_location_type,
                    refseq_seq_reference,
                    germ_allele_seq_to_refseq_alternative,
                    source_allele_seq_to_refseq_alternative,
                    mutant_allele_seq_to_refseq_alternative,
                    refseq_left_index,
                    refseq_right_index,
                ) = adjust_vcf.alignment_based_on_global_alignment(
                    all_alleles[0],
                    allele_seq[0],
                    allele_seq[0],
                    allele_seq[1],
                    refseq_left_padding_1bp,
                    PADDING_BPS,
                )
                seq0_length = len(all_alleles[germ_allele])
                seq1_length = len(all_alleles[germ_allele])
                seq2_length = len(all_alleles[mut_allele])
                if len(all_alleles[germ_allele]) == len(
                    all_alleles[mut_allele]
                ):
                    (
                        _,
                        mis_location,
                    ) = myutils.cal_edit_distance_based_on_mismatch(
                        all_alleles[germ_allele], all_alleles[mut_allele]
                    )
                else:
                    mis_location = []
            elif muttype == "germlinehom":
                allele_seq = [
                    all_alleles[germ_allele],
                    all_alleles[mut_allele],
                ]
                seq0_length = len(all_alleles[germ_allele])
                seq1_length = len(all_alleles[germ_allele])
                seq2_length = len(all_alleles[mut_allele])
                alleles_mut_type = "NA"
                mutation_site_seq1 = "NA"
                mutation_site_seq2 = "NA"
                source_seq = "NA"
                target_seq = "NA"
                mutation_site_type = "NA"
                mutation_location_type = "NA"
                refseq_seq_reference = "NA"
                germ_allele_seq_to_refseq_alternative = "NA"
                source_allele_seq_to_refseq_alternative = "NA"
                mutant_allele_seq_to_refseq_alternative = "NA"
                refseq_left_index = "NA"
                refseq_right_index = "NA"
                mis_location = []
            else:
                raise ValueError("Invalid mutation type")
            # 计算突变发生的位置在哪（mutation site 在距离 STR flanking hap 序列末端第几位的位置，1-based mutation site 距离 hap 末端）
            all_samples_all_features_dict[bamname] = {}
            all_samples_all_features_dict[bamname][
                "alleles_mut_type"
            ] = alleles_mut_type
            all_samples_all_features_dict[bamname][
                "mis_location"
            ] = mis_location
            all_samples_all_features_dict[bamname]["seq0_length"] = seq0_length
            all_samples_all_features_dict[bamname]["seq1_length"] = seq1_length
            all_samples_all_features_dict[bamname]["seq2_length"] = seq2_length
            all_samples_all_features_dict[bamname]["germ_seq"] = all_alleles[
                germ_allele
            ]
            all_samples_all_features_dict[bamname]["source_seq"] = all_alleles[
                source_allele
            ]
            all_samples_all_features_dict[bamname]["mut_seq"] = all_alleles[
                mut_allele
            ]
            if alleles_mut_type in [
                "SNV",
                "noncontinuous_MNV",
                "continuous_MNV",
            ]:
                all_samples_all_features_dict[bamname][
                    "source_is_recurrent_motif"
                ] = judge_recurrent_motif(
                    all_alleles[source_allele], mis_location, seg_locus.motif
                )
                all_samples_all_features_dict[bamname][
                    "mut_is_recurrent_motif"
                ] = judge_recurrent_motif(
                    all_alleles[mut_allele], mis_location, seg_locus.motif
                )
            else:
                all_samples_all_features_dict[bamname][
                    "source_is_recurrent_motif"
                ] = np.nan
                all_samples_all_features_dict[bamname][
                    "mut_is_recurrent_motif"
                ] = np.nan
            all_samples_all_features_dict[bamname]["muttype"] = muttype
            all_samples_all_features_dict[bamname][
                "mutation_site_type"
            ] = mutation_site_type
            all_samples_all_features_dict[bamname][
                "mutation_location_type"
            ] = mutation_location_type
            if mutation_site_seq1 == "NA":
                mutation_start_site_seq1 = "NA"
                mutation_end_site_seq1 = "NA"
            elif mutation_site_seq1[0] > seq1_length / 2:
                mutation_start_site_seq1 = (
                    seq1_length - mutation_site_seq1[0]
                )  # 1-based 变异位点，仍然是 left align，倒数第 mutation_start_site_seq1 到倒数第 mutation_end_site_seq1 位
                mutation_end_site_seq1 = (
                    seq1_length - mutation_site_seq1[1]
                )  # 1-based 变异位点
            else:
                mutation_start_site_seq1 = (
                    mutation_site_seq1[0] + 1
                )  # 1-based 变异位点
                mutation_end_site_seq1 = (
                    mutation_site_seq1[1] + 1
                )  # 1-based 变异位点
            if (
                mutation_site_seq2 == "NA"
            ):  # BUG: TypeError: '>' not supported between instances of 'str' and 'float'
                mutation_start_site_seq2 = "NA"
                mutation_end_site_seq2 = "NA"
            elif mutation_site_seq2[0] > seq2_length / 2:
                mutation_start_site_seq2 = (
                    seq2_length - mutation_site_seq2[0]
                )  # 1-based 变异位点
                mutation_end_site_seq2 = seq2_length - mutation_site_seq2[1]
            else:
                mutation_start_site_seq2 = (
                    mutation_site_seq2[0] + 1
                )  # 1-based 变异位点
                mutation_end_site_seq2 = mutation_site_seq2[1] + 1
            all_samples_all_features_dict[bamname][
                "mutation_site_start_seq1"
            ] = mutation_start_site_seq1
            all_samples_all_features_dict[bamname][
                "mutation_site_end_seq1"
            ] = mutation_end_site_seq1
            all_samples_all_features_dict[bamname][
                "mutation_site_start_seq2"
            ] = mutation_start_site_seq2
            all_samples_all_features_dict[bamname][
                "mutation_site_end_seq2"
            ] = mutation_end_site_seq2
            all_samples_all_features_dict[bamname][
                "mosaic_fraction"
            ] = var_record.samples[bamname]["MF"]
            phasable = var_record.samples[bamname]["PHA"]
            all_samples_all_features_dict[bamname]["phasable"] = phasable
            if phasable == "Yes":
                (
                    hSNP_features_dict,
                    all_reads_nearbysnp_dict,
                    all_reads_nearbysnp_list_dict,
                ) = get_sample_reads_nearby_snp(
                    chrom,
                    var_record,
                    bamname,
                    bamfile,
                    pysam_fasta,
                    wgs_mean_depth_dict[bamname],
                    sample_median_depth,
                    sample_wgs_point_depth_list,
                    sample_chrom_median_point_depth,
                    sample_chrom_point_depth_list,
                )
                germ_snp1, germ_snp2, mosaic_snp = get_phased_mosaic_GT_allele(
                    var_record, bamname
                )
                pass  # TODO: Get phasable-related features
            else:
                (
                    hSNP_features_dict,
                    all_reads_nearbysnp_dict,
                    all_reads_nearbysnp_list_dict,
                ) = ({}, {}, {})
            overall_sample_after_filtered_read_dp = 0
            overall_sample_filtered_out_read_dp = 0
            for read in bamfile.fetch(chrom, seg_locus.start, seg_locus.end):
                # multiple_iterators=False
                # Set multiple_iterators to true if you will be using multiple iterators on the same file at the same time. The iterator returned will receive its own copy of a filehandle to the file effectively re-opening the file. Re-opening a file creates some overhead, so beware.
                per_read_feature = extract_allele.PerReadFeature(
                    read, seg_locus.start, seg_locus.end
                )
                per_read_feature.read_is_usable_for_bwa()
                if READ_FILTER:
                    if per_read_feature.is_usable:
                        use_read = True
                    else:
                        use_read = False
                else:
                    if (read.reference_end is None) or (
                        read.query_alignment_qualities is None
                    ):
                        use_read = False
                    else:
                        if SPANNING_FILTER:
                            if per_read_feature.read_spanning:
                                use_read = True
                            else:
                                use_read = False
                        else:
                            use_read = True
                if OVERALL_FEATURES_READ_FILTER:
                    if per_read_feature.is_usable:
                        overall_use_read = True
                    else:
                        overall_use_read = False
                else:
                    if (read.reference_end is None) or (
                        read.query_alignment_qualities is None
                    ):
                        overall_use_read = False
                    else:
                        if OVERALL_SPANNING_FILTER:
                            if per_read_feature.read_spanning:
                                overall_use_read = True
                            else:
                                overall_use_read = False
                        else:
                            overall_use_read = True
                # TODO alleles 之间突变的容易性 ok，STR 区域的 indels 和 mismatches num ok
                # TODO mis num/indel or not/indel size ok
                # TODO other phasing-related features
                # TODO 如何区分 germhet 和 mosaic，有哪些特征可以用来区分 germline 和 mosaic
                # TODO germ het 特征 和 mosaic homhet 有哪些特征可以用来区分
                # TODO phasing-related features 哪些可以用
                # TODO artifacts 的特征 和 germline 和 mosaic 的特征有哪些可以用来区分
                # TODO share variants 和 cell-specific variants 有哪些特征可以用来区分
                # TODO read-level features 有哪些可以用来区分
                if overall_use_read:
                    readsegger = coordinate_segger.CoordinateSegRead(
                        seg_locus, read
                    )
                    readsegger_usable = readsegger.is_usable
                else:
                    readsegger_usable = False
                if overall_use_read and readsegger_usable:
                    per_read_feature.read_level_features()
                    read_str_flanking_seq = readsegger.STR_flanking_seq
                    read_str_flanking_seq_error_rate = (
                        readsegger.STR_flanking_error_rate
                    )
                    read_str_flanking_seq_accuracy = (
                        1 - read_str_flanking_seq_error_rate
                    )
                    all_allele_log_lik = {}
                    all_allele_mis_score = {}
                    all_allele_mis_fraction = {}
                    base_accuracy_dict = {}
                    # if alleles_mut_type == "SNV":
                    #     mutation_site_index_based_on_hap_zero_based = (
                    #         mutation_site_seq1[0]
                    #     )
                    # else:
                    #     mutation_site_index_based_on_hap_zero_based = "NA"
                    if alleles_mut_type in [
                        "SNV",
                        "noncontinuous_MNV",
                        "continuous_MNV",
                    ]:
                        mutation_site_index_based_on_hap_zero_based_list = (
                            mis_location
                        )
                    else:
                        mutation_site_index_based_on_hap_zero_based_list = []
                    for expected_allele in allele_seq:
                        (
                            log_str_align_prob,
                            mis_score,
                            max_alignment_length,
                            base_accuracy_list,
                        ) = myutils.cal_per_read_log_likelihood_given_allele_seq_based(
                            expected_allele,
                            read_str_flanking_seq,
                            read_str_flanking_seq_accuracy,
                            log_no_stutter_lik,
                            seg_locus.motif_length,
                            inframe_stutter_model_dict,
                            outframe_stutter_model_dict,
                            mutation_site_index_based_on_hap_zero_based_list,
                        )
                        if len(base_accuracy_list) > 0:
                            base_accuracy = np.mean(base_accuracy_list)
                        else:
                            base_accuracy = "NA"
                        all_allele_log_lik[
                            expected_allele
                        ] = log_str_align_prob
                        all_allele_mis_score[expected_allele] = mis_score
                        all_allele_mis_fraction[expected_allele] = (
                            mis_score / max_alignment_length
                        )
                        if BASEQ_OR_ACCURACY == "accuracy":
                            base_accuracy_dict[expected_allele] = base_accuracy
                        else:
                            base_accuracy_dict[
                                expected_allele
                            ] = accuracy_to_phred(base_accuracy)
                    all_allele_log_lik_sorted_dict = sorted(
                        all_allele_log_lik.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    if (
                        len(all_allele_log_lik_sorted_dict) < 2
                    ):  # BUG: IndexError: list index out of range
                        all_samples_all_features_dict[bamname].setdefault(
                            "str_mis_num", []
                        ).append(
                            all_allele_mis_score[
                                all_allele_log_lik_sorted_dict[0][0]
                            ]
                        )
                        all_samples_all_features_dict[bamname].setdefault(
                            "str_mis_fraction", []
                        ).append(
                            all_allele_mis_fraction[
                                all_allele_log_lik_sorted_dict[0][0]
                            ]
                        )
                        if (
                            all_allele_log_lik_sorted_dict[0][0]
                            == allele_seq[0]
                        ):
                            all_samples_all_features_dict[bamname].setdefault(
                                "read_hap_assignment", []
                            ).append("germ")
                            all_samples_all_features_dict[bamname].setdefault(
                                "mis_base_accuracy", []
                            ).append(base_accuracy_dict[allele_seq[0]])
                            # all_samples_all_features_dict[bamname].setdefault(
                            #     "all_reads", []
                            # ).append("")
                        else:  # XXX: 理论上不会发生
                            all_samples_all_features_dict[bamname].setdefault(
                                "read_hap_assignment", []
                            ).append("NA")
                            all_samples_all_features_dict[bamname].setdefault(
                                "mis_base_accuracy", []
                            ).append("NA")
                            # all_samples_all_features_dict[bamname].setdefault(
                            #     "all_reads", []
                            # ).append("NA")
                        if (
                            read_str_flanking_seq.upper()
                            == all_allele_log_lik_sorted_dict[0][0].upper()
                        ):
                            all_samples_all_features_dict[bamname].setdefault(
                                "noise_type", []
                            ).append("no_noise")
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size", []
                            ).append(0)
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size_per_motif_length", []
                            ).append(0)
                        elif len(read_str_flanking_seq) == len(
                            all_allele_log_lik_sorted_dict[0][0]
                        ):
                            all_samples_all_features_dict[bamname].setdefault(
                                "noise_type", []
                            ).append("mismatch")
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size", []
                            ).append(0)
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size_per_motif_length", []
                            ).append(0)
                        elif len(read_str_flanking_seq) > len(
                            all_allele_log_lik_sorted_dict[0][0]
                        ):
                            all_samples_all_features_dict[bamname].setdefault(
                                "noise_type", []
                            ).append("insertion")
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size", []
                            ).append(
                                len(read_str_flanking_seq)
                                - len(all_allele_log_lik_sorted_dict[0][0])
                            )
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size_per_motif_length", []
                            ).append(
                                (
                                    len(read_str_flanking_seq)
                                    - len(all_allele_log_lik_sorted_dict[0][0])
                                )
                                / seg_locus.motif_length
                            )
                        else:
                            all_samples_all_features_dict[bamname].setdefault(
                                "noise_type", []
                            ).append("deletion")
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size", []
                            ).append(
                                len(all_allele_log_lik_sorted_dict[0][0])
                                - len(read_str_flanking_seq)
                            )
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size_per_motif_length", []
                            ).append(
                                (
                                    len(all_allele_log_lik_sorted_dict[0][0])
                                    - len(read_str_flanking_seq)
                                )
                                / seg_locus.motif_length
                            )
                    elif (
                        all_allele_log_lik_sorted_dict[0][1]
                        == all_allele_log_lik_sorted_dict[1][1]
                    ):
                        all_samples_all_features_dict[bamname].setdefault(
                            "read_hap_assignment", []
                        ).append("NA")
                        all_samples_all_features_dict[bamname].setdefault(
                            "noise_type", []
                        ).append("NA")
                        all_samples_all_features_dict[bamname].setdefault(
                            "indel_size", []
                        ).append("NA")
                        all_samples_all_features_dict[bamname].setdefault(
                            "indel_size_per_motif_length", []
                        ).append("NA")
                        all_samples_all_features_dict[bamname].setdefault(
                            "str_mis_num", []
                        ).append("NA")
                        all_samples_all_features_dict[bamname].setdefault(
                            "str_mis_fraction", []
                        ).append("NA")
                        all_samples_all_features_dict[bamname].setdefault(
                            "mis_base_accuracy", []
                        ).append("NA")
                        # all_samples_all_features_dict[bamname].setdefault(
                        #     "all_reads", []
                        # ).append("NA")
                    else:
                        all_samples_all_features_dict[bamname].setdefault(
                            "str_mis_num", []
                        ).append(
                            all_allele_mis_score[
                                all_allele_log_lik_sorted_dict[0][0]
                            ]
                        )
                        all_samples_all_features_dict[bamname].setdefault(
                            "str_mis_fraction", []
                        ).append(
                            all_allele_mis_fraction[
                                all_allele_log_lik_sorted_dict[0][0]
                            ]
                        )
                        if muttype == "hethet":
                            if (
                                all_allele_log_lik_sorted_dict[0][0]
                                == allele_seq[0]
                            ):
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("read_hap_assignment", []).append(
                                    "germ"
                                )
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("mis_base_accuracy", []).append(
                                    base_accuracy_dict[allele_seq[0]]
                                )
                                # all_samples_all_features_dict[bamname].setdefault(
                                #     "all_reads", []
                                #     ).append("")
                            elif (
                                all_allele_log_lik_sorted_dict[0][0]
                                == allele_seq[1]
                            ):
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("read_hap_assignment", []).append(
                                    "source"
                                )
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("mis_base_accuracy", []).append(
                                    base_accuracy_dict[allele_seq[1]]
                                )
                                # all_samples_all_features_dict[bamname].setdefault(
                                #     "all_reads", []
                                #     ).append(per_read_feature)
                            else:
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("read_hap_assignment", []).append(
                                    "mut"
                                )
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("mis_base_accuracy", []).append(
                                    base_accuracy_dict[allele_seq[2]]
                                )
                                # all_samples_all_features_dict[bamname].setdefault(
                                #     "all_reads", []
                                #     ).append(per_read_feature)
                        else:
                            if (
                                all_allele_log_lik_sorted_dict[0][0]
                                == allele_seq[0]
                            ):
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("read_hap_assignment", []).append(
                                    "germ"
                                )
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("mis_base_accuracy", []).append(
                                    base_accuracy_dict[allele_seq[0]]
                                )
                                # all_samples_all_features_dict[bamname].setdefault(
                                #     "all_reads", []
                                #     ).append("")
                            else:  # homhet 和 germline het 除了 germ allele 之外的 allele 都是 mut，按照前面的代码和定义是没有问题的
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("read_hap_assignment", []).append(
                                    "mut"
                                )
                                all_samples_all_features_dict[
                                    bamname
                                ].setdefault("mis_base_accuracy", []).append(
                                    base_accuracy_dict[allele_seq[1]]
                                )
                                # all_samples_all_features_dict[bamname].setdefault(
                                #     "all_reads", []
                                #     ).append(per_read_feature)
                        if (
                            read_str_flanking_seq.upper()
                            == all_allele_log_lik_sorted_dict[0][0].upper()
                        ):
                            all_samples_all_features_dict[bamname].setdefault(
                                "noise_type", []
                            ).append("no_noise")
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size", []
                            ).append(0)
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size_per_motif_length", []
                            ).append(0)
                        elif len(read_str_flanking_seq) == len(
                            all_allele_log_lik_sorted_dict[0][0]
                        ):
                            all_samples_all_features_dict[bamname].setdefault(
                                "noise_type", []
                            ).append("mismatch")
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size", []
                            ).append(0)
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size_per_motif_length", []
                            ).append(0)
                        elif len(read_str_flanking_seq) > len(
                            all_allele_log_lik_sorted_dict[0][0]
                        ):
                            all_samples_all_features_dict[bamname].setdefault(
                                "noise_type", []
                            ).append("insertion")
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size", []
                            ).append(
                                len(read_str_flanking_seq)
                                - len(all_allele_log_lik_sorted_dict[0][0])
                            )
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size_per_motif_length", []
                            ).append(
                                (
                                    len(read_str_flanking_seq)
                                    - len(all_allele_log_lik_sorted_dict[0][0])
                                )
                                / seg_locus.motif_length
                            )
                        else:
                            all_samples_all_features_dict[bamname].setdefault(
                                "noise_type", []
                            ).append("deletion")
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size", []
                            ).append(
                                len(all_allele_log_lik_sorted_dict[0][0])
                                - len(read_str_flanking_seq)
                            )
                            all_samples_all_features_dict[bamname].setdefault(
                                "indel_size_per_motif_length", []
                            ).append(
                                (
                                    len(all_allele_log_lik_sorted_dict[0][0])
                                    - len(read_str_flanking_seq)
                                )
                                / seg_locus.motif_length
                            )
                    overall_sample_after_filtered_read_dp += 1
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_query_name", []
                    ).append(read.query_name)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_genotyping_use_read", []
                    ).append(use_read)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_feature_use_read", []
                    ).append(overall_use_read)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_template_length", []
                    ).append(read.template_length)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_mapQ", []
                    ).append(read.mapping_quality)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_proper_pair", []
                    ).append(per_read_feature.read_is_proper_pair)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_secondary_read", []
                    ).append(read.is_secondary)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_supplementary_read", []
                    ).append(read.is_supplementary)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_duplicate_read", []
                    ).append(read.is_duplicate)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_unmapped_read", []
                    ).append(read.is_unmapped)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_qcfail_read", []
                    ).append(read.is_qcfail)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_baseq_read", []
                    ).append(np.mean(read.query_qualities))
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_softclipped_read", []
                    ).append(per_read_feature.read_soft_clipping_num)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_hardclipped_read", []
                    ).append(
                        per_read_feature.read_hard_clipping_num
                    )  # > READS_HARD_CLIPPING_CUTOFF
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_clipped_read", []
                    ).append(
                        (
                            (
                                per_read_feature.read_hard_clipping_num
                                > READS_HARD_CLIPPING_CUTOFF
                            )
                            or (
                                per_read_feature.read_soft_clipping_num
                                > READS_SOFT_CLIPPING_CUTOFF
                            )
                        )
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_flanking_indel", []
                    ).append(
                        per_read_feature.read_features["is_flanking_indel"]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_spanning_str_flanking_indel", []
                    ).append(
                        per_read_feature.read_features[
                            "is_spanning_str_flanking_indel"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_strand", []
                    ).append(per_read_feature.read_features["strand"])
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read1", []
                    ).append(per_read_feature.read_features["read1"])
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_flanking_sbs_num", []
                    ).append(
                        per_read_feature.read_features["read_flanking_sbs_num"]
                    )
                    if (
                        per_read_feature.read_features["left_flanking_bps_num"]
                        + per_read_feature.read_features[
                            "right_flanking_bps_num"
                        ]
                    ) == 0:
                        all_samples_all_features_dict[bamname].setdefault(
                            "overall_read_flanking_sbs_num_per_bp", []
                        ).append("NA")
                        all_samples_all_features_dict[bamname].setdefault(
                            "overall_read_flanking_indel_num_per_bp", []
                        ).append("NA")
                    else:
                        all_samples_all_features_dict[bamname].setdefault(
                            "overall_read_flanking_sbs_num_per_bp", []
                        ).append(
                            per_read_feature.read_features[
                                "read_flanking_sbs_num"
                            ]
                            / (
                                per_read_feature.read_features[
                                    "left_flanking_bps_num"
                                ]
                                + per_read_feature.read_features[
                                    "right_flanking_bps_num"
                                ]
                            )
                        )
                        all_samples_all_features_dict[bamname].setdefault(
                            "overall_read_flanking_indel_num_per_bp", []
                        ).append(
                            per_read_feature.read_features[
                                "read_flanking_indel_num"
                            ]
                            / (
                                per_read_feature.read_features[
                                    "left_flanking_bps_num"
                                ]
                                + per_read_feature.read_features[
                                    "right_flanking_bps_num"
                                ]
                            )
                        )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_flanking_bps_num", []
                    ).append(
                        per_read_feature.read_features["left_flanking_bps_num"]
                        + per_read_feature.read_features[
                            "right_flanking_bps_num"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_flanking_indel_num", []
                    ).append(
                        per_read_feature.read_features[
                            "read_flanking_indel_num"
                        ]
                    )
                    # TODO: Need Normalize by Read Length # Finish
                    # TODO: Need Check For This Features Output # Finish
                    # TODO: Overall mis fraction for a STR loci # Finish
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_all_sbs_num_based_on_MD", []
                    ).append(
                        per_read_feature.read_features[
                            "read_all_sbs_num_based_on_MD"
                        ]
                        / per_read_feature.read_features["read_length"]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_all_sbs_num_based_NM_edit_distance", []
                    ).append(
                        per_read_feature.read_features[
                            "read_all_sbs_num_based_NM_edit_distance"
                        ]
                        / per_read_feature.read_features["read_length"]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_left_flanking_bps_num", []
                    ).append(
                        per_read_feature.read_features["left_flanking_bps_num"]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_right_flanking_bps_num", []
                    ).append(
                        per_read_feature.read_features[
                            "right_flanking_bps_num"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_length", []
                    ).append(per_read_feature.read_features["read_length"])
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_str_pcr_start", []
                    ).append(per_read_feature.read_features["str_pcr_start"])
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_str_pcr_end", []
                    ).append(per_read_feature.read_features["str_pcr_end"])
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_str_padding_pcr_start", []
                    ).append(
                        per_read_feature.read_features["str_padding_pcr_start"]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_str_padding_pcr_end", []
                    ).append(
                        per_read_feature.read_features["str_padding_pcr_end"]
                    )

                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_pcr_cycle_percentage_sum", []
                    ).append(
                        (
                            (
                                per_read_feature.read_features[
                                    "str_padding_pcr_start"
                                ]
                                + per_read_feature.read_features[
                                    "str_padding_pcr_end"
                                ]
                            )
                            / per_read_feature.read_features["read_length"]
                        )
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_read_pcr_cycle_sum", []
                    ).append(
                        (
                            per_read_feature.read_features[
                                "str_padding_pcr_start"
                            ]
                            + per_read_feature.read_features[
                                "str_padding_pcr_end"
                            ]
                        )
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_str_pcr_start_terminal1", []
                    ).append(
                        per_read_feature.read_features[
                            "str_pcr_start_terminal1"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_str_pcr_end_terminal2", []
                    ).append(
                        per_read_feature.read_features["str_pcr_end_terminal2"]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_str_padding_pcr_start_terminal1", []
                    ).append(
                        per_read_feature.read_features[
                            "str_padding_pcr_start_terminal1"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_str_padding_pcr_end_terminal2", []
                    ).append(
                        per_read_feature.read_features[
                            "str_padding_pcr_end_terminal2"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_zero_based_leftmost_coordinate", []
                    ).append(
                        seg_locus.start
                        - per_read_feature.read_features[
                            "zero_based_leftmost_coordinate"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_reference_start", []
                    ).append(per_read_feature.reference_start)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_reference_end", []
                    ).append(per_read_feature.reference_end)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_left_flanking_bps_num", []
                    ).append(
                        per_read_feature.read_features["left_flanking_bps_num"]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_right_flanking_bps_num", []
                    ).append(
                        per_read_feature.read_features[
                            "right_flanking_bps_num"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_left_flanking_sbs_num", []
                    ).append(
                        per_read_feature.read_features["left_flanking_sbs_num"]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_right_flanking_sbs_num", []
                    ).append(
                        per_read_feature.read_features[
                            "right_flanking_sbs_num"
                        ]
                    )
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_left_flanking_baseq", []
                    ).append(readsegger.all_left_flanking_baseq)
                    all_samples_all_features_dict[bamname].setdefault(
                        "overall_right_flanking_baseq", []
                    ).append(readsegger.all_right_flanking_baseq)
                    # TODO
                    # 需不需要提取突变位置碱基的 baseq 来比较防止整个 hap 关键位置的 low baseQ 被其他非关键位置的 baseQ 给稀释了？
                    # 太麻烦了，目前不考虑这个问题，因为涉及到 stutter reads anchor 突变位置进行重比对的问题，并且 anchor 位置可能不准确
                else:
                    overall_sample_filtered_out_read_dp += 1
            # TODO
            # mutant and source mismatch 左边和右边的 baseQ 以及 mismatch number 数量比较
            # Mismatch iclude SNV/MNV/non-continuous MNV
            # 先找到 mismatch 的 location 对于 query_alignnment 的序列位值，再找到基因组位置，再根据基因组位值找到左半边和右半边的 baseQ 和 mismatch
            # SNV 和 MNV 可以取左半边和右半边
            # non-continuous MNV 可以取起始和终止的左边和右边
            # 突变的位置要不要算入，暂时不算入，因为不知道左边还是右边
            # 需不需要和左边 1bp 和 右边 1bp 做 test呢
            # 需不需要知道这个 base 是不是 recurrent motif 的 base 呢
            # 注意对长度进行标准化
            # 如果 mutant reads 很少，怎么做 test，test 会不会有 bias
            # 到底是做 test 好，还是卡大于特定阀值的 fraction 值，所有平均值 或是 中位数值 或是单一数值的 fraction 值硬卡比较好呢 ？
            # Test and code begin
            # 最好最后能保存一份包含标题的文件，防止 features 标题对应不上
            # 具体怎么做呢？基于重比对，基于坐标，还是基于 reference 重比对，还是基于 reference indel 位置呢
            # 使用 reference 的坐标是没问题的
            # begin code
            # 第一找 read exact the same，第二和 reference 比较假如一致长假如不一致长，否则怎么做，这三部曲解决这个问题
            # HACK: 左右两边如果有 germline variants，会影响 mismatches 的计数
            # 所以同时满足多个条件才过滤（and / or）,需要进行精心设计
            # Features 后面需要取 log 值还是 -log 值哈
            # 使用突变位点的左边和右边可能会有 bias 对于 germline STR het
            # germline 位点问题无法避免在非 gennotype 的 STR 区域，目前先忽略不记
            # 当前先比较左边 flanking mis baseQ 与 右边 flanking mis baseQ 吧
            all_samples_all_features_dict[bamname][
                "overall_mis_num_per_bp_all_reads_mean_based_on_NM"
            ] = np.mean(
                all_samples_all_features_dict[bamname][
                    "overall_read_all_sbs_num_based_NM_edit_distance"
                ]
            )
            all_samples_all_features_dict[bamname][
                "overall_mis_num_per_bp_all_reads_mean_based_on_MD"
            ] = np.mean(
                all_samples_all_features_dict[bamname][
                    "overall_read_all_sbs_num_based_on_MD"
                ]
            )
            all_samples_all_features_dict[bamname][
                "overall_sample_after_filtered_read_dp"
            ] = overall_sample_after_filtered_read_dp
            all_samples_all_features_dict[bamname][
                "overall_sample_filtered_out_read_dp"
            ] = overall_sample_filtered_out_read_dp
            all_samples_all_features_dict[bamname][
                "overall_used_reads_fraction"
            ] = overall_sample_after_filtered_read_dp / (
                overall_sample_after_filtered_read_dp
                + overall_sample_filtered_out_read_dp
            )
            mle_mosaic_allele_vaf_single_locus = (
                get_mle_mosaic_allele_frequency_single_locus(
                    all_samples_all_features_dict[bamname]
                )
            )
            all_samples_all_features_dict[bamname][
                "mle_mosaic_allele_vaf_single_locus"
            ] = mle_mosaic_allele_vaf_single_locus
            used_read_num_in_genotyping = np.sum(
                all_samples_all_features_dict[bamname][
                    "overall_genotyping_use_read"
                ]
            )
            reference_str_padding_length = len(
                seg_locus.ref_STR_flanking_sequence
            )
            sample_str_used_depth = str_used_mean_depth_dict[bamname]
            sample_wgs_used_depth = wgs_mean_depth_dict[bamname]
            depth_norm_factor = (
                np.median(
                    all_samples_all_features_dict[bamname][
                        "overall_read_length"
                    ]
                )
                - reference_str_padding_length
            ) / np.median(
                all_samples_all_features_dict[bamname]["overall_read_length"]
            )
            used_read_num_in_feature = overall_sample_after_filtered_read_dp
            expected_str_used_depth = sample_wgs_used_depth * depth_norm_factor
            expected_str_used_depth_ratio = myutils.avoid_zero_divide_error(used_read_num_in_feature, expected_str_used_depth)
            all_samples_all_features_dict[bamname][
                "used_read_num_in_feature"
            ] = used_read_num_in_feature
            all_samples_all_features_dict[bamname][
                "used_read_num_in_genotyping"
            ] = used_read_num_in_genotyping
            all_samples_all_features_dict[bamname][
                "sample_str_used_depth"
            ] = sample_str_used_depth
            all_samples_all_features_dict[bamname][
                "sample_wgs_used_depth"
            ] = sample_wgs_used_depth
            all_samples_all_features_dict[bamname][
                "depth_norm_factor"
            ] = depth_norm_factor
            all_samples_all_features_dict[bamname][
                "expected_str_used_depth"
            ] = expected_str_used_depth
            all_samples_all_features_dict[bamname][
                "expected_str_used_depth_ratio"
            ] = expected_str_used_depth_ratio
            if (
                used_read_num_in_feature > 0
            ):  # BUG: RuntimeWarning: divide by zero encountered in long_scalars
                all_samples_all_features_dict[bamname][
                    "used_read_fraction_in_genotyping"
                ] = (used_read_num_in_genotyping / used_read_num_in_feature)
            else:
                all_samples_all_features_dict[bamname][
                    "used_read_fraction_in_genotyping"
                ] = 0
            vcf_based_features = get_features_from_vcf(var_record, bamname)
            # str_nearby_raw_depth_list = get_chrom_depth(
            #     chrom, pysam_fasta, bamfile
            # )
            str_padding_unfiltered_raw_depth = np.mean(
                np.array(
                    bamfile.count_coverage(
                        chrom,
                        seg_locus.start - PADDING_BPS,
                        seg_locus.end + PADDING_BPS,
                        quality_threshold=0,
                    )
                ).sum(axis=0)
            )
            # median_str_nearby_raw_depth = np.median(str_nearby_raw_depth_list)
            median_str_wgs_point_raw_depth = sample_median_depth
            median_str_nearby_raw_depth = sample_chrom_median_point_depth
            str_dp_diff = (
                str_padding_unfiltered_raw_depth - median_str_nearby_raw_depth
            )
            if median_str_nearby_raw_depth == 0:
                str_dp_diff_fraction = "NA"
            else:
                str_dp_diff_fraction = (
                    str_dp_diff / median_str_nearby_raw_depth
                )
            final_padding_str_depth_percentile = percentile_rank(
                sample_wgs_point_depth_list, str_padding_unfiltered_raw_depth
            )
            final_padding_str_chrom_depth_percentile = percentile_rank(
                sample_chrom_point_depth_list, str_padding_unfiltered_raw_depth
            )
            # XXX: undirectional depth percentile
            if DIRECTIONAL_STR_DEPTH_PERCENTILE:
                pass
            else:
                if final_padding_str_depth_percentile > 0.5:
                    final_padding_str_depth_percentile = (
                        final_padding_str_depth_percentile
                    )
                else:
                    final_padding_str_depth_percentile = (
                        1 - final_padding_str_depth_percentile
                    )
                if final_padding_str_chrom_depth_percentile > 0.5:
                    final_padding_str_chrom_depth_percentile = (
                        final_padding_str_chrom_depth_percentile
                    )
                else:
                    final_padding_str_chrom_depth_percentile = (
                        1 - final_padding_str_chrom_depth_percentile
                    )
            if sample_wgs_used_depth == 0:
                raw_wgs_expected_depth_ratio = "NA"
            else:
                raw_wgs_expected_depth_ratio = (
                    str_padding_unfiltered_raw_depth / sample_wgs_used_depth
                )
            if median_str_nearby_raw_depth == 0:
                nearby_raw_expected_depth_ratio = "NA"
            else:
                nearby_raw_expected_depth_ratio = (
                    str_padding_unfiltered_raw_depth
                    / median_str_nearby_raw_depth
                )
            if median_str_wgs_point_raw_depth == 0:
                raw_wgs_point_expected_depth_ratio = "NA"
            else:
                raw_wgs_point_expected_depth_ratio = (
                    str_padding_unfiltered_raw_depth
                    / median_str_wgs_point_raw_depth
                )
            all_samples_all_features_dict[bamname][
                "str_padding_unfiltered_raw_depth"
            ] = str_padding_unfiltered_raw_depth
            all_samples_all_features_dict[bamname][
                "median_str_nearby_raw_depth"
            ] = median_str_nearby_raw_depth
            all_samples_all_features_dict[bamname]["str_dp_diff"] = str_dp_diff
            all_samples_all_features_dict[bamname][
                "str_dp_diff_fraction"
            ] = str_dp_diff_fraction
            all_samples_all_features_dict[bamname][
                "final_padding_str_depth_percentile"
            ] = final_padding_str_depth_percentile
            all_samples_all_features_dict[bamname][
                "raw_wgs_expected_depth_ratio"
            ] = raw_wgs_expected_depth_ratio
            all_samples_all_features_dict[bamname][
                "nearby_raw_expected_depth_ratio"
            ] = nearby_raw_expected_depth_ratio
            all_samples_all_features_dict[bamname][
                "raw_wgs_point_expected_depth_ratio"
            ] = raw_wgs_point_expected_depth_ratio
            all_samples_all_features_dict[bamname][
                "final_padding_str_chrom_depth_percentile"
            ] = final_padding_str_chrom_depth_percentile
            (
                str_features_dict,
                germ_mut_feature_dict,
                germ_source_feature_dict,
            ) = statistic_test_for_str_loci_features(
                all_samples_all_features_dict[bamname],
                muttype,
                seg_locus.start - PADDING_BPS,
                seg_locus.end + PADDING_BPS,
            )
            # count_coverage(self, contig, start=None, stop=None, region=None, quality_threshold=15, read_callback='all', reference=None, end=None)
            # contig (string) – reference_name of the genomic region (chromosome)
            # start (int) – start of the genomic region (0-based inclusive). If not given, count from the start of the chromosome.
            # stop (int) – end of the genomic region (0-based exclusive). If not given, count to the end of the chromosome.
            # region (string) – a region string.
            # quality_threshold (int) – quality_threshold is the minimum quality score (in phred) a base has to reach to be counted.
            # read_callback (string or function) –
            # select a call-back to ignore reads when counting. It can be either a string with the following values:
            # all
            # skip reads in which any of the following flags are set: BAM_FUNMAP, BAM_FSECONDARY, BAM_FQCFAIL, BAM_FDUP
            # nofilter
            # uses every single read
            # Alternatively, read_callback can be a function check_read(read) that should return True only for those reads that shall be included in the counting.
            # reference (string) – backward compatible synonym for contig
            # end (int) – backward compatible synonym for stop
            if phasable == "Yes":
                other_phasing_related_features_dict = (
                    other_phasing_related_features(
                        all_reads_nearbysnp_dict,
                        muttype,
                        all_samples_all_features_dict[bamname],
                        germ_snp1,
                        germ_snp2,
                        mosaic_snp,
                    )
                )
            else:
                other_phasing_related_features_dict = {}
            writed_line = write_features_to_csv(
                bamname,
                per_loci_level_features_dict,
                hSNP_features_dict,
                all_samples_all_features_dict[bamname],
                str_features_dict,
                germ_mut_feature_dict,
                germ_source_feature_dict,
                other_phasing_related_features_dict,
                vcf_based_features,
                pysam_bed_row_dict,
            )
            writed_line = writed_line.strip()
            # writed_line = pysam_bed_row_string + "," + writed_line
            writed_line = writed_line + "\n"
            with open(output_features_file, "a") as f:
                if myutils.acquire_lock(f):
                    f.write(writed_line)
                    myutils.release_lock(f)
    else:
        if BED_MAPPABILITY:
            pass
        else:
            py_bw_k24.close()
            py_bw_k100.close()
        logger_features.info(
            "Locus %s %s:%d-%d is not usable because %s"
            % (
                seg_locus.STR_id,
                seg_locus.chrom,
                seg_locus.start,
                seg_locus.end,
                seg_locus.unusable_reason,
            )
        )
        with open(fail_file, "a") as f:
            if myutils.acquire_lock(f):
                f.write(
                    "%s\t%d\t%d\t%s\t%s\n"
                    % (
                        seg_locus.chrom,
                        seg_locus.start,
                        seg_locus.end,
                        seg_locus.STR_id,
                        seg_locus.unusable_reason,
                    )
                )
                myutils.release_lock(f)


def write_features_to_csv(
    sample_name,
    per_loci_level_features_dict,
    hSNP_features_dict,
    sample_all_features_dict,
    str_features_dict,
    germ_mut_feature_dict,
    germ_source_feature_dict,
    other_phasing_related_features_dict,
    vcf_based_features,
    pysam_bed_row_dict,
):
    line = ""
    vcf_based_features_list = [
        "chrom",
        "STRSTART",
        "STREND",
        "MOTIF_length",
        "PERIOD",
        "str_id",
        "MOTIF",
        "GT",
        "MGT",
        "MP",
        "CMP",
        "variant_type",
        "frame",
        "mosaic_fraction",
        "MF_hom2het_het2het",
        "perfect_ref_str",
        "depth",
        "filtered_depth",
        "SegmentConditionFail",
        "SegmentResultFail",
        "ReadN",
        "library_issue",
        "map_issue",
        "spanning_issue",
        "unmap_issue",
        "duplicate_same",
        "duplicate_diff",
        'read_pair_same',
        'read_pair_diff',
        "duplicate_loose_same",
        "duplicate_loose_diff",
        "duplicate_complex_1",
        "duplicate_complex_2",
        "duplicate_complex_3",
        "duplicate_complex_4",
        "EAF",
        "observed_mosaic_allele_vaf_single_locus",
        "NSTUTTER",
        "DSTUTTER",
        "NFSTUTTER",
        "DFSTUTTER",
        "HVAF",
        "PS",
        "PP",
        "PEAF",
        "MBP",
        "PHA",
        "PPP",
        "PODR",
        "PODHR",
        "PMAP",
        "PMLEEH",
        "PMLEEHR",
        "PMLEDR",
        "PFAH",
        "MAPLR",
        "MAPLRTP",
        "MAPLRS",
        "MAPLRTPS",
        "POMDR",
        "PSOMDR",
        "PGOMDR",
        "PSMMDR",
        "PGMMDR",
        "PMMDR",
        "NHSNPN",
        "NHSNPINDELN",
        "mle_mosaic_allele_vaf_two_loci",
        "observed_mosaic_allele_vaf_two_loci",
        "GIOAF0",
        "GIOAF1",
        "GIMLEAF0",
        "GIMLEAF1",
        "GIUAF",
        "PRHN",
        "MLEPEAF",
        "EMLEAF",
        "NALLELES_PER_STR_BP",
        "MLEUAF",
        "MLEPUAF",
        "PMB",
        "POB",
        "PGHAF",
        "PMHAF",
        "PH3AF",
        "PGD",
        "PSD",
        "PMD",
        "GQ_OR",
        "GQ_PVAL",
        "GG_OR",
        "GG_PVAL",
        "GI",
        "GIP",
        "GIQ",
        "GIQP",
        "AMP",
        "ALLSINGLE",
        "HPRHN",
        "obs_phase_state",
        "obs_hap_state",
        "obs_hap_count",
        "mle_phase_state",
        "mle_hap_state",
        "mle_hap_count",
        "NORMGERM1",
        "NORMGERM2",
        "NORMMOSAIC",
        "NORMSECOND",
        "SECISGERM",
        "obs_mut_order",
        "obs_depth_string",
        "mle_mut_order",
        "mle_depth_string",
        "PDP",
        "PGT0",
        "PGT1",
        "PMGT0",
        "PMGT1",
        "ALLELE_BARCODE_UMI",
        "ALLELE_BARCODE_MOSAIC"
    ]
    for feature_name in vcf_based_features_list:
        line += f"{vcf_based_features.get(feature_name, 'NA')},"
    line += f"{sample_name},"
    per_loci_level_features_name = [
        "average_mappability_score_k24",
        "average_mappability_score_k100",
        "inframe_single_step_prob",
        "inframe_ins_prob",
        "inframe_del_prob",
        "outframe_single_step_prob",
        "outframe_ins_prob",
        "outframe_del_prob",
        "all_samples_hSNP_INDEL_num",
        "left_flanking_context_entropy_A",
        "left_flanking_context_entropy_T",
        "left_flanking_context_entropy_C",
        "left_flanking_context_entropy_G",
        "right_flanking_context_entropy_A",
        "right_flanking_context_entropy_T",
        "right_flanking_context_entropy_C",
        "right_flanking_context_entropy_G",
        "flanking_context_GC_content",
        "STR_flanking_context_GC_content",
        "motif_length",
    ]
    for feature_name in per_loci_level_features_name:
        line += f"{per_loci_level_features_dict.get(feature_name, 'NA')},"
    hSNP_features_name = [
        "snp1_af",
        "snp2_af",
        "snp3_af",
        "nearby_snp_base1_is_ref",
        "nearby_snp_base2_is_ref",
        "nearby_snp_base3_is_ref",
        "snp_overall_mean_mapQ",
        "snp_overall_mean_baseQ",
        "snp_overall_strand_fraction",
        "dp_diff",
        "dp_diff_fraction",
        "final_chrom_point_depth_percentile",
        "final_wgs_point_depth_percentile",
        "nearby_hsnp_depth_ratio",
        "nearby_hsnp_chrom_point_median_depth_ratio",
        "nearby_hsnp_wgs_point_median_depth_ratio",
        "mapQ_diff",
        "mapQ_statistic",
        "mapQ_pvalue",
        "baseQ_diff",
        "baseQ_statistic",
        "baseQ_pvalue",
        "strand_oddsratio",
        "strand_pvalue",
        "strand_fraction_diff",
        "snp1_positive_strand_fraction",
        "snp2_positive_strand_fraction",
    ]
    for feature_name in hSNP_features_name:
        line += f"{hSNP_features_dict.get(feature_name, 'NA')},"
    sample_scattered_features = [
        "alleles_mut_type",
        "seq0_length",
        "seq1_length",
        "seq2_length",
        "muttype",
        "mutation_site_type",
        "mutation_location_type",
        "mutation_site_start_seq1",
        "mutation_site_end_seq1",
        "mutation_site_start_seq2",
        "mutation_site_end_seq2",
        # "mosaic_fraction",
        # "phasable",
        "overall_sample_after_filtered_read_dp",
        "overall_sample_filtered_out_read_dp",
        "overall_used_reads_fraction",
        "mle_mosaic_allele_vaf_single_locus",
        "used_read_fraction_in_genotyping",
        "str_padding_unfiltered_raw_depth",
        "median_str_nearby_raw_depth",
        "str_dp_diff",
        "str_dp_diff_fraction",
        "final_padding_str_depth_percentile",
        "overall_mis_num_per_bp_all_reads_mean_based_on_NM",
        "overall_mis_num_per_bp_all_reads_mean_based_on_MD",
        "germ_seq",
        "source_seq",
        "mut_seq",
        "source_is_recurrent_motif",
        "mut_is_recurrent_motif",
    ]
    for feature_name in sample_scattered_features:
        line += f"{sample_all_features_dict.get(feature_name, 'NA')},"
    str_features_list = [
        "unknown_hap_read_fraction",
        "germ_read_fraction",
        "source_read_fraction",
        "mut_read_fraction",
        "read_length",
        "germ_str_padding_length",
        "mut_str_padding_length",
        "source_str_padding_length",
        "germ_mut_str_padding_length_diff",
        "source_mut_str_padding_length_diff",
        "germ_source_str_padding_length_diff",
        "overall_mean_base_accuracy",
        "source_mut_diff_base_accuracy",
        "source_mut_base_accuracy_statistic",
        "source_mut_base_accuracy_pvalue",
        "germ_allele_left_start_diff",
        "germ_allele_right_end_diff",
        "mut_allele_left_start_diff",
        "mut_allele_right_end_diff",
        "source_allele_left_start_diff",
        "source_allele_right_end_diff",
        "germ_allele_ks_start_stat",
        "germ_allele_ks_start_p_value",
        "germ_allele_ks_end_stat",
        "germ_allele_ks_end_p_value",
        "mut_allele_ks_start_stat",
        "mut_allele_ks_start_p_value",
        "mut_allele_ks_end_stat",
        "mut_allele_ks_end_p_value",
        "source_allele_ks_start_stat",
        "source_allele_ks_start_p_value",
        "source_allele_ks_end_stat",
        "source_allele_ks_end_p_value",
        "germ_allele_chi_square_start_stat",
        "germ_allele_chi_square_start_p_value",
        "germ_allele_chi_square_end_stat",
        "germ_allele_chi_square_end_p_value",
        "mut_allele_chi_square_start_stat",
        "mut_allele_chi_square_start_p_value",
        "mut_allele_chi_square_end_stat",
        "mut_allele_chi_square_end_p_value",
        "source_allele_chi_square_start_stat",
        "source_allele_chi_square_start_p_value",
        "source_allele_chi_square_end_stat",
        "source_allele_chi_square_end_p_value",
        "used_read_num_in_feature",
        "used_read_num_in_genotyping",
        "sample_str_used_depth",
        "sample_wgs_used_depth",
        "depth_norm_factor",
        "expected_str_used_depth",
        "expected_str_used_depth_ratio",
        "raw_wgs_expected_depth_ratio",
        "nearby_raw_expected_depth_ratio",
        "raw_wgs_point_expected_depth_ratio",
        "final_padding_str_chrom_depth_percentile",
        "source_mean_baseq",
        "source_fraction_baseq_more_cutoff",
        "mut_mean_baseq",
        "mut_fraction_baseq_more_cutoff",
        "germ_allele_unique_start_coordinate_number",
        "germ_allele_unique_end_coordinate_number",
        "mut_allele_unique_start_coordinate_number",
        "mut_allele_unique_end_coordinate_number",
        "source_allele_unique_start_coordinate_number",
        "source_allele_unique_end_coordinate_number",
    ]
    for feature_name in str_features_list:
        line += f"{str_features_dict.get(feature_name, 'NA')},"
    other_phasing_related_features_list = [
        "phasing_fraction",
        "germ_phasing_fraction",
        "mut_phasing_fraction",
        "source_phasing_fraction",
        "germ_hsnp_fraction",
        "germ_dis_fraction",
        "mut_hsnp_fraction",
        "mut_dis_fraction",
        "source_hsnp_fraction",
        "source_dis_fraction",
        "all_dis_fraction",
    ]

    for feature_name in other_phasing_related_features_list:
        line += (
            f"{other_phasing_related_features_dict.get(feature_name, 'NA')},"
        )
    pysam_bed_row_used_features = [
        "AAD",  # HACK: compatible with previous version
        "uncallable_num",
        "uncallable_frac",
        "recurrent_num",
        "recurrent_fraction",
        "sample_num",
        "germ_popAF",
        "source_popAF",
        "mosaic_popAF",
        "muttype_hf",
        "external_germ_popAF",
        "external_source_popAF",
        "external_mosaic_popAF",
        "allele_length_dp",
        "ref_allele_length_padding_flanking",
        "median_genotyping_dp",
    ]
    for feature_name in pysam_bed_row_used_features:
        line += f"{pysam_bed_row_dict.get(feature_name, 'NA')},"
    germ_mut_feature_list = [
        "overall_mean_mapQ",
        "germ_mut_diff_mapQ",
        "germ_mut_mapQ_statistic",
        "germ_mut_mapQ_pvalue",
        "overall_mean_mean_baseQ",
        "germ_mut_diff_mean_baseQ",
        "germ_mut_mean_baseQ_statistic",
        "germ_mut_mean_baseQ_pvalue",
        "overall_mean_str_mis_num",
        "germ_mut_diff_mean_str_mis_num",
        "germ_mut_str_mis_num_statistic",
        "germ_mut_str_mis_num_pvalue",
        "overall_mean_str_mis_fraction",
        "germ_mut_diff_mean_str_mis_fraction",
        "germ_mut_str_mis_fraction_statistic",
        "germ_mut_str_mis_fraction_pvalue",
        "overall_mean_indel_size",
        "germ_mut_diff_mean_indel_size",
        "germ_mut_indel_size_statistic",
        "germ_mut_indel_size_pvalue",
        "overall_mean_indel_size_per_motif_length",
        "germ_mut_diff_mean_indel_size_per_motif_length",
        "germ_mut_indel_size_per_motif_length_statistic",
        "germ_mut_indel_size_per_motif_length_pvalue",
        "overall_mean_flanking_sbs_num_per_bp",
        "germ_mut_diff_mean_flanking_sbs_num_per_bp",
        "germ_mut_flanking_sbs_num_per_bp_statistic",
        "germ_mut_flanking_sbs_num_per_bp_pvalue",
        "overall_mean_flanking_indel_num_per_bp",
        "germ_mut_diff_mean_flanking_indel_num_per_bp",
        "germ_mut_flanking_indel_num_per_bp_statistic",
        "germ_mut_flanking_indel_num_per_bp_pvalue",
        "overall_mean_flanking_bps_num",
        "germ_mut_diff_mean_flanking_bps_num",
        "germ_mut_flanking_bps_num_statistic",
        "germ_mut_flanking_bps_num_pvalue",
        "overall_mean_read_pcr_cycle_percentage_sum",
        "germ_mut_diff_mean_read_pcr_cycle_percentage_sum",
        "germ_mut_read_pcr_cycle_percentage_sum_statistic",
        "germ_mut_read_pcr_cycle_percentage_sum_pvalue",
        "overall_mean_read_pcr_start_num",
        "germ_mut_diff_mean_read_pcr_start_num",
        "germ_mut_read_pcr_start_num_statistic",
        "germ_mut_read_pcr_start_num_pvalue",
        "overall_mean_read_pcr_cycle_sum",
        "germ_mut_diff_mean_read_pcr_cycle_sum",
        "germ_mut_read_pcr_cycle_sum_statistic",
        "germ_mut_read_pcr_cycle_sum_pvalue",
        "overall_non_second_fraction",
        "germ_non_second_fraction",
        "mut_non_second_fraction",
        "germ_mut_diff_non_second_fraction",
        "overall_non_supplementary_fraction",
        "germ_non_supplementary_fraction",
        "mut_non_supplementary_fraction",
        "germ_mut_diff_non_supplementary_fraction",
        "overall_non_duplicate_fraction",
        "germ_non_duplicate_fraction",
        "mut_non_duplicate_fraction",
        "germ_mut_diff_non_duplicate_fraction",
        "overall_non_unmapped_fraction",
        "germ_non_unmapped_fraction",
        "mut_non_unmapped_fraction",
        "germ_mut_diff_non_unmapped_fraction",
        "overall_non_qcfail_fraction",
        "germ_non_qcfail_fraction",
        "mut_non_qcfail_fraction",
        "germ_mut_diff_non_qcfail_fraction",
        "overall_non_clipped_fraction",
        "germ_non_clipped_fraction",
        "mut_non_clipped_fraction",
        "germ_mut_diff_non_clipped_fraction",
        "overall_non_flanking_indel_fraction",
        "germ_non_flanking_indel_fraction",
        "mut_non_flanking_indel_fraction",
        "germ_mut_diff_non_flanking_indel_fraction",
        "overall_non_spanning_str_flanking_indel_fraction",
        "germ_non_spanning_str_flanking_indel_fraction",
        "mut_non_spanning_str_flanking_indel_fraction",
        "germ_mut_diff_spanning_str_flanking_indel_fraction",
        "overall_strand_fraction",
        "germ_strand_fraction",
        "mut_strand_fraction",
        "germ_mut_abs_diff_strand_fraction",
        "germ_mut_strand_oddsratio",
        "germ_mut_strand_pvalue",
        "overall_orientation_fraction",
        "germ_orientation_fraction",
        "mut_orientation_fraction",
        "germ_mut_abs_diff_orientation_fraction",
        "germ_mut_orientation_oddsratio",
        "germ_mut_orientation_pvalue",
        "overall_proper_pair_fraction",
        "germ_proper_pair_fraction",
        "mut_proper_pair_fraction",
        "germ_mut_diff_proper_pair",
        "overall_indel",
        "overall_mis",
        "overall_errors",
        "germ_mut_diff_indel",
        "germ_mut_diff_mis",
        "germ_mut_diff_errors",
        "germ_indel",
        "germ_mis",
        "germ_errors",
        "mut_indel",
        "mut_mis",
        "mut_errors",
        "exact_match_odds_ratio",
        "exact_match_p_value",
        "left_flanking_baseq_mean",
        "right_flanking_baseq_mean",
        "flanking_abs_diff_baseq_mean",
        "flanking_baseq_mean_statistic",
        "flanking_baseq_mean_p",
        "left_flanking_sbs_num_mean",
        "right_flanking_sbs_num_mean",
        "flanking_abs_diff_sbs_num_mean",
        "flanking_sbs_num_mean_statistic",
        "flanking_sbs_num_mean_p",
        "germ_overall_baseQ_mean",
        "germ_overall_baseQ_more_fraction",
        "germ_mapQ_mean",
        "germ_mapQ_more_fraction",
        "mut_overall_baseQ_mean",
        "mut_overall_baseQ_more_fraction",
        "mut_mapQ_mean",
        "mut_mapQ_more_fraction",
        "overall_mean_temp_length",
        "germ_mut_diff_mean_temp_length",
        "germ_mut_temp_length_statistic",
        "germ_mut_temp_length_pvalue",
        "left_flanking_bps_num_mut_feature_mean",
        "right_flanking_bps_num_mut_feature_mean",
        "left_flanking_bps_num_mut_feature_less_five_fraction",
        "right_flanking_bps_num_mut_feature_less_five_fraction",
        "left_flanking_bps_num_mut_feature_sd",
        "right_flanking_bps_num_mut_feature_sd",
        "left_flanking_bps_num_mut_feature_mad",
        "right_flanking_bps_num_mut_feature_mad",
        "germ_mean_flanking_sbs_num_per_bp",
        "germ_flanking_sbs_num_per_bp_more_fraction",
        "mut_mean_flanking_sbs_num_per_bp",
        "mut_flanking_sbs_num_per_bp_more_fraction",
    ]
    for feature_name in germ_mut_feature_list:
        line += f"{germ_mut_feature_dict.get(feature_name, 'NA')},"
    germ_source_feature_list = [
        i + "_germ_source" for i in germ_mut_feature_list
    ]
    for feature_name in germ_source_feature_list:
        line += f"{germ_source_feature_dict.get(feature_name, 'NA')},"
    line = line[:-1] + "\n"
    return line


def task_generator(pysam_bed, per_locus_parse_params):
    for arg in pysam_bed.fetch():
        yield (per_locus_parse_params, arg)


def task_generator_given_region(
    pysam_bed,
    per_locus_parse_params,
    chrom,
    start,
    end,
):
    for arg in pysam_bed.fetch(chrom, start, end):
        yield (per_locus_parse_params, arg)


def features_extract_prepare(parsed_options):
    genome_wide_info_dict = {}
    features_extract_all_params = {}
    # features_extract_all_params["parsed_options"] = parsed_options
    features_extract_all_params[
        "mappability_file"
    ] = parsed_options.mappability_annotation
    ref_file = parsed_options.reference_genome
    if "37" in ref_file:
        genome_wide_info_dict["reference_version"] = "GRCH37"
    elif "38" in ref_file:
        genome_wide_info_dict["reference_version"] = "GRCH38"
    elif "19" in ref_file:
        genome_wide_info_dict["reference_version"] = "hg19"
    # python_command = ' '.join([sys.executable] + sys.argv)
    # logging.info('Command: %s', python_command)
    genome_wide_info_dict["ref_fa"] = parsed_options.reference_genome
    metadata = pd.read_csv(parsed_options.metadata, sep=",", index_col=0)
    if parsed_options.sample_name is not None:
        metadata = metadata[metadata['Sample Name']==parsed_options.sample_name]
    bam_files = metadata['bam_path']
    wgs_mean_depth = metadata['mosdepth_wgs_mean_depth']
    str_used_mean_depth = metadata['used_genotyping_str_mean_depth']
    features_extract_all_params["bam_files"] = bam_files
    features_extract_all_params["wgs_mean_depth"] = wgs_mean_depth
    features_extract_all_params["str_used_mean_depth"] = str_used_mean_depth
    features_extract_all_params["results_vcf_file"] = parsed_options.variant_info
    sample_name_list = metadata['Sample Name'].to_list()
    features_extract_all_params["bam_name"] = sample_name_list
    genome_wide_info_dict["sample_name_list"] = sample_name_list
    log_dir = parsed_options.output_dir
    result_dir = parsed_options.output_dir
    os.system("mkdir -p " + parsed_options.output_dir)
    os.system("mkdir -p " + log_dir)
    os.system("mkdir -p " + result_dir)
    # bed_name = parsed_options.bed_panel.split("/")[-1].split(".")[0]
    if parsed_options.chrom:
        uid = f"{parsed_options.sample_name}_{parsed_options.chrom}_{parsed_options.start}_{parsed_options.end}"
    else:
        uid = parsed_options.sample_name
    fail_file = (
        log_dir
        + "/"
        + uid
        + "_features_extract_fail_loci.log"
    )
    features_extract_all_params["output_features_file"] = (
        result_dir
        + "/"
        + uid
        + ".csv"
    )
    features_extract_all_params["fail_file"] = fail_file
    logfile = log_dir + "/" + uid + ".log"
    features_extract_all_params["log_file"] = logfile
    features_extract_all_params[
        "stutter_file"
    ] = parsed_options.stutter_model_in
    fasta_chromosome_name = myutils.check_reference_fasta_name(
        parsed_options.reference_genome
    )
    bed_chromosome_name = myutils.check_reference_bed_name(
        parsed_options.bed_panel
    )
    if "chr" in fasta_chromosome_name and "chr" not in bed_chromosome_name:
        add_chr_character = True
        remove_chr_character = False
    elif "chr" not in fasta_chromosome_name and "chr" in bed_chromosome_name:
        remove_chr_character = True
        add_chr_character = False
    else:
        add_chr_character = False
        remove_chr_character = False
    if "chr" in fasta_chromosome_name:
        all_chroms_list = ["chr" + str(i) for i in range(1, 23)]
    else:
        all_chroms_list = [str(i) for i in range(1, 23)]
    pysam_fasta = pysam.FastaFile(ref_file)
    all_samples_median_depth_dict = {}
    all_samples_point_depth_dict = {}
    all_samples_all_chroms_point_depth_dict = {}
    for bam_name, bam in zip(sample_name_list, bam_files):
        if "bam" in bam:
            pysam_bam = pysam.AlignmentFile(bam, "rb")
        else:
            pysam_bam = pysam.AlignmentFile(
                bam, "rc", reference_filename=ref_file
            )
        sample_all_chroms_depth_list = []
        all_samples_all_chroms_point_depth_dict[bam_name] = {}
        for chrom_id in all_chroms_list:
            chrom_depth_list = get_chrom_depth(
                chrom_id, pysam_fasta, pysam_bam
            )
            sample_all_chroms_depth_list.extend(chrom_depth_list)
            all_samples_all_chroms_point_depth_dict[bam_name][
                chrom_id
            ] = chrom_depth_list
        all_samples_point_depth_dict[bam_name] = sample_all_chroms_depth_list
        all_samples_median_depth_dict[bam_name] = np.median(
            sample_all_chroms_depth_list
        )
    features_extract_all_params[
        "all_samples_median_depth_dict"
    ] = all_samples_median_depth_dict
    features_extract_all_params[
        "all_samples_point_depth_dict"
    ] = all_samples_point_depth_dict
    features_extract_all_params[
        "all_samples_all_chroms_point_depth_dict"
    ] = all_samples_all_chroms_point_depth_dict
    features_extract_all_params["add_chr_character"] = add_chr_character
    features_extract_all_params["remove_chr_character"] = remove_chr_character
    features_extract_all_params["loglevel"] = parsed_options.loglevel
    log_level = getattr(logging, parsed_options.loglevel.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {parsed_options.loglevel}")
    logger_features_extract = logging.getLogger("FeaturesExtract")
    logger_features_extract.setLevel(log_level)
    features_extract_all_params["log_to_file"] = parsed_options.log_to_file
    if parsed_options.log_to_file:
        logger_features_extract_file_handler = myutils.LockedFileHandler(
            logfile
        )
    else:
        logger_features_extract_file_handler = logging.StreamHandler()
    # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger_features_extract_file_handler.setFormatter(formatter)
    if not logger_features_extract.handlers:
        logger_features_extract.addHandler(
            logger_features_extract_file_handler
        )
    # features_extract_all_params[
    #     "logger_features_extract"
    # ] = logger_features_extract
    logger_features_extract.info(
        "FeatureExtractor: Extract Features From BAM files and Other files"
    )
    logger_features_extract.info("Version: 1.0")
    logger_features_extract.info(
        "Start time: %s",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
    )
    # STR_refpanel_bed = pysam.TabixFile(parsed_options.bed_panel)
    # features_extract_all_params["STR_refpanel_bed"] = STR_refpanel_bed
    sex = metadata.iloc[0, 1]  # HACK: don't consider sex chromosome currently
    features_extract_all_params["sex"] = sex
    features_extract_all_params["fasta_file"] = ref_file
    return genome_wide_info_dict, features_extract_all_params


def run_extract_features(
    metadata,
    reference_genome,
    bed_panel,
    output_dir,
    # Optional inputs
    sample_name=None,
    mode="RNA-seq",
    stutter_model="EM",
    variant_info=None,
    output_prefix=None,
    mappability_annotation=None,
    gene_model=None,
    stutter_model_in=None,
    phasing=None,
    allele_imbalance=None,
    # Optional outputs
    loglevel="INFO",
    log_to_file=True,
    stutter_model_out=None,
    vcf_out=None,
    viz_out=None,
    # Other options
    chrom="",
    start=0,
    end=1000000000,
    threads=1,
    verbose=0,
    debug=False,
    # 控制是否单线程（调试用）
    single_thread=False,
):
    # 构造 Namespace 对象，模拟 argparse 解析结果
    args = argparse.Namespace(
        metadata=metadata,
        reference_genome=reference_genome,
        bed_panel=bed_panel,
        output_dir=output_dir,
        sample_name=sample_name,
        mode=mode,
        stutter_model=stutter_model,
        variant_info=variant_info,
        output_prefix=output_prefix,
        mappability_annotation=mappability_annotation,
        gene_model=gene_model,
        stutter_model_in=stutter_model_in,
        phasing=phasing,
        allele_imbalance=allele_imbalance,
        loglevel=loglevel,
        log_to_file=log_to_file,
        stutter_model_out=stutter_model_out,
        vcf_out=vcf_out,
        viz_out=viz_out,
        chrom=chrom,
        start=start,
        end=end,
        threads=threads if threads > 0 else os.cpu_count(),
        verbose=verbose,
        debug=debug,
    )


    # 模拟 Python 命令（用于日志记录）
    python_command = f"python -m extract_features [function call]"


    # 调用 prepare 函数（假设已定义）
    genome_wide_info_dict, features_extract_all_params = features_extract_prepare(args)
    genome_wide_info_dict["commands"] = python_command


    # 打开 BED 注释文件（需 tabix 索引）
    STR_refpanel_bed = pysam.TabixFile(args.bed_panel)


    # 设置日志
    logger_features_extract = logger_config.feature_extract_logger(
        features_extract_all_params["loglevel"],
        features_extract_all_params["log_to_file"],
        features_extract_all_params["log_file"],
    )


    start_time = time.time()
    logger_features_extract.info("Start time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    logger_features_extract.info("Parameters: %s", args)
    logger_features_extract.info("Command: %s", python_command)


    MAX_THREADS = args.threads


    try:
        if args.chrom == "":
            total_tasks = sum(1 for _ in STR_refpanel_bed.fetch())
            STR_refpanel_bed = pysam.TabixFile(args.bed_panel)  # 重新打开以重置迭代器


            if single_thread or MAX_THREADS == 1:
                for task in task_generator(STR_refpanel_bed, features_extract_all_params):
                    main_per_locus_feature_extract(task[0], task[1])
            else:
                with Manager() as manager:
                    with Pool(processes=MAX_THREADS) as pool:
                        progress_bar = tqdm(total=total_tasks)
                        def update_progress(_):
                            progress_bar.update(1)


                        for task in task_generator(STR_refpanel_bed, features_extract_all_params):
                            pool.apply_async(
                                main_per_locus_feature_extract,
                                args=task,
                                callback=update_progress,
                                error_callback=lambda e: logger_features_extract.error(str(e))
                            )
                        pool.close()
                        pool.join()
                        progress_bar.close()
        else:
            total_tasks = sum(1 for _ in STR_refpanel_bed.fetch(args.chrom, args.start, args.end))
            STR_refpanel_bed = pysam.TabixFile(args.bed_panel)  # 重置


            if single_thread or MAX_THREADS == 1:
                for task in task_generator_given_region(
                    STR_refpanel_bed,
                    features_extract_all_params,
                    args.chrom,
                    args.start,
                    args.end
                ):
                    main_per_locus_feature_extract(task[0], task[1])
            else:
                with Manager() as manager:
                    with Pool(processes=MAX_THREADS) as pool:
                        progress_bar = tqdm(total=total_tasks)
                        def update_progress(_):
                            progress_bar.update(1)


                        for task in task_generator_given_region(
                            STR_refpanel_bed,
                            features_extract_all_params,
                            args.chrom,
                            args.start,
                            args.end
                        ):
                            pool.apply_async(
                                main_per_locus_feature_extract,
                                args=task,
                                callback=update_progress,
                                error_callback=lambda e: logger_features_extract.error(str(e))
                            )
                        pool.close()
                        pool.join()
                        progress_bar.close()


        logger_features_extract.info("Finish all STR loci features extraction.")
    except Exception as e:
        logger_features_extract.error("Error during feature extraction: %s", str(e))
        traceback.print_exc()
        raise
    finally:
        end_time = time.time()
        logger_features_extract.info("End time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
        total_time = end_time - start_time
        logger_features_extract.info("Total time spent: %.2f seconds", total_time)
        STR_refpanel_bed.close()


def cmd_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="FeatureExtract: Extract Features From BAM FILEs",
        prog="extract_features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="%(prog)s [options]",
        allow_abbrev=True,
        add_help=False,
        conflict_handler="error",
    )
    help_args = parser.add_argument_group("Help arguments")
    help_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")


    required_args = parser.add_argument_group("Required arguments")
    required_args.add_argument("-i", "--metadata", required=True, help="Metadata (CSV file)")
    required_args.add_argument("-r", "--reference_genome", required=True, help="Reference genome (FASTA file)")
    required_args.add_argument("-b", "--bed_panel", required=True, help="STR genome annotation (BED file)")
    required_args.add_argument("-o", "--output_dir", required=True, help="Output path")
    required_args.add_argument("-m", "--mode", choices=["RNA-seq", "WGS", "WES"], default="RNA-seq", help="Sequencing type")
    required_args.add_argument("-u", "--stutter_model", choices=["Common", "1KG", "GTEx", "EM"], default="EM", help="Stutter model")
    required_args.add_argument("-vi", "--variant_info", help="Variant info (VCF)")
    required_args.add_argument("-op", "--output_prefix", help="Prefix of output feature file")


    optional_input_args = parser.add_argument_group("Optional input arguments")
    parser.add_argument("-s", "--sample_name", type=str, help="Sample name")
    optional_input_args.add_argument("-ma", "--mappability_annotation", help="Mappability (Wig)")
    optional_input_args.add_argument("-g", "-gene_model", help="Gene model (GTF/GFF)")
    optional_input_args.add_argument("-si", "--stutter_model_in", help="Input stutter model")
    optional_input_args.add_argument("-p", "--phasing", help="Phased SNP VCF")
    optional_input_args.add_argument("-a", "--allele_imbalance", help="Allele imbalance info")


    optional_output_args = parser.add_argument_group("Optional output arguments")
    optional_output_args.add_argument("-ll", "--loglevel", default="INFO", choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    optional_output_args.add_argument("-lf", "--log_to_file", action="store_true", help="Output log to file")
    optional_output_args.add_argument("-q", "--stutter_model_out", help="Output stutter model")
    optional_output_args.add_argument("-f", "--vcf_out", help="Output VCF")
    optional_output_args.add_argument("-z", "--viz_out", help="Visualization output")


    other_optional_args = parser.add_argument_group("Other optional arguments")
    other_optional_args.add_argument("-c", "--chrom", type=str, default="", help="Selected chromosome")
    other_optional_args.add_argument("-s", "--start", type=int, default=0, help="Start position")
    other_optional_args.add_argument("-e", "--end", type=int, default=1000000000, help="End position")
    other_optional_args.add_argument("-t", "--threads", type=int, default=1, help="Number of threads (-1 for all)")
    other_optional_args.add_argument("-v", "--verbose", action="count", default=0, help="Verbose level")
    other_optional_args.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    other_optional_args.add_argument("-n", "--versions", action="version", version=f"%(prog)s version '{__VERSION__}'")


    return parser.parse_args(args)



def main():
    args = cmd_args()
    run_extract_features(**vars(args))


if __name__ == "__main__":
    main()