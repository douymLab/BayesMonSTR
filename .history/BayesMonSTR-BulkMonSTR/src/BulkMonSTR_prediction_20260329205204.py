# referred to publications/sampling_hg002/04.extract_random_forest_iput_from_extract_features.py
# Notice that whether to do MapQ Normalization(BWA 60 Novaalign 70)

import pandas as pd
import argparse
from scipy import stats
import pickle
import numpy as np
import pysam
import scipy
import myutils
import adjust_vcf
import config_params
import final_hard_filter_config_params

parser = argparse.ArgumentParser(description="BulkMonSTR prediction")

# 定义命令行参数
parser.add_argument("-i", "--input_file", help="Input file")
parser.add_argument("-o", "--output_file", help="Output file")
parser.add_argument(
    "-l",
    "--label",
    default="both",
    help="mismatch or indel or both",
    choices=["mismatch", "indel", "both"],
)
parser.add_argument("-mm", "--model_mis", help="mismatch model")
parser.add_argument("-mid", "--model_indel", help="indel model")
parser.add_argument(
    "-r", "--reference_fasta", help="Reference genome FASTA file"
)
parser.add_argument(
    "-m",
    "--mode",
    help="Random forest prediction mode or Hard filter prediction mode",
    default="both",
    choices=["rf", "hard_filter", "both", "either", "all"],
)
parser.add_argument(
    "-hnf",
    "--het_no_filter",
    action="store_true",
    help=(
        "No filter out Heterozygous for compatible with clonal mosaic"
        " mutations with VAF approximately equal to 0.5"
    ),
)
parser.add_argument(
    "-ss",
    "--save_sort_features",
    action="store_true",
    help="Save sorted features",
)
parser.add_argument(
    "-sp",
    "--save_prediction_features",
    action="store_true",
    help="Save prediction features",
)


# 解析参数
args = parser.parse_args()
mis_or_indel = args.label
input_file = args.input_file
output_file = args.output_file
mismatch_model = args.model_mis
indel_model = args.model_indel
reference_fasta = args.reference_fasta
mode = args.mode
het_no_filter = args.het_no_filter
save_sort_features = args.save_sort_features
save_prediction_features = args.save_prediction_features

ALLELE_EXTRACT = config_params.ALLELE_EXTRACT
PADDING_BPS = ALLELE_EXTRACT["PADDING_BPS"]  # 5 # 10 # revise_padding
ALLOW_NUMPY_MIN_VALUE = np.finfo(float).tiny  # 1e-300

if het_no_filter:
    FINAL_CUTOFF_PARAMS = (
        final_hard_filter_config_params.FINAL_HARD_FILTER_CONFIG_PARAMS_COMPATIBLE_CLONAL_HET
    )
else:
    FINAL_CUTOFF_PARAMS = (
        final_hard_filter_config_params.FINAL_HARD_FILTER_CONFIG_PARAMS
    )

MUTATION_SIZE_MAX = FINAL_CUTOFF_PARAMS["MUTATION_SIZE_MAX"]
MUTATION_PERIOD_MAX = FINAL_CUTOFF_PARAMS["MUTATION_PERIOD_MAX"]
MF_HOM2HET_HET2HET_MAX = FINAL_CUTOFF_PARAMS["MF_HOM2HET_HET2HET_MAX"]
MF_HOM2HET_HET2HET_MIN = FINAL_CUTOFF_PARAMS["MF_HOM2HET_HET2HET_MIN"]
OBSERVED_MOSAIC_ALLELE_VAF_SINGLE_LOCUS_MIN = FINAL_CUTOFF_PARAMS[
    "OBSERVED_MOSAIC_ALLELE_VAF_SINGLE_LOCUS_MIN"
]
OBSERVED_MOSAIC_ALLELE_VAF_SINGLE_LOCUS_MAX = FINAL_CUTOFF_PARAMS[
    "OBSERVED_MOSAIC_ALLELE_VAF_SINGLE_LOCUS_MAX"
]
NO_NORM_LEN_REVISE_DEPTH_RATIO_MIN = FINAL_CUTOFF_PARAMS[
    "NO_NORM_LEN_REVISE_DEPTH_RATIO_MIN"
]
NO_NORM_LEN_REVISE_DEPTH_RATIO_MAX = FINAL_CUTOFF_PARAMS[
    "NO_NORM_LEN_REVISE_DEPTH_RATIO_MAX"
]
MUTANT_DP_MIN = FINAL_CUTOFF_PARAMS["MUTANT_DP_MIN"]
USED_READ_NUM_IN_GENOTYPING_MIN = FINAL_CUTOFF_PARAMS[
    "USED_READ_NUM_IN_GENOTYPING_MIN"
]
POST_HOMOPOLYMER_MISMATCH = FINAL_CUTOFF_PARAMS["POST_HOMOPOLYMER_MISMATCH"]
MUTATION_TYPE_INCLUDE = FINAL_CUTOFF_PARAMS["MUTATION_TYPE_INCLUDE"]
FLANKING_MIS_EXCLUDE = FINAL_CUTOFF_PARAMS["FLANKING_MIS_EXCLUDE"]
INDEL_POPAF_MAX = FINAL_CUTOFF_PARAMS["INDEL_POPAF_MAX"]
MISMATCH_POPAF_MAX = FINAL_CUTOFF_PARAMS["MISMATCH_POPAF_MAX"]
DECISION_EXCLUDE_PATTERNS = FINAL_CUTOFF_PARAMS["DECISION_EXCLUDE_PATTERNS"]
MUT_MEAN_BASEQ_MIN = DECISION_EXCLUDE_PATTERNS["BASEQ_HIGH_BASEQ_FRAC"][
    "BASEQ_MIN"
]
MUT_FRACTION_BASEQ_MORE_CUTOFF = DECISION_EXCLUDE_PATTERNS[
    "BASEQ_HIGH_BASEQ_FRAC"
]["HIGH_BASEQ_FRAC_MIN"]
USED_READ_FRACTION_IN_GENOTYPING_MIN = FINAL_CUTOFF_PARAMS[
    "USED_READ_FRACTION_IN_GENOTYPING_MIN"
]
UNKNOWN_HAP_READ_FRACTION_MAX = FINAL_CUTOFF_PARAMS[
    "UNKNOWN_HAP_READ_FRACTION_MAX"
]
OVERALL_NON_FLANKING_INDEL_FRACTION_MIN = FINAL_CUTOFF_PARAMS[
    "OVERALL_NON_FLANKING_INDEL_FRACTION_MIN"
]
FLANKING_ABS_DIFF_BASEQ_MEAN_MAX = FINAL_CUTOFF_PARAMS[
    "FLANKING_ABS_DIFF_BASEQ_MEAN_MAX"
]
MUT_NON_CLIPPED_FRACTION_GERM_SOURCE_MIN = FINAL_CUTOFF_PARAMS[
    "MUT_NON_CLIPPED_FRACTION_GERM_SOURCE_MIN"
]
MUT_NON_CLIPPED_FRACTION_MIN = FINAL_CUTOFF_PARAMS[
    "MUT_NON_CLIPPED_FRACTION_MIN"
]
GERM_NON_CLIPPED_FRACTION_MIN = FINAL_CUTOFF_PARAMS[
    "GERM_NON_CLIPPED_FRACTION_MIN"
]
MUT_INDEL_GERM_SOURCE_MAX = FINAL_CUTOFF_PARAMS["MUT_INDEL_GERM_SOURCE_MAX"]
MUT_INDEL_MAX = FINAL_CUTOFF_PARAMS["MUT_INDEL_MAX"]
GERM_INDEL_MAX = FINAL_CUTOFF_PARAMS["GERM_INDEL_MAX"]
MUT_NON_FLANKING_INDEL_FRACTION_GERM_SOURCE_MIN = FINAL_CUTOFF_PARAMS[
    "MUT_NON_FLANKING_INDEL_FRACTION_GERM_SOURCE_MIN"
]
MUT_NON_FLANKING_INDEL_FRACTION_MIN = FINAL_CUTOFF_PARAMS[
    "MUT_NON_FLANKING_INDEL_FRACTION_MIN"
]
RECURRENT_CUTOFF = FINAL_CUTOFF_PARAMS["RECURRENT_CUTOFF"]
ONLY_HOMHET = FINAL_CUTOFF_PARAMS["ONLY_HOMHET"]
FLANKING_BPS_MEAN_MIN = FINAL_CUTOFF_PARAMS["FLANKING_BPS_MEAN_MIN"]
FLANKING_BPS_LESS_FIVE_FRACTION_MAX = FINAL_CUTOFF_PARAMS[
    "FLANKING_BPS_LESS_FIVE_FRACTION_MAX"
]
COORDINATE_NUMBER_MIN = FINAL_CUTOFF_PARAMS["COORDINATE_NUMBER_MIN"]
FLANKING_BPS_SD_MIN = FINAL_CUTOFF_PARAMS["FLANKING_BPS_SD_MIN"]
FLANKING_BPS_MAD_MIN = FINAL_CUTOFF_PARAMS["FLANKING_BPS_MAD_MIN"]
HIGHER_THAN_STUTTER_ERROR_FILTER = FINAL_CUTOFF_PARAMS[
    "HIGHER_THAN_STUTTER_ERROR_FILTER"
]
VAF_CORRECTION = FINAL_CUTOFF_PARAMS["VAF_CORRECTION"]
CALLABLE_NUM_MIN = FINAL_CUTOFF_PARAMS["CALLABLE_NUM_MIN"]
CALLABLE_FRACTION_MIN = FINAL_CUTOFF_PARAMS["CALLABLE_FRACTION_MIN"]
RECURRENT_NUMBER_MAX = FINAL_CUTOFF_PARAMS["RECURRENT_NUMBER_MAX"]
MUTATION_SITE_TYPE_EXCLUDE = FINAL_CUTOFF_PARAMS["MUTATION_SITE_TYPE_EXCLUDE"]
RECURRENT_FRACTION_INDEL_MAX = FINAL_CUTOFF_PARAMS[
    "RECURRENT_FRACTION_INDEL_MAX"
]
RECURRENT_FRACTION_MISMATCH_MAX = FINAL_CUTOFF_PARAMS[
    "RECURRENT_FRACTION_MISMATCH_MAX"
]
MUT_MEAN_BASEQ_MIN_SINGLE = FINAL_CUTOFF_PARAMS["MUT_MEAN_BASEQ_MIN_SINGLE"]
STRAND_FRAC_MAX = FINAL_CUTOFF_PARAMS["STRAND_FRAC_MAX"]
POST_HOMOPOLYMER_BASEQ_MIN = FINAL_CUTOFF_PARAMS["POST_HOMOPOLYMER_BASEQ_MIN"]
POST_HOMOPOLYMER_BASEQ_FRAC_MIN = FINAL_CUTOFF_PARAMS[
    "POST_HOMOPOLYMER_BASEQ_FRAC_MIN"
]
POP_AF_MAX = FINAL_CUTOFF_PARAMS["POP_AF_MAX"]
POPAF_AF_INDEL_POPAF_MIN = FINAL_CUTOFF_PARAMS["DECISION_EXCLUDE_PATTERNS"][
    "POPAF_AF_INDEL"
]["POPAF_MIN"]
POPAF_AF_INDEL_AF_MIN = FINAL_CUTOFF_PARAMS["DECISION_EXCLUDE_PATTERNS"][
    "POPAF_AF_INDEL"
]["AF_MIN"]
POPAF_AF_MISMATCH_POPAF_MIN = FINAL_CUTOFF_PARAMS["DECISION_EXCLUDE_PATTERNS"][
    "POPAF_AF_MISMATCH"
]["POPAF_MIN"]
POPAF_AF_MISMATCH_AF_MIN = FINAL_CUTOFF_PARAMS["DECISION_EXCLUDE_PATTERNS"][
    "POPAF_AF_MISMATCH"
]["AF_MIN"]
PHASE_HARD_FILTER = FINAL_CUTOFF_PARAMS["PHASE_HARD_FILTER"]
SAMPLE_NUM_MIN = FINAL_CUTOFF_PARAMS["SAMPLE_NUM_MIN"]
DP_RATIO_AF_AF_MIN = FINAL_CUTOFF_PARAMS["DECISION_EXCLUDE_PATTERNS"][
    "DP_RATIO_AF"
]["AF_MIN"]
DP_RATIO_AF_DP_RATIO_MIN = FINAL_CUTOFF_PARAMS["DECISION_EXCLUDE_PATTERNS"][
    "DP_RATIO_AF"
]["DP_RATIO_MIN"]
MUTATIONSIZE_AF_AF_MIN = FINAL_CUTOFF_PARAMS["DECISION_EXCLUDE_PATTERNS"][
    "MUTATIONSIZE_AF"
]["AF_MIN"]
MUTATIONSIZE_AF_MUTATIONSIZE_MIN = FINAL_CUTOFF_PARAMS[
    "DECISION_EXCLUDE_PATTERNS"
]["MUTATIONSIZE_AF"]["MUTATIONSIZE_MIN"]
MODEL_VAF_CORRECTION = FINAL_CUTOFF_PARAMS["MODEL_VAF_CORRECTION"]
MAQ_MEAN_MIN = FINAL_CUTOFF_PARAMS["MAQ_MEAN_MIN"]
HIGH_MAPQ_FRAC_MIN = FINAL_CUTOFF_PARAMS["HIGH_MAPQ_FRAC_MIN"]
MISMATCHES_NUM_PER_BP_PER_READ_MAX = FINAL_CUTOFF_PARAMS[
    "MISMATCHES_NUM_PER_BP_PER_READ_MAX"
]
PER_ALLELE_MIN_DP = FINAL_CUTOFF_PARAMS["PER_ALLELE_MIN_DP"]
NORM1_AS_HET = True  # change NORM1 as Het in homhet or hethet, hethet 概率最大值为 NORM1, homhet 假如 GT Het 则 概率最大值为 NORM1，否则 概率最小值为 NORM1，GermlineHet 时,赋值 NORM1 0.5 NORM2 0 Mosaic 0.5 Second 设置为多少设置为不变目前?
# Test in features distribution
MISMATCH_CATE = ["SNV", "noncontinuous_MNV", "continuous_MNV"]
INDEL_CATE = [
    "Insertion",
    "Deletion",
    "Insertion_SNV",
    "Deletion_SNV",
    "Insertion_MNV",
    "Deletion_MNV",
]


germ_mut = [
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
source_mut = []
for fea in germ_mut:
    source_mut.append(fea + "_germ_source")

features_1 = [
    "chrom",
    "STRSTART",
    "STREND",
    "MOTIF_length",
    "PERIOD",
    "str_id",
    "MOTIF",
    "GT",
    # "GT1",
    "MGT",
    # "MGT1",
    "MP",
    "CMP",
    "variant_type",
    "frame",
    "VCF_mosaic_fraction",
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
    "EAF0",
    "EAF1",
    "EAF2",
    "EAF3",
    "observed_mosaic_allele_vaf_single_locus",
    "NSTUTTER",  # stutter allele num
    "DSTUTTER",  # stutter allele depth
    "NFSTUTTER",  # stutter allele num fraction
    "DFSTUTTER",  # stutter allele depth fraction
    "HVAF",  # hSNP af
    "PS",  # phase location
    "PP",  # phase posterior
    "PEAF0",
    "PEAF1",
    "PEAF2",
    "PEAF3",
    # "MLEPEAF0",
    # "MLEPEAF1",
    # "MLEPEAF2",
    # "MLEPEAF3",
    "MBP",  # mut base pair
    "PHA",  # Whether phase or not
    "PPP",  # phase proportion
    "PODR",  # observed_discordant_reads_rate
    "PODHR",  # observed_discordant_hap_rate
    "PMAP",  # phased_mosaic_allele_proportion
    "PMLEEH",  # mle_hap_number (>1)
    "PMLEEHR",  # mle_hap_number / mle_all_hap_number
    "PMLEDR",  # discordant_mle_rate
    "PFAH",  # filter_hap_number (>1)
    "MAPLR",  # Don't use i think issue
    "MAPLRTP",  # Don't use i think issue
    "MAPLRS",  # Don't use i think issue
    "MAPLRTPS",  # Don't use i think issue
    "POMDR",  # observed_mutant_discordant_rate
    "PSOMDR",  # observed_source_discordant_rate
    "PGOMDR",
    "PSMMDR",
    "PGMMDR",
    "PMMDR",  # mle_mutant_discordant_rate
    "NHSNPN",  # filtered
    "NHSNPINDELN",  # unfiltered
    "mle_mosaic_allele_vaf_two_loci",
    "observed_mosaic_allele_vaf_two_loci",
    "GIOAF1",  # TODO: DEBUG OK!
    "GIOAF2",  # TODO: DEBUG OK!
    "GIMLEAF1",  # TODO: DEBUG OK!
    "GIMLEAF2",  # TODO: DEBUG OK!
    "GIUAF",
    "PRHN",
    "MLEPEAF0",
    "MLEPEAF1",
    "MLEPEAF2",
    "MLEPEAF3",
    "EMLEAF0",
    "EMLEAF1",
    "EMLEAF2",
    "EMLEAF3",
    "NALLELES_PER_STR_BP",
    "MLEUAF",
    "MLEPUAF",
    "PMB",  # phasing_hap_MLE_balance h1/(h1+h2)
    "POB",  # phasing_hap_observe_balance h1/(h1+h2)
    "PGHAF",  # germ_hsnp_af (spanning reads)
    "PMHAF",  # mut_hsnp_af (spanning reads)
    "PH3AF",  # hsnp3_af (spanning reads)
    "PGD",  # germ_dis 这个 features 算得有点问题需要 DEBUG，因为忽略了 unphase reads 和 assign 问题 # TODO unphase observed and phase mle 有问题不使用
    "PSD",  # source_dis 这个 features 算得有点问题需要 DEBUG，因为忽略了 unphase reads 和 assign 问题 # TODO unphase observed and phase mle 有问题不使用
    "PMD",  # mut_dis 这个 features 算得有点问题需要 DEBUG，因为忽略了 unphase reads 和 assign 问题 # TODO unphase observed and phase mle 有问题不使用
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
    "sample_name",
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
    "snp1_af",  # (all reads)
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
    # "mosaic_fraction", # 特征已去掉
    # "phasable", # 特征已去掉
    "overall_sample_after_filtered_read_dp",
    "overall_sample_filtered_out_read_dp",
    "overall_used_reads_fraction",  # spanning and unspanning
    "mle_mosaic_allele_vaf_single_locus",
    "used_read_fraction_in_genotyping",  # (used_read_num_in_genotyping / used_read_num_in_feature), both spanning,but one is filtered genotyping,one is unfilter features extraction
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
    "used_read_num_in_feature",  # features
    "used_read_num_in_genotyping",  # genotyping
    "sample_str_used_depth",  # mosdepth/samtools
    "sample_wgs_used_depth",  # mosdepth/samtools
    "depth_norm_factor",  # (read_length - str_length) / read_length
    "expected_str_used_depth",  # sample_wgs_used_depth * depth_norm_factor
    "expected_str_used_depth_ratio",  # used_read_num_in_feature / expected_str_used_depth
    "raw_wgs_expected_depth_ratio",  # str_padding_unfiltered_raw_depth / sample_wgs_used_depth
    "nearby_raw_expected_depth_ratio",  # str_padding_unfiltered_raw_depth / median_str_nearby_raw_depth
    "raw_wgs_point_expected_depth_ratio",  # str_padding_unfiltered_raw_depth / median_str_wgs_point_raw_depth
    "final_padding_str_chrom_depth_percentile",  # percentile_rank(sample_chrom_point_depth_list, str_padding_unfiltered_raw_depth)
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
    "phasing_fraction",  # len(nearby_snp_seq_list_rm_NA) / len(nearby_snp_seq_list) all str reads phasable reads
    "germ_phasing_fraction",  # germ str reads phasable reads, single locus assign reads
    "mut_phasing_fraction",  # mut str reads phasable reads, single locus assign reads
    "source_phasing_fraction",  # source str reads phasable reads, single locus assign reads
    "germ_hsnp_fraction",
    "germ_dis_fraction",
    "mut_hsnp_fraction",
    "mut_dis_fraction",
    "source_hsnp_fraction",
    "source_dis_fraction",
    "all_dis_fraction",
    "AAD",  # HACK: TODO: add AAD
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
    "median_genotyping_depth",  # from all STR loci median genotyping depth calculation
]
fea_columns = features_1 + germ_mut + source_mut
# ========== Step 1: 读取与预处理 ==========
GIAB_fp = pd.read_csv(input_file, sep=",", header=None)
GIAB_fp.columns = fea_columns
GIAB_fp = GIAB_fp.drop_duplicates(subset=["str_id"], keep="first")
# HACK: homhet and hethet all use
# GIAB_fp = GIAB_fp[
#     (GIAB_fp["muttype"] == "homhet") & (GIAB_fp["CMP"].astype(float) > 0.5)
# ]
# 解析GT与MGT为两个等位基因
GIAB_fp[["MGT0", "MGT1"]] = GIAB_fp["MGT"].str.split("/", expand=True)
GIAB_fp[["GT0", "GT1"]] = GIAB_fp["GT"].str.split("/", expand=True)

# 修正Mosaic VAF
GIAB_fp["genotyping_mle_mosaic_allele_vaf_single_locus"] = GIAB_fp["EMLEAF3"]
mask_hom = GIAB_fp["MGT0"] == GIAB_fp["MGT1"]
GIAB_fp.loc[
    mask_hom, "genotyping_mle_mosaic_allele_vaf_single_locus"
] = GIAB_fp.loc[mask_hom, "EMLEAF1"]

GIAB_fp["genotyping_obs_germ_allele_vaf_single_locus"] = GIAB_fp["EAF0"]
GIAB_fp["genotyping_obs_source_allele_vaf_single_locus"] = GIAB_fp["EAF1"]
GIAB_fp["genotyping_obs_mutant_allele_vaf_single_locus"] = GIAB_fp["EAF3"]
GIAB_fp.loc[
    mask_hom, "genotyping_obs_source_allele_vaf_single_locus"
] = GIAB_fp.loc[mask_hom, "EAF3"]
GIAB_fp.loc[
    mask_hom, "genotyping_obs_mutant_allele_vaf_single_locus"
] = GIAB_fp.loc[mask_hom, "EAF1"]


# ========== Step 2: MAPQ 归一化处理 ==========
max_mapq = GIAB_fp["overall_mean_mapQ"].astype(float).max()
scaling_factor = 60 / max_mapq if max_mapq > 61 else 1

for col in ["overall_mean_mapQ", "mut_mapQ_mean", "mut_mapQ_mean_germ_source"]:
    GIAB_fp[col] = GIAB_fp[col].astype(float) * scaling_factor

# ========== Step 3: 衍生统计特征 ==========
GIAB_fp["germ_read_number"] = (
    GIAB_fp["used_read_num_in_feature"] * GIAB_fp["germ_read_fraction"]
)
GIAB_fp["mut_read_number"] = (
    GIAB_fp["used_read_num_in_feature"] * GIAB_fp["mut_read_fraction"]
)
GIAB_fp["germ_error_number"] = (
    GIAB_fp["germ_read_number"] * GIAB_fp["germ_errors"]
)
GIAB_fp["mut_error_number"] = (
    GIAB_fp["mut_read_number"] * GIAB_fp["mut_errors"]
)


# ========== Step 4: Fisher 精确检验 ==========
def fisher_test(row):
    if np.any(
        np.isnan(
            [
                row["germ_read_number"],
                row["germ_error_number"],
                row["mut_read_number"],
                row["mut_error_number"],
            ]
        )
    ):
        return pd.Series(
            {
                "exact_match_odds_ratio_renew": np.nan,
                "exact_match_p_value_renew": np.nan,
            }
        )
    table = [
        [
            row["germ_read_number"] - row["germ_error_number"],
            row["germ_error_number"],
        ],
        [
            row["mut_read_number"] - row["mut_error_number"],
            row["mut_error_number"],
        ],
    ]
    odds_ratio, p_value = stats.fisher_exact(table)
    return pd.Series(
        {
            "exact_match_odds_ratio_renew": odds_ratio,
            "exact_match_p_value_renew": p_value,
        }
    )


GIAB_fp = GIAB_fp.join(GIAB_fp.apply(fisher_test, axis=1))


if het_no_filter:
    mask_clonalhet = GIAB_fp["muttype_hf"] == "clonalhet"
    eaf_swap = GIAB_fp["EAF0"] < GIAB_fp["EAF1"]

    MF = 1 - GIAB_fp["MF_hom2het_het2het"]
    MF = np.where(
        GIAB_fp["MF_hom2het_het2het"] >= 0.5, GIAB_fp["MF_hom2het_het2het"], MF
    )

    delta = GIAB_fp["mut_seq"].str.len() - GIAB_fp["source_seq"].str.len()

    # 初始化结果列为原始值
    GIAB_fp.loc[mask_clonalhet, "MBP"] = delta[mask_clonalhet]
    GIAB_fp.loc[mask_clonalhet, "MF_hom2het_het2het"] = MF[mask_clonalhet]

    swap_mask = mask_clonalhet & eaf_swap
    noswap_mask = mask_clonalhet & ~eaf_swap

    # swap
    GIAB_fp.loc[
        swap_mask, "observed_mosaic_allele_vaf_single_locus"
    ] = GIAB_fp.loc[swap_mask, "EAF0"]
    GIAB_fp.loc[
        swap_mask, "genotyping_obs_germ_allele_vaf_single_locus"
    ] = GIAB_fp.loc[swap_mask, "EAF1"]
    GIAB_fp.loc[
        swap_mask, "genotyping_obs_source_allele_vaf_single_locus"
    ] = GIAB_fp.loc[swap_mask, "EAF1"]
    GIAB_fp.loc[
        swap_mask, "genotyping_obs_mutant_allele_vaf_single_locus"
    ] = GIAB_fp.loc[swap_mask, "EAF0"]
    GIAB_fp.loc[
        swap_mask, "genotyping_mle_mosaic_allele_vaf_single_locus"
    ] = GIAB_fp.loc[swap_mask, "EMLEAF0"]
    GIAB_fp.loc[swap_mask, "GT"] = (
        GIAB_fp.loc[swap_mask, "GT1"] + "/" + GIAB_fp.loc[swap_mask, "GT0"]
    )
    GIAB_fp.loc[swap_mask, "MGT"] = (
        GIAB_fp.loc[swap_mask, "MGT1"] + "/" + GIAB_fp.loc[swap_mask, "MGT0"]
    )

    # no swap
    GIAB_fp.loc[
        noswap_mask, "observed_mosaic_allele_vaf_single_locus"
    ] = GIAB_fp.loc[noswap_mask, "EAF1"]
    GIAB_fp.loc[
        noswap_mask, "genotyping_obs_germ_allele_vaf_single_locus"
    ] = GIAB_fp.loc[noswap_mask, "EAF0"]
    GIAB_fp.loc[
        noswap_mask, "genotyping_obs_source_allele_vaf_single_locus"
    ] = GIAB_fp.loc[noswap_mask, "EAF0"]
    GIAB_fp.loc[
        noswap_mask, "genotyping_obs_mutant_allele_vaf_single_locus"
    ] = GIAB_fp.loc[noswap_mask, "EAF1"]
    GIAB_fp.loc[
        noswap_mask, "genotyping_mle_mosaic_allele_vaf_single_locus"
    ] = GIAB_fp.loc[noswap_mask, "EMLEAF1"]
    GIAB_fp.loc[noswap_mask, "GT"] = (
        GIAB_fp.loc[noswap_mask, "GT0"] + "/" + GIAB_fp.loc[noswap_mask, "GT1"]
    )
    GIAB_fp.loc[noswap_mask, "MGT"] = (
        GIAB_fp.loc[noswap_mask, "MGT0"]
        + "/"
        + GIAB_fp.loc[noswap_mask, "MGT1"]
    )

GIAB_fp["MBP"] = GIAB_fp["mut_seq"].str.len() - GIAB_fp["source_seq"].str.len()
GIAB_fp["MBP_abs"] = GIAB_fp["MBP"].apply(lambda x: abs(int(x)))
GIAB_fp["MStep_abs"] = GIAB_fp["MBP_abs"] / GIAB_fp["MOTIF_length"]


def get_noise_fraction(row):
    max_noise = max(
        row["inframe_ins_prob"],
        row["inframe_del_prob"],
        row["outframe_ins_prob"],
        row["outframe_del_prob"],
    )
    dp = row["depth"]
    mosaic_dp = dp * row["genotyping_mle_mosaic_allele_vaf_single_locus"]
    diff = row["genotyping_mle_mosaic_allele_vaf_single_locus"] - max_noise
    pval = stats.binom_test(mosaic_dp, dp, max_noise, alternative="greater")
    return pd.Series(
        {"mut_noise_diff_frac": diff, "binomial_noise_p_value": pval}
    )


GIAB_fp = GIAB_fp.join(GIAB_fp.apply(get_noise_fraction, axis=1))

GIAB_fp["sample_median_dp"] = (
    GIAB_fp["str_padding_unfiltered_raw_depth"]
    / GIAB_fp["raw_wgs_point_expected_depth_ratio"]
)
GIAB_fp["sample_median_dp_norm_str_length"] = (
    GIAB_fp["depth_norm_factor"] * GIAB_fp["sample_median_dp"]
)
GIAB_fp["revise_depth_ratio"] = (
    GIAB_fp["used_read_num_in_genotyping"]
    / GIAB_fp["sample_median_dp_norm_str_length"]
)
median_depth_ratio = np.median(GIAB_fp["revise_depth_ratio"])
GIAB_fp["norm_revise_depth_ratio"] = (
    GIAB_fp["revise_depth_ratio"] / median_depth_ratio
)
GIAB_fp["no_norm_len_depth_ratio"] = (
    GIAB_fp["used_read_num_in_genotyping"] / GIAB_fp["sample_median_dp"]
)
no_norm_median_depth_ratio = np.median(GIAB_fp["no_norm_len_depth_ratio"])
GIAB_fp["no_norm_len_revise_depth_ratio"] = (
    GIAB_fp["no_norm_len_depth_ratio"] / no_norm_median_depth_ratio
)
# GIAB_fp["median_genotyping_depth"] = np.median(GIAB_fp["used_read_num_in_genotyping"])


mut_dp = GIAB_fp["used_read_num_in_feature"] * GIAB_fp["mut_read_fraction"]
indel_dp = (
    GIAB_fp["used_read_num_in_feature"]
    * GIAB_fp["germ_read_fraction"]
    * GIAB_fp["germ_indel"]
)
GIAB_fp["stutter_ratio"] = np.where(mut_dp > 0, indel_dp / mut_dp, np.nan)
baseq_test_alternative = "greater"
stat = GIAB_fp["germ_mut_mean_baseQ_statistic"]
pval = GIAB_fp["germ_mut_mean_baseQ_pvalue"]
one_tail = np.where(
    ((stat >= 0) & (baseq_test_alternative == "greater"))
    | ((stat <= 0) & (baseq_test_alternative == "less")),
    pval / 2,
    1 - pval / 2,
)
GIAB_fp["germ_mut_mean_baseQ_statistic_single_tail"] = stat
GIAB_fp["germ_mut_mean_baseQ_pvalue_single_tail"] = one_tail


if NORM1_AS_HET:
    # 先处理 "clonalhet" 的情况
    mask_clonalhet = GIAB_fp["muttype_hf"] == "clonalhet"
    GIAB_fp.loc[mask_clonalhet, ["NORMGERM1", "NORMGERM2", "NORMMOSAIC"]] = [
        0.5,
        0,
        0.5,
    ]

    # 处理 homhet 特殊逻辑: 只有在 NORMGERM2 > NORMGERM1 时交换
    mask_homhet_swap = (GIAB_fp["muttype_hf"] == "homhet") & (
        GIAB_fp["NORMGERM1"] > GIAB_fp["NORMGERM2"]
    )
    GIAB_fp.loc[mask_homhet_swap, ["NORMGERM1", "NORMGERM2"]] = GIAB_fp.loc[
        mask_homhet_swap, ["NORMGERM2", "NORMGERM1"]
    ].values

    # 对于 hethet 和 hethom 的情况，直接交换 NORMGERM1 和 NORMGERM2
    mask_swap = GIAB_fp["muttype_hf"].isin(["hethet", "hethom"]) & (
        GIAB_fp["NORMGERM2"] > GIAB_fp["NORMGERM1"]
    )
    GIAB_fp.loc[mask_swap, ["NORMGERM1", "NORMGERM2"]] = GIAB_fp.loc[
        mask_swap, ["NORMGERM2", "NORMGERM1"]
    ].values


GIAB_fp.index = GIAB_fp["str_id"]
GIAB_fp["genotyping_obs_mutant_dp"] = (
    GIAB_fp["genotyping_obs_mutant_allele_vaf_single_locus"]
    * GIAB_fp["used_read_num_in_genotyping"]
)
GIAB_fp["genotyping_obs_source_dp"] = (
    GIAB_fp["genotyping_obs_source_allele_vaf_single_locus"]
    * GIAB_fp["used_read_num_in_genotyping"]
)
GIAB_fp["genotyping_obs_germ_dp"] = (
    GIAB_fp["genotyping_obs_germ_allele_vaf_single_locus"]
    * GIAB_fp["used_read_num_in_genotyping"]
)


def process_input(mis_or_indel, GIAB_fp):
    if mis_or_indel == "indel":
        if MODEL_VAF_CORRECTION:
            mosaic_fraction_use = "mosaic_fraction_correction"
            obs_vaf_use = "obs_vaf_correction"
        else:
            mosaic_fraction_use = "MF_hom2het_het2het"
            obs_vaf_use = "observed_mosaic_allele_vaf_single_locus"
        GIAB_fp_indel = GIAB_fp[GIAB_fp["alleles_mut_type"].isin(INDEL_CATE)]
        used_mis_feas = [
            # "CMP", ## HACK: No used for double check
            mosaic_fraction_use,
            ##"NALLELES_PER_STR_BP",
            #   "MBP_abs", ## HACK: No used for double check
            # "MStep_abs", ## HACK: No used for double check
            # "inframe_single_step_prob",
            # "inframe_ins_prob",
            # "inframe_del_prob",
            "stutter_ratio",
            "mut_noise_diff_frac",
            "binomial_noise_p_value",
            obs_vaf_use,
            ##"DFSTUTTER",
            #   "genotyping_mle_mosaic_allele_vaf_single_locus", ## HACK: No used for double check
            #   "MLEUAF", # 分布感觉差不多 ## HACK: No used for double check mismatch can use
            "NORMGERM1",
            "NORMGERM2",
            "NORMMOSAIC",
            "NORMSECOND",
            "average_mappability_score_k24",  # loci-based
            #   "flanking_context_GC_content", # loci-based  ## HACK: No used for double check
            #   "STR_flanking_context_GC_content", # loci-based  ## HACK: No used for double check
            #   "used_read_fraction_in_genotyping",
            #   "overall_mis_num_per_bp_all_reads_mean_based_on_NM",
            ##"raw_wgs_expected_depth_ratio",
            "overall_mean_mapQ",  # 需要标准化到最大值 60
            "germ_mut_diff_mapQ",
            "germ_mut_mean_baseQ_statistic_single_tail",
            # "germ_mut_mean_baseQ_statistic",
            #   "overall_mean_str_mis_fraction",
            "germ_mut_str_mis_fraction_statistic",
            "overall_mean_flanking_sbs_num_per_bp",
            "germ_mut_flanking_sbs_num_per_bp_statistic",
            #   "overall_non_clipped_fraction",
            "germ_non_clipped_fraction",
            "mut_non_clipped_fraction",
            "germ_mut_diff_non_clipped_fraction",
            # "overall_non_flanking_indel_fraction",
            "mut_non_flanking_indel_fraction",
            "germ_mut_diff_non_flanking_indel_fraction",
            # "overall_non_spanning_str_flanking_indel_fraction",
            # "mut_non_spanning_str_flanking_indel_fraction",## HACK: No used for double check
            # "germ_mut_diff_spanning_str_flanking_indel_fraction",## HACK: No used for double check
            # "germ_strand_fraction", ## HACK: No used for double check
            "mut_strand_fraction",
            # "germ_mut_strand_oddsratio",
            # "germ_orientation_fraction",## HACK: No used for double check
            "mut_orientation_fraction",
            # "germ_mut_orientation_oddsratio",
            "mut_proper_pair_fraction",
            "germ_mut_diff_proper_pair",
            # "overall_indel",
            # "overall_mis",
            # "overall_errors",
            "germ_mut_diff_indel",
            "germ_mut_diff_mis",
            "germ_mut_diff_errors",
            "mut_indel",
            "mut_mis",
            "mut_errors",
            # "exact_match_odds_ratio_renew",
            "flanking_abs_diff_baseq_mean",
            "flanking_baseq_mean_statistic",
            "flanking_abs_diff_sbs_num_mean",
            "flanking_sbs_num_mean_statistic",
            "mut_mapQ_mean",  # 需要标准化到最大值 60
            # "mut_mapQ_more_fraction", ## HACK: No used for double check
            # "germ_overall_baseQ_more_fraction",## HACK: No used for double check
            # "germ_mapQ_more_fraction", ## HACK: No used for double check
            "mut_overall_baseQ_more_fraction",
            # "germ_mut_temp_length_statistic",## HACK: No used for double check
            "mut_allele_ks_start_stat",
            "mut_allele_ks_end_stat",
            # "germ_mut_flanking_bps_num_statistic",
            # "germ_mut_read_pcr_cycle_percentage_sum_statistic",
            # "germ_mut_read_pcr_start_num_statistic",
            ## "GG_PVAL",
            ## "GQ_PVAL",
            # "source_mut_base_accuracy_pvalue",
            "germ_mut_mapQ_pvalue",
            "germ_mut_mean_baseQ_pvalue_single_tail",
            # "germ_mut_mean_baseQ_pvalue",
            "germ_mut_flanking_sbs_num_per_bp_pvalue",
            "germ_mut_strand_pvalue",
            "germ_mut_orientation_pvalue",
            "exact_match_p_value_renew",
            "flanking_baseq_mean_p",
            "flanking_sbs_num_mean_p",
            "germ_mut_temp_length_pvalue",
            "germ_mut_str_mis_fraction_pvalue",
            "mut_allele_ks_start_p_value",
            "mut_allele_ks_end_p_value",
            # "germ_mut_flanking_bps_num_pvalue",
            # "germ_mut_read_pcr_cycle_percentage_sum_pvalue",
            # "germ_mut_read_pcr_start_num_pvalue",
            ##"GQ_OR",
            ##"GG_OR",
            # "SECISGERM"## HACK: No used for double check
        ]
        GIAB_fp_indel.index = GIAB_fp_indel["str_id"]
        GIAB_fp_indel = GIAB_fp_indel[used_mis_feas]
        for col in [
            "binomial_noise_p_value",
            "germ_mut_mapQ_pvalue",
            "germ_mut_mean_baseQ_pvalue_single_tail",
            "germ_mut_flanking_sbs_num_per_bp_pvalue",
            "germ_mut_strand_pvalue",
            "germ_mut_orientation_pvalue",
            "exact_match_p_value_renew",
            "flanking_baseq_mean_p",
            "flanking_sbs_num_mean_p",
            "germ_mut_temp_length_pvalue",
            "germ_mut_str_mis_fraction_pvalue",
            "mut_allele_ks_start_p_value",
            "mut_allele_ks_end_p_value",
        ]:
            # 获取当前列中非零最小值
            # non_zero_min = GIAB_fp_2[GIAB_fp_2[col] > 0][col].min()

            # 将当前列中的零值替换为该非零最小值
            GIAB_fp_indel.loc[
                GIAB_fp_indel[col] == 0, col
            ] = ALLOW_NUMPY_MIN_VALUE  #  non_zero_min
        GIAB_fp_indel["stutter_ratio"] = GIAB_fp_indel["stutter_ratio"].fillna(
            GIAB_fp_indel["stutter_ratio"].max()
        )  # HACK: revise
        GIAB_fp_indel["flanking_baseq_mean_statistic"] = GIAB_fp_indel[
            "flanking_baseq_mean_statistic"
        ].fillna(GIAB_fp_indel["flanking_baseq_mean_statistic"].abs().max())
        GIAB_fp_indel["flanking_sbs_num_mean_statistic"] = GIAB_fp_indel[
            "flanking_sbs_num_mean_statistic"
        ].fillna(GIAB_fp_indel["flanking_sbs_num_mean_statistic"].abs().max())
        GIAB_fp_indel["flanking_abs_diff_baseq_mean"] = GIAB_fp_indel[
            "flanking_abs_diff_baseq_mean"
        ].fillna(GIAB_fp_indel["flanking_abs_diff_baseq_mean"].max())
        GIAB_fp_indel["flanking_abs_diff_sbs_num_mean"] = GIAB_fp_indel[
            "flanking_abs_diff_sbs_num_mean"
        ].fillna(GIAB_fp_indel["flanking_abs_diff_sbs_num_mean"].max())
        GIAB_fp_indel["flanking_baseq_mean_p"] = GIAB_fp_indel[
            "flanking_baseq_mean_p"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_indel["flanking_baseq_mean_p"].min()
        GIAB_fp_indel["flanking_sbs_num_mean_p"] = GIAB_fp_indel[
            "flanking_sbs_num_mean_p"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_indel["flanking_sbs_num_mean_p"].min()
        GIAB_fp_indel["exact_match_p_value_renew"] = GIAB_fp_indel[
            "exact_match_p_value_renew"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_indel["exact_match_p_value_renew"].min()
        # 这些也是因为 flanking 没有base，而且是因为两边都没有 base，以为 reads length 和 allele length 完全相等
        # 感觉可能是假的
        # ['germ_mut_flanking_sbs_num_per_bp_statistic',
        #        'flanking_abs_diff_baseq_mean', 'flanking_baseq_mean_statistic',
        #        'flanking_abs_diff_sbs_num_mean', 'flanking_sbs_num_mean_statistic',
        #        'mut_allele_ks_start_stat', 'mut_allele_ks_end_stat',
        #        'germ_mut_flanking_sbs_num_per_bp_pvalue', 'flanking_baseq_mean_p',
        #        'flanking_sbs_num_mean_p', 'mut_allele_ks_start_p_value',
        #        'mut_allele_ks_end_p_value']
        GIAB_fp_indel[
            "germ_mut_flanking_sbs_num_per_bp_statistic"
        ] = GIAB_fp_indel["germ_mut_flanking_sbs_num_per_bp_statistic"].fillna(
            GIAB_fp_indel["germ_mut_flanking_sbs_num_per_bp_statistic"]
            .abs()
            .max()
        )
        GIAB_fp_indel[
            "germ_mut_flanking_sbs_num_per_bp_pvalue"
        ] = GIAB_fp_indel["germ_mut_flanking_sbs_num_per_bp_pvalue"].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_indel["germ_mut_flanking_sbs_num_per_bp_pvalue"].abs().min()
        GIAB_fp_indel["mut_allele_ks_start_stat"] = GIAB_fp_indel[
            "mut_allele_ks_start_stat"
        ].fillna(GIAB_fp_indel["mut_allele_ks_start_stat"].abs().max())
        GIAB_fp_indel["mut_allele_ks_end_stat"] = GIAB_fp_indel[
            "mut_allele_ks_end_stat"
        ].fillna(GIAB_fp_indel["mut_allele_ks_end_stat"].abs().max())
        GIAB_fp_indel["mut_allele_ks_start_p_value"] = GIAB_fp_indel[
            "mut_allele_ks_start_p_value"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_indel["mut_allele_ks_start_p_value"].abs().min()
        GIAB_fp_indel["mut_allele_ks_end_p_value"] = GIAB_fp_indel[
            "mut_allele_ks_end_p_value"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_indel["mut_allele_ks_end_p_value"].abs().min()

        # used_mis_feas_dict = {
        #     "MF_hom2het_het2het": "EM Mosaic Fraction",
        #     # "MStep_abs": "Mut Step", ## HACK: No used for double check
        #     "stutter_ratio": "Stutter Mut VAF Ratio",
        #     "mut_noise_diff_frac": "Mut VAF Stutter Diff",
        #     "binomial_noise_p_value": "Binomial Test Noise p-value",
        #     "observed_mosaic_allele_vaf_single_locus": "Observed Mut VAF",
        #     "NORMGERM1": "Germline1 Likelihood",
        #     "NORMGERM2": "Germline2 Likelihood",
        #     "NORMMOSAIC": "Mosaic Likelihood",
        #     "NORMSECOND": "Second Max Mosaic Likelihood",
        #     "average_mappability_score_k24": "Mappability Score k24",
        #     "overall_mean_mapQ": "Overall Mean mapQ",
        #     "germ_mut_diff_mapQ": "Germ Mut Diff MapQ",
        #     "germ_mut_mean_baseQ_statistic_single_tail": (
        #         "Germ Mut Diff BaseQ Statistic"
        #     ),
        #     "germ_mut_str_mis_fraction_statistic": (
        #         "Germ Mut Diff STR Mismatches Statistic"
        #     ),
        #     "overall_mean_flanking_sbs_num_per_bp": "Overall Flanking Mismatches",
        #     "germ_mut_flanking_sbs_num_per_bp_statistic": (
        #         "Germ Mut Diff Flanking Mismatches Statistic"
        #     ),
        #     "germ_non_clipped_fraction": "Germ Non Clipped Fraction",
        #     "mut_non_clipped_fraction": "Mut Non Clipped Fraction",
        #     "germ_mut_diff_non_clipped_fraction": (
        #         "Germ Mut Diff Non Clipped Fraction"
        #     ),
        #     "mut_non_flanking_indel_fraction": "Mut Non Flanking Indel Fraction",
        #     "germ_mut_diff_non_flanking_indel_fraction": (
        #         "Germ Mut Diff Non Flanking Indel Fraction"
        #     ),
        #     "mut_strand_fraction": "Mut Strand Fraction",
        #     "mut_orientation_fraction": "Mut Orientation Fraction",
        #     "mut_proper_pair_fraction": "Mut Proper Pair Fraction",
        #     "germ_mut_diff_proper_pair": "Germ Mut Diff Proper Pair",
        #     "germ_mut_diff_indel": "Germ Mut Diff Indels",
        #     "germ_mut_diff_mis": "Germ Mut Diff Mismatches",
        #     "germ_mut_diff_errors": "Germ Mut Diff Errors",
        #     "mut_indel": "Mut InDel Fraction",
        #     "mut_mis": "Mut Mismatch Fraction",
        #     "mut_errors": "Mut Errors",
        #     "flanking_abs_diff_baseq_mean": "Mut Left-Right Flanking Diff BaseQ ",
        #     "flanking_baseq_mean_statistic": (
        #         "Mut Left-Right Flanking BaseQ Statistic"
        #     ),
        #     "flanking_abs_diff_sbs_num_mean": (
        #         "Mut Left-Right Flanking Diff Mismatches "
        #     ),
        #     "flanking_sbs_num_mean_statistic": (
        #         "Mut Left-Right Flanking Mismatches Statistic"
        #     ),
        #     "mut_mapQ_mean": "Mut Mean MapQ",
        #     "mut_overall_baseQ_more_fraction": "Mut BaseQ More Fraction",
        #     "mut_allele_ks_start_stat": "Mut Left Start Shape KS Statistic",
        #     "mut_allele_ks_end_stat": "Mut Right End Shape KS Statistic",
        #     "germ_mut_mapQ_pvalue": "Germ Mut Diff MapQ p-value",
        #     "germ_mut_mean_baseQ_pvalue_single_tail": (
        #         "Germ Mut Diff BaseQ p-value"
        #     ),
        #     "germ_mut_flanking_sbs_num_per_bp_pvalue": (
        #         "Germ Mut Diff Mismatches p-value"
        #     ),
        #     "germ_mut_strand_pvalue": "Germ Mut Diff Strand p-value",
        #     "germ_mut_orientation_pvalue": "Germ Mut Diff Orientation p-value",
        #     "exact_match_p_value_renew": "Germ Mut Diff Exact Match p-value",
        #     "flanking_baseq_mean_p": "Mut Left-Right Diff Flanking BaseQ p-value",
        #     "flanking_sbs_num_mean_p": (
        #         "Mut Left-Right Diff Flanking Mismatches p-value"
        #     ),
        #     "germ_mut_temp_length_pvalue": "Germ Mut Diff Template Length p-value",
        #     "germ_mut_str_mis_fraction_pvalue": (
        #         "Germ Mut Diff STR Mismatches Fraction p-value"
        #     ),
        #     "mut_allele_ks_start_p_value": "Mut Left Start Shape KS p-value",
        #     "mut_allele_ks_end_p_value": "Mut Right End Shape KS p-value",
        # }

        # final_df = GIAB_fp_indel
        import numpy as np

        # norm_pvalue
        no_norm_pvalue = []
        norm_pvalue = []
        # HACK: 这边更改了 features 的顺序了，与 used_mis_feas 不一致了，注意不要搞错了，这样训练和预测会出问题的
        # HACK: 注意模型 features 顺序，模型预测输出概率顺序，used_feature_list used_feature_dict 的顺序可能不一致，需要检查和调整在一些步骤中
        for i in [
            "binomial_noise_p_value",
            "germ_mut_mapQ_pvalue",
            "germ_mut_mean_baseQ_pvalue_single_tail",
            "germ_mut_flanking_sbs_num_per_bp_pvalue",
            "germ_mut_strand_pvalue",
            "germ_mut_orientation_pvalue",
            "exact_match_p_value_renew",
            "flanking_baseq_mean_p",
            "flanking_sbs_num_mean_p",
            "germ_mut_temp_length_pvalue",
            "germ_mut_str_mis_fraction_pvalue",
            "mut_allele_ks_start_p_value",
            "mut_allele_ks_end_p_value",
        ]:
            no_norm_pvalue.append(i)
            norm_pvalue.append(f"{i}_norm")
            GIAB_fp_indel[f"{i}_norm"] = -np.log10(GIAB_fp_indel[i])

        GIAB_fp_indel = GIAB_fp_indel.drop(
            columns=no_norm_pvalue
        )  # + ["MStep_abs"]
        # features_merged_df_selected = final_df_norm_pvalue
        # 检查是否存在缺失值（NaN）
        missing_values = GIAB_fp_indel.isnull().any().any()
        if missing_values:
            print("数据框中存在缺失值 (NaN)")
            # features_merged_df_selected = features_merged_df_selected.fillna(features_merged_df_selected.mean())
        else:
            pass
            # print("数据框中不存在缺失值 (NaN)")

        cols_with_issues = GIAB_fp_indel.columns[
            GIAB_fp_indel.isna().any()
        ].tolist()
        if cols_with_issues:
            print("包含缺失值的列:", cols_with_issues)
        # 检查是否存在无穷值 (inf 或 -inf)
        infinite_values = GIAB_fp_indel.apply(
            lambda col: np.isinf(col).any()
        ).any()
        if infinite_values:
            # final_df.loc[final_df["mut_allele_ks_end_p_value_norm"]==np.inf,"mut_allele_ks_end_p_value_norm"]=final_df[final_df["mut_allele_ks_end_p_value_norm"]!=np.inf]["mut_allele_ks_end_p_value_norm"].max()
            # final_df.loc[final_df["mut_allele_ks_start_p_value_norm"]==np.inf,"mut_allele_ks_start_p_value_norm"]=final_df[final_df["mut_allele_ks_start_p_value_norm"]!=np.inf]["mut_allele_ks_start_p_value_norm"].max()
            # final_df.loc[final_df["binomial_noise_p_value_norm"]==np.inf,"binomial_noise_p_value_norm"]=final_df[final_df["binomial_noise_p_value_norm"]!=np.inf]["binomial_noise_p_value_norm"].max()
            print("数据框中存在无穷值 (inf 或 -inf)")
        else:
            pass
            # print("数据框中不存在无穷值 (inf 或 -inf)")
        # 检查每一列是否包含无穷值
        columns_with_inf = GIAB_fp_indel.columns[
            GIAB_fp_indel.apply(lambda col: np.isinf(col).any())
        ].tolist()
        if columns_with_inf:
            print("以下列包含无穷值:", columns_with_inf)
        else:
            pass
            # print("没有列包含无穷值")
        # X = GIAB_fp_indel
        GIAB_fp_indel["str_id"] = GIAB_fp_indel.index
        GIAB_fp_indel["germ_seq"] = list(
            GIAB_fp.loc[list(GIAB_fp_indel["str_id"]), "germ_seq"]
        )
        GIAB_fp_indel["source_seq"] = list(
            GIAB_fp.loc[list(GIAB_fp_indel["str_id"]), "source_seq"]
        )
        GIAB_fp_indel["mut_seq"] = list(
            GIAB_fp.loc[list(GIAB_fp_indel["str_id"]), "mut_seq"]
        )
        GIAB_fp_indel.dropna(
            subset=["germ_non_clipped_fraction"], inplace=True
        )
        GIAB_fp_indel.dropna(subset=["mut_non_clipped_fraction"], inplace=True)
        GIAB_fp_indel.dropna(
            subset=["overall_mean_flanking_sbs_num_per_bp"], inplace=True
        )
        return GIAB_fp_indel
    elif mis_or_indel == "mismatch":
        GIAB_fp_mis = GIAB_fp[GIAB_fp["alleles_mut_type"].isin(MISMATCH_CATE)]
        used_mis_feas = [
            # "CMP",
            "MF_hom2het_het2het",
            "NALLELES_PER_STR_BP",
            "observed_mosaic_allele_vaf_single_locus",
            # "DFSTUTTER",
            "stutter_ratio",
            "mut_noise_diff_frac",
            # "genotyping_mle_mosaic_allele_vaf_single_locus",  # HACK: no use for dup
            "MLEUAF",  # 分布感觉差不多
            "NORMGERM1",
            "NORMGERM2",
            "NORMMOSAIC",
            "NORMSECOND",
            "average_mappability_score_k24",  # loci-based
            # "flanking_context_GC_content", # loci-based ## HACK: No used for double check
            # "STR_flanking_context_GC_content", # loci-based ## HACK: No used for double check
            # "used_read_fraction_in_genotyping",
            "source_mut_diff_base_accuracy",
            "source_mut_base_accuracy_statistic",
            # "raw_wgs_expected_depth_ratio",  # HACK: No use because simulation data is quite good for this value
            "mut_fraction_baseq_more_cutoff",
            "overall_mean_mapQ",  # 需要标准化到最大值 60
            "germ_mut_diff_mapQ",
            "germ_mut_mean_baseQ_statistic_single_tail",
            # "germ_mut_mean_baseQ_statistic", # HACK: no use for dup
            # "overall_mean_str_mis_fraction",
            "overall_mean_flanking_sbs_num_per_bp",
            "germ_mut_flanking_sbs_num_per_bp_statistic",
            # "overall_non_clipped_fraction",
            "germ_non_clipped_fraction",
            "mut_non_clipped_fraction",
            "germ_mut_diff_non_clipped_fraction",
            "overall_non_flanking_indel_fraction",
            "mut_non_flanking_indel_fraction",
            "germ_mut_diff_non_flanking_indel_fraction",
            # "overall_non_spanning_str_flanking_indel_fraction", ## HACK: No used for double check
            # "mut_non_spanning_str_flanking_indel_fraction", ## HACK: No used for double check
            # "germ_mut_diff_spanning_str_flanking_indel_fraction", ## HACK: No used for double check
            # "germ_strand_fraction", ## HACK: No used for double check
            "mut_strand_fraction",
            # "germ_mut_strand_oddsratio",
            # "germ_orientation_fraction", ## HACK: No used for double check
            "mut_orientation_fraction",
            # "germ_mut_orientation_oddsratio",
            "mut_proper_pair_fraction",
            "germ_mut_diff_proper_pair",
            # "overall_indel",
            # "overall_mis",
            # "overall_errors",
            "germ_mut_diff_indel",
            "germ_mut_diff_mis",
            "germ_mut_diff_errors",
            "mut_indel",
            "mut_mis",
            "mut_errors",
            # "exact_match_odds_ratio_renew",
            "flanking_abs_diff_baseq_mean",
            "flanking_baseq_mean_statistic",
            "flanking_abs_diff_sbs_num_mean",
            "flanking_sbs_num_mean_statistic",
            "mut_mapQ_mean",  # 需要标准化到最大值 60
            # "mut_mapQ_more_fraction",  ## HACK: No used for double check
            # "germ_overall_baseQ_more_fraction", ## HACK: No used for double check
            # "germ_mapQ_more_fraction", ## HACK: No used for double check
            # "germ_mut_temp_length_statistic",  ## HACK: No used for double check
            "mut_allele_ks_start_stat",
            "mut_allele_ks_end_stat",
            "germ_mut_flanking_bps_num_statistic",
            # "germ_mut_read_pcr_cycle_percentage_sum_statistic", ## HACK: No used for double check
            "germ_mut_read_pcr_start_num_statistic",
            # "GG_PVAL",
            # "GQ_PVAL",
            "source_mut_base_accuracy_pvalue",
            "germ_mut_mapQ_pvalue",
            # "germ_mut_mean_baseQ_pvalue", ## HACK: No used for double check
            "germ_mut_mean_baseQ_pvalue_single_tail",
            "germ_mut_flanking_sbs_num_per_bp_pvalue",
            "germ_mut_strand_pvalue",
            "germ_mut_orientation_pvalue",
            "exact_match_p_value_renew",
            "flanking_baseq_mean_p",
            "flanking_sbs_num_mean_p",
            "germ_mut_temp_length_pvalue",
            "mut_allele_ks_start_p_value",
            "mut_allele_ks_end_p_value",
            "germ_mut_flanking_bps_num_pvalue",
            "germ_mut_read_pcr_cycle_percentage_sum_pvalue",
            "germ_mut_read_pcr_start_num_pvalue",
            "binomial_noise_p_value",
            # "GQ_OR",
            # "GG_OR",
            # "SECISGERM", ## HACK: No used for double check
            # "mut_is_recurrent_motif",
            # "source_is_recurrent_motif"
        ]
        GIAB_fp_mis.index = GIAB_fp_mis["str_id"]
        GIAB_fp_mis = GIAB_fp_mis[used_mis_feas]
        for col in [
            "source_mut_base_accuracy_pvalue",
            "germ_mut_mapQ_pvalue",
            # "germ_mut_mean_baseQ_pvalue",
            "germ_mut_mean_baseQ_pvalue_single_tail",
            "germ_mut_flanking_sbs_num_per_bp_pvalue",
            "germ_mut_strand_pvalue",
            "germ_mut_orientation_pvalue",
            "exact_match_p_value_renew",
            "flanking_baseq_mean_p",
            "flanking_sbs_num_mean_p",
            "germ_mut_temp_length_pvalue",
            "mut_allele_ks_start_p_value",
            "mut_allele_ks_end_p_value",
            "germ_mut_flanking_bps_num_pvalue",
            "germ_mut_read_pcr_cycle_percentage_sum_pvalue",
            "germ_mut_read_pcr_start_num_pvalue",
            "binomial_noise_p_value",
        ]:
            # 获取当前列中非零最小值
            # non_zero_min = GIAB_fp_2[GIAB_fp_2[col] > 0][col].min()

            # 将当前列中的零值替换为该非零最小值
            GIAB_fp_mis.loc[
                GIAB_fp_mis[col] == 0, col
            ] = ALLOW_NUMPY_MIN_VALUE  # non_zero_min
        ## 按照逻辑填补缺失值
        # 这些缺失值是因为 flanking 没有 base 很可能，flanking 一条 reads 都没有 base 很可能是假的，可能 flanking 完全在一边，另外一边没有 base
        # 感觉可能是假的
        GIAB_fp_mis["stutter_ratio"] = GIAB_fp_mis["stutter_ratio"].fillna(
            GIAB_fp_mis["stutter_ratio"].max()
        )  # HACK: revise
        GIAB_fp_mis["flanking_baseq_mean_statistic"] = GIAB_fp_mis[
            "flanking_baseq_mean_statistic"
        ].fillna(GIAB_fp_mis["flanking_baseq_mean_statistic"].abs().max())
        GIAB_fp_mis["flanking_sbs_num_mean_statistic"] = GIAB_fp_mis[
            "flanking_sbs_num_mean_statistic"
        ].fillna(GIAB_fp_mis["flanking_sbs_num_mean_statistic"].abs().max())
        GIAB_fp_mis["flanking_abs_diff_baseq_mean"] = GIAB_fp_mis[
            "flanking_abs_diff_baseq_mean"
        ].fillna(GIAB_fp_mis["flanking_abs_diff_baseq_mean"].max())
        GIAB_fp_mis["flanking_abs_diff_sbs_num_mean"] = GIAB_fp_mis[
            "flanking_abs_diff_sbs_num_mean"
        ].fillna(GIAB_fp_mis["flanking_abs_diff_sbs_num_mean"].max())
        GIAB_fp_mis["flanking_baseq_mean_p"] = GIAB_fp_mis[
            "flanking_baseq_mean_p"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_mis["flanking_baseq_mean_p"].min()
        GIAB_fp_mis["flanking_sbs_num_mean_p"] = GIAB_fp_mis[
            "flanking_sbs_num_mean_p"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_mis["flanking_sbs_num_mean_p"].min()
        GIAB_fp_mis["exact_match_p_value_renew"] = GIAB_fp_mis[
            "exact_match_p_value_renew"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_mis["exact_match_p_value_renew"].min()
        # 这些也是因为 flanking 没有base，而且是因为两边都没有 base，以为 reads length 和 allele length 完全相等
        # 感觉可能是假的
        # ['germ_mut_flanking_sbs_num_per_bp_statistic',
        #        'flanking_abs_diff_baseq_mean', 'flanking_baseq_mean_statistic',
        #        'flanking_abs_diff_sbs_num_mean', 'flanking_sbs_num_mean_statistic',
        #        'mut_allele_ks_start_stat', 'mut_allele_ks_end_stat',
        #        'germ_mut_flanking_sbs_num_per_bp_pvalue', 'flanking_baseq_mean_p',
        #        'flanking_sbs_num_mean_p', 'mut_allele_ks_start_p_value',
        #        'mut_allele_ks_end_p_value']
        GIAB_fp_mis[
            "germ_mut_flanking_sbs_num_per_bp_statistic"
        ] = GIAB_fp_mis["germ_mut_flanking_sbs_num_per_bp_statistic"].fillna(
            GIAB_fp_mis["germ_mut_flanking_sbs_num_per_bp_statistic"]
            .abs()
            .max()
        )
        GIAB_fp_mis["germ_mut_flanking_sbs_num_per_bp_pvalue"] = GIAB_fp_mis[
            "germ_mut_flanking_sbs_num_per_bp_pvalue"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_mis["germ_mut_flanking_sbs_num_per_bp_pvalue"].abs().min()
        GIAB_fp_mis["mut_allele_ks_start_stat"] = GIAB_fp_mis[
            "mut_allele_ks_start_stat"
        ].fillna(GIAB_fp_mis["mut_allele_ks_start_stat"].abs().max())
        GIAB_fp_mis["mut_allele_ks_end_stat"] = GIAB_fp_mis[
            "mut_allele_ks_end_stat"
        ].fillna(GIAB_fp_mis["mut_allele_ks_end_stat"].abs().max())
        GIAB_fp_mis["mut_allele_ks_start_p_value"] = GIAB_fp_mis[
            "mut_allele_ks_start_p_value"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_mis["mut_allele_ks_start_p_value"].abs().min()
        GIAB_fp_mis["mut_allele_ks_end_p_value"] = GIAB_fp_mis[
            "mut_allele_ks_end_p_value"
        ].fillna(
            ALLOW_NUMPY_MIN_VALUE
        )  # GIAB_fp_mis["mut_allele_ks_end_p_value"].abs().min()
        # used_mis_feas_dict = {
        #     "MF_hom2het_het2het": "EM Mosaic Fraction",
        #     "NALLELES_PER_STR_BP": "Normalized Alleles Number",
        #     "MLEUAF": "Unassign Fraction",
        #     "MStep_abs": "Mut Step",
        #     "stutter_ratio": "Stutter Mut VAF Ratio",
        #     "genotyping_mle_mosaic_allele_vaf_single_locus": "MLE Mut VAF",
        #     "mut_noise_diff_frac": "Mut VAF Stutter Diff",
        #     "binomial_noise_p_value": "Binomial Test Noise p-value",
        #     "observed_mosaic_allele_vaf_single_locus": "Observed Mut VAF",
        #     "NORMGERM1": "Germline1 Likelihood",
        #     "NORMGERM2": "Germline2 Likelihood",
        #     "NORMMOSAIC": "Mosaic Likelihood",
        #     "NORMSECOND": "Second Max Mosaic Likelihood",
        #     "average_mappability_score_k24": "Mappability Score k24",
        #     "source_mut_diff_base_accuracy": "Mismatch Germ Mut Diff BaseQ",
        #     "source_mut_base_accuracy_statistic": (
        #         "Mismatch Germ Mut Diff BaseQ Statistic"
        #     ),
        #     "raw_wgs_expected_depth_ratio": "Expected Depth ratio",
        #     "mut_fraction_baseq_more_cutoff": "Mut High BaseQ Fraction",
        #     "overall_mean_mapQ": "Overall Mean mapQ",
        #     "germ_mut_diff_mapQ": "Germ Mut Diff MapQ",
        #     "germ_mut_mean_baseQ_statistic_single_tail": (
        #         "Germ Mut Diff BaseQ Statistic"
        #     ),
        #     "germ_mut_str_mis_fraction_statistic": (
        #         "Germ Mut Diff STR Mismatches Statistic"
        #     ),
        #     "overall_mean_flanking_sbs_num_per_bp": "Overall Flanking Mismatches",
        #     "germ_mut_flanking_sbs_num_per_bp_statistic": (
        #         "Germ Mut Diff Flanking Mismatches Statistic"
        #     ),
        #     "germ_non_clipped_fraction": "Germ Non Clipped Fraction",
        #     "mut_non_clipped_fraction": "Mut Non Clipped Fraction",
        #     "germ_mut_diff_non_clipped_fraction": (
        #         "Germ Mut Diff Non Clipped Fraction"
        #     ),
        #     "overall_non_flanking_indel_fraction": (
        #         "Overall Non Flanking Indel Fraction"
        #     ),
        #     "mut_non_flanking_indel_fraction": "Mut Non Flanking Indel Fraction",
        #     "germ_mut_diff_non_flanking_indel_fraction": (
        #         "Germ Mut Diff Non Flanking Indel Fraction"
        #     ),
        #     "overall_non_spanning_str_flanking_indel_fraction": (
        #         "Overall Non Flanking Span Indel Fraction"
        #     ),
        #     "mut_non_spanning_str_flanking_indel_fraction": (
        #         "Mut Non Flanking Span Indel Fraction"
        #     ),
        #     "germ_mut_diff_spanning_str_flanking_indel_fraction": (
        #         "Germ Mut Diff Flanking Span Indel Fraction"
        #     ),
        #     "germ_strand_fraction": "Germ Strand Fraction",
        #     "mut_strand_fraction": "Mut Strand Fraction",
        #     "germ_orientation_fraction": "Germ Orientation Fraction",
        #     "mut_orientation_fraction": "Mut Orientation Fraction",
        #     "mut_proper_pair_fraction": "Mut Proper Pair Fraction",
        #     "germ_mut_diff_proper_pair": "Germ Mut Diff Proper Pair",
        #     "germ_mut_diff_indel": "Germ Mut Diff Indels",
        #     "germ_mut_diff_mis": "Germ Mut Diff Mismatches",
        #     "germ_mut_diff_errors": "Germ Mut Diff Errors",
        #     "mut_indel": "Mut InDel Fraction",
        #     "mut_mis": "Mut Mismatch Fraction",
        #     "mut_errors": "Mut Errors",
        #     "flanking_abs_diff_baseq_mean": "Mut Left-Right Flanking Diff BaseQ",
        #     "flanking_baseq_mean_statistic": (
        #         "Mut Left-Right Flanking BaseQ Statistic"
        #     ),
        #     "flanking_abs_diff_sbs_num_mean": (
        #         "Mut Left-Right Flanking Diff Mismatches "
        #     ),
        #     "flanking_sbs_num_mean_statistic": (
        #         "Mut Left-Right Flanking Mismatches Statistic"
        #     ),
        #     "mut_mapQ_mean": "Mut Mean MapQ",
        #     "mut_overall_baseQ_more_fraction": "Mut BaseQ More Fraction",
        #     "mut_allele_ks_start_stat": "Mut Left Start Shape KS Statistic",
        #     "mut_allele_ks_end_stat": "Mut Right End Shape KS Statistic",
        #     "germ_mut_flanking_bps_num_statistic": (
        #         "Germ Mut Flanking Len Statistic"
        #     ),
        #     "germ_mut_read_pcr_cycle_percentage_sum_statistic": (
        #         "Germ Mut STR PCR Cycle Percentage Statistic"
        #     ),
        #     "germ_mut_read_pcr_start_num_statistic": (
        #         "Germ Mut STR PCR Start Site Statistic"
        #     ),
        #     "source_mut_base_accuracy_pvalue": (
        #         "Germ Mut Diff Mismatches BaseQ p-value"
        #     ),
        #     "germ_mut_mapQ_pvalue": "Germ Mut Diff MapQ p-value",
        #     "germ_mut_mean_baseQ_pvalue_single_tail": (
        #         "Germ Mut Diff BaseQ p-value"
        #     ),
        #     "germ_mut_flanking_sbs_num_per_bp_pvalue": (
        #         "Germ Mut Diff Mismatches p-value"
        #     ),
        #     "germ_mut_strand_pvalue": "Germ Mut Diff Strand p-value",
        #     "germ_mut_orientation_pvalue": "Germ Mut Diff Orientation p-value",
        #     "exact_match_p_value_renew": "Germ Mut Diff Exact Match p-value",
        #     "flanking_baseq_mean_p": "Mut Left-Right Diff Flanking BaseQ p-value",
        #     "flanking_sbs_num_mean_p": (
        #         "Mut Left-Right Diff Flanking Mismatches p-value"
        #     ),
        #     "germ_mut_temp_length_pvalue": "Germ Mut Diff Template Length p-value",
        #     "germ_mut_str_mis_fraction_pvalue": (
        #         "Germ Mut Diff STR Mismatches Fraction p-value"
        #     ),
        #     "mut_allele_ks_start_p_value": "Mut Left Start Shape KS p-value",
        #     "mut_allele_ks_end_p_value": "Mut Right End Shape KS p-value",
        #     "germ_mut_flanking_bps_num_pvalue": "Germ Mut Flanking Len p-value",
        #     "germ_mut_read_pcr_cycle_percentage_sum_pvalue": (
        #         "Germ Mut STR PCR Cycle Percentage p-value"
        #     ),
        #     "germ_mut_read_pcr_start_num_pvalue": (
        #         "Germ Mut STR PCR Start Site p-value"
        #     ),
        # }
        # final_df = GIAB_fp_mis.copy()
        import numpy as np

        no_norm_pvalue = []
        norm_pvalue = []
        for i in [
            "germ_mut_mapQ_pvalue",
            "germ_mut_mean_baseQ_pvalue_single_tail",
            "germ_mut_flanking_sbs_num_per_bp_pvalue",
            "germ_mut_strand_pvalue",
            "germ_mut_orientation_pvalue",
            "exact_match_p_value_renew",
            "flanking_baseq_mean_p",
            "flanking_sbs_num_mean_p",
            "germ_mut_temp_length_pvalue",
            "mut_allele_ks_start_p_value",
            "mut_allele_ks_end_p_value",
            "source_mut_base_accuracy_pvalue",
            "germ_mut_flanking_bps_num_pvalue",
            "germ_mut_read_pcr_cycle_percentage_sum_pvalue",  # HACK: no use for dup
            "germ_mut_read_pcr_start_num_pvalue",
            "binomial_noise_p_value",
        ]:
            no_norm_pvalue.append(i)
            norm_pvalue.append(f"{i}_norm")
            GIAB_fp_mis[f"{i}_norm"] = -np.log10(GIAB_fp_mis[i])
        GIAB_fp_mis = GIAB_fp_mis.drop(columns=no_norm_pvalue)
        # features_merged_df_selected = final_df_norm_pvalue
        # 检查是否存在缺失值（NaN）
        missing_values = GIAB_fp_mis.isnull().any().any()
        if missing_values:
            print("数据框中存在缺失值 (NaN)")
            # features_merged_df_selected = features_merged_df_selected.fillna(features_merged_df_selected.mean())
        # else:
        #     print("数据框中不存在缺失值 (NaN)")

        cols_with_issues = GIAB_fp_mis.columns[
            GIAB_fp_mis.isna().any()
        ].tolist()
        if cols_with_issues:
            print("包含缺失值的列:", cols_with_issues)
        # 检查是否存在无穷值 (inf 或 -inf)
        infinite_values = GIAB_fp_mis.apply(
            lambda col: np.isinf(col).any()
        ).any()
        if infinite_values:
            # final_df.loc[final_df["mut_allele_ks_end_p_value_norm"]==np.inf,"mut_allele_ks_end_p_value_norm"]=final_df[final_df["mut_allele_ks_end_p_value_norm"]!=np.inf]["mut_allele_ks_end_p_value_norm"].max()
            # final_df.loc[final_df["mut_allele_ks_start_p_value_norm"]==np.inf,"mut_allele_ks_start_p_value_norm"]=final_df[final_df["mut_allele_ks_start_p_value_norm"]!=np.inf]["mut_allele_ks_start_p_value_norm"].max()
            # final_df.loc[final_df["binomial_noise_p_value_norm"]==np.inf,"binomial_noise_p_value_norm"]=final_df[final_df["binomial_noise_p_value_norm"]!=np.inf]["binomial_noise_p_value_norm"].max()
            print("数据框中存在无穷值 (inf 或 -inf)")
        # else:
        #     print("数据框中不存在无穷值 (inf 或 -inf)")
        # 检查每一列是否包含无穷值
        columns_with_inf = GIAB_fp_mis.columns[
            GIAB_fp_mis.apply(lambda col: np.isinf(col).any())
        ].tolist()
        if columns_with_inf:
            print("以下列包含无穷值:", columns_with_inf)
        # else:
        #     print("没有列包含无穷值")
        #  X = final_df_norm_pvalue
        GIAB_fp_mis["str_id"] = GIAB_fp_mis.index
        GIAB_fp_mis["germ_seq"] = list(
            GIAB_fp.loc[list(GIAB_fp_mis["str_id"]), "germ_seq"]
        )
        GIAB_fp_mis["source_seq"] = list(
            GIAB_fp.loc[list(GIAB_fp_mis["str_id"]), "source_seq"]
        )
        GIAB_fp_mis["mut_seq"] = list(
            GIAB_fp.loc[list(GIAB_fp_mis["str_id"]), "mut_seq"]
        )
        GIAB_fp_mis.dropna(subset=["germ_non_clipped_fraction"], inplace=True)
        GIAB_fp_mis.dropna(subset=["mut_non_clipped_fraction"], inplace=True)
        GIAB_fp_mis.dropna(
            subset=["overall_mean_flanking_sbs_num_per_bp"], inplace=True
        )
        return GIAB_fp_mis
    else:
        raise ValueError(f"Invalid mis_or_indel parameter: {mis_or_indel}")


# 太高的 stutter 可能导致 germ 或者 mut reads assignment 错误和缺失，导致没有 germ 或者 mut，从而导致缺失值这儿（7/605 出现）
# STR 长度等于 read length 缺失可能导致 flanking 没有碱基导致缺失值这儿（1/605 出现），我们没办法处理这么长的 STR，或者也可能是上述的原因，更可能是上述的原因
# [wangweixiang@login01 log_pre]$ cat *|grep 包含缺失值的列
# 包含缺失值的列: ['germ_mut_diff_mapQ', 'germ_mut_mean_baseQ_statistic_single_tail', 'germ_mut_str_mis_fraction_statistic', 'germ_non_clipped_fraction', 'germ_mut_diff_non_clipped_fraction', 'germ_mut_diff_non_flanking_indel_fraction', 'germ_mut_diff_proper_pair', 'germ_mut_diff_indel', 'germ_mut_diff_mis', 'germ_mut_diff_errors', 'germ_mut_mapQ_pvalue_norm', 'germ_mut_mean_baseQ_pvalue_single_tail_norm', 'germ_mut_strand_pvalue_norm', 'germ_mut_orientation_pvalue_norm', 'germ_mut_temp_length_pvalue_norm', 'germ_mut_str_mis_fraction_pvalue_norm']
# 包含缺失值的列: ['source_mut_diff_base_accuracy', 'source_mut_base_accuracy_statistic', 'mut_fraction_baseq_more_cutoff', 'germ_mut_diff_mapQ', 'germ_mut_mean_baseQ_statistic_single_tail', 'mut_non_clipped_fraction', 'germ_mut_diff_non_clipped_fraction', 'mut_non_flanking_indel_fraction', 'germ_mut_diff_non_flanking_indel_fraction', 'mut_strand_fraction', 'mut_orientation_fraction', 'mut_proper_pair_fraction', 'germ_mut_diff_proper_pair', 'germ_mut_diff_indel', 'germ_mut_diff_mis', 'germ_mut_diff_errors', 'mut_indel', 'mut_mis', 'mut_errors', 'mut_mapQ_mean', 'germ_mut_flanking_bps_num_statistic', 'germ_mut_read_pcr_start_num_statistic', 'germ_mut_mapQ_pvalue_norm', 'germ_mut_mean_baseQ_pvalue_single_tail_norm', 'germ_mut_strand_pvalue_norm', 'germ_mut_orientation_pvalue_norm', 'germ_mut_temp_length_pvalue_norm', 'source_mut_base_accuracy_pvalue_norm', 'germ_mut_flanking_bps_num_pvalue_norm', 'germ_mut_read_pcr_cycle_percentage_sum_pvalue_norm', 'germ_mut_read_pcr_start_num_pvalue_norm']
# 包含缺失值的列: ['overall_mean_flanking_sbs_num_per_bp']
# 包含缺失值的列: ['germ_mut_diff_mapQ', 'germ_mut_mean_baseQ_statistic_single_tail', 'germ_mut_str_mis_fraction_statistic', 'mut_non_clipped_fraction', 'germ_mut_diff_non_clipped_fraction', 'mut_non_flanking_indel_fraction', 'germ_mut_diff_non_flanking_indel_fraction', 'mut_strand_fraction', 'mut_orientation_fraction', 'mut_proper_pair_fraction', 'germ_mut_diff_proper_pair', 'germ_mut_diff_indel', 'germ_mut_diff_mis', 'germ_mut_diff_errors', 'mut_indel', 'mut_mis', 'mut_errors', 'mut_mapQ_mean', 'mut_overall_baseQ_more_fraction', 'germ_mut_mapQ_pvalue_norm', 'germ_mut_mean_baseQ_pvalue_single_tail_norm', 'germ_mut_strand_pvalue_norm', 'germ_mut_orientation_pvalue_norm', 'germ_mut_temp_length_pvalue_norm', 'germ_mut_str_mis_fraction_pvalue_norm']


if MODEL_VAF_CORRECTION:
    mosaic_fraction_use = "mosaic_fraction_correction"
    obs_vaf_use = "obs_vaf_correction"
else:
    mosaic_fraction_use = "MF_hom2het_het2het"
    obs_vaf_use = "observed_mosaic_allele_vaf_single_locus"

indel_model_features_order = [
    mosaic_fraction_use,  # "MF_hom2het_het2het", 使用校正的 VAF
    "stutter_ratio",
    "mut_noise_diff_frac",
    obs_vaf_use,  # "observed_mosaic_allele_vaf_single_locus",  使用校正的 VAF
    "NORMGERM1",
    "NORMGERM2",
    "NORMMOSAIC",
    "NORMSECOND",
    "average_mappability_score_k24",
    "overall_mean_mapQ",
    "germ_mut_diff_mapQ",
    "germ_mut_mean_baseQ_statistic_single_tail",
    "germ_mut_str_mis_fraction_statistic",
    "overall_mean_flanking_sbs_num_per_bp",
    "germ_mut_flanking_sbs_num_per_bp_statistic",
    "germ_non_clipped_fraction",
    "mut_non_clipped_fraction",
    "germ_mut_diff_non_clipped_fraction",
    "mut_non_flanking_indel_fraction",
    "germ_mut_diff_non_flanking_indel_fraction",
    "mut_strand_fraction",
    "mut_orientation_fraction",
    "mut_proper_pair_fraction",
    "germ_mut_diff_proper_pair",
    "germ_mut_diff_indel",
    "germ_mut_diff_mis",
    "germ_mut_diff_errors",
    "mut_indel",
    "mut_mis",
    "mut_errors",
    "flanking_abs_diff_baseq_mean",
    "flanking_baseq_mean_statistic",
    "flanking_abs_diff_sbs_num_mean",
    "flanking_sbs_num_mean_statistic",
    "mut_mapQ_mean",
    "mut_overall_baseQ_more_fraction",
    "mut_allele_ks_start_stat",
    "mut_allele_ks_end_stat",
    "binomial_noise_p_value_norm",
    "germ_mut_mapQ_pvalue_norm",
    "germ_mut_mean_baseQ_pvalue_single_tail_norm",
    "germ_mut_flanking_sbs_num_per_bp_pvalue_norm",
    "germ_mut_strand_pvalue_norm",
    "germ_mut_orientation_pvalue_norm",
    "exact_match_p_value_renew_norm",
    "flanking_baseq_mean_p_norm",
    "flanking_sbs_num_mean_p_norm",
    "germ_mut_temp_length_pvalue_norm",
    "germ_mut_str_mis_fraction_pvalue_norm",
    "mut_allele_ks_start_p_value_norm",
    "mut_allele_ks_end_p_value_norm",
]

mis_model_features_order = [
    "MF_hom2het_het2het",
    "NALLELES_PER_STR_BP",
    "observed_mosaic_allele_vaf_single_locus",
    "stutter_ratio",
    "mut_noise_diff_frac",
    "MLEUAF",
    "NORMGERM1",
    "NORMGERM2",
    "NORMMOSAIC",
    "NORMSECOND",
    "average_mappability_score_k24",
    "source_mut_diff_base_accuracy",
    "source_mut_base_accuracy_statistic",
    "mut_fraction_baseq_more_cutoff",
    "overall_mean_mapQ",
    "germ_mut_diff_mapQ",
    "germ_mut_mean_baseQ_statistic_single_tail",
    "overall_mean_flanking_sbs_num_per_bp",
    "germ_mut_flanking_sbs_num_per_bp_statistic",
    "germ_non_clipped_fraction",
    "mut_non_clipped_fraction",
    "germ_mut_diff_non_clipped_fraction",
    "overall_non_flanking_indel_fraction",
    "mut_non_flanking_indel_fraction",
    "germ_mut_diff_non_flanking_indel_fraction",
    "mut_strand_fraction",
    "mut_orientation_fraction",
    "mut_proper_pair_fraction",
    "germ_mut_diff_proper_pair",
    "germ_mut_diff_indel",
    "germ_mut_diff_mis",
    "germ_mut_diff_errors",
    "mut_indel",
    "mut_mis",
    "mut_errors",
    "flanking_abs_diff_baseq_mean",
    "flanking_baseq_mean_statistic",
    "flanking_abs_diff_sbs_num_mean",
    "flanking_sbs_num_mean_statistic",
    "mut_mapQ_mean",
    "mut_allele_ks_start_stat",
    "mut_allele_ks_end_stat",
    "germ_mut_flanking_bps_num_statistic",
    "germ_mut_read_pcr_start_num_statistic",
    "germ_mut_mapQ_pvalue_norm",
    "germ_mut_mean_baseQ_pvalue_single_tail_norm",
    "germ_mut_flanking_sbs_num_per_bp_pvalue_norm",
    "germ_mut_strand_pvalue_norm",
    "germ_mut_orientation_pvalue_norm",
    "exact_match_p_value_renew_norm",
    "flanking_baseq_mean_p_norm",
    "flanking_sbs_num_mean_p_norm",
    "germ_mut_temp_length_pvalue_norm",
    "mut_allele_ks_start_p_value_norm",
    "mut_allele_ks_end_p_value_norm",
    "source_mut_base_accuracy_pvalue_norm",
    "germ_mut_flanking_bps_num_pvalue_norm",
    "germ_mut_read_pcr_cycle_percentage_sum_pvalue_norm",
    "germ_mut_read_pcr_start_num_pvalue_norm",
    "binomial_noise_p_value_norm",
]


# def get_popAF(population_panel, reference_fasta, final_mosaic_merge_df):
#     if population_panel:
#         bed_pysam = pysam.TabixFile(population_panel)
#         pysam_fasta = pysam.FastaFile(reference_fasta)
#         final_mosaic_merge_df["mutant_popAF"] = "NA"
#         final_mosaic_merge_df["germ_popAF"] = "NA"
#         final_mosaic_merge_df["source_popAF"] = "NA"
#         for idx, row in final_mosaic_merge_df.iterrows():
#             STRSTART = int(row["STRSTART"])
#             STREND = int(row["STREND"])
#             chrom = str(row["chrom"])
#             for j in bed_pysam.fetch(chrom, STRSTART, STREND):
#                 row_list = j.split("\t")
#                 allele_start = int(row_list[1])
#                 allele_end = int(row_list[2])
#                 allele_af = row_list[9:]
#                 allele_af_dict = {}
#                 if (
#                     (allele_start >= STRSTART - 5)
#                     and (allele_start <= STRSTART + 5)
#                     and (allele_end >= STREND - 5)
#                     and (allele_end <= STREND + 5)
#                 ):
#                     start_offset = allele_start - (STRSTART - 5)
#                     end_offset = (STREND + 5) - allele_end
#                     for k in allele_af:
#                         i_list = k.split(":")
#                         allele_seq = i_list[0]
#                         allele_frq = float(i_list[1])
#                         if start_offset > 0:
#                             left_padding_seq = pysam_fasta.fetch(
#                                 chrom, STRSTART - 5, allele_start
#                             )
#                         else:
#                             left_padding_seq = ""
#                         if end_offset > 0:
#                             right_padding_seq = pysam_fasta.fetch(
#                                 chrom, allele_end, STREND + 5
#                             )
#                         else:
#                             right_padding_seq = ""
#                         allele_seq_padding = (
#                             left_padding_seq + allele_seq + right_padding_seq
#                         )
#                         allele_af_dict[allele_seq_padding.upper()] = allele_frq
#                     germ_allele_af = allele_af_dict.get(row["germ_seq"].upper(), 0)
#                     source_allele_af = allele_af_dict.get(
#                         row["source_seq"].upper(), 0
#                     )
#                     mosaic_allele_af = allele_af_dict.get(
#                         row["mut_seq"].upper(), 0
#                     )
#                     final_mosaic_merge_df.loc[
#                         idx, "mutant_popAF"
#                     ] = mosaic_allele_af
#                     final_mosaic_merge_df.loc[idx, "germ_popAF"] = germ_allele_af
#                     final_mosaic_merge_df.loc[
#                         idx, "source_popAF"
#                     ] = source_allele_af
#                     if mosaic_allele_af != 0:
#                         break
#                 else:
#                     if final_mosaic_merge_df.loc[idx, "mutant_popAF"] != "NA":
#                         pass
#                     else:
#                         final_mosaic_merge_df.loc[idx, "mutant_popAF"] = "NoItem"
#                         final_mosaic_merge_df.loc[idx, "germ_popAF"] = "NoItem"
#                         final_mosaic_merge_df.loc[idx, "source_popAF"] = "NoItem"
#     else:
#         final_mosaic_merge_df["mutant_popAF"] = "NA"
#         final_mosaic_merge_df["germ_popAF"] = "NA"
#         final_mosaic_merge_df["source_popAF"] = "NA"
#     return final_mosaic_merge_df


def get_reference_chr_pos_ref_alt_per_row(row, pysam_fasta):
    if PADDING_BPS > 0:
        refseq_left_padding_1bp = pysam_fasta.fetch(
            str(row["chrom"]),
            row["STRSTART"] - 1 - PADDING_BPS,
            row["STRSTART"] - PADDING_BPS,
        )
        reference_seq = pysam_fasta.fetch(
            str(row["chrom"]),
            row["STRSTART"] - PADDING_BPS,
            row["STREND"] + PADDING_BPS,
        )
    else:
        refseq_left_padding_1bp = pysam_fasta.fetch(
            str(row["chrom"]), row["STRSTART"] - 1, row["STRSTART"]
        )
        reference_seq = pysam_fasta.fetch(
            str(row["chrom"]), row["STRSTART"], row["STREND"]
        )
    muttype = row["muttype"]
    germ_allele = row["germ_seq"]
    source_allele = row["source_seq"]
    mut_allele = row["mut_seq"]

    if muttype == "homhet" or muttype == "germlinehet":
        allele_seq = [
            germ_allele,
            mut_allele,
        ]
        (
            alleles_mut_type,
            mutation_site_seq1,
            mutation_site_seq2,
            source_seq,
            target_seq,
        ) = adjust_vcf.STR_specific_alignment(
            germ_allele,
            mut_allele,
            refseq_left_padding_1bp,
        )  # HACK: 0-based mutation sites
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
            reference_seq,
            allele_seq[0],
            allele_seq[0],
            allele_seq[1],
            refseq_left_padding_1bp,
            PADDING_BPS,
        )
        seq0_length = len(germ_allele)
        seq1_length = len(germ_allele)
        seq2_length = len(mut_allele)
        if seq0_length == seq2_length:
            (
                _,
                mis_location,
            ) = myutils.cal_edit_distance_based_on_mismatch(
                germ_allele, mut_allele
            )
        else:
            mis_location = []
    elif muttype == "hethet":
        allele_seq = [
            germ_allele,
            source_allele,
            mut_allele,
        ]
        (
            alleles_mut_type,
            mutation_site_seq1,
            mutation_site_seq2,
            source_seq,
            target_seq,
        ) = adjust_vcf.STR_specific_alignment(
            source_allele,
            mut_allele,
            refseq_left_padding_1bp,
        )  # HACK: 0-based mutation sites
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
            reference_seq,
            allele_seq[0],
            allele_seq[1],
            allele_seq[2],
            refseq_left_padding_1bp,
            PADDING_BPS,
        )
        seq0_length = len(germ_allele)
        seq1_length = len(source_allele)
        seq2_length = len(mut_allele)
        if seq1_length == seq2_length:
            (
                _,
                mis_location,
            ) = myutils.cal_edit_distance_based_on_mismatch(
                source_allele, mut_allele
            )
        else:
            mis_location = []
    mut_source_seq = source_seq
    mut_target_seq = target_seq
    mut_source_seq_pos_tuple_0_based_leftrightclose = mutation_site_seq1
    mut_target_seq_pos_tuple_0_based_leftrightclose = mutation_site_seq2
    reference_start_index_0_based_include = refseq_left_index
    reference_end_index_0_based_include = refseq_right_index
    reference_start_coordinate_1_based_include = (
        row["STRSTART"]
        - PADDING_BPS
        + reference_start_index_0_based_include
        + 1
    )
    reference_end_coordinate_1_based_include = (
        row["STRSTART"] - PADDING_BPS + reference_end_index_0_based_include + 1
    )
    (
        mutated_reference_seq,
        relative_to_reference_germ_seq,
        relative_to_reference_source_seq,
        relative_to_reference_mut_seq,
        reference_start_coordinate_1_based_include,
        reference_end_coordinate_1_based_include,
    ) = adjust_vcf.normalize_global_vcf(
        refseq_seq_reference,
        germ_allele_seq_to_refseq_alternative,
        source_allele_seq_to_refseq_alternative,
        mutant_allele_seq_to_refseq_alternative,
        reference_start_coordinate_1_based_include,
        reference_end_coordinate_1_based_include,
    )
    return (
        mut_source_seq,
        mut_target_seq,
        mut_source_seq_pos_tuple_0_based_leftrightclose,
        mut_target_seq_pos_tuple_0_based_leftrightclose,
        reference_start_index_0_based_include,
        reference_end_index_0_based_include,
        reference_start_coordinate_1_based_include,
        reference_end_coordinate_1_based_include,
        mutated_reference_seq,
        relative_to_reference_germ_seq,
        relative_to_reference_source_seq,
        relative_to_reference_mut_seq,
    )


def VAF_correction(muttype, vaf, mutation_size, read_length, obs_gt_dp):
    if muttype == "hethet":
        return vaf
    else:
        if mutation_size > 0:
            mutation_size_ratio = 1 - (mutation_size / read_length)
            mut_obs_dp = obs_gt_dp * vaf
            mut_real_dp = mut_obs_dp / mutation_size_ratio
            mut_real_vaf = mut_real_dp / (obs_gt_dp + mut_real_dp - mut_obs_dp)
            return mut_real_vaf
        elif mutation_size < 0:
            mutation_size_ratio = 1 - (abs(mutation_size) / read_length)
            germ_obs_dp = obs_gt_dp * (1 - vaf)
            germ_real_dp = germ_obs_dp / mutation_size_ratio
            germ_real_vaf = germ_real_dp / (
                obs_gt_dp + germ_real_dp - germ_obs_dp
            )
            return 1 - germ_real_vaf
        else:
            return vaf


def VAF_correction_to_df(row):
    muttype = row["muttype"]
    vaf = row["observed_mosaic_allele_vaf_single_locus"]
    mutation_size = row["MBP"]
    read_length = row["read_length"]
    obs_gt_dp = row["depth"]
    return VAF_correction(muttype, vaf, mutation_size, read_length, obs_gt_dp)


def check_nan(value, default_value):
    if pd.isna(value):
        return default_value
    else:
        return value


def check_no_item(value, default_value):
    if (value == "NoItem") or (value == "NA") or (value == ".") or (value is None):
        return default_value
    else:
        return value


def hard_filter_mode(row):
    # HARD_FILTER_DETAILS and HARD_FILTER_CATEGORY
    hard_filter_details = ""
    hard_filter_category = []
    # LOW MUTANT NOISE FILTER
    if VAF_CORRECTION:
        obs_vaf = VAF_correction(
            row["muttype"],
            row["observed_mosaic_allele_vaf_single_locus"],
            row["MBP"],
            row["read_length"],
            row["depth"],
        )
        mosaic_fraction = (
            VAF_correction(
                row["muttype"],
                row["MF_hom2het_het2het"] / 2,
                row["MBP"],
                row["read_length"],
                row["depth"],
            )
            * 2
        )
    else:
        obs_vaf = row["observed_mosaic_allele_vaf_single_locus"]
        mosaic_fraction = row["MF_hom2het_het2het"]
    mutation_size = row["MBP"]
    mutation_period = mutation_size / row["MOTIF_length"]
    depth_ratio = row["no_norm_len_revise_depth_ratio"]
    muttype = row["muttype"]
    germ_pop_af = check_nan(row["germ_popAF"], 0)
    mutant_pop_af = check_nan(row["mosaic_popAF"], 0)
    source_pop_af = check_nan(row["source_popAF"], 0)
    external_germ_pop_af = check_nan(row["external_germ_popAF"], 0)
    external_mosaic_pop_af = check_nan(row["external_mosaic_popAF"], 0)
    external_source_pop_af = check_nan(row["external_source_popAF"], 0)
    germ_pop_af = float(check_no_item(germ_pop_af, 0))
    mutant_pop_af = float(check_no_item(mutant_pop_af, 0))
    source_pop_af = float(check_no_item(source_pop_af, 0))
    external_germ_pop_af = float(check_no_item(external_germ_pop_af, 0))
    external_mosaic_pop_af = float(check_no_item(external_mosaic_pop_af, 0))
    external_source_pop_af = float(check_no_item(external_source_pop_af, 0))
    mutant_dp = row["observed_mosaic_allele_vaf_single_locus"] * row["depth"]
    genotyping_source_dp = row["genotyping_obs_source_dp"]
    genotyping_germ_dp = row["genotyping_obs_germ_dp"]
    genotyping_mutant_dp = row["genotyping_obs_mutant_dp"]
    mutation_site_type = row["mutation_site_type"]
    mutation_location_type = row["mutation_location_type"]
    alleles_mut_type = row["alleles_mut_type"]
    motif_length = row["MOTIF_length"]
    used_read_num_in_genotyping = row["used_read_num_in_genotyping"]
    used_read_fraction_in_genotyping = row["used_read_fraction_in_genotyping"]
    unknown_hap_read_fraction = row["unknown_hap_read_fraction"]
    # overall_non_flanking_indel_fraction = row["overall_non_flanking_indel_fraction"]
    flanking_abs_diff_baseq_mean = row["flanking_abs_diff_baseq_mean"]
    flanking_abs_diff_baseq_mean_germ_source = row[
        "flanking_abs_diff_baseq_mean_germ_source"
    ]
    mut_non_clipped_fraction_germ_source = row[
        "mut_non_clipped_fraction_germ_source"
    ]
    mut_non_clipped_fraction = row["mut_non_clipped_fraction"]
    germ_non_clipped_fraction = row["germ_non_clipped_fraction"]
    mut_indel_germ_source = row["mut_indel_germ_source"]
    germ_indel = row["germ_indel"]
    mut_indel = row["mut_indel"]
    stutter_ratio = row["stutter_ratio"]
    mut_non_flanking_indel_fraction_germ_source = row[
        "mut_non_flanking_indel_fraction_germ_source"
    ]
    mut_non_flanking_indel_fraction = row["mut_non_flanking_indel_fraction"]
    mut_allele_unique_start_coordinate_number = row[
        "mut_allele_unique_start_coordinate_number"
    ]
    mut_allele_unique_end_coordinate_number = row[
        "mut_allele_unique_end_coordinate_number"
    ]
    source_allele_unique_start_coordinate_number = row[
        "source_allele_unique_start_coordinate_number"
    ]
    source_allele_unique_end_coordinate_number = row[
        "source_allele_unique_end_coordinate_number"
    ]
    left_flanking_bps_num_mut_feature_mean = row[
        "left_flanking_bps_num_mut_feature_mean"
    ]
    right_flanking_bps_num_mut_feature_mean = row[
        "right_flanking_bps_num_mut_feature_mean"
    ]
    left_flanking_bps_num_mut_feature_less_five_fraction = row[
        "left_flanking_bps_num_mut_feature_less_five_fraction"
    ]
    right_flanking_bps_num_mut_feature_less_five_fraction = row[
        "right_flanking_bps_num_mut_feature_less_five_fraction"
    ]
    left_flanking_bps_num_mut_feature_mean_germ_source = row[
        "left_flanking_bps_num_mut_feature_mean_germ_source"
    ]
    right_flanking_bps_num_mut_feature_mean_germ_source = row[
        "right_flanking_bps_num_mut_feature_mean_germ_source"
    ]
    left_flanking_bps_num_mut_feature_less_five_fraction_germ_source = row[
        "left_flanking_bps_num_mut_feature_less_five_fraction_germ_source"
    ]
    right_flanking_bps_num_mut_feature_less_five_fraction_germ_source = row[
        "right_flanking_bps_num_mut_feature_less_five_fraction_germ_source"
    ]
    overall_mean_mapQ = row["overall_mean_mapQ"]
    mut_mapQ_mean = row["mut_mapQ_mean"]
    mut_mapQ_more_fraction = row["mut_mapQ_more_fraction"]
    mut_mapQ_more_fraction_germ_source = row[
        "mut_mapQ_more_fraction_germ_source"
    ]
    mut_mapQ_mean_germ_source = row["mut_mapQ_mean_germ_source"]
    overall_mean_flanking_sbs_num_per_bp = row[
        "overall_mean_flanking_sbs_num_per_bp"
    ]
    mut_mean_flanking_sbs_num_per_bp = row["mut_mean_flanking_sbs_num_per_bp"]
    mut_mean_flanking_sbs_num_per_bp_germ_source = row[
        "mut_mean_flanking_sbs_num_per_bp_germ_source"
    ]
    if row["uncallable_frac"] == ".":
        uncallable_frac = 0
        callable_frac = 1
    else:
        uncallable_frac = float(row["uncallable_frac"])
        callable_frac = 1 - uncallable_frac
    if row["recurrent_num"] == ".":
        recurrent_num = 0
        recurrent_fraction = 0
    else:
        recurrent_num = row["recurrent_num"]
        recurrent_fraction = row["recurrent_fraction"]
    if row["sample_num"] == ".":
        uncallable_num = 0
        sample_num = 1
        callable_num = 1
    else:
        uncallable_num = row["uncallable_num"]
        sample_num = row["sample_num"]
        callable_num = sample_num - uncallable_num
    in_or_out_frame = row["frame"]
    if mutation_size > 0:
        if in_or_out_frame == "inframe":
            stutter_probs = row["inframe_ins_prob"]
        else:
            stutter_probs = row["outframe_ins_prob"]
    elif mutation_size < 0:
        if in_or_out_frame == "inframe":
            stutter_probs = row["inframe_del_prob"]
        else:
            stutter_probs = row["outframe_del_prob"]
    else:
        stutter_probs = 0
    source_site_start = row[
        "mutation_site_start_seq1"
    ]  # HACK 1-based left-right flanking 1bp is the same value
    source_site_end = row[
        "mutation_site_end_seq1"
    ]  # HACK 1-based left-right flanking is the same value
    mutation_site_start = row[
        "mutation_site_start_seq2"
    ]  # HACK 1-based left-right flanking is the same value
    mutation_site_end = row[
        "mutation_site_end_seq2"
    ]  # HACK 1-based left-right flanking is the same value
    if alleles_mut_type in MISMATCH_CATE:
        mut_mean_baseq = row["mut_mean_baseq"]
        mut_fraction_baseq_more_cutoff = row["mut_fraction_baseq_more_cutoff"]
    else:
        mut_mean_baseq = 30
        mut_fraction_baseq_more_cutoff = 1
    mutant_strand_frac = row["mut_strand_fraction"]
    hard_filter_decision_list = []
    if obs_vaf < 0.01 or mosaic_fraction < 0.02:
        hard_filter_details = hard_filter_details + "AF<0.01;"
        hard_filter_category.append("LowMutantArtifacts")
    elif obs_vaf < 0.02 or mosaic_fraction < 0.04:
        hard_filter_details = hard_filter_details + "AF<0.02;"
        hard_filter_category.append("LowMutantArtifacts")
    elif obs_vaf < 0.03 or mosaic_fraction < 0.06:
        hard_filter_details = hard_filter_details + "AF<0.03;"
    if (
        obs_vaf < OBSERVED_MOSAIC_ALLELE_VAF_SINGLE_LOCUS_MIN
        or mosaic_fraction < MF_HOM2HET_HET2HET_MIN
    ):
        hard_filter_decision_list.append("Artifacts")
    if (overall_mean_mapQ < MAQ_MEAN_MIN) or (mut_mapQ_mean < MAQ_MEAN_MIN):
        hard_filter_details = hard_filter_details + f"MeanMapQ<{MAQ_MEAN_MIN};"
        hard_filter_category.append("LowMapQArtifacts")
        hard_filter_decision_list.append("MappingErrors")
    if mut_mapQ_more_fraction < HIGH_MAPQ_FRAC_MIN:
        hard_filter_details = (
            hard_filter_details + f"HighMapQFrac<{HIGH_MAPQ_FRAC_MIN};"
        )
        hard_filter_category.append("LowMapQArtifacts")
        hard_filter_decision_list.append("MappingErrors")
    if (
        overall_mean_flanking_sbs_num_per_bp
        > MISMATCHES_NUM_PER_BP_PER_READ_MAX
    ) or (
        mut_mean_flanking_sbs_num_per_bp > MISMATCHES_NUM_PER_BP_PER_READ_MAX
    ):
        hard_filter_details = (
            hard_filter_details
            + f"MisPerBP>{MISMATCHES_NUM_PER_BP_PER_READ_MAX};"
        )
        hard_filter_category.append("MuchMisArtifacts")
        hard_filter_decision_list.append("SequencingErrors")
    if mutant_dp < 1.2:
        hard_filter_details = hard_filter_details + "mut=1;"
        hard_filter_category.append("LowMutantArtifacts")
    elif mutant_dp < (MUTANT_DP_MIN - 0.1):
        hard_filter_details = hard_filter_details + f"mut<{MUTANT_DP_MIN};"
        hard_filter_category.append("LowMutantArtifacts")
    if mutant_dp < (MUTANT_DP_MIN - 0.1):
        hard_filter_decision_list.append("Artifacts")
    if (
        (genotyping_mutant_dp < PER_ALLELE_MIN_DP - 0.1)
        or (genotyping_source_dp < PER_ALLELE_MIN_DP - 0.1)
        or (genotyping_germ_dp < PER_ALLELE_MIN_DP - 0.1)
    ):
        hard_filter_details = hard_filter_details + f"adp<{PER_ALLELE_MIN_DP};"
        hard_filter_category.append("LowAlleleDPArtifacts")
        hard_filter_decision_list.append("Artifacts")
    if mut_mean_baseq < MUT_MEAN_BASEQ_MIN_SINGLE:
        hard_filter_details = (
            hard_filter_details + f"baseq<{MUT_MEAN_BASEQ_MIN_SINGLE};"
        )
        hard_filter_category.append("LowBaseQArtifacts")
        hard_filter_decision_list.append("Artifacts")
    elif (mut_mean_baseq < MUT_MEAN_BASEQ_MIN) or (
        mut_fraction_baseq_more_cutoff < MUT_FRACTION_BASEQ_MORE_CUTOFF
    ):
        hard_filter_details = (
            hard_filter_details
            + f"(baseq<{MUT_MEAN_BASEQ_MIN})|((high_baseq_frac<{MUT_FRACTION_BASEQ_MORE_CUTOFF}));"
        )
        hard_filter_category.append("LowBaseQArtifacts")
        hard_filter_decision_list.append("Artifacts")
    if (
        mosaic_fraction > MF_HOM2HET_HET2HET_MAX
        or obs_vaf > OBSERVED_MOSAIC_ALLELE_VAF_SINGLE_LOCUS_MAX
    ):
        hard_filter_details = (
            hard_filter_details
            + f"AF>{OBSERVED_MOSAIC_ALLELE_VAF_SINGLE_LOCUS_MAX};"
        )
        hard_filter_category.append("HighAFGermline")
        hard_filter_decision_list.append("GermHet")
    if depth_ratio > NO_NORM_LEN_REVISE_DEPTH_RATIO_MAX:
        hard_filter_details = (
            hard_filter_details
            + f"(dp_ratio>{NO_NORM_LEN_REVISE_DEPTH_RATIO_MAX});"
        )
        hard_filter_category.append("ExtraHighDepth")
        hard_filter_decision_list.append("Repeat")
    elif depth_ratio < NO_NORM_LEN_REVISE_DEPTH_RATIO_MIN:
        hard_filter_details = (
            hard_filter_details
            + f"(dp_ratio<{NO_NORM_LEN_REVISE_DEPTH_RATIO_MIN});"
        )
        hard_filter_category.append("ExtraLowDepth")
        hard_filter_decision_list.append("LowEvidence")
    if used_read_num_in_genotyping < USED_READ_NUM_IN_GENOTYPING_MIN:
        hard_filter_details = (
            hard_filter_details + f"dp<{USED_READ_NUM_IN_GENOTYPING_MIN};"
        )
        hard_filter_category.append("ExtraLowDepth")
        hard_filter_decision_list.append("LowEvidence")
    if (mutation_location_type == "flanking") and (
        alleles_mut_type in MISMATCH_CATE
    ):
        hard_filter_details = hard_filter_details + "flanking_mis;"
        hard_filter_category.append("FlankingMis")
        if FLANKING_MIS_EXCLUDE:
            hard_filter_decision_list.append("FlankingSequencingErrors")
    if (
        (int(source_site_start) == 5)
        and (int(source_site_end) == 5)
        and (int(mutation_site_start) == 5)
        and (int(mutation_site_end) == 5)
        and (int(motif_length) == 1)
        and (alleles_mut_type == "SNV")
        and (mutant_strand_frac > STRAND_FRAC_MAX)
        and (
            (mut_mean_baseq < POST_HOMOPOLYMER_BASEQ_MIN)
            or (
                mut_fraction_baseq_more_cutoff
                < POST_HOMOPOLYMER_BASEQ_FRAC_MIN
            )
        )
    ):
        hard_filter_details = (
            hard_filter_details + "post_homopolymer_mismatch;"
        )
        hard_filter_category.append("FlankingMis")
        if POST_HOMOPOLYMER_MISMATCH:
            hard_filter_decision_list.append("FlankingSequencingErrors")
    if ONLY_HOMHET:
        if muttype == "hethet":
            hard_filter_details = hard_filter_details + "hethet;"
            hard_filter_category.append("hethet")
            hard_filter_decision_list.append("HetHet")
    # print(external_mosaic_pop_af)
    if muttype == "homhet":
        if (external_mosaic_pop_af > POP_AF_MAX) or (
            (mutant_pop_af > POP_AF_MAX)
            and (callable_num > CALLABLE_NUM_MIN)
        ):
            hard_filter_details = hard_filter_details + f"popAF>{POP_AF_MAX};"
            hard_filter_category.append("HighPopAFGermline")
            hard_filter_decision_list.append("GermHet")
        if (
            (external_mosaic_pop_af > POPAF_AF_INDEL_POPAF_MIN)
            or (
                (mutant_pop_af > POPAF_AF_INDEL_POPAF_MIN)
                and (callable_num > CALLABLE_NUM_MIN)
            )
        ) and (alleles_mut_type in INDEL_CATE):
            if (mosaic_fraction / 2) > POPAF_AF_INDEL_AF_MIN or (
                obs_vaf > POPAF_AF_INDEL_AF_MIN
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"(popAF>{POPAF_AF_INDEL_POPAF_MIN})|(AF>{POPAF_AF_INDEL_AF_MIN})&(indel);"
                )
                hard_filter_category.append("HighPopAFGermline")
                hard_filter_decision_list.append("GermHet")
            else:
                hard_filter_details = (
                    hard_filter_details
                    + f"(popAF>{POPAF_AF_INDEL_POPAF_MIN})&(indel);"
                )
        if (
            (external_mosaic_pop_af > POPAF_AF_MISMATCH_POPAF_MIN)
            or (
                (mutant_pop_af > POPAF_AF_MISMATCH_POPAF_MIN)
                and (callable_num > CALLABLE_NUM_MIN)
            )
        ) and (alleles_mut_type in MISMATCH_CATE):
            if (mosaic_fraction / 2) > POPAF_AF_MISMATCH_AF_MIN or (
                obs_vaf > POPAF_AF_MISMATCH_AF_MIN
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"(popAF>{POPAF_AF_MISMATCH_POPAF_MIN})|(AF>{POPAF_AF_MISMATCH_AF_MIN})&(mismatch);"
                )
                hard_filter_category.append("HighPopAFGermline")
                hard_filter_decision_list.append("GermHet")
            else:
                hard_filter_details = (
                    hard_filter_details
                    + f"(popAF>{POPAF_AF_MISMATCH_POPAF_MIN})&(mismatch);"
                )
                hard_filter_category.append("HighPopAFGermline")
                hard_filter_decision_list.append("GermHet")
    elif muttype == "hethet" or muttype == "clonalhet":
        if (
            (external_mosaic_pop_af > POP_AF_MAX)
            or (
                (mutant_pop_af > POP_AF_MAX)
                and (callable_num > CALLABLE_NUM_MIN)
            )
        ) and (
            (external_source_pop_af > POP_AF_MAX)
            or (
                (source_pop_af > POP_AF_MAX)
                and (callable_num > CALLABLE_NUM_MIN)
            )
        ):
            hard_filter_details = hard_filter_details + f"popAF>{POP_AF_MAX};"
            hard_filter_category.append("HighPopAFGermline")
            hard_filter_decision_list.append("GermHet")
        if (
            (
                (external_mosaic_pop_af > POPAF_AF_INDEL_POPAF_MIN)
                or (
                    (mutant_pop_af > POPAF_AF_INDEL_POPAF_MIN)
                    and (callable_num > CALLABLE_NUM_MIN)
                )
            )
            and (
                (external_source_pop_af > POPAF_AF_INDEL_POPAF_MIN)
                or (
                    (source_pop_af > POPAF_AF_INDEL_POPAF_MIN)
                    and (callable_num > CALLABLE_NUM_MIN)
                )
            )
            and (alleles_mut_type in INDEL_CATE)
        ):
            if (mosaic_fraction / 2) > POPAF_AF_INDEL_AF_MIN or (
                obs_vaf > POPAF_AF_INDEL_AF_MIN
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"(popAF>{POPAF_AF_INDEL_POPAF_MIN})|(AF>{POPAF_AF_INDEL_AF_MIN})&(indel);"
                )
                hard_filter_category.append("HighPopAFGermline")
                hard_filter_decision_list.append("GermHet")
            else:
                hard_filter_details = (
                    hard_filter_details
                    + f"(popAF>{POPAF_AF_INDEL_POPAF_MIN})&(indel);"
                )
        if (
            (
                (external_mosaic_pop_af > POPAF_AF_MISMATCH_POPAF_MIN)
                or (
                    (mutant_pop_af > POPAF_AF_MISMATCH_POPAF_MIN)
                    and (callable_num > CALLABLE_NUM_MIN)
                )
            )
            and (
                (external_source_pop_af > POPAF_AF_MISMATCH_POPAF_MIN)
                or (
                    (source_pop_af > POPAF_AF_MISMATCH_POPAF_MIN)
                    and (callable_num > CALLABLE_NUM_MIN)
                )
            )
            and (alleles_mut_type in MISMATCH_CATE)
        ):
            if (mosaic_fraction / 2) > POPAF_AF_MISMATCH_AF_MIN or (
                obs_vaf > POPAF_AF_MISMATCH_AF_MIN
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"(popAF>{POPAF_AF_MISMATCH_POPAF_MIN})|(AF>{POPAF_AF_MISMATCH_AF_MIN})&(mismatch);"
                )
                hard_filter_category.append("HighPopAFGermline")
                hard_filter_decision_list.append("GermHet")
            else:
                hard_filter_details = (
                    hard_filter_details
                    + f"(popAF>{POPAF_AF_MISMATCH_POPAF_MIN})&(mismatch);"
                )
                hard_filter_category.append("HighPopAFGermline")
                hard_filter_decision_list.append("GermHet")
        if muttype == "hethet":
            if (
                mut_non_clipped_fraction_germ_source
                < MUT_NON_CLIPPED_FRACTION_GERM_SOURCE_MIN
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"source_clip_frac>{1-MUT_NON_CLIPPED_FRACTION_GERM_SOURCE_MIN};"
                )
                hard_filter_category.append("MuchClipped")
                hard_filter_decision_list.append("MappingErrors")
            if mut_indel_germ_source > MUT_INDEL_GERM_SOURCE_MAX:
                hard_filter_details = (
                    hard_filter_details
                    + f"source_indel_fraction>{MUT_INDEL_GERM_SOURCE_MAX};"
                )
                hard_filter_category.append("MuchStutter")
                hard_filter_decision_list.append("StutterErrors")
            if (
                mut_non_flanking_indel_fraction_germ_source
                < MUT_NON_FLANKING_INDEL_FRACTION_MIN
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"source_flanking_indel_frac>{1-MUT_NON_FLANKING_INDEL_FRACTION_MIN};"
                )
                hard_filter_category.append("MuchFlankingIndel")
                hard_filter_decision_list.append("MappingErrors")
            if (
                source_allele_unique_start_coordinate_number
                < COORDINATE_NUMBER_MIN
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"src_start_read_dis_num<{COORDINATE_NUMBER_MIN};"
                )
                hard_filter_category.append("UniqueReadDistribution")
                hard_filter_decision_list.append("MappingErrors")
            if (
                source_allele_unique_end_coordinate_number
                < COORDINATE_NUMBER_MIN
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"src_end_read_dis_num<{COORDINATE_NUMBER_MIN};"
                )
                hard_filter_category.append("UniqueReadDistribution")
                hard_filter_decision_list.append("MappingErrors")
            if (
                left_flanking_bps_num_mut_feature_mean_germ_source
                < FLANKING_BPS_MEAN_MIN
                or right_flanking_bps_num_mut_feature_mean_germ_source
                < FLANKING_BPS_MEAN_MIN
                or left_flanking_bps_num_mut_feature_less_five_fraction_germ_source
                > FLANKING_BPS_LESS_FIVE_FRACTION_MAX
                or right_flanking_bps_num_mut_feature_less_five_fraction_germ_source
                > FLANKING_BPS_LESS_FIVE_FRACTION_MAX
            ):
                hard_filter_details = (
                    hard_filter_details + "bias_read_distribution;"
                )
                hard_filter_category.append("BiasReadDistribution")
                hard_filter_decision_list.append("MappingErrors")
            if mut_mapQ_mean_germ_source < MAQ_MEAN_MIN:
                hard_filter_details = (
                    hard_filter_details + f"MeanMapQ<{MAQ_MEAN_MIN};"
                )
                hard_filter_category.append("LowMapQArtifacts")
                hard_filter_decision_list.append("MappingErrors")
            if mut_mapQ_more_fraction_germ_source < HIGH_MAPQ_FRAC_MIN:
                hard_filter_details = (
                    hard_filter_details + f"HighMapQFrac<{HIGH_MAPQ_FRAC_MIN};"
                )
                hard_filter_category.append("LowMapQArtifacts")
                hard_filter_decision_list.append("MappingErrors")
            if (
                mut_mean_flanking_sbs_num_per_bp_germ_source
                > MISMATCHES_NUM_PER_BP_PER_READ_MAX
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"MisPerBP>{MISMATCHES_NUM_PER_BP_PER_READ_MAX};"
                )
                hard_filter_category.append("MuchMisArtifacts")
                hard_filter_decision_list.append("SequencingErrors")
            if (
                flanking_abs_diff_baseq_mean_germ_source
                > FLANKING_ABS_DIFF_BASEQ_MEAN_MAX
            ):
                hard_filter_details = (
                    hard_filter_details
                    + f"flanking_baseq_diff>{FLANKING_ABS_DIFF_BASEQ_MEAN_MAX};"
                )
                hard_filter_category.append("ImbalanceBaseQ")
                hard_filter_decision_list.append("SequencingErrors")
    if used_read_fraction_in_genotyping < USED_READ_FRACTION_IN_GENOTYPING_MIN:
        hard_filter_details = (
            hard_filter_details
            + f"used_read_fraction<{USED_READ_FRACTION_IN_GENOTYPING_MIN};"
        )
        hard_filter_category.append("MuchDirtyReads")
        hard_filter_decision_list.append("DirtyLoci")
    if unknown_hap_read_fraction > UNKNOWN_HAP_READ_FRACTION_MAX:
        hard_filter_details = (
            hard_filter_details
            + f"unassigned_read_fraction>{UNKNOWN_HAP_READ_FRACTION_MAX};"
        )
        hard_filter_category.append("MuchUnAssignedReads")
        hard_filter_decision_list.append("MultiAlleles")
    # if overall_non_flanking_indel_fraction < OVERALL_NON_FLANKING_INDEL_FRACTION_MIN:
    #     hard_filter_details = hard_filter_details + f"flanking_indel_fraction>{1-OVERALL_NON_FLANKING_INDEL_FRACTION_MIN};"
    #     hard_filter_category.append("MuchFlankingIndel")
    if flanking_abs_diff_baseq_mean > FLANKING_ABS_DIFF_BASEQ_MEAN_MAX:
        hard_filter_details = (
            hard_filter_details
            + f"flanking_baseq_diff>{FLANKING_ABS_DIFF_BASEQ_MEAN_MAX};"
        )
        hard_filter_category.append("ImbalanceBaseQ")
        hard_filter_decision_list.append("SequencingErrors")
    if mut_non_flanking_indel_fraction < MUT_NON_FLANKING_INDEL_FRACTION_MIN:
        hard_filter_details = (
            hard_filter_details
            + f"mut_flanking_indel_fraction>{1-MUT_NON_FLANKING_INDEL_FRACTION_MIN};"
        )
        hard_filter_category.append("MuchFlankingIndel")
        hard_filter_decision_list.append("MappingErrors")
    if mut_non_clipped_fraction < MUT_NON_CLIPPED_FRACTION_MIN:
        hard_filter_details = (
            hard_filter_details
            + f"mut_clip_frac>{1-MUT_NON_CLIPPED_FRACTION_MIN};"
        )
        hard_filter_category.append("MuchClipped")
        hard_filter_decision_list.append("MappingErrors")
    if germ_non_clipped_fraction < GERM_NON_CLIPPED_FRACTION_MIN:
        hard_filter_details = (
            hard_filter_details
            + f"germ_clip_frac>{1-GERM_NON_CLIPPED_FRACTION_MIN};"
        )
        hard_filter_category.append("MuchClipped")
        hard_filter_decision_list.append("MappingErrors")
    if germ_indel > GERM_INDEL_MAX:
        hard_filter_details = (
            hard_filter_details + f"germ_indel>{GERM_INDEL_MAX};"
        )
        hard_filter_category.append("MuchStutter")
        hard_filter_decision_list.append("StutterErrors")
    if mut_indel > MUT_INDEL_MAX:
        hard_filter_details = (
            hard_filter_details + f"mut_indel>{MUT_INDEL_MAX};"
        )
        hard_filter_category.append("MuchStutter")
        hard_filter_decision_list.append("StutterErrors")
    # HACK: Add low mutation size stutter ratio filter
    if ((stutter_ratio >= STUTTER_RATIO_MAX) and (abs(mutation_size) == 1)):
        hard_filter_details = (
            hard_filter_details + f"stutter_ratio>={STUTTER_RATIO_MAX};"
        )
        hard_filter_category.append("MuchStutter")
        hard_filter_decision_list.append("StutterErrors")
    if (abs(mutation_size) > MUTATION_SIZE_MAX) or (
        abs(mutation_period) > MUTATION_PERIOD_MAX
    ):
        hard_filter_details = (
            hard_filter_details
            + f"(mutation_size>{MUTATION_SIZE_MAX})|(mutation_period>{MUTATION_PERIOD_MAX});"
        )
        hard_filter_category.append("BigInDelSize")
        hard_filter_decision_list.append("BigInDelGermline")
    if mosaic_fraction / 2 <= stutter_probs:
        hard_filter_details = hard_filter_details + "stutter_error;"
        hard_filter_category.append("StutterError")
        if HIGHER_THAN_STUTTER_ERROR_FILTER:
            hard_filter_decision_list.append("StutterErrors")
    if mut_allele_unique_start_coordinate_number < COORDINATE_NUMBER_MIN:
        hard_filter_details = (
            hard_filter_details
            + f"mut_start_read_dis_num<{COORDINATE_NUMBER_MIN};"
        )
        hard_filter_category.append("UniqueReadDistribution")
        hard_filter_decision_list.append("MappingErrors")
    if mut_allele_unique_end_coordinate_number < COORDINATE_NUMBER_MIN:
        hard_filter_details = (
            hard_filter_details
            + f"mut_end_read_dis_num<{COORDINATE_NUMBER_MIN};"
        )
        hard_filter_category.append("UniqueReadDistribution")
        hard_filter_decision_list.append("MappingErrors")
    if (
        left_flanking_bps_num_mut_feature_mean < FLANKING_BPS_MEAN_MIN
        or right_flanking_bps_num_mut_feature_mean < FLANKING_BPS_MEAN_MIN
        or left_flanking_bps_num_mut_feature_less_five_fraction
        > FLANKING_BPS_LESS_FIVE_FRACTION_MAX
        or right_flanking_bps_num_mut_feature_less_five_fraction
        > FLANKING_BPS_LESS_FIVE_FRACTION_MAX
    ):
        hard_filter_details = hard_filter_details + "bias_read_distribution;"
        hard_filter_category.append("BiasReadDistribution")
        hard_filter_decision_list.append("MappingErrors")
    obs_hap_num = row["obs_hap_state"]
    obs_hap_state = row["obs_hap_count"]
    mle_hap_num = row["mle_hap_count"]
    mle_hap_state = row["mle_hap_state"]
    if ((obs_hap_state == "Pass") and (mle_hap_state == "Pass")) and (
        (obs_hap_num != 3) and (mle_hap_num != 3)
    ):
        phase_fail = True
    else:
        phase_fail = False
    if phase_fail:
        if obs_hap_num != 3:
            hap_num = f"Hap{obs_hap_num}"
        elif mle_hap_num != 3:
            hap_num = f"Hap{mle_hap_num}"
        hard_filter_category.append("HaplotypePhaseFail")
        hard_filter_details = hard_filter_details + f"hap{hap_num};"
        if PHASE_HARD_FILTER:
            hard_filter_decision_list.append("PhasingFails")
    if mutation_site_type == "Complex":
        hard_filter_details = hard_filter_details + "Complex;"
        hard_filter_category.append(f"Complex,{alleles_mut_type}")
        if MUTATION_SITE_TYPE_EXCLUDE:
            hard_filter_decision_list.append("MappingErrors")
    if (callable_frac < CALLABLE_FRACTION_MIN) and (
        sample_num > SAMPLE_NUM_MIN
    ):
        hard_filter_details = (
            hard_filter_details + f"callable_frac<{CALLABLE_FRACTION_MIN};"
        )
        hard_filter_category.append("LowCallableRate")
        hard_filter_decision_list.append("LowCallableRate")
    if (
        (recurrent_num > RECURRENT_NUMBER_MAX)
        and (recurrent_fraction > RECURRENT_FRACTION_INDEL_MAX)
        and (callable_num > CALLABLE_NUM_MIN)
        and (alleles_mut_type in INDEL_CATE)
    ):
        hard_filter_details = (
            hard_filter_details
            + f"recurrent_fraction>{RECURRENT_FRACTION_INDEL_MAX};"
        )
        hard_filter_category.append("HighRecurrentRate")
        hard_filter_decision_list.append("GermHetOrRecurrentArtifacts")
    if (
        (recurrent_num > RECURRENT_NUMBER_MAX)
        and (recurrent_fraction > RECURRENT_FRACTION_MISMATCH_MAX)
        and (callable_num > CALLABLE_NUM_MIN)
        and (alleles_mut_type in MISMATCH_CATE)
    ):
        hard_filter_details = (
            hard_filter_details
            + f"recurrent_fraction>{RECURRENT_FRACTION_MISMATCH_MAX};"
        )
        hard_filter_category.append("HighRecurrentRate")
        hard_filter_decision_list.append("GermHetOrRecurrentArtifacts")
    # depth fraction ok
    # mutation site revise
    # mutation seq revise
    # 改为叠加效应 ok
    if (depth_ratio > DP_RATIO_AF_DP_RATIO_MIN) and (
        (mosaic_fraction / 2 > DP_RATIO_AF_AF_MIN)
        or (obs_vaf > DP_RATIO_AF_AF_MIN)
    ):
        hard_filter_details = (
            hard_filter_details
            + f"(dp_ratio>{DP_RATIO_AF_DP_RATIO_MIN})&(AF>{DP_RATIO_AF_AF_MIN});"
        )
        hard_filter_category.append("AbnormalDepthAndHighAF")
        hard_filter_decision_list.append("Repeats")
    if abs(mutation_size) > MUTATIONSIZE_AF_MUTATIONSIZE_MIN:
        if (mosaic_fraction / 2 > MUTATIONSIZE_AF_AF_MIN) or (
            obs_vaf > MUTATIONSIZE_AF_AF_MIN
        ):
            hard_filter_details = (
                hard_filter_details
                + f"(mutation_size>{MUTATIONSIZE_AF_MUTATIONSIZE_MIN})&(AF>{MUTATIONSIZE_AF_AF_MIN});"
            )
            hard_filter_category.append("BigInDelHighAFGermline")
            hard_filter_decision_list.append("BigInDelGermline")
    hard_filter_category = list(set(hard_filter_category))
    hard_filter_category_string = ";".join(hard_filter_category)
    if len(set(hard_filter_decision_list)) == 0:
        hard_filter_decision_list.append("Mosaic mutations")
    hard_filter_decision = ";".join(list(set(hard_filter_decision_list)))
    return pd.Series(
        {
            "obs_vaf_correction": obs_vaf,
            "mosaic_fraction_correction": mosaic_fraction,
            "hard_filter_details": hard_filter_details.strip(";"),
            "hard_filter_category": hard_filter_category_string,
            "hard_filter_decision": hard_filter_decision,
        }
    )


# def bulkmonstr_decision(row):
#     # VAF CI no, SD no, SSR no, MutationSize yes
#     depth_ratio = row["no_norm_len_revise_depth_ratio"]
#     mosaic_fraction = row["MF_hom2het_het2het"]
#     obs_vaf = row["observed_mosaic_allele_vaf_single_locus"]
#     pop_af = row["mutant_popAF"]
#     mutation_size = abs(row["MBP"])
#     mutant_dp = row["mutant_dp"]
#     all_singleton = row["ALLSINGLE"]
#     mutation_site_type = row["mutation_site_type"]
#     mutation_location_type = row["mutation_location_type"]
#     alleles_mut_type = row["alleles_mut_type"]
#     confidence = row["confidence"]
#     motif_length = row["MOTIF_length"]
#     variant_type = row["variant_type"]
#     source_site_start = row[
#         "mutation_site_start_seq1"
#     ]  # HACK 1-based left-right flanking 1bp is the same value
#     source_site_end = row[
#         "mutation_site_end_seq1"
#     ]  # HACK 1-based left-right flanking is the same value
#     mutation_site_start = row[
#         "mutation_site_start_seq2"
#     ]  # HACK 1-based left-right flanking is the same value
#     mutation_site_end = row[
#         "mutation_site_end_seq2"
#     ]  # HACK 1-based left-right flanking is the same value
#     mutant_strand_frac = row["mut_strand_fraction"]
#     if variant_type == "mismatch":
#         mut_mean_baseq = row["mut_mean_baseq"]
#         mut_fraction_baseq_more_cutoff = row["mut_fraction_baseq_more_cutoff"]
#     else:
#         mut_mean_baseq = 30
#         mut_fraction_baseq_more_cutoff = 1
#     if (
#         (int(source_site_start) == 5)
#         and (int(source_site_end) == 5)
#         and (int(mutation_site_start) == 5)
#         and (int(mutation_site_end) == 5)
#         and (int(motif_length) == 1)
#         and (alleles_mut_type == "SNV")
#         and (mutant_strand_frac > 0.9)
#         and ((mut_mean_baseq < 20) or (mut_fraction_baseq_more_cutoff < 0.5))
#     ):
#         post_homopolymer_mismatch = True
#     else:
#         post_homopolymer_mismatch = False
#     obs_hap_num = row["obs_hap_state"]
#     obs_hap_state = row["obs_hap_count"]
#     mle_hap_num = row["mle_hap_count"]
#     mle_hap_state = row["mle_hap_state"]
#     total_genotyping_depth = row["used_read_num_in_genotyping"]
#     if ((obs_hap_state == "Pass") and (mle_hap_state == "Pass")) and (
#         (obs_hap_num != 3) and (mle_hap_num != 3)
#     ):
#         phase_fail = True
#     else:
#         phase_fail = False
#     # depth fraction ok
#     # mutation site revise
#     # mutation seq revise
#     # 改为叠加效应 ok
#     if (pop_af == "NoItem") or (pop_af == "NA"):
#         pop_af = 0

#     decision = row["Prediction"]
#     if obs_vaf < 0.01:
#         decision = decision + ";AF<0.01"
#     elif obs_vaf < 0.02:
#         decision = decision + ";AF<0.02"
#     elif obs_vaf < 0.03:
#         decision = decision + ";AF<0.03"
#     if mosaic_fraction > 0.6:
#         decision = decision + ";AF>0.3"
#     if mut_mean_baseq < 13:
#         decision = decision + ";baseq<13"
#     elif (mut_mean_baseq < 20) or (mut_fraction_baseq_more_cutoff < 0.5):
#         decision = decision + ";(baseq<20)|(high_baseq_frac<0.5)"
#     if (float(mutant_dp) < 1.5) or (
#         all_singleton == "all_singleton_obs_allele"
#     ):
#         decision = decision + ";mut=1"
#     elif float(mutant_dp) < 2.5:
#         decision = decision + ";mut<3"
#     if depth_ratio > 2:
#         decision = decision + ";dp_ratio>2"
#     elif (depth_ratio > 1.5) and (mosaic_fraction > 0.4):
#         decision = decision + ";(dp_ratio>1.5)&(AF>0.2)"
#     elif depth_ratio < 0.5:
#         decision = decision + ";dp_ratio<0.5"
#     if total_genotyping_depth < 10:
#         decision = decision + ";dp<10"
#     if mutation_size > 12:
#         if mosaic_fraction > 0.4:
#             decision = decision + ";(MutationSize>12)&(AF>0.2)"
#         else:
#             decision = decision + ";MutationSize>12"
#     if mutation_site_type == "Complex":
#         decision = decision + ";Complex_haplotypes_" + alleles_mut_type
#     if post_homopolymer_mismatch:
#         decision = decision + ";post_homopolymer_mismatch"
#     if confidence == "Low":
#         decision = decision + ";het2het"
#     if (pop_af > 0.1) and (row["mutation_type"] == "InDel"):
#         if mosaic_fraction > 0.4:
#             decision = decision + ";(popAF>0.1)&(AF>0.2)&(indel)"
#         else:
#             decision = decision + ";(popAF>0.1)&(indel)"
#     if (pop_af > 0.01) and (row["mutation_type"] == "Mismatch"):
#         if mosaic_fraction > 0.4:
#             decision = decision + ";(popAF>0.01)&(AF>0.2)&(mismatch)"
#         else:
#             decision = decision + ";(popAF>0.01)&(mismatch)"
#     if (mutation_location_type == "flanking") and (
#         row["variant_type"] == "mismatch"
#     ):
#         decision = decision + ";flanking_mis"
#     if phase_fail:
#         decision = decision + ";hap" + str(mle_hap_num)
#     return decision


Output_features = [
    "chrom",
    "STRSTART",
    "STREND",
    "MOTIF_length",
    "PERIOD",
    "str_id",
    "MOTIF",
    "GT",
    # "GT1",
    "MGT",
    # "MGT1",
    # "MP",
    # "CMP",
    # "alleles_mut_type",
    "frame",
    "perfect_ref_str",
    # "depth",
    # "filtered_depth",
    # "EAF0",
    # "EAF1",
    # "EAF2",
    # "EAF3",
    "MBP",
    # "EMLEAF0",
    # "EMLEAF1",
    # "EMLEAF2",
    # "EMLEAF3",
    # "GQ_PVAL",
    # "GG_PVAL",
    "ALLSINGLE",
    "obs_phase_state",
    "obs_hap_state",
    "obs_hap_count",
    "mle_phase_state",
    "mle_hap_state",
    "mle_hap_count",
    # "SECISGERM",
    "obs_mut_order",
    "obs_depth_string",
    "mle_mut_order",
    "mle_depth_string",
    "sample_name",
    "average_mappability_score_k24",
    "average_mappability_score_k100",
    "inframe_single_step_prob",
    "inframe_ins_prob",
    "inframe_del_prob",
    "outframe_single_step_prob",
    "outframe_ins_prob",
    "outframe_del_prob",
    "alleles_mut_type",
    # "seq0_length",
    # "seq1_length",
    # "seq2_length",
    # "muttype",
    "mutation_site_type",
    "mutation_location_type",
    "mutation_site_start_seq1",
    "mutation_site_end_seq1",
    "mutation_site_start_seq2",
    "mutation_site_end_seq2",
    "used_read_fraction_in_genotyping",  # (used_read_num_in_genotyping / used_read_num_in_feature), both spanning,but one is filtered genotyping,one is unfilter features extraction
    "germ_seq",
    "source_seq",
    "mut_seq",
    "source_is_recurrent_motif",
    "mut_is_recurrent_motif",
    "unknown_hap_read_fraction",
    "read_length",
    "germ_str_padding_length",
    "mut_str_padding_length",
    "source_str_padding_length",
    # "used_read_num_in_feature",  # features
    "used_read_num_in_genotyping",  # genotyping
    # "nearby_raw_expected_depth_ratio",  # str_padding_unfiltered_raw_depth / median_str_nearby_raw_depth
    # "phasing_fraction",  # len(nearby_snp_seq_list_rm_NA) / len(nearby_snp_seq_list) all str reads phasable reads
    # "germ_phasing_fraction",  # germ str reads phasable reads, single locus assign reads
    # "mut_phasing_fraction",  # mut str reads phasable reads, single locus assign reads
    # "source_phasing_fraction",  # source str reads phasable reads, single locus assign reads
    "overall_non_flanking_indel_fraction",
    # "no_norm_len_revise_depth_ratio"
    # "norm_revise_depth_ratio",
    "genotyping_mle_mosaic_allele_vaf_single_locus",
    "genotyping_obs_mutant_dp",
    "genotyping_obs_source_dp",
    "genotyping_obs_germ_dp",
    # "overall_mis_num_per_bp_all_reads_mean_based_on_NM",
    # "overall_mis_num_per_bp_all_reads_mean_based_on_MD",
    # "overall_mean_flanking_bps_num",
    "germ_mut_diff_mean_flanking_bps_num",
    # "NALLELES_PER_STR_BP",
    "MLEUAF",
    # "GQ_OR",
    # "GQ_PVAL",
    # "GG_OR",
    # "GG_PVAL",
    # "overall_mean_base_accuracy",
    "source_mut_diff_base_accuracy",
    # "source_mut_base_accuracy_pvalue",
    # "sample_str_used_depth",  # mosdepth/samtools
    # "sample_wgs_used_depth",  # mosdepth/samtools
    # "source_mean_baseq",
    # "source_fraction_baseq_more_cutoff",
    "mut_mean_baseq",
    "mut_fraction_baseq_more_cutoff",
    # "germ_dis_fraction",
    # "mut_dis_fraction",
    # "source_dis_fraction",
    # "all_dis_fraction",
    "overall_mean_mapQ",
    "mut_mapQ_more_fraction",
    "mut_mapQ_more_fraction_germ_source",
    "mut_mapQ_mean_germ_source",
    "germ_mean_flanking_sbs_num_per_bp",
    "mut_mean_flanking_sbs_num_per_bp",
    "mut_mean_flanking_sbs_num_per_bp_germ_source",
    "no_norm_len_depth_ratio",
    "no_norm_len_revise_depth_ratio",
    "median_genotyping_depth",  # from before initial filtering
    # "mut_allele_ks_start_p_value",
    # "mut_allele_ks_end_p_value",
    # "source_allele_ks_start_p_value",
    # "source_allele_ks_end_p_value",
    "left_flanking_baseq_mean",
    "right_flanking_baseq_mean",
    "left_flanking_sbs_num_mean",
    "right_flanking_sbs_num_mean",
    "flanking_abs_diff_baseq_mean_germ_source",
    # "germ_allele_left_start_diff",
    # "germ_allele_right_end_diff",
    # "mut_allele_left_start_diff",
    # "mut_allele_right_end_diff",
    # "source_allele_left_start_diff",
    # "source_allele_right_end_diff",
    "mut_non_clipped_fraction_germ_source",
    "mut_indel_germ_source",
    "mut_non_flanking_indel_fraction_germ_source",
    "overall_indel",
    "germ_indel",
    "observed_mosaic_allele_vaf_single_locus",
    "MF_hom2het_het2het",
    "obs_vaf_correction",
    "mosaic_fraction_correction",
    "AAD",  # HACK: TODO: add AAD
    "uncallable_num",
    "uncallable_frac",
    "recurrent_num",
    "recurrent_fraction",
    "sample_num",
    "germ_popAF",
    "source_popAF",
    "mosaic_popAF",
    "external_germ_popAF",
    "external_source_popAF",
    "external_mosaic_popAF",
    "allele_length_dp",
    "ref_allele_length_padding_flanking",
    "muttype",
    "hard_filter_details",
    "hard_filter_category",
    "hard_filter_decision",
]
Output_RF_features = [
    "stutter_ratio",
    "mut_noise_diff_frac",
    # "average_mappability_score_k24",
    "overall_mean_flanking_sbs_num_per_bp",
    "germ_non_clipped_fraction",
    "mut_non_clipped_fraction",
    "mut_non_flanking_indel_fraction",
    "mut_strand_fraction",
    "mut_orientation_fraction",
    "mut_proper_pair_fraction",
    "mut_indel",
    "mut_mis",
    # "mut_errors",
    "flanking_abs_diff_baseq_mean",
    "flanking_abs_diff_sbs_num_mean",
    "mut_mapQ_mean",
    "NORMGERM1",
    "NORMGERM2",
    "NORMMOSAIC",
    "NORMSECOND",
    "mutation_type",
    "Artifacts",
    "Germline Het",
    "Mosaic mutations",
    "RF_Prediction",
]

hard_filter_result = GIAB_fp.apply(hard_filter_mode, axis=1)

# 将结果合并回原数据框
GIAB_fp = pd.concat([GIAB_fp, hard_filter_result], axis=1)

if save_sort_features:
    output_file_sort = ".".join(output_file.split(".")[:-1]) + ".sortup.csv"
    GIAB_fp.to_csv(output_file_sort, header=True, index=True)

if mis_or_indel == "indel":
    indel_df = process_input(mis_or_indel, GIAB_fp)
    indel_df = indel_df[indel_model_features_order]
    with open(indel_model, "rb") as file:
        all_depth_indel_model = pickle.load(file)
    indel_model_features_order_for_rf_name = indel_model_features_order.copy()
    indel_model_features_order_for_rf_name[0] = "MF_hom2het_het2het"
    indel_model_features_order_for_rf_name[
        3
    ] = "observed_mosaic_allele_vaf_single_locus"
    indel_df.columns = indel_model_features_order_for_rf_name
    indel_class_name = all_depth_indel_model.named_steps["classifier"].classes_
    if not indel_df.empty:
        indel_prediction = all_depth_indel_model.predict(indel_df)
        indel_prediction_prob = all_depth_indel_model.predict_proba(indel_df)
        indel_df["RF_Prediction"] = indel_prediction
        indel_df[indel_class_name[0]] = indel_prediction_prob[:, 0]
        indel_df[indel_class_name[1]] = indel_prediction_prob[:, 1]
        indel_df[indel_class_name[2]] = indel_prediction_prob[:, 2]
        features_predict_file = (
            ".".join(output_file.split(".")[:-1]) + ".indel.csv"
        )
        indel_df["mutation_type"] = "InDel"
        if save_prediction_features:
            indel_df.to_csv(features_predict_file, header=True, index=True)
        mosaic_df_select_rf = indel_df[Output_RF_features]
    else:
        mosaic_df_select_rf = pd.DataFrame(columns=Output_RF_features)
elif mis_or_indel == "mismatch":
    with open(mismatch_model, "rb") as file:
        all_depth_mis_model = pickle.load(file)
    mis_df = process_input(mis_or_indel, GIAB_fp)
    mis_df = mis_df[mis_model_features_order]
    mis_class_name = all_depth_mis_model.named_steps["classifier"].classes_
    if not mis_df.empty:
        mis_prediction = all_depth_mis_model.predict(mis_df)
        mis_prediction_prob = all_depth_mis_model.predict_proba(mis_df)
        mis_df["RF_Prediction"] = mis_prediction
        mis_df[mis_class_name[0]] = mis_prediction_prob[:, 0]
        mis_df[mis_class_name[1]] = mis_prediction_prob[:, 1]
        mis_df[mis_class_name[2]] = mis_prediction_prob[:, 2]
        features_predict_file = (
            ".".join(output_file.split(".")[:-1]) + ".mis.csv"
        )
        mis_df["mutation_type"] = "Mismatch"
        if save_prediction_features:
            mis_df.to_csv(features_predict_file, header=True, index=True)
        mosaic_df_select_rf = mis_df[Output_RF_features]
    else:
        mosaic_df_select_rf = pd.DataFrame(columns=Output_RF_features)
elif mis_or_indel == "both":
    indel_df = process_input("indel", GIAB_fp)
    mis_df = process_input("mismatch", GIAB_fp)
    indel_df = indel_df[indel_model_features_order]
    mis_df = mis_df[mis_model_features_order]
    with open(mismatch_model, "rb") as file:
        all_depth_mis_model = pickle.load(file)
    with open(indel_model, "rb") as file:
        all_depth_indel_model = pickle.load(file)
    indel_class_name = all_depth_indel_model.named_steps["classifier"].classes_
    mis_class_name = all_depth_mis_model.named_steps["classifier"].classes_
    indel_model_features_order_for_rf_name = indel_model_features_order.copy()
    indel_model_features_order_for_rf_name[0] = "MF_hom2het_het2het"
    indel_model_features_order_for_rf_name[
        3
    ] = "observed_mosaic_allele_vaf_single_locus"
    indel_df.columns = indel_model_features_order_for_rf_name
    if not indel_df.empty:
        indel_prediction_prob = all_depth_indel_model.predict_proba(indel_df)
        indel_prediction = all_depth_indel_model.predict(indel_df)
        indel_df["RF_Prediction"] = indel_prediction
        indel_df[indel_class_name[0]] = indel_prediction_prob[:, 0]
        indel_df[indel_class_name[1]] = indel_prediction_prob[:, 1]
        indel_df[indel_class_name[2]] = indel_prediction_prob[:, 2]
        features_predict_indel_file = (
            ".".join(output_file.split(".")[:-1]) + ".indel.csv"
        )
        indel_df["mutation_type"] = "InDel"
        if save_prediction_features:
            indel_df.to_csv(
                features_predict_indel_file, header=True, index=True
            )
        indel_mosaic_df_select = indel_df[Output_RF_features]
    else:
        indel_mosaic_df_select = pd.DataFrame(columns=Output_RF_features)
    if not mis_df.empty:
        mis_prediction_prob = all_depth_mis_model.predict_proba(mis_df)
        mis_prediction = all_depth_mis_model.predict(mis_df)
        mis_df["RF_Prediction"] = mis_prediction
        mis_df[mis_class_name[0]] = mis_prediction_prob[:, 0]
        mis_df[mis_class_name[1]] = mis_prediction_prob[:, 1]
        mis_df[mis_class_name[2]] = mis_prediction_prob[:, 2]
        features_predict_mis_file = (
            ".".join(output_file.split(".")[:-1]) + ".mis.csv"
        )
        mis_df["mutation_type"] = "Mismatch"
        if save_prediction_features:
            mis_df.to_csv(features_predict_mis_file, header=True, index=True)
        mis_mosaic_df_select = mis_df[Output_RF_features]
    else:
        mis_mosaic_df_select = pd.DataFrame(columns=Output_RF_features)
    mosaic_df_select_rf = pd.concat(
        [mis_mosaic_df_select, indel_mosaic_df_select]
    )
else:
    raise ValueError(f"Invalid label: {mis_or_indel}")

GIAB_selected_df = GIAB_fp[Output_features]
if mode == "rf":
    if het_no_filter:
        mosaic_df_select_rf_mosaic = mosaic_df_select_rf[
            (mosaic_df_select_rf["RF_Prediction"] == "Mosaic mutations")
            | (mosaic_df_select_rf["RF_Prediction"] == "Germline Het")
        ]
    else:
        mosaic_df_select_rf_mosaic = mosaic_df_select_rf[
            mosaic_df_select_rf["RF_Prediction"] == "Mosaic mutations"
        ]
    mosaic_selected_df = GIAB_selected_df[
        GIAB_selected_df["str_id"].isin(mosaic_df_select_rf_mosaic.index)
    ]
elif mode == "hard_filter":
    mosaic_selected_df = GIAB_selected_df[
        GIAB_selected_df["hard_filter_decision"] == "Mosaic mutations"
    ]
elif mode == "both":
    if het_no_filter:
        mosaic_df_select_rf_mosaic = mosaic_df_select_rf[
            (mosaic_df_select_rf["RF_Prediction"] == "Mosaic mutations")
            | (mosaic_df_select_rf["RF_Prediction"] == "Germline Het")
        ]
    else:
        mosaic_df_select_rf_mosaic = mosaic_df_select_rf[
            mosaic_df_select_rf["RF_Prediction"] == "Mosaic mutations"
        ]
    mosaic_selected_df = GIAB_selected_df[
        (GIAB_selected_df["str_id"].isin(mosaic_df_select_rf_mosaic.index))
        & (GIAB_selected_df["hard_filter_decision"] == "Mosaic mutations")
    ]
elif mode == "either":
    if het_no_filter:
        mosaic_df_select_rf_mosaic = mosaic_df_select_rf[
            (mosaic_df_select_rf["RF_Prediction"] == "Mosaic mutations")
            | (mosaic_df_select_rf["RF_Prediction"] == "Germline Het")
        ]
    else:
        mosaic_df_select_rf_mosaic = mosaic_df_select_rf[
            mosaic_df_select_rf["RF_Prediction"] == "Mosaic mutations"
        ]
    mosaic_selected_df = GIAB_selected_df[
        (GIAB_selected_df["str_id"].isin(mosaic_df_select_rf_mosaic.index))
        | (GIAB_selected_df["hard_filter_decision"] == "Mosaic mutations")
    ]
elif mode == "all":
    mosaic_selected_df = GIAB_selected_df
else:
    raise ValueError(f"Invalid mode: {mode}")

mosaic_selected_df.index = mosaic_selected_df["str_id"]
mosaic_df_select_rf_mosaic_selected = mosaic_df_select_rf[
    mosaic_df_select_rf.index.isin(list(mosaic_selected_df.index))
]
mosaic_df_select_rf_mosaic_selected = mosaic_df_select_rf_mosaic_selected.loc[
    mosaic_selected_df["str_id"]
]

final_mosaic_merge_df = pd.concat(
    [mosaic_selected_df, mosaic_df_select_rf_mosaic_selected], axis=1
)
# final_mosaic_merge_df_add_popAF = get_popAF(
#     population_panel, reference_fasta, final_mosaic_merge_df
# )

pysam_fasta = pysam.FastaFile(reference_fasta)
if final_mosaic_merge_df.empty:
    GIAB_selected_df.index = GIAB_selected_df["str_id"]
    mosaic_df_select_rf_selected = mosaic_df_select_rf[mosaic_df_select_rf.index.isin(list(GIAB_selected_df.index))]
    mosaic_df_select_rf_selected = mosaic_df_select_rf_selected.loc[GIAB_selected_df["str_id"]]
    final_mosaic_merge_df = pd.concat(
        [GIAB_selected_df, mosaic_df_select_rf_selected], axis=1
    )
    print("Warning: The resulting DataFrame is empty.")
    print("This suggests that all sites were predicted as Artifacts or Germline heterozygous,")
    print("and therefore no somatic or mosaic variants were identified for further analysis.")
    print("We therefore save all the prediction results.")
final_mosaic_merge_df[
    [
        "mut_source_seq",
        "mut_target_seq",
        "mut_source_seq_pos_tuple_0_based_leftrightclose",
        "mut_target_seq_pos_tuple_0_based_leftrightclose",
        "reference_start_index_0_based_include",
        "reference_end_index_0_based_include",
        "reference_start_coordinate_1_based_include",
        "reference_end_coordinate_1_based_include",
        "mutated_reference_seq",
        "relative_to_reference_germ_seq",
        "relative_to_reference_source_seq",
        "relative_to_reference_mut_seq",
    ]
] = final_mosaic_merge_df.apply(
    lambda row: pd.Series(
        get_reference_chr_pos_ref_alt_per_row(row, pysam_fasta)
    ),
    axis=1,
)
# final_mosaic_merge_df["confidence"] = "High"
# final_mosaic_merge_df.loc[
#     final_mosaic_merge_df["muttype"] == "hethet", "confidence"
# ] = "Low"
final_mosaic_merge_df["mutant_dp"] = (
    final_mosaic_merge_df["observed_mosaic_allele_vaf_single_locus"]
    * final_mosaic_merge_df["used_read_num_in_genotyping"]
)

# final_mosaic_merge_df[
#     "Decision"
# ] = final_mosaic_merge_df.apply(bulkmonstr_decision, axis=1)
order_columns = [
    "chrom",
    "STRSTART",
    "STREND",
    "MOTIF_length",
    "PERIOD",
    "str_id",
    "MOTIF",
    "sample_name",
    "GT",
    # "GT1",
    "MGT",
    # "MGT1",
    # "MP",
    # "CMP",
    # "alleles_mut_type",
    "frame",
    "perfect_ref_str",
    # "depth",
    # "filtered_depth",
    # "EAF0",
    # "EAF1",
    # "EAF2",
    # "EAF3",
    "MBP",
    # "EMLEAF0",
    # "EMLEAF1",
    # "EMLEAF2",
    # "EMLEAF3",
    # "GQ_PVAL",
    # "GG_PVAL",
    "ALLSINGLE",
    # "SECISGERM",
    "average_mappability_score_k24",
    "average_mappability_score_k100",
    "inframe_single_step_prob",
    "inframe_ins_prob",
    "inframe_del_prob",
    "outframe_single_step_prob",
    "outframe_ins_prob",
    "outframe_del_prob",
    "alleles_mut_type",
    # "seq0_length",
    # "seq1_length",
    # "seq2_length",
    # "muttype",
    "mutation_site_type",
    "mutation_location_type",
    "mutation_site_start_seq1",
    "mutation_site_end_seq1",
    "mutation_site_start_seq2",
    "mutation_site_end_seq2",
    "used_read_fraction_in_genotyping",  # (used_read_num_in_genotyping / used_read_num_in_feature), both spanning,but one is filtered genotyping,one is unfilter features extraction
    "germ_seq",
    "source_seq",
    "mut_seq",
    "mut_source_seq",
    "mut_target_seq",
    "mut_source_seq_pos_tuple_0_based_leftrightclose",
    "mut_target_seq_pos_tuple_0_based_leftrightclose",
    "reference_start_index_0_based_include",
    "reference_end_index_0_based_include",
    "reference_start_coordinate_1_based_include",
    "reference_end_coordinate_1_based_include",
    "mutated_reference_seq",
    "relative_to_reference_germ_seq",
    "relative_to_reference_source_seq",
    "relative_to_reference_mut_seq",
    "source_is_recurrent_motif",
    "mut_is_recurrent_motif",
    "unknown_hap_read_fraction",
    "read_length",
    "germ_str_padding_length",
    "source_str_padding_length",
    "mut_str_padding_length",
    # "used_read_num_in_feature",  # features
    "used_read_num_in_genotyping",  # genotyping
    # "nearby_raw_expected_depth_ratio",  # str_padding_unfiltered_raw_depth / median_str_nearby_raw_depth
    # "phasing_fraction",  # len(nearby_snp_seq_list_rm_NA) / len(nearby_snp_seq_list) all str reads phasable reads
    # "germ_phasing_fraction",  # germ str reads phasable reads, single locus assign reads
    # "mut_phasing_fraction",  # mut str reads phasable reads, single locus assign reads
    # "source_phasing_fraction",  # source str reads phasable reads, single locus assign reads
    "overall_non_flanking_indel_fraction",
    # "no_norm_len_revise_depth_ratio"
    # "norm_revise_depth_ratio",
    "genotyping_mle_mosaic_allele_vaf_single_locus",
    "genotyping_obs_mutant_dp",
    "genotyping_obs_source_dp",
    "genotyping_obs_germ_dp",
    # "overall_mis_num_per_bp_all_reads_mean_based_on_NM",
    # "overall_mis_num_per_bp_all_reads_mean_based_on_MD",
    # "overall_mean_flanking_bps_num",
    "germ_mut_diff_mean_flanking_bps_num",
    # "NALLELES_PER_STR_BP",
    "MLEUAF",
    # "GQ_OR",
    # "GQ_PVAL",
    # "GG_OR",
    # "GG_PVAL",
    # "overall_mean_base_accuracy",
    "source_mut_diff_base_accuracy",
    # "source_mut_base_accuracy_pvalue",
    # "sample_str_used_depth",  # mosdepth/samtools
    # "sample_wgs_used_depth",  # mosdepth/samtools
    # "source_mean_baseq",
    # "source_fraction_baseq_more_cutoff",
    "mut_mean_baseq",
    "mut_fraction_baseq_more_cutoff",
    # "germ_dis_fraction",
    # "mut_dis_fraction",
    # "source_dis_fraction",
    # "all_dis_fraction",
    "overall_mean_mapQ",
    "no_norm_len_depth_ratio",
    "no_norm_len_revise_depth_ratio",
    "median_genotyping_depth",  # from before initial filtering
    # "mut_allele_ks_start_p_value",
    # "mut_allele_ks_end_p_value",
    # "source_allele_ks_start_p_value",
    # "source_allele_ks_end_p_value",
    "left_flanking_baseq_mean",
    "right_flanking_baseq_mean",
    "left_flanking_sbs_num_mean",
    "right_flanking_sbs_num_mean",
    "flanking_abs_diff_baseq_mean_germ_source",
    # "germ_allele_left_start_diff",
    # "germ_allele_right_end_diff",
    # "mut_allele_left_start_diff",
    # "mut_allele_right_end_diff",
    # "source_allele_left_start_diff",
    # "source_allele_right_end_diff",
    "mut_non_clipped_fraction_germ_source",
    "mut_indel_germ_source",
    "mut_non_flanking_indel_fraction_germ_source",
    "overall_indel",
    "germ_indel",
    "mosaic_fraction_correction",
    "obs_vaf_correction",
    "AAD",  # HACK: TODO: add AAD
    "uncallable_num",
    "uncallable_frac",
    "recurrent_num",
    "recurrent_fraction",
    "sample_num",
    "germ_popAF",
    "source_popAF",
    "mosaic_popAF",
    "external_germ_popAF",
    "external_source_popAF",
    "external_mosaic_popAF",
    "allele_length_dp",
    "ref_allele_length_padding_flanking",
    "MF_hom2het_het2het",
    "observed_mosaic_allele_vaf_single_locus",
    "mutant_dp",
    "stutter_ratio",
    "mut_noise_diff_frac",
    # "average_mappability_score_k24",
    "overall_mean_flanking_sbs_num_per_bp",
    "germ_mean_flanking_sbs_num_per_bp",
    "mut_mean_flanking_sbs_num_per_bp",
    "mut_mean_flanking_sbs_num_per_bp_germ_source",
    "germ_non_clipped_fraction",
    "mut_non_clipped_fraction",
    "mut_non_flanking_indel_fraction",
    "mut_strand_fraction",
    "mut_orientation_fraction",
    "mut_proper_pair_fraction",
    "mut_indel",
    "mut_mis",
    # "mut_errors",
    "flanking_abs_diff_baseq_mean",
    "flanking_abs_diff_sbs_num_mean",
    "mut_mapQ_mean",
    "mut_mapQ_more_fraction",
    "mut_mapQ_more_fraction_germ_source",
    "mut_mapQ_mean_germ_source",
    "NORMGERM1",
    "NORMGERM2",
    "NORMMOSAIC",
    "NORMSECOND",
    "mutation_type",
    "Artifacts",
    "Germline Het",
    "Mosaic mutations",
    "obs_mut_order",
    "obs_depth_string",
    "mle_mut_order",
    "mle_depth_string",
    "obs_phase_state",
    "obs_hap_state",
    "obs_hap_count",
    "mle_phase_state",
    "mle_hap_state",
    "mle_hap_count",
    "hard_filter_details",
    "hard_filter_category",
    "muttype",
    "hard_filter_decision",
    "RF_Prediction",
]
final_mosaic_merge_df = final_mosaic_merge_df[order_columns]
final_mosaic_merge_df["Confidence"] = "All PASS"
if het_no_filter:
    final_mosaic_merge_df.loc[
        (
            (final_mosaic_merge_df["RF_Prediction"] != "Artifacts")
            & (
                final_mosaic_merge_df["hard_filter_decision"]
                != "Mosaic mutations"
            )
        ),
        "Confidence",
    ] = "ONLY RF PASS"
    final_mosaic_merge_df.loc[
        (final_mosaic_merge_df["RF_Prediction"] == "Artifacts")
        & (
            final_mosaic_merge_df["hard_filter_decision"] == "Mosaic mutations"
        ),
        "Confidence",
    ] = "ONLY HARD FILTER PASS"
    final_mosaic_merge_df.loc[
        (final_mosaic_merge_df["RF_Prediction"] == "Artifacts")
        & (
            final_mosaic_merge_df["hard_filter_decision"] != "Mosaic mutations"
        ),
        "Confidence",
    ] = "Fail"
    final_mosaic_merge_df["Final Decision"] = "Artifacts"
    final_mosaic_merge_df.loc[
        (final_mosaic_merge_df["RF_Prediction"] != "Artifacts")
        & (
            final_mosaic_merge_df["hard_filter_decision"] == "Mosaic mutations"
        ),
        "Final Decision",
    ] = "Mosaic"
else:
    final_mosaic_merge_df.loc[
        (final_mosaic_merge_df["RF_Prediction"] == "Mosaic mutations")
        & (
            final_mosaic_merge_df["hard_filter_decision"] != "Mosaic mutations"
        ),
        "Confidence",
    ] = "ONLY RF PASS"
    final_mosaic_merge_df.loc[
        (final_mosaic_merge_df["RF_Prediction"] != "Mosaic mutations")
        & (
            final_mosaic_merge_df["hard_filter_decision"] == "Mosaic mutations"
        ),
        "Confidence",
    ] = "ONLY HARD FILTER PASS"
    final_mosaic_merge_df.loc[
        (final_mosaic_merge_df["RF_Prediction"] != "Mosaic mutations")
        & (
            final_mosaic_merge_df["hard_filter_decision"] != "Mosaic mutations"
        ),
        "Confidence",
    ] = "Fail"
    final_mosaic_merge_df["Final Decision"] = "Heterzygous or Artifacts"
    final_mosaic_merge_df.loc[
        (final_mosaic_merge_df["RF_Prediction"] == "Mosaic mutations")
        & (
            final_mosaic_merge_df["hard_filter_decision"] == "Mosaic mutations"
        ),
        "Final Decision",
    ] = "Mosaic"
final_mosaic_merge_df.to_csv(output_file, header=True, index=True)
