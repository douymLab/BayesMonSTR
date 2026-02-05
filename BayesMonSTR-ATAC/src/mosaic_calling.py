# read_name 法和 mate 法提取 spanning reads
# 提取候选区域的 SNPs 从 VCF 文件中
# 按照 loci 的顺序把每一个人样本都跑了，因为 allele 编号需要统一的问题
# 检查是否存在 SNP，检查是否存在 spanning reads 在这个 STR loci

import pysam
import textwrap

# from ..configs import config_params
import config_params
import myutils
import numpy as np
import extract_allele
import hmmsegger
import coordinate_segger
from scipy import stats
import logging
from collections import defaultdict
import mosaic_fraction_estimation_unphase
import mosaic_fraction_estimation_phase
import mutation_filtering
import sys
import os
import pandas as pd
import time
import argparse
from multiprocessing import Pool, Manager
from tqdm import tqdm
import traceback
import output_vcf
import logger_config
from scipy.special import logsumexp

DEBUG = False
SINGLETHREAD = True

MOSAIC_FRACTION_ESTIMATION_PARAMS = (
    config_params.MOSAIC_FRACTION_ESTIMATION_PARAMS
)
PHASING_PARAMS = config_params.PHASING_PARAMS
SNP_SPANNING_RANGE = PHASING_PARAMS["SNP_SPANNING_RANGE"]  # 1000
LIKELIHOOD_MODE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "likelihood_mode"
]  # "seq-based"
EXCLUDED_SNP_FLANKING_LENGTH = PHASING_PARAMS[
    "EXCLUDED_SNP_FLANKING_LENGTH"
]  # 15
hSNP_VAF_MAX = PHASING_PARAMS["hSNP_VAF_MAX"]  # 0.6
hSNP_VAF_MIN = PHASING_PARAMS["hSNP_VAF_MIN"]  # 0.4
hSNP_MIN_DEPTH = PHASING_PARAMS[
    "hSNP_MIN_DEPTH"
]  # 10 # 20 # HACK: revise 20 to 10
PHASING_STRATEDY = PHASING_PARAMS["PHASING_STRATEDY"]
AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH = PHASING_PARAMS[
    "AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH"
]  # 5
SPANNING_hSNP_VAF_MIN = PHASING_PARAMS["SPANNING_hSNP_VAF_MIN"]  # 0.2
SPANNING_hSNP_VAF_MAX = PHASING_PARAMS["SPANNING_hSNP_VAF_MAX"]  # 0.8
SPANNING_DEPTH_CUTOFF = PHASING_PARAMS["SPANNING_DEPTH_CUTOFF"]  # 10
AF_DIFFERENCE_CUTOFF = PHASING_PARAMS["AF_DIFFERENCE_CUTOFF"]  # 10
ADD_FLANKING_PROB = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "ADD_FLANKING_PROB"
]  # False
MUTATION_FILTER = config_params.MUTATION_FILTER
ALLELE_BALANCE = MUTATION_FILTER["ALLELE_BALANCE"]  # 1/2
SNP_SEQUENCING_ERROR_RATE_PRIOR = MUTATION_FILTER[
    "SNP_SEQUENCING_ERROR_RATE_PRIOR"
]  # 0.001
MAX_MOSAIC_POSTERIOR_CUTOFF = MUTATION_FILTER[
    "MAX_MOSAIC_POSTERIOR_CUTOFF"
]  # 0.9
ALL_MOSAIC_POSTERIOR_CUTOFF = MUTATION_FILTER[
    "ALL_MOSAIC_POSTERIOR_CUTOFF"
]  # 0.9
LRT_PVALUE_CUTOFF = MUTATION_FILTER["LRT_PVALUE_CUTOFF"]  # 0.05
BINOMIAL_TEST_PVALUE_CUTOFF = MUTATION_FILTER[
    "BINOMIAL_TEST_PVALUE_CUTOFF"
]  # 0.05
# PHASE_POSTERIOR_CUTOFF = MUTATION_FILTER["PHASE_POSTERIOR_CUTOFF"] # 0.25
UNPHASE_PHASE_OUTPUT = config_params.UNPHASE_PHASE_OUTPUT
PROB_BASED_GERMLINE = MUTATION_FILTER["PROB_BASED_GERMLINE"]  # True
COUNT_HAP_NUM_DEPTH_CUTOFF = PHASING_PARAMS["COUNT_HAP_NUM_DEPTH_CUTOFF"]  # 1
COUNT_MLE_HAP_NUM_DEPTH_CUTOFF = PHASING_PARAMS[
    "COUNT_MLE_HAP_NUM_DEPTH_CUTOFF"
]  # 1
LRT_FREEDOM = PHASING_PARAMS[
    "LRT_FREEDOM"
]  # {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
DEBUG = False
PHASE_POSTERIOR_CUTOFF = PHASING_PARAMS["PHASE_POSTERIOR_CUTOFF"]  # 0.9
ANYWAY_LOOK_NSNP = PHASING_PARAMS["ANYWAY_LOOK_NSNP"]  # False
NEARBY_SNP_READ_FILTER = True
NEARBYSNP_MAPQ_BASEQ_CUTOFF = 20
PILEUP_STEPPER = "all"
VCF_DIS_DP_THIRD_ALLELE_CUTOFF = 0.2
BINOMIAL_TEST_PVALUE_CUTOFF = 0.05
SPANNING_NEARBY_SNP_FILTER = True
USE_DEFAULT_MAX_STEP = True  # HACK: Temp for Mix Samples and GIAB Samples
DEFAULT_MAX_STEP = (
    100  # HACK: For default steps and max step limitation for KeyErrors
)
__VERSION__ = "0.1"
DONT_ASSIGNMENT_FOR_IDENTICAL_PROBS = True
PHASE_MUST_TWO_SPAN_ALLELE = True
HAP_IDENTIFY_POA = True
REFINE_HAP_COUNT_HETHET = False
HIGH_QUALITY_PHASE_DEPTH_CUTOFF = 3
MIX = False  # HACK: temp for Mix Samples
# HACK: temp for Mix Samples, Mix Samples are "PCR-free" and use "PCR-free" version temporarily
# Use "PCR" for PCR-free samples because may exist optical duplicates
COMMON_VARIANTS_CUTOFF = 0.01
HARSH_AF_DIFF_CUTOFF = 0.6 / 0.4
MILD_AF_DIFF_CUTOFF = 0.7 / 0.3
MILD_hSNP_DEPTH_DIFF_CUTOFF = 10
PACBIO_PHASING_DEPTH_CUTOFF = 10
PACBIO_PHASING_BALANCE_CUTOFF = 0.8
GIAB = False
GIAB_STUTTER = False
PSEUDOBULK = False
INIT_STUTTER_MODEL = config_params.INIT_STUTTER_MODEL  # 0.9 0.05 0.01 0.000001
PHASE_COUNT_SCALE_FACTOR = 5
GIAB_NON_HG008 = False


def cmd_args(args=sys.argv[1:]):
    """Get the command line arguments

    Args:
        args (list, optional): A list with arguments name and value in order.
        Defaults to sys.argv[1:].

    Returns:
        argparse.Namespace object: A namespace object with the command line
        arguments as attributes.
    """
    parser = argparse.ArgumentParser(
        prog="MosaicSTR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="%(prog)s [options] ",
        allow_abbrev=True,
        add_help=False,
        conflict_handler="error",
    )
    # * Help option is default if add_help is True
    help_args = parser.add_argument_group("Help arguments")
    help_args.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    # Add the command-line arguments Required arguments
    required_args = parser.add_argument_group("Required arguments")
    required_args.add_argument(
        "-i",
        "--metadata",
        required=True,
        help=(
            "Metadata(CSV file) including ind, sex, tissue, sequencing_type,"
            " bam_path and, others columns"
        ),
    )
    required_args.add_argument(
        "-r",
        "--reference_genome",
        required=True,
        help="Reference genome(FASTA file)",
    )
    # required_args.add_argument(
    #     "-b",
    #     "--bed_panel",
    #     action="append",
    #     nargs="+",
    #     required=True,
    #     help="STR genome annotation(BED file)",
    # )
    required_args.add_argument(
        "-b",
        "--bed_panel",
        required=True,
        help="STR genome annotation(BED file)",
    )
    required_args.add_argument(
        "-o",
        "--output_dir",
        required=True,
        # help="STR mutation calling output(Bgzipped VCF file)",
        help="STR mutation calling output path",
    )
    # Optional input arguments
    optional_input_args = parser.add_argument_group("Optional input arguments")
    optional_input_args.add_argument(
        "-g",
        "-gene_model",
        help=(
            "Gene model(GTF/GFF file) for gene-based read backed phasing "
            "using allele imbalance/dropout information from amplification "
            "bias or allelic specific expression"
        ),
    )
    optional_input_args.add_argument(
        "-s",
        "--stutter_model",
        help=(
            "Use stutter models from big cohorts (Current available: 1KG"
            " HipSTR model and GTEx HipSTR model, Estimate via EM algorithm"
            " for datasets specific stutter model is better option)"
        ),
        default="",
    )
    optional_input_args.add_argument(
        "-gn",
        "--gnomad_freq_in",
        help=(
            "Path to the gnomAD population frequency file. This file should"
            " contain allele frequency data from gnomAD"
        ),
        default="",
        type=str,  # 这会将输入解析为字符串
    )
    optional_input_args.add_argument(
        "-p",
        "--phasing",
        help=(
            "Phased SNP information(Bgzipped VCF file) from germline variants"
            " genotyping and phasing"
        ),
        default="",
    )
    optional_input_args.add_argument(
        "-a",
        "--allele_imbalance",
        help=(
            "Allele imbalance information from Gaussian process regression"
            " (GPR) model"
        ),
    )
    # Optional output arguments
    optional_output_args = parser.add_argument_group(
        "Optional output arguments"
    )
    # optional_output_args.add_argument(
    #     "-l", "--log", action="extend", nargs="*", help="Log files"
    # )
    optional_output_args.add_argument(
        "-ll",
        "--loglevel",
        help="Sets the threshold for this logger to level.",
        default="INFO",
        type=str,
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    optional_output_args.add_argument(
        "-lf",
        "--log_to_file",
        action="store_true",
        help="Output log to file",
    )
    optional_output_args.add_argument(
        "-f",
        "--vcf_out",
        help="Prefiex of output vcf files of mosaic calling",
    )
    # Other optional arguments
    other_optional_args = parser.add_argument_group("Other optional arguments")
    other_optional_args.add_argument(
        "-c", "--chrom", type=str, default="", help="Selected chromosome"
    )
    other_optional_args.add_argument(
        "-s", "--start", type=int, default=0, help="Selected start"
    )
    other_optional_args.add_argument(
        "-e", "--end", type=int, default=1000000000, help="Selected end"
    )
    other_optional_args.add_argument(
        "-t",
        "--threads",
        type=int,
        default=-1,
        help="Number of threads to use (default: -1, use all available cores)",
    )
    other_optional_args.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="verbose-level of log",
    )
    other_optional_args.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    other_optional_args.add_argument(
        "-n",
        "--versions",
        action="version",
        version="%(prog)s" + " version '" + __VERSION__ + "'",
    )
    # Parse the command-line arguments
    options = parser.parse_args(args)
    return options


def mosaic_fraction_estimate_prepare(parsed_options):
    genome_wide_info_dict = {}
    mosaic_fraction_all_params = {}
    # mosaic_fraction_all_params["parsed_options"] = parsed_options
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
    bam_files = metadata['bam_path']
    sample_name_list = metadata['Sample Name'].to_list()
    mosaic_fraction_all_params["bam_files"] = bam_files
    mosaic_fraction_all_params["vcf_files"] = parsed_options.phasing
    # if MIX:  # HACK: temp for Mix Samples
    #     sample_name_list = [
    #         "_".join(bam.split("/")[-1].split(".")[0:2]) for bam in bam_files
    #     ]
    # elif GIAB:
    #     sample_name_list = [
    #         "_".join(bam.split("/")[-1].split(".")[0:-1]) for bam in bam_files
    #     ]
    # elif PSEUDOBULK:
    #     sample_name_list = [
    #         "_".join(bam.split("/")[-1].split(".")[0:-1]) for bam in bam_files
    #     ]
    # else:
    #     sample_name_list = [
    #         bam.split("/")[-1].split(".")[0] for bam in bam_files
    #     ]
    mosaic_fraction_all_params["bam_name"] = sample_name_list
    genome_wide_info_dict["sample_name_list"] = sample_name_list

    log_dir = parsed_options.output_dir + "/log"
    result_dir = (
        parsed_options.output_dir + "/results"
    )
    os.system("mkdir -p " + parsed_options.output_dir)
    os.system("mkdir -p " + log_dir)
    os.system("mkdir -p " + result_dir)
    bed_name = parsed_options.bed_panel.split("/")[-1].split(".")[0]
    if parsed_options.chrom:
        uid = f"{parsed_options.chrom}_{parsed_options.start}_{parsed_options.end}"
    else:
        uid = bed_name
    fail_file = (
        log_dir
        + "/"
        + uid
        + "_mosaic_calling_fail_loci.log"
    )
    mosaic_fraction_all_params["vcf_output"] = (
        result_dir
        + "/"
        + uid
        + "_mosaic_calling.vcf"
    )
    mosaic_fraction_all_params["fail_file"] = fail_file
    logfile = log_dir + "/" + uid + ".log"
    mosaic_fraction_all_params["log_file"] = logfile
    mosaic_fraction_all_params[
        "stutter_file"
    ] = parsed_options.stutter_model
    mosaic_fraction_all_params[
        "gnomad_freq_in"
    ] = parsed_options.gnomad_freq_in
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
    mosaic_fraction_all_params["add_chr_character"] = add_chr_character
    mosaic_fraction_all_params["remove_chr_character"] = remove_chr_character
    mosaic_fraction_all_params["loglevel"] = parsed_options.loglevel
    log_level = getattr(logging, parsed_options.loglevel.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {parsed_options.loglevel}")
    logger_mosaic_fraction = logging.getLogger("MosaicFractionEstimator")
    logger_mosaic_fraction.setLevel(log_level)
    mosaic_fraction_all_params["log_to_file"] = parsed_options.log_to_file
    if parsed_options.log_to_file:
        logger_mosaic_fraction_file_handler = myutils.LockedFileHandler(
            logfile
        )
    else:
        logger_mosaic_fraction_file_handler = logging.StreamHandler()
    # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger_mosaic_fraction_file_handler.setFormatter(formatter)
    if not logger_mosaic_fraction.handlers:
        logger_mosaic_fraction.addHandler(logger_mosaic_fraction_file_handler)
    # mosaic_fraction_all_params[
    #     "logger_mosaic_fraction"
    # ] = logger_mosaic_fraction
    logger_mosaic_fraction.info(
        "MosaicFractionEstimator: Mosaic Fraction Estimation Using Individual"
        " Data"
    )
    logger_mosaic_fraction.info("Version: 1.0")
    logger_mosaic_fraction.info(
        "Start time: %s",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
    )
    # STR_refpanel_bed = pysam.TabixFile(parsed_options.bed_panel)
    # mosaic_fraction_all_params["STR_refpanel_bed"] = STR_refpanel_bed
    sex = metadata.iloc[
        0, 1
    ]  # HACK: don't mosaic calling for sex chromosome for current version
    mosaic_fraction_all_params["sex"] = sex
    if metadata.shape[1] == 8:
        sequencing_method = metadata.iloc[:, 7]
        mosaic_fraction_all_params["sequencing_method"] = list(
            sequencing_method
        )
    mosaic_fraction_all_params["fasta_file"] = ref_file
    return genome_wide_info_dict, mosaic_fraction_all_params

    # self.hmm_seg_init = hmmsegger.hmm_seg_init


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


# ? check reads or mate spanning nearby SNPs loci ?#
def find_nearby_snp(pysam_read, pos, bam):
    # ??????? Calling this method will change the file position. This might interfere with any iterators that have not re-opened the file. ?????? #
    # ??????? This method is too slow for high-throughput processing. If a read needs to be processed with its mate, work from a read name sorted file or, better, cache reads. ?????? #
    try:
        if (
            pysam_read.reference_start + AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH
            <= pos
            < pysam_read.reference_end - AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH
        ):
            aligned_pairs = pysam_read.get_aligned_pairs(
                with_seq=True, matches_only=True
            )
            # a list of aligned read (query) and reference positions.
            # Each item in the returned list is a tuple consisting of the 0-based offset from the start of the read sequence followed by the 0-based reference position.
            refpos_queryindex = {i[1]: [i[0], i[2]] for i in aligned_pairs}
            query_index = refpos_queryindex.get(pos, None)
            if query_index is not None:
                [index, snp_seq] = refpos_queryindex.get(pos)
                snp_baseq = pysam_read.query_qualities[index]
                snp_seq_accuracy = 1 - myutils.baseq_2_error_rate(snp_baseq)
                return pysam_read.query_name, snp_seq, snp_seq_accuracy
        if not pysam_read.mate_is_unmapped:
            mate = bam.mate(pysam_read)
            if (
                mate.reference_start + AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH
                <= pos
                < mate.reference_start - AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH
            ):
                aligned_pairs = mate.get_aligned_pairs(
                    with_seq=True, matches_only=True
                )
                refpos_queryindex = {i[1]: [i[0], i[2]] for i in aligned_pairs}
                query_index = refpos_queryindex.get(pos, None)
                if query_index is not None:
                    [index, snp_seq] = refpos_queryindex.get(pos)
                    snp_baseq = mate.query_qualities[index]
                    snp_seq_accuracy = 1 - myutils.baseq_2_error_rate(
                        snp_baseq
                    )
                    return mate.query_name, snp_seq, snp_seq_accuracy
    except Exception as e:
        return False, False, False
    return False, False, False


# XXX: don't do this currently otherwise will produce an unforeseen mistake #
def read_name_adjust_for_spanning_find(read_query_name):  # TODO
    if (
        read_query_name.endswith("/1")
        or read_query_name.endswith("/2")
        or read_query_name.endswith(".1")
        or read_query_name.endswith(".2")
        or read_query_name.endswith("_1")
        or read_query_name.endswith("_2")
        or read_query_name.endswith(":1")
        or read_query_name.endswith(":2")
    ):
        read_query_name = read_query_name[:-2]
    return read_query_name


def nearby_SNP_quality_control():
    # Done in VCF records
    # Third allele fraction 0.2
    # DP 10
    # AF 0.2 - 0.8 or AF Bino test 0.05 (biomial depth 越高要求的 af 越高)
    pass  # TODO: Get high quality nearbysnp according to IGV Hints


def nearby_SNP_spanning_reads_quality_control():
    # Done in code
    # Third allele fraction 0.2
    # DP 5
    # AF 0.2 - 0.8 or AF Bino test 0.05 (biomial depth 越高要求的 af 越高)
    pass  # TODO: Get high quality nearbysnp according to IGV Hints


def keey_nearby_snp_with_all_str_allele():
    # Done in code
    pass


def judge_nearbysnp_popAF(
    gn_bed, snp1, snp2, chrom, snp_pos_zero_based, ref_snp
):
    # Default for ref allele is common allele #
    alt_allele_af_dict = {}
    if gn_bed:
        for i in gn_bed.fetch(
            chrom, snp_pos_zero_based, snp_pos_zero_based + 1
        ):
            row_list = i.split("\t")
            if len(row_list[3]) == len(row_list[4]):
                if (int(row_list[1]) == snp_pos_zero_based) and (
                    int(row_list[2]) == (snp_pos_zero_based + 1)
                ):
                    if (row_list[4] == snp1) or (row_list[4] == snp2):
                        alt_allele_af_dict[row_list[4]] = float(row_list[5])
        if snp1 == ref_snp:
            if alt_allele_af_dict.get(snp2, 0) >= COMMON_VARIANTS_CUTOFF:
                return True
            else:
                return False
        elif snp2 == ref_snp:
            if alt_allele_af_dict.get(snp1, 0) >= COMMON_VARIANTS_CUTOFF:
                return True
            else:
                return False
        else:
            if (
                alt_allele_af_dict.get(snp1, 0) >= COMMON_VARIANTS_CUTOFF
            ) and (alt_allele_af_dict.get(snp2, 0) >= COMMON_VARIANTS_CUTOFF):
                return True
            else:
                return False
    else:
        return True


def select_best_nearby_snp(
    reads_snp_seq, nearby_snp_rec, gn_bed, ref_snp_dict
):
    best_af_diff = AF_DIFFERENCE_CUTOFF
    best_var_id = False
    best_af = False
    best_depth = 0
    if DEBUG:
        print(reads_snp_seq)
        print(nearby_snp_rec)
    reads_snp_seq = dict(
        sorted(reads_snp_seq.items(), key=lambda item: len(item[1]))
    )
    for var_id, var_rec in reads_snp_seq.items():
        ref_snp = ref_snp_dict[var_id]
        snp_seq = list(var_rec.values())
        depth = len(snp_seq)
        if depth == 0:
            continue
        allele1_counts = snp_seq.count(nearby_snp_rec[var_id][0])
        allele2_counts = snp_seq.count(nearby_snp_rec[var_id][1])
        third_allele = depth - allele1_counts - allele2_counts
        allele1_af = allele1_counts / depth
        if GIAB_NON_HG008:
            if GIAB:
                pass
        else:
            if SPANNING_NEARBY_SNP_FILTER:  # HACK: 防止 span snp 只存在 h1 或者 h2
                if (
                    SPANNING_hSNP_VAF_MAX
                    >= allele1_af
                    >= SPANNING_hSNP_VAF_MIN
                ) or stats.binomtest(
                    allele1_counts, depth, 0.5, alternative="two-sided"
                ).pvalue > BINOMIAL_TEST_PVALUE_CUTOFF:
                    pass
                else:
                    continue
            if (third_allele / depth) > VCF_DIS_DP_THIRD_ALLELE_CUTOFF:
                continue
            if (
                depth < SPANNING_DEPTH_CUTOFF
            ):  # XXX: Min Spanning Depth To Meet: 10 (make the hSNP more confident and available)
                continue
        if allele1_counts == 0 or allele2_counts == 0:
            continue
        snp1 = var_id.split("_")[4]
        snp2 = var_id.split("_")[5]
        snp_pos_zero_based = var_id.split("_")[1]
        chrom = var_id.split("_")[0]
        judge_common_variants = judge_nearbysnp_popAF(
            gn_bed, snp1, snp2, chrom, snp_pos_zero_based, ref_snp
        )
        if judge_common_variants:
            pass
        else:
            continue
        af_diff = abs(
            max([allele1_counts, allele2_counts])
            / min([allele1_counts, allele2_counts])
        )
        if af_diff >= best_af_diff:
            if af_diff <= HARSH_AF_DIFF_CUTOFF:
                best_af_diff = af_diff
                best_var_id = var_id
                best_af = allele1_counts / depth
                best_depth = depth
            elif (af_diff <= MILD_AF_DIFF_CUTOFF) and (
                (depth - best_depth) >= MILD_hSNP_DEPTH_DIFF_CUTOFF
            ):
                best_af_diff = af_diff
                best_var_id = var_id
                best_af = allele1_counts / depth
                best_depth = depth
            else:
                continue
        else:
            best_af_diff = af_diff
            best_var_id = var_id
            best_af = allele1_counts / depth
            best_depth = depth
    return best_var_id, best_af


def count_all_samples_hSNP_INDEL_num(vcf_record):
    if vcf_record == []:
        all_samples_hSNP_INDEL_num = 0
    else:
        all_samples_hSNP_INDEL_num = 0
        for var in vcf_record:
            all_alleles = var.alleles
            if len(all_alleles) > 1:
                all_samples_hSNP_INDEL_num += 1
    return all_samples_hSNP_INDEL_num


def sort_var_record(vcf_record, str_start, str_end):
    dis_list = []
    for var in vcf_record:
        pos = var.pos
        if pos < str_start:
            dis = str_start - pos
        elif pos > str_end:
            dis = pos - str_end
        dis_list.append(dis)
    sort_dis_index_list = np.argsort(
        dis_list
    )  # XXX: np.argsort 默认从小到大，先找近的 nearby SNP 再找远的 nearby SNP
    sort_var_record = [vcf_record[i] for i in sort_dis_index_list]
    return sort_var_record


def select_high_confidence_nearby_snp(
    vcf_record,
    sample,
    spanning_reads_name,
    spanning_pysam_reads_list,
    pysam_bam,
    spanning_reads_dict,
    mosaic_allele_list,
    gn_bed,
):
    # 每个样本有各自的 nearby snp，所以分开看，先统一管理 vcf 去掉一些低质量的条目，再在这些条目中筛选样本中最好的位点
    # 如果满足条件则认为这是个 phasable 位点，并且同时计算 phasable 和 unphasable 的概率
    # 否则只计算 unphasable 的概率
    # 根据两个位点的深度信息进一步的过滤 hSNP
    # 判断是否为最好 hSNP 的标准，最近，还是深度最高，hSNP 包括两个 alleles
    # 打印每一步 hSNP 的过滤信息来判断 phase 是否有问题和足够的信息
    # 使用最多的 reads 和 最多的 alleles 复杂性
    # 最后判断被选中的 phase spanning 的情况，假如不满足条件，则使用 phase mode
    # 不一定 mate 本身 reads 也要检查是否存在 nearby snp
    sample_hSNP_num = 0
    sample_hSNP_INDEL_num = 0
    if vcf_record == []:
        return (
            False,
            False,
            False,
            False,
            sample_hSNP_num,
            sample_hSNP_INDEL_num,
        )
    reads_snp_seq = {}
    reads_snp_seq_accuracy = {}
    nearby_snp_rec = {}
    ref_snp_dict = {}
    if MIX:
        sample = "TCGA-D5-6927-10A-01D-A91Z-36"
    elif GIAB:
        if "HG002" in sample:
            sample = "HG002"
        elif "HG003" in sample:
            sample = "HG003"
        elif "HG004" in sample:
            sample = "HG004"
        elif "HG005" in sample:
            sample = "HG005"
        elif "HG006" in sample:
            sample = "HG006"
        elif "HG007" in sample:
            sample = "HG007"
        elif "HG008" in sample:
            sample = "HG008-N-D"
        else:
            sample = "HG002"
    elif PSEUDOBULK:
        sample = "SRR13989893"
    for var in vcf_record:
        # HACK: filter field must be filtered out before input my program for saving PASS snps only
        gt = var.samples[sample]["GT"]
        if gt[0] == gt[-1]:
            continue
        chrom = var.chrom
        # record start position on chrom/contig (1-based inclusive),so minus 1 to 0-based
        pos = var.pos - 1
        dp = var.samples[sample].get("DP", 0)
        # DEBUG: Finish, DP Format may not exist
        dp = dp if isinstance(dp, int) else 0
        ad = var.samples[sample]["AD"][1]
        ad = ad if isinstance(ad, int) else 0
        if GIAB_NON_HG008:
            if GIAB:
                pass
        else:
            if 0 in (ad, dp):
                continue
            other_allele_dp = dp - np.sum(var.samples[sample]["AD"])
            if (other_allele_dp / dp) > VCF_DIS_DP_THIRD_ALLELE_CUTOFF:
                continue
        all_alleles = var.alleles
        # nearby_snp_alleles = (all_alleles[int(gt[0])], all_alleles[int(gt[-1])])
        sample_hSNP_INDEL_num += 1
        if (
            len(all_alleles[int(gt[0])]) != 1
            or len(all_alleles[int(gt[-1])]) != 1
        ):
            continue
        if GIAB_NON_HG008:
            if GIAB:
                pass
        else:
            try:
                gq = int(var.samples[sample]["GQ"])
            except:
                gq = 100

            if (
                (
                    (hSNP_VAF_MAX >= ad / dp >= hSNP_VAF_MIN)
                    or stats.binomtest(
                        ad, dp, 0.5, alternative="two-sided"
                    ).pvalue
                    > BINOMIAL_TEST_PVALUE_CUTOFF
                )
                and (gq > 20)
                and (dp >= hSNP_MIN_DEPTH)
            ):
                pass
            else:
                continue
        sample_hSNP_num += 1
        var_loci_name = (
            str(chrom)
            + "_"
            + str(pos)
            + "_"
            + str(gt[0])
            + "_"
            + str(gt[-1])
            + "_"
            + str(all_alleles[int(gt[0])])
            + "_"
            + str(all_alleles[int(gt[-1])])
        )
        reads_snp_seq[var_loci_name] = {}
        reads_snp_seq_accuracy[var_loci_name] = {}
        nearby_snp_rec[var_loci_name] = [
            all_alleles[int(gt[0])],
            all_alleles[int(gt[-1])],
            dp,
        ]
        ref_snp = all_alleles[0]
        ref_snp_dict[var_loci_name] = ref_snp
        spanning_str_allele_list = []
        if PHASING_STRATEDY == "READ_NAME":
            # HACK In genotyping, filter low quality snp and str, but in read-level features,filtering won't be done.
            # HACK: mosaic calling nearby snp features 过滤 reads, stepper = PILEUP_STEPPER, ignore_overlaps = True,ignore_orphans = True,min_base_quality = 20,min_mapping_quality = 20
            # TODO: if PE reads both overlap and spanning with STR, i don't cosider this status here, and use both reads for genotyping and mosaic callings STR
            for pileupcolumn in pysam_bam.pileup(
                chrom,
                pos,
                pos + 1,
                stepper=PILEUP_STEPPER,
                ignore_overlaps=True,
                ignore_orphans=True,
                min_base_quality=20,
                min_mapping_quality=20,
                truncate=True,
                max_depth=8000,
            ):
                # perform a pileup within a region. The region is specified by contig, start and stop (using 0-based indexing).
                if pileupcolumn.reference_pos == pos:
                    # the position in the reference sequence (0-based).
                    for pileupread in pileupcolumn.pileups:
                        if (
                            (not pileupread.is_del)
                            and (not pileupread.is_refskip)
                        ) and (
                            pileupread.alignment.query_name
                            in spanning_reads_name
                        ):
                            # query position is None if is_del or is_refskip is set.
                            if (
                                pileupread.alignment.reference_start
                                + AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH
                                <= pos
                                < pileupread.alignment.reference_end
                                - AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH
                            ):
                                snp_seq = pileupread.alignment.query_sequence.upper()[
                                    pileupread.query_position
                                ]
                                snp_baseq = (
                                    pileupread.alignment.query_qualities[
                                        pileupread.query_position
                                    ]
                                )
                                snp_seq_accuracy = (
                                    1 - myutils.baseq_2_error_rate(snp_baseq)
                                )
                                reads_snp_seq[var_loci_name][
                                    pileupread.alignment.query_name
                                ] = snp_seq.upper()
                                reads_snp_seq_accuracy[var_loci_name][
                                    pileupread.alignment.query_name
                                ] = snp_seq_accuracy
                                spanning_str_allele_list.append(
                                    spanning_reads_dict[
                                        pileupread.alignment.query_name
                                    ]
                                )
            # for read in pysam_bam.fetch(chrom, pos, pos+1):
            #     if read.query_name in spanning_reads_name:
            #         reads_snp_seq[var_loci_name][read.query_name] =
        else:
            for read in spanning_pysam_reads_list:
                query_name, snp_seq, snp_seq_accuracy = find_nearby_snp(
                    read, pos, pysam_bam
                )
                if query_name:
                    reads_snp_seq[var_loci_name][query_name] = snp_seq.upper()
                    reads_snp_seq_accuracy[var_loci_name][
                        query_name
                    ] = snp_seq_accuracy
                    spanning_str_allele_list.append(
                        spanning_reads_dict[query_name]
                    )
        if PHASE_MUST_TWO_SPAN_ALLELE:
            for str_allele in mosaic_allele_list:
                if str_allele not in spanning_str_allele_list:
                    del reads_snp_seq[var_loci_name]
                    del reads_snp_seq_accuracy[var_loci_name]
                    del nearby_snp_rec[var_loci_name]
                    break
                else:
                    continue
        else:
            pass

    best_var_id, best_af1 = select_best_nearby_snp(
        reads_snp_seq, nearby_snp_rec, gn_bed, ref_snp_dict
    )
    if best_var_id:
        return (
            best_var_id,
            best_af1,
            reads_snp_seq[best_var_id],
            reads_snp_seq_accuracy[best_var_id],
            sample_hSNP_num,
            sample_hSNP_INDEL_num,
        )
    else:
        return (
            False,
            False,
            False,
            False,
            sample_hSNP_num,
            sample_hSNP_INDEL_num,
        )


def pacbio_phasing(
    HP_tag_dict,
    pacbio_phasing_allele_list,
    mosaic_allele_list,
    PS_tag_dict,
    chrom,
):
    # TODO: Pacbio phasing
    judge_all_spanning_allele = True
    if PHASE_MUST_TWO_SPAN_ALLELE:
        for str_allele in mosaic_allele_list:
            if str_allele not in pacbio_phasing_allele_list:
                judge_all_spanning_allele = False
            else:
                continue
    all_hap_tag_list = list(HP_tag_dict.values())
    # all_depth = len(all_hap_tag_list)
    phasable_depth = all_hap_tag_list.count("1") + all_hap_tag_list.count("2")
    # un_phasable_depth = all_hap_tag_list.count("NA")
    # phasable_fraction = phasable_depth / all_depth
    if phasable_depth >= PACBIO_PHASING_DEPTH_CUTOFF:
        judge_depth = True
        phasable_balance = (
            max([all_hap_tag_list.count("1"), all_hap_tag_list.count("2")])
            / phasable_depth
        )
        if phasable_balance <= PACBIO_PHASING_BALANCE_CUTOFF:
            judge_balance = True
        else:
            judge_balance = False
    else:
        judge_depth = False
        judge_balance = False
    if judge_all_spanning_allele and judge_depth and judge_balance:
        phasable_hap_dict = {}
        phasable_hap_accuracy_dict = {}
        phase_block = str(list(PS_tag_dict.values())[0])
        best_var_id = (
            str(chrom) + "_" + str(phase_block) + "_0" + "_1" + "_1" + "_2"
        )
        # HACK: Assumping spanning reads share the same phase block in a high probability
        # But this is always true ? so we need to check the phase block for each read if this hypothesis is not true
        # TODO: check phase block consistence for phasable check
        # 没发现问题时暂先这么使用，因为 prancSTR 也只使用了 HP tag 在 pacbio phasing 中，并没有使用 PS tag
        for query_name, haptag in HP_tag_dict.items():
            if haptag != "NA":
                phasable_hap_dict[query_name] = haptag
                phasable_hap_accuracy_dict[query_name] = 1
        return (
            best_var_id,
            all_hap_tag_list.count("1") / phasable_depth,
            phasable_hap_dict,
            phasable_hap_accuracy_dict,
            "NA",
            "NA",
        )
    else:
        return (
            False,
            False,
            False,
            False,
            "NA",
            "NA",
        )


def main_per_locus_estimation(parse_params, pysam_bed_row):
    # zero_based_snp_pos = parse_params["zero_based_snp_pos"]
    fasta_file = parse_params["fasta_file"]
    pysam_fasta = pysam.FastaFile(fasta_file)
    bam_name = parse_params["bam_name"]
    add_chr_character = parse_params["add_chr_character"]
    remove_chr_character = parse_params["remove_chr_character"]
    bam_files = parse_params["bam_files"]
    vcf_files = parse_params["vcf_files"]
    output_vcf_files = parse_params["vcf_output"]
    pysam_bam = [bam for bam in bam_files]
    sex = parse_params["sex"]
    sequencing_method = parse_params.get("sequencing_method", ["unknown"])
    stutter_file = parse_params["stutter_file"]
    gnomad_in_bed_file = parse_params["gnomad_freq_in"]
    fail_file = parse_params["fail_file"]
    log_level = getattr(logging, parse_params["loglevel"].upper(), None)
    logfile = parse_params["log_file"]
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {parse_params['loglevel']}")
    logger_mosaic = logging.getLogger("MosaicFractionEstimator")
    logger_mosaic.setLevel(log_level)
    if parse_params["log_to_file"]:
        logger_mosaic_file_handler = myutils.LockedFileHandler(logfile)
    else:
        logger_mosaic_file_handler = logging.StreamHandler()
    # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger_mosaic_file_handler.setFormatter(formatter)
    if not logger_mosaic.handlers:  # 避免重复添加handler
        logger_mosaic.addHandler(logger_mosaic_file_handler)
    # XXX try:
    # all_samples_all_alleles_dict = {}
    # all_samples_fail_reads_dict = defaultdict(int)
    # all_samples_dp = 0
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
        if vcf_files:
            pysam_vcf = pysam.VariantFile(vcf_files)
            # pos_MS = round((seg_locus.start + seg_locus.end) / 2)
            records = list(
                pysam_vcf.fetch(
                    contig=chrom,
                    start=seg_locus.start - SNP_SPANNING_RANGE,
                    stop=seg_locus.start - EXCLUDED_SNP_FLANKING_LENGTH,
                )  # which are 0-based, half-open
            ) + list(
                pysam_vcf.fetch(
                    contig=chrom,
                    start=seg_locus.end + EXCLUDED_SNP_FLANKING_LENGTH,
                    stop=seg_locus.end + SNP_SPANNING_RANGE,
                )
            )
            vcf_all_samples = list(pysam_vcf.header.samples)
        else:
            records = []
            vcf_all_samples = []
        all_samples_hSNP_INDEL_num = count_all_samples_hSNP_INDEL_num(records)
        if gnomad_in_bed_file:
            gn_bed = pysam.TabixFile(gnomad_in_bed_file)
        else:
            gn_bed = ""
        if stutter_file:
            if GIAB_STUTTER:
                stutter_errors_bed = pd.read_csv(
                    stutter_file, sep=",", header=0, index_col=0
                )
                stutter_model_params = {}
                stutter_loci_df = stutter_errors_bed[
                    (
                        stutter_errors_bed["MOTIF_LEN"]
                        == int(seg_locus.motif_length)
                    )
                    & (
                        stutter_errors_bed["left_border"]
                        <= seg_locus.STR_length
                    )
                    & (
                        stutter_errors_bed["right_border"]
                        > seg_locus.STR_length
                    )
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
                inframe_stutter_model = myutils.stutter_model(
                    stutter_model_params["inframe_single_step_prob"],
                    stutter_model_params["inframe_ins_prob"],
                    stutter_model_params["inframe_del_prob"],
                    inframe_max_stutter_steps,
                )
                outframe_stutter_model = myutils.stutter_model(
                    stutter_model_params["outframe_single_step_prob"],
                    stutter_model_params["outframe_ins_prob"],
                    stutter_model_params["outframe_del_prob"],
                    outframe_max_stutter_steps,
                )
            else:
                stutter_errors_bed = pysam.TabixFile(stutter_file)
                stutter_model_params = {}
                for stutter_pysam_bed_row in stutter_errors_bed.fetch(
                    chrom, seg_locus.start, seg_locus.end
                ):
                    row_list = stutter_pysam_bed_row.split("\t")
                    if row_list[5] == seg_locus.STR_id:
                        stutter_model_params[
                            "inframe_single_step_prob"
                        ] = float(row_list[24])
                        stutter_model_params["inframe_ins_prob"] = float(
                            row_list[22]
                        )
                        stutter_model_params["inframe_del_prob"] = float(
                            row_list[23]
                        )
                        stutter_model_params[
                            "outframe_single_step_prob"
                        ] = float(row_list[27])
                        stutter_model_params["outframe_ins_prob"] = float(
                            row_list[25]
                        )
                        stutter_model_params["outframe_del_prob"] = float(
                            row_list[26]
                        )
                        # try:  # XXX: when run renew noise estimation,just use except max steps
                        #     inframe_max_stutter_steps = int(float(row_list[30]))
                        #     outframe_max_stutter_steps = int(float(row_list[31]))
                        # except:
                        if (
                            USE_DEFAULT_MAX_STEP
                        ):  # HACK: When use for mix samples or no stutter estimate samples,please use this parameter instead
                            inframe_max_stutter_steps = DEFAULT_MAX_STEP
                            outframe_max_stutter_steps = DEFAULT_MAX_STEP
                        else:
                            inframe_max_stutter_steps = int(
                                float(row_list[-1])
                            )
                            outframe_max_stutter_steps = int(
                                float(row_list[-1])
                            )
                        inframe_stutter_model = myutils.stutter_model(
                            stutter_model_params["inframe_single_step_prob"],
                            stutter_model_params["inframe_ins_prob"],
                            stutter_model_params["inframe_del_prob"],
                            inframe_max_stutter_steps,
                        )
                        outframe_stutter_model = myutils.stutter_model(
                            stutter_model_params["outframe_single_step_prob"],
                            stutter_model_params["outframe_ins_prob"],
                            stutter_model_params["outframe_del_prob"],
                            outframe_max_stutter_steps,
                        )
                        stutter_model_state = str(row_list[15])
                        break
        else:
            stutter_model_params = INIT_STUTTER_MODEL
            inframe_stutter_model = myutils.stutter_model(
                INIT_STUTTER_MODEL["inframe_single_step_prob"],
                INIT_STUTTER_MODEL["inframe_ins_prob"],
                INIT_STUTTER_MODEL["inframe_del_prob"],
                DEFAULT_MAX_STEP,
            )
            if seg_locus.motif_length == 1:
                outframe_stutter_model = myutils.stutter_model(
                    INIT_STUTTER_MODEL["outframe_single_step_prob"],
                    INIT_STUTTER_MODEL["homopolymer_outframe_ins_prob"],
                    INIT_STUTTER_MODEL["homopolymer_outframe_del_prob"],
                    DEFAULT_MAX_STEP,
                )
            else:
                outframe_stutter_model = myutils.stutter_model(
                    INIT_STUTTER_MODEL["outframe_single_step_prob"],
                    INIT_STUTTER_MODEL["outframe_ins_prob"],
                    INIT_STUTTER_MODEL["outframe_del_prob"],
                    DEFAULT_MAX_STEP,
                )
            stutter_model_state = "DefaultStutterModel"
        if stutter_model_params:
            if (
                stutter_model_state == "MaxIterationReached"
            ):  # TODO: Need revise estimate_noise.py to change the Maximum iteration times
                logger_mosaic.info(
                    "Locus %s %s:%d-%d is not usable because stutter model"
                    " MaxIterationReached\n"
                    % (
                        seg_locus.STR_id,
                        seg_locus.chrom,
                        seg_locus.start,
                        seg_locus.end,
                    )
                )
                with open(fail_file, "a") as f:
                    if myutils.acquire_lock(f):
                        f.write(
                            "ALL\t%s\t%s\t%d\t%d\t%s\t%s\n"
                            % (
                                "ALL",
                                seg_locus.chrom,
                                seg_locus.start,
                                seg_locus.end,
                                seg_locus.STR_id,
                                "MaxIterationReached",
                            )
                        )
                        myutils.release_lock(f)
                # return  # HACK: Need DEBUG, maybe just use the stutter model but not skip this loci to mosaic calling even MaxIterationReached
            else:
                pass
        else:
            logger_mosaic.info(
                "Locus %s %s:%d-%d is not usable because no stutter model"
                " found\n"
                % (
                    seg_locus.STR_id,
                    seg_locus.chrom,
                    seg_locus.start,
                    seg_locus.end,
                )
            )
            with open(fail_file, "a") as f:
                if myutils.acquire_lock(f):
                    f.write(
                        "ALL\t%s\t%s\t%d\t%d\t%s\t%s\n"
                        % (
                            "ALL",
                            seg_locus.chrom,
                            seg_locus.start,
                            seg_locus.end,
                            seg_locus.STR_id,
                            "NoAvaliableStutterModel",
                        )
                    )
                    myutils.release_lock(f)
                    # ALL samples 的过滤情况写入日记文件，single sample 的过滤情况既写入日记文件，又写入 VCF 文件
                    # Phase 的有无情况只写入 VCF 文件，不写入日记文件
            return
        # noise class prep
        other_params = {}
        locus_infos = {}
        # other_params["log_file"] = parse_params["log_file"]
        # other_params["loglevel"] = parse_params["loglevel"]
        # other_params["log_to_file"] = parse_params["log_to_file"]
        other_params["log_file_path"] = logfile
        # other_params["total_unfiltered_depth"] = all_samples_dp
        locus_infos["motif"] = seg_locus.motif
        if sex == "male":
            if (
                (
                    "X" in chrom
                    and seg_locus.end <= config_params.X_PAR1_start_zero_based
                )
                or (
                    "X" in chrom
                    and seg_locus.end <= config_params.X_PAR2_start_zero_based
                    and seg_locus.start >= config_params.X_PAR1_end_zero_based
                )
                or (
                    "X" in chrom
                    and seg_locus.start >= config_params.X_PAR2_end_zero_based
                )
                or (
                    "Y" in chrom
                    and seg_locus.end <= config_params.Y_PAR1_start_zero_based
                )
                or (
                    "Y" in chrom
                    and seg_locus.end <= config_params.Y_PAR2_start_zero_based
                    and seg_locus.start >= config_params.Y_PAR1_end_zero_based
                )
                or (
                    "Y" in chrom
                    and seg_locus.start >= config_params.Y_PAR2_end_zero_based
                )
            ):
                locus_infos["ploidy"] = 1
            else:
                locus_infos["ploidy"] = 2
        else:
            locus_infos["ploidy"] = 2
        locus_infos["motif_length"] = seg_locus.motif_length
        locus_infos["STR_id"] = seg_locus.STR_id
        locus_infos["str_zero_based_start_included"] = seg_locus.start
        locus_infos["str_zero_based_end_excluded"] = seg_locus.end
        locus_infos["chr"] = chrom
        locus_infos["period"] = seg_locus.period
        locus_infos[
            "str_total_length"
        ] = seg_locus.STR_length  # XXX: STR_length is only STR length
        locus_infos["stutter_model_params"] = stutter_model_params
        locus_infos["ref_allele_length"] = seg_locus.ref_allele_length
        if LIKELIHOOD_MODE == "length-based":
            all_alleles_list_for_gt_output = [
                seg_locus.ref_allele_length
            ]  # XXX: if padding, ref_allele_length include padding flanking length
        else:
            if ADD_FLANKING_PROB:
                all_alleles_list_for_gt_output = [
                    (
                        seg_locus.left_flanking_padding_bp,
                        seg_locus.ref_allele_seq,
                        seg_locus.right_flanking_padding_bp,
                    )
                ]

            else:
                all_alleles_list_for_gt_output = [
                    ("", seg_locus.ref_allele_seq, "")
                ]
        variant_info = {}
        samples_dict = {}
        if sequencing_method != ["unknown"]:
            pass
        else:
            sequencing_method = ["illumina"] * len(bam_name)
        seq_index = 0
        for bamname, bamfile_nopysam in zip(bam_name, pysam_bam):
            # all_samples_all_alleles_dict[bamname] = {}
            if "bam" in bamfile_nopysam:
                bamfile = pysam.AlignmentFile(bamfile_nopysam, "rb")
            else:
                bamfile = pysam.AlignmentFile(
                    bamfile_nopysam, "rc", reference_filename=fasta_file
                )
            sample_seq_method = sequencing_method[seq_index]
            seq_index += 1
            alleles_depth_dict = {}
            sample_fail_reads_dict = defaultdict(int)
            reads_list = []
            reads_accuracy_list = []
            pysam_reads_list = []
            alleles_pysam_read_dict = {}
            alleles_reads_accuracy_dict = {}
            alleles_STR_length_list = []
            spanning_reads_name = []
            spanning_pysam_read = []
            # allele_read_name_dict = {}
            spanning_reads_dict = {}
            spanning_reads_accuracy_dict = {}
            filtered_out_reads_depth = 0
            mosaic_allele_list = []
            HP_tag_dict = {}
            PS_tag_dict = {}
            pacbio_phasing_allele_list = []
            # 添加一个新的字典来记录每个allele的barcode/umi信息
            alleles_barcode_umi_dict = {}
            alleles_barcode_umi_dict_all = {}
            for read in bamfile.fetch(chrom, seg_locus.start, seg_locus.end):
                # all_samples_dp += 1
                per_read_feature = extract_allele.PerReadFeature(
                    read, seg_locus.start, seg_locus.end
                )
                per_read_feature.read_is_usable_for_bwa()
                if per_read_feature.is_usable:
                    if config_params.ALLELE_EXTRACT["SegMethods"] == "HMM":
                        readsegger = hmmsegger.SegRead(
                            hmmsegger.hmm_seg_init, seg_locus, read
                        )
                    else:
                        readsegger = coordinate_segger.CoordinateSegRead(
                            seg_locus, read
                        )
                    if readsegger.is_usable:
                        read_name = read.query_name
                        pysam_read = readsegger.pysam_read
                        start = pysam_read.reference_start
                        end = pysam_read.reference_end
                        barcode = per_read_feature.barcode if per_read_feature.barcode else 'NA'
                        hp_tag = read.get_tag("HP") if read.has_tag("HP") else None
                        ps_tag = read.get_tag("PS") if read.has_tag("PS") else None

                        # =============== Step 1: 确定 allele 和 accuracy ===============
                        allele = None
                        read_str_hap_accuracy = None

                        if config_params.ALLELE_EXTRACT["SegMethods"] == "HMM":
                            read_str_hap_accuracy = (
                                    readsegger.left_flanking_allele_bp_baseq_accuray,
                                    readsegger.STR_baseq_accuray,
                                    readsegger.right_flanking_allele_bp_baseq_accuray,
                                )
                            if LIKELIHOOD_MODE == "length-based":
                                allele = readsegger.STR_length
                            else:
                                allele = (
                                    readsegger.left_flanking_allele_bp,
                                    readsegger.STR_seq,
                                    readsegger.right_flanking_allele_bp,
                                ) if ADD_FLANKING_PROB else ("", readsegger.STR_seq, "")
                                
                        else:
                            read_str_hap_accuracy = (
                                1 - readsegger.left_flanking_error_rate,
                                readsegger.read_allele_accuracy,
                                1 - readsegger.right_flanking_error_rate,
                            )
                            if LIKELIHOOD_MODE == "length-based":
                                allele = readsegger.read_allele_length
                            else:
                                allele = (
                                    readsegger.left_flanking_seq,
                                    readsegger.read_allele_seq,
                                    readsegger.right_flanking_seq,
                                ) if ADD_FLANKING_PROB else ("", readsegger.read_allele_seq, "")

                        alleles_barcode_umi_dict_all.setdefault(allele, []).append({
                            'read_name': read_name.replace(':', '-'),
                            'barcode': barcode,
                            'start': start,
                            'end': end
                        })

                        # =============== Step 2: 去重检查 ===============
                        # 重要，防止比较自身
                        existing_records = alleles_barcode_umi_dict_all.get(allele, [])[:-1]

                        is_read_pair = any(
                            record['read_name'] == read_name.replace(':', '-')
                            for record in existing_records
                        )
                        if is_read_pair:
                            sample_fail_reads_dict["read_pair_same"] += 1
                            filtered_out_reads_depth += 1
                            continue

                        is_read_pair_overall = any(
                            other_allele != allele and  # 排除当前 allele
                            any(
                                record['read_name'] == read_name.replace(':', '-')
                                for record in alleles_barcode_umi_dict_all.get(other_allele, [])
                            )
                            for other_allele in alleles_barcode_umi_dict_all
                        )
                        if is_read_pair_overall:
                            sample_fail_reads_dict["read_pair_diff"] += 1

                        is_duplicate = any(
                            record['barcode'] == barcode and
                            record['start'] == start and
                            record['end'] == end
                            for record in existing_records
                        )
                        is_duplicate_loose = any(
                            record['barcode'] == barcode and
                            (record['start'] == start or
                            record['end'] == end)
                            for record in existing_records
                        )
                        is_duplicate_overall = any(
                            other_allele != allele and  # 排除当前 allele
                            any(
                                record['barcode'] == barcode and
                                record['start'] == start and
                                record['end'] == end
                                for record in alleles_barcode_umi_dict_all.get(other_allele, [])
                            )
                            for other_allele in alleles_barcode_umi_dict_all
                        )
                        is_duplicate_overall_loose = any(
                            other_allele != allele and  # 排除当前 allele
                            any(
                                record['barcode'] == barcode and
                                (record['start'] == start or
                                record['end'] == end)
                                for record in alleles_barcode_umi_dict_all.get(other_allele, [])
                            )
                            for other_allele in alleles_barcode_umi_dict_all
                        )

                        flags_1 = [is_duplicate_overall, is_duplicate]
                        flags_2 = [is_duplicate_overall_loose, is_duplicate]
                        flags_3 = [is_duplicate_overall, is_duplicate_loose]
                        flags_4 = [is_duplicate_overall_loose, is_duplicate_loose]
                        if sum(flags_1) >= 2:
                            sample_fail_reads_dict["duplicate_complex_1"] += 1
                        elif sum(flags_2) >= 2:
                            sample_fail_reads_dict["duplicate_complex_2"] += 1
                        elif sum(flags_3) >= 2:
                            sample_fail_reads_dict["duplicate_complex_3"] += 1
                        elif sum(flags_4) >= 2:
                            sample_fail_reads_dict["duplicate_complex_4"] += 1
                        else:
                            if is_duplicate:
                                sample_fail_reads_dict["duplicate_same"] += 1
                            elif is_duplicate_loose:
                                sample_fail_reads_dict["duplicate_loose_same"] += 1
                            if is_duplicate_overall:
                                sample_fail_reads_dict["duplicate_diff"] += 1
                            elif is_duplicate_overall_loose:
                                sample_fail_reads_dict["duplicate_loose_diff"] += 1

                        if is_duplicate or is_duplicate_loose:
                            filtered_out_reads_depth += 1
                            continue

                        # =============== Step 3: 统一添加所有信息 ===============
                        HP_tag_dict[read_name] = str(hp_tag) if hp_tag is not None else "NA"
                        PS_tag_dict[read_name] = str(ps_tag) if ps_tag is not None else "NA"

                        spanning_reads_name.append(read_name)
                        spanning_pysam_read.append(pysam_read)

                        # 注意：alleles_STR_length_list 似乎只用于 length，不管 mode
                        # 根据原始逻辑，HMM 用 STR_length，非 HMM 用 read_allele_length
                        length_value = readsegger.STR_length if config_params.ALLELE_EXTRACT["SegMethods"] == "HMM" else readsegger.read_allele_length
                        alleles_STR_length_list.append(length_value)

                        reads_accuracy_list.append(read_str_hap_accuracy)
                        pysam_reads_list.append(read)

                        # 更新 allele 相关字典
                        alleles_depth_dict[allele] = alleles_depth_dict.get(allele, 0) + 1
                        alleles_pysam_read_dict.setdefault(allele, []).append(read)
                        alleles_reads_accuracy_dict.setdefault(allele, []).append(read_str_hap_accuracy)

                        # 添加带位置信息的 barcode 记录
                        alleles_barcode_umi_dict.setdefault(allele, []).append({
                            'read_name': read_name.replace(':', '-'),
                            'barcode': barcode,
                            'start': start,
                            'end': end
                        })

                        reads_list.append(allele)
                        spanning_reads_dict[read_name] = allele
                        spanning_reads_accuracy_dict[read_name] = None if LIKELIHOOD_MODE == "length-based" else read_str_hap_accuracy

                        # PacBio phasing：直接 append allele 即可
                        if hp_tag is not None:
                            pacbio_phasing_allele_list.append(allele)

                    else:
                        sample_fail_reads_dict[readsegger.unusable_reason] += 1
                        filtered_out_reads_depth += 1

                else:
                    sample_fail_reads_dict[
                        per_read_feature.unusable_reason
                    ] += 1
                    filtered_out_reads_depth += 1
            samples_dict[bamname] = {}
            samples_dict[bamname][
                "SegmentConditionFail"
            ] = sample_fail_reads_dict.get("SegmentConditionFail", 0)
            samples_dict[bamname][
                "SegmentResultFail"
            ] = sample_fail_reads_dict.get("SegmentResultFail", 0)
            samples_dict[bamname]["ReadN"] = sample_fail_reads_dict.get(
                "ReadN", 0
            )
            samples_dict[bamname][
                "library_issue"
            ] = sample_fail_reads_dict.get("library_issue", 0)
            samples_dict[bamname]["map_issue"] = sample_fail_reads_dict.get(
                "map_issue", 0
            )
            samples_dict[bamname][
                "spanning_issue"
            ] = sample_fail_reads_dict.get("spanning_issue", 0)
            samples_dict[bamname]["unmap_issue"] = sample_fail_reads_dict.get(
                "unmap_issue", 0
            )
            samples_dict[bamname]["duplicate_same"] = sample_fail_reads_dict.get(
                "duplicate_same", 0
            )
            samples_dict[bamname]["duplicate_diff"] = sample_fail_reads_dict.get(
                "duplicate_diff", 0
            )
            samples_dict[bamname]["read_pair_same"] = sample_fail_reads_dict.get(
                "read_pair_same", 0
            )
            samples_dict[bamname]["read_pair_diff"] = sample_fail_reads_dict.get(
                "read_pair_diff", 0
            )
            samples_dict[bamname]["duplicate_loose_same"] = sample_fail_reads_dict.get(
                "duplicate_loose_same", 0
            )
            samples_dict[bamname]["duplicate_loose_diff"] = sample_fail_reads_dict.get(
                "duplicate_loose_diff", 0
            )
            samples_dict[bamname]["duplicate_complex_1"] = sample_fail_reads_dict.get(
                "duplicate_complex_1", 0
            )
            samples_dict[bamname]["duplicate_complex_2"] = sample_fail_reads_dict.get(
                "duplicate_complex_2", 0
            )
            samples_dict[bamname]["duplicate_complex_3"] = sample_fail_reads_dict.get(
                "duplicate_complex_3", 0
            )
            samples_dict[bamname]["duplicate_complex_4"] = sample_fail_reads_dict.get(
                "duplicate_complex_4", 0
            )
            samples_dict[bamname]["FDPC"] = (
                str(samples_dict[bamname]["SegmentConditionFail"])
                + ","
                + str(samples_dict[bamname]["SegmentResultFail"])
                + ","
                + str(samples_dict[bamname]["ReadN"])
                + ","
                + str(samples_dict[bamname]["library_issue"])
                + ","
                + str(samples_dict[bamname]["map_issue"])
                + ","
                + str(samples_dict[bamname]["spanning_issue"])
                + ","
                + str(samples_dict[bamname]["unmap_issue"])
                + ","
                + str(samples_dict[bamname]["duplicate_same"])
                + ","
                + str(samples_dict[bamname]["duplicate_diff"])
                + ","
                + str(samples_dict[bamname]["read_pair_same"])
                + ","
                + str(samples_dict[bamname]["read_pair_diff"])
                + ","
                + str(samples_dict[bamname]["duplicate_loose_same"])
                + ","
                + str(samples_dict[bamname]["duplicate_loose_diff"])
                + ","
                + str(samples_dict[bamname]["duplicate_complex_1"])
                + ","
                + str(samples_dict[bamname]["duplicate_complex_2"])
                + ","
                + str(samples_dict[bamname]["duplicate_complex_3"])
                + ","
                + str(samples_dict[bamname]["duplicate_complex_4"])
            )
            new_alleles = list(
                set(alleles_depth_dict.keys())
                - set(all_alleles_list_for_gt_output)
            )
            all_alleles_list_for_gt_output.extend(new_alleles)
            # Flanking in alleles set parameters in this code for alleles definition
            # stutter model is per locus level calculation but not per sample levels calculation
            # TODO: 修改不反复建立 stutter model
            # alleles = alleles_depth_dict.keys()
            samples_dict[bamname]["NALLELES"] = len(alleles_depth_dict)
            unphase_calling = mosaic_fraction_estimation_unphase.MosaicFractionPerSampleEstimator(
                reads_list,
                reads_accuracy_list,
                pysam_reads_list,
                alleles_STR_length_list,
                alleles_pysam_read_dict,
                alleles_reads_accuracy_dict,
                alleles_depth_dict,
                all_alleles_list_for_gt_output,
                stutter_model_params,
                inframe_stutter_model,
                outframe_stutter_model,
                locus_infos,
                other_params,
            )
            if HAP_IDENTIFY_POA:
                all_alleles_list_for_gt_output.extend(
                    unphase_calling.extra_consensus_allele_from_poa
                )
                samples_dict[bamname][
                    "ALLSINGLE"
                ] = unphase_calling.singleton_judge
            else:
                if max(alleles_depth_dict.values()) > 1:
                    samples_dict[bamname][
                        "ALLSINGLE"
                    ] = "not_all_singleton_obs_allele"
                else:
                    samples_dict[bamname][
                        "ALLSINGLE"
                    ] = "all_singleton_obs_allele"
            if unphase_calling.callable:
                unphase_calling.mosaic_fraction_estimation()
                (
                    MaxMosaicGT_for_output,
                    MaxMosaicLogPosterior,
                    MaxMosaicLogLik,
                    SecondMosaicGT_for_output,
                    SecondMosaicLogPosterior,
                    SecondMosaicLogLik,
                    GermlineGT_for_output,
                    GermlineLogPosterior,
                    GermlineLogLik,
                    GermlineLogLik1,
                    GermlineLogLik2,
                    Non_Germ2germLogPosterior,
                    Non_Germ2germLogLikFraction,
                    ConsistentMaxMosaicLogPosterior,
                ) = unphase_calling.cal_somatic_log_posterior()
                # 需要的信息：germline mosaic GT genotype posterior phase and unphase
                # type: germline or mosaic
                # two LRTs，Phase confidence（binominal test and two phase h1/h1+h2 and p1/p1+p2）
                # to pop allele index output
                # TODO: Assumping that big mosaic fraction is germline GT, small mosaic fraction is mosaic GT, mosaic fraction <= 0.5 ???##
                # Posterior germline and mosaic GT
                if MaxMosaicGT_for_output:
                    if sorted(list(SecondMosaicGT_for_output[0])) == sorted(
                        list(SecondMosaicGT_for_output[1])
                    ):
                        if (
                            sorted(list(SecondMosaicGT_for_output[0]))
                            == sorted(list(MaxMosaicGT_for_output[0]))
                        ) or (
                            sorted(list(SecondMosaicGT_for_output[0]))
                            == sorted(list(MaxMosaicGT_for_output[1]))
                        ):
                            samples_dict[bamname][
                                "SECISGERM"
                            ] = "SecondIsMosaicGerm"
                        else:
                            samples_dict[bamname][
                                "SECISGERM"
                            ] = "SecondIsNotMosaicGerm"
                    else:
                        samples_dict[bamname]["SECISGERM"] = "SecondIsNotGerm"
                    if MaxMosaicGT_for_output[0] == MaxMosaicGT_for_output[1]:
                        unphase_mosaic_type = "germline"
                        mosaic_vs_germline_or, mosaic_vs_germline_p = 0, 1
                        mosaic_vs_second_or, mosaic_vs_second_p = 0, 1
                    else:
                        (
                            mosaic_vs_germline_or,
                            mosaic_vs_germline_p,
                        ) = mutation_filtering.ht_log_likelihood_ratio_test(
                            MaxMosaicLogLik, GermlineLogLik, 2
                        )
                        if mosaic_vs_germline_p < 0.05:
                            if (
                                MaxMosaicGT_for_output[0]
                                == SecondMosaicGT_for_output[0]
                            ) or (
                                MaxMosaicGT_for_output[1]
                                == SecondMosaicGT_for_output[1]
                            ):  # 对，包括一个 GT 不一样的情况
                                if (
                                    SecondMosaicGT_for_output[0]
                                    == SecondMosaicGT_for_output[1]
                                ):
                                    freedom_num = 1  # XXX: 1 or 2
                                else:
                                    freedom_num = 1
                            else:
                                freedom_num = 2  # 对，包括两个 GT 不一样的情况
                            # 三个参数 GT/MGT/MF
                            (
                                mosaic_vs_second_or,
                                mosaic_vs_second_p,
                            ) = mutation_filtering.ht_log_likelihood_ratio_test(
                                MaxMosaicLogLik,
                                SecondMosaicLogLik,
                                freedom_num,
                            )
                            if mosaic_vs_second_p < 0.05:
                                unphase_mosaic_type = "mosaic"
                            else:
                                unphase_mosaic_type = (  # "germline_artifacts_2_mosaic"
                                    "artifact"
                                )
                        else:
                            unphase_mosaic_type = "germline"
                            mosaic_vs_second_or, mosaic_vs_second_p = 0, 1
                    if DEBUG:
                        print(MaxMosaicLogLik)  # HACK: 0.06181240393947037
                        print(GermlineLogLik)  # HACK: 0.0 why ?
                    (
                        MaxGermlineGT,
                        MaxGermlineLogPosterior,
                        MaxGermlineLogLik,
                        SecondGermlineGT,
                        SecondGermlineLogPosterior,
                        SecondGermlineLogLik,
                    ) = unphase_calling.cal_germline_log_posterior()
                    germline_freedom = 2 - len(
                        set(MaxGermlineGT) & set(SecondGermlineGT)
                    )
                    (
                        best_germ_vs_second_germ_or,
                        best_germ_vs_second_germ_pvalue,
                    ) = mutation_filtering.ht_log_likelihood_ratio_test(
                        MaxGermlineLogLik,
                        SecondGermlineLogLik,
                        germline_freedom,
                    )
                    # if ((unphase_mosaic_type == "germline") or (unphase_mosaic_type == "artifacts")):
                    #     # germline output
                    #     # germline posterior
                    #     # germline best vs second

                    # elif unphase_mosaic_type == "mosaic":
                    mosaic_gts_alleles = [
                        tuple(
                            [
                                unphase_calling.alleles_used_for_gt[
                                    MaxMosaicGT_for_output[0][0]
                                ],
                                unphase_calling.alleles_used_for_gt[
                                    MaxMosaicGT_for_output[0][1]
                                ],
                            ]
                        ),
                        tuple(
                            [
                                unphase_calling.alleles_used_for_gt[
                                    MaxMosaicGT_for_output[1][0]
                                ],
                                unphase_calling.alleles_used_for_gt[
                                    MaxMosaicGT_for_output[1][1]
                                ],
                            ]
                        ),
                    ]
                    MaxGermGT_for_pop_output = (
                        unphase_calling.get_final_gt_index(MaxGermlineGT)
                    )
                    MaxGermGT_from_MosaicGT_for_pop_output = (
                        unphase_calling.get_final_gt_index(
                            GermlineGT_for_output
                        )
                    )  # 不输出到 VCF 文件
                    MaxMosaicGT_for_pop_output = (
                        unphase_calling.get_final_mosaic_index(
                            MaxMosaicGT_for_output
                        )
                    )
                    max_germ_gt = MaxMosaicGT_for_output[0]
                    max_mosaic_gt = MaxMosaicGT_for_output[1]
                    max_germ_gt_index = unphase_calling.genotypes.index(
                        tuple(max_germ_gt)
                    )
                    max_mosaic_gt_index = unphase_calling.genotypes.index(
                        tuple(max_mosaic_gt)
                    )
                    max_germ_mosaic_gt_index = (
                        unphase_calling.mosaic_genotypes.index(
                            tuple([max_germ_gt_index, max_mosaic_gt_index])
                        )
                    )
                    max_mosaic_mosaic_fraction = (
                        unphase_calling.estimated_mosaic_fraction[
                            max_germ_mosaic_gt_index
                        ]
                    )
                    all_mosaic_alleles = [
                        MaxMosaicGT_for_output[0][0],
                        MaxMosaicGT_for_output[0][1],
                        MaxMosaicGT_for_output[1][0],
                        MaxMosaicGT_for_output[1][1],
                    ]
                    # hap_depth_list = myutils.cal_mle_haplotype_depth_dict_phase(
                    #     unphase_calling.all_reads_all_alleles_log_lik_array)
                    # samples_dict[bamname]["DSTUTTER"] = sum(
                    #     hap_depth_list)-sum(hap_depth_list[all_mosaic_alleles])
                    # all_mosaic_allele_depth = alleles_depth_dict[mosaic_gts_alleles[0][0]] + alleles_depth_dict[mosaic_gts_alleles[0][1]] + alleles_depth_dict[mosaic_gts_alleles[1][0]] + alleles_depth_dict[mosaic_gts_alleles[1][1]]
                    mosaic_gt_allele_depth = 0
                    mosaic_allele_num = 0
                    for mosaic_allele in set(
                        [
                            mosaic_gts_alleles[0][0],
                            mosaic_gts_alleles[0][1],
                            mosaic_gts_alleles[1][0],
                            mosaic_gts_alleles[1][1],
                        ]
                    ):
                        mosaic_gt_allele_depth += alleles_depth_dict.get(
                            mosaic_allele, 0
                        )
                        if alleles_depth_dict.get(mosaic_allele, 0) > 0:
                            mosaic_allele_num += 1
                    observed_stutter_depth = (
                        sum(alleles_depth_dict.values())
                        - mosaic_gt_allele_depth
                    )
                    observed_stutter_fraction = observed_stutter_depth / sum(
                        alleles_depth_dict.values()
                    )
                    observed_stutter_allele_num = (
                        len(alleles_depth_dict) - mosaic_allele_num
                    )
                    observed_stutter_allele_fraction = (
                        observed_stutter_allele_num / len(alleles_depth_dict)
                    )
                    # mask = np.ones_like(hap_depth_list, dtype=bool)
                    # mask[all_mosaic_alleles] = False
                    # count_nonzero = np.count_nonzero(hap_depth_list[mask])
                    # samples_dict[bamname]["NSTUTTER"] = count_nonzero
                    # normalized_lik
                    log_normalized_factor = logsumexp(
                        [GermlineLogLik1, GermlineLogLik2, MaxMosaicLogLik]
                    )  # GermlineLogLik1 GermlineLogLik2 MaxMosaicLogLik All are loglik frac
                    normalized_germ1 = np.exp(
                        GermlineLogLik1 - log_normalized_factor
                    )
                    normalized_germ2 = np.exp(
                        GermlineLogLik2 - log_normalized_factor
                    )
                    normalized_max = np.exp(
                        MaxMosaicLogLik - log_normalized_factor
                    )
                    normalized_second = np.exp(
                        SecondMosaicLogLik
                        - logsumexp([SecondMosaicLogLik, MaxMosaicLogLik])
                    )
                    # XXX: 注意更大的 NORMGERM 值对应 GT 的 lik frac，更小的 NORMGERM 值对应 MGT 的 lik frac
                    # HACK: 这边的 NORMGERM1 和 NORMGERM2 是无序的，不知道 NORMGERM1 大还是 NORMGERM2 大，也不知道 NORMGERM1 对应 GT 还是 NORMGERM2 对应 GT
                    # 也不知道 NORMGERM1 对应 Het 还是 NORMGERM2 对应 Het, 并且 Het2Het 时可以都是 Het
                    samples_dict[bamname]["NORMGERM1"] = normalized_germ1
                    samples_dict[bamname]["NORMGERM2"] = normalized_germ2
                    samples_dict[bamname]["NORMMOSAIC"] = normalized_max
                    samples_dict[bamname]["NORMSECOND"] = normalized_second
                    samples_dict[bamname]["DSTUTTER"] = observed_stutter_depth
                    samples_dict[bamname][
                        "DFSTUTTER"
                    ] = observed_stutter_fraction
                    samples_dict[bamname][
                        "NSTUTTER"
                    ] = observed_stutter_allele_num
                    samples_dict[bamname][
                        "NFSTUTTER"
                    ] = observed_stutter_allele_fraction
                    if (
                        MaxMosaicGT_for_pop_output[0]
                        == MaxMosaicGT_for_pop_output[1]
                    ):
                        # mosaic_type = "Germline"
                        samples_dict[bamname]["MUTP"] = "NA"
                        samples_dict[bamname]["MN"] = 0
                        samples_dict[bamname]["FRAME"] = "NA"
                        samples_dict[bamname]["MBP"] = 0
                        # XXX: Here is observed GT depth and not expected depth (considering stutter error reads depth when use expected)
                        EAD = ""
                        EAF = ""
                        for mosaic_allele_pop_out_index in [
                            MaxMosaicGT_for_pop_output[0][0],
                            MaxMosaicGT_for_pop_output[0][1],
                            MaxMosaicGT_for_pop_output[1][0],
                            MaxMosaicGT_for_pop_output[1][1],
                        ]:
                            mosaic_allele = all_alleles_list_for_gt_output[
                                mosaic_allele_pop_out_index
                            ]
                            EAD += (
                                str(alleles_depth_dict.get(mosaic_allele, 0))
                                + ","
                            )
                            EAF += (
                                str(
                                    alleles_depth_dict.get(mosaic_allele, 0)
                                    / unphase_calling.reads_number
                                )
                                + ","
                            )
                        # for adp in hap_depth_list[all_mosaic_alleles]:
                        #     EAD = str(adp)+","
                        #     EAF = str(adp/unphase_calling.reads_number)+","
                        EAD = EAD[:-1]
                        EAF = EAF[:-1]
                        # TODO Allele pool number of sample, Done!
                        # TODO phase calling allele filter, Done!
                        # TODO EMLEAD, EMLEAF, MLE unknown reads number unphase, Done!
                        if (
                            MaxMosaicGT_for_pop_output[0][0]
                            == MaxMosaicGT_for_pop_output[0][1]
                        ):
                            allele_depth = str(unphase_calling.reads_number)
                            EMLEAD = (
                                allele_depth
                                + ","
                                + allele_depth
                                + ","
                                + allele_depth
                                + ","
                                + allele_depth
                            )
                            EMLEAF = "1,1,1,1"
                            unphase_gt_unknown_assign_num = 0
                        else:
                            mle_allele_index_list = [
                                unphase_calling.alleles_used_for_gt.index(
                                    all_alleles_list_for_gt_output[
                                        MaxMosaicGT_for_pop_output[0][0]
                                    ]
                                ),
                                unphase_calling.alleles_used_for_gt.index(
                                    all_alleles_list_for_gt_output[
                                        MaxMosaicGT_for_pop_output[0][1]
                                    ]
                                ),
                            ]
                            (
                                unphase_gt_hap_depth_list,
                                unphase_gt_unknown_assign_num,
                            ) = myutils.cal_mle_haplotype_depth_dict_given_mosaicgt_phase(
                                unphase_calling.all_reads_all_alleles_log_lik_array,
                                mle_allele_index_list,
                            )
                            EMLEAD = (
                                str(unphase_gt_hap_depth_list[0])
                                + ","
                                + str(unphase_gt_hap_depth_list[1])
                                + ","
                                + str(unphase_gt_hap_depth_list[0])
                                + ","
                                + str(unphase_gt_hap_depth_list[1])
                            )
                            EMLEAF = (
                                str(
                                    unphase_gt_hap_depth_list[0]
                                    / unphase_calling.reads_number
                                )
                                + ","
                                + str(
                                    unphase_gt_hap_depth_list[1]
                                    / unphase_calling.reads_number
                                )
                                + ","
                                + str(
                                    unphase_gt_hap_depth_list[0]
                                    / unphase_calling.reads_number
                                )
                                + ","
                                + str(
                                    unphase_gt_hap_depth_list[1]
                                    / unphase_calling.reads_number
                                )
                            )
                        samples_dict[bamname]["GT"] = (
                            str(MaxMosaicGT_for_pop_output[0][0])
                            + "/"
                            + str(MaxMosaicGT_for_pop_output[0][1])
                        )
                        samples_dict[bamname]["MGT"] = (
                            str(MaxMosaicGT_for_pop_output[1][0])
                            + "/"
                            + str(MaxMosaicGT_for_pop_output[1][1])
                        )
                        samples_dict[bamname][
                            "MF"
                        ] = max_mosaic_mosaic_fraction
                        update_mosaic_gts_alleles = MaxMosaicGT_for_pop_output
                        update_mosaic_fraction = max_mosaic_mosaic_fraction
                    else:
                        # if len(set(MaxMosaicGT_for_pop_output[0]) & set(MaxMosaicGT_for_pop_output[1])) == 0:
                        #     mosaic_type = "TwoAlleleMosaic"
                        #     samples_dict[bamname]["MUTP"] = "."
                        #     samples_dict[bamname]["MN"] = 2
                        #     samples_dict[bamname]["FRAME"] = "."
                        # elif len(set(MaxMosaicGT_for_pop_output[0]) & set(MaxMosaicGT_for_pop_output[1])) == 1:
                        samples_dict[bamname]["MN"] = 1
                        # mosaic_type = "OneAlleleMosaic"
                        # XXX We don't allow Het2Hom and assumping het-hom is mosaic allele,but in fact germGT and mosaicGT maybe inconsistent with mosaic allele,i will refine later, ## TODO
                        # 杂合变杂合时，germ pro 较大的为 germ
                        # 杂合变纯合或纯合变杂合时
                        # mosaic fraction，germline GT from mosaic GT，germline GT from gt lik
                        # het2hom，hom2het，het2het
                        # mosaic GT 的方向和 mosaic fraction 和 mosaic 输出应该一致
                        # 注意 tuple, list 和 set 是不相等的哈
                        # TypeError: 'set' object is not subscriptable
                        # XXX: 目前 MosaicGT 的顺根据 Germ 的概率来排序，所以可能出现 HetHom 的 MosaicGT，而 update_mosaic_fraction 的顺序和 MosaicGT 的顺序一致
                        # XXX: 所以 update_mosaic_fraction 可能有 HetHom 的 mosaic fraction 因为这边的 mosaic fraction germ 取决于概率而不管是不是 homhet hethet hethom
                        # XXX: 所以计算 features 时有一个额外的 MF_hom2het_het2het 的 mosaic fraction，只考虑 homhet hethet，hethom 会计算成 1 - update_mosaic_fraction 作为 MF_hom2het_het2het
                        if tuple(MaxMosaicGT_for_pop_output[0]) == tuple(
                            MaxGermGT_from_MosaicGT_for_pop_output
                        ):  # 两个都是tuple
                            if (
                                len(set(MaxMosaicGT_for_pop_output[1])) == 1
                            ):  # HetHom
                                if PROB_BASED_GERMLINE:  # True
                                    mosaic_allele = MaxMosaicGT_for_pop_output[
                                        1
                                    ][0]
                                    mosaic_source_germline_allele = list(
                                        set(MaxMosaicGT_for_pop_output[0])
                                        - set(MaxMosaicGT_for_pop_output[1])
                                    )[0]
                                    # XXX: Let mosaic allele in the second allele, adjust the GT order and Mosaic order
                                    # update_mosaic_gts_alleles = MaxMosaicGT_for_pop_output
                                    # another_germline_allele = list(
                                    #     set(MaxMosaicGT_for_pop_output[0]) & set(MaxMosaicGT_for_pop_output[1]))[0]
                                    # update_mosaic_gts_alleles = tuple([tuple([another_germline_allele,mosaic_source_germline_allele]),MaxMosaicGT_for_pop_output[1]])
                                    another_germline_allele = list(
                                        set(MaxMosaicGT_for_pop_output[0])
                                        & set(MaxMosaicGT_for_pop_output[1])
                                    )[0]
                                    # XXX: Use this because allow germ2germ and single allele mutation
                                    update_mosaic_gts_alleles = tuple(
                                        [
                                            tuple(
                                                [
                                                    another_germline_allele,
                                                    mosaic_source_germline_allele,
                                                ]
                                            ),
                                            tuple(
                                                [
                                                    another_germline_allele,
                                                    mosaic_allele,
                                                ]
                                            ),
                                        ]
                                    )
                                    update_mosaic_fraction = (
                                        max_mosaic_mosaic_fraction
                                    )
                                else:
                                    mosaic_allele = list(
                                        set(MaxMosaicGT_for_pop_output[0])
                                        - set(MaxMosaicGT_for_pop_output[1])
                                    )[0]
                                    mosaic_source_germline_allele = (
                                        MaxMosaicGT_for_pop_output[1][0]
                                    )
                                    another_germline_allele = list(
                                        set(MaxMosaicGT_for_pop_output[0])
                                        & set(MaxMosaicGT_for_pop_output[1])
                                    )[0]
                                    update_mosaic_gts_alleles = tuple(
                                        [
                                            tuple(
                                                [
                                                    another_germline_allele,
                                                    mosaic_source_germline_allele,
                                                ]
                                            ),
                                            tuple(
                                                [
                                                    another_germline_allele,
                                                    mosaic_allele,
                                                ]
                                            ),
                                        ]
                                    )
                                    update_mosaic_fraction = (
                                        1 - max_mosaic_mosaic_fraction
                                    )
                            else:  # HomHet or HetHet
                                mosaic_allele = list(
                                    set(MaxMosaicGT_for_pop_output[1])
                                    - set(MaxMosaicGT_for_pop_output[0])
                                )[0]
                                if (
                                    len(set(MaxMosaicGT_for_pop_output[0]))
                                    == 1
                                ):
                                    mosaic_source_germline_allele = (
                                        MaxMosaicGT_for_pop_output[0][0]
                                    )
                                else:
                                    mosaic_source_germline_allele = list(
                                        set(MaxMosaicGT_for_pop_output[0])
                                        - set(MaxMosaicGT_for_pop_output[1])
                                    )[0]
                                another_germline_allele = list(
                                    set(MaxMosaicGT_for_pop_output[0])
                                    & set(MaxMosaicGT_for_pop_output[1])
                                )[0]
                                update_mosaic_gts_alleles = tuple(
                                    [
                                        tuple(
                                            [
                                                another_germline_allele,
                                                mosaic_source_germline_allele,
                                            ]
                                        ),
                                        tuple(
                                            [
                                                another_germline_allele,
                                                mosaic_allele,
                                            ]
                                        ),
                                    ]
                                )
                                update_mosaic_fraction = (
                                    max_mosaic_mosaic_fraction
                                )
                        else:
                            if (
                                len(set(MaxMosaicGT_for_pop_output[0])) == 1
                            ):  # HetHom
                                if PROB_BASED_GERMLINE:
                                    mosaic_allele = MaxMosaicGT_for_pop_output[
                                        0
                                    ][0]
                                    mosaic_source_germline_allele = list(
                                        set(MaxMosaicGT_for_pop_output[1])
                                        - set(MaxMosaicGT_for_pop_output[0])
                                    )[0]
                                    # update_mosaic_gts_alleles = tuple([MaxMosaicGT_for_pop_output[1], MaxMosaicGT_for_pop_output[0]])
                                    another_germline_allele = list(
                                        set(MaxMosaicGT_for_pop_output[0])
                                        & set(MaxMosaicGT_for_pop_output[1])
                                    )[0]
                                    update_mosaic_gts_alleles = tuple(
                                        [
                                            tuple(
                                                [
                                                    another_germline_allele,
                                                    mosaic_source_germline_allele,
                                                ]
                                            ),
                                            tuple(
                                                [
                                                    another_germline_allele,
                                                    mosaic_allele,
                                                ]
                                            ),
                                        ]
                                    )
                                    update_mosaic_fraction = (
                                        1 - max_mosaic_mosaic_fraction
                                    )
                                else:
                                    mosaic_allele = list(
                                        set(MaxMosaicGT_for_pop_output[1])
                                        - set(MaxMosaicGT_for_pop_output[0])
                                    )[0]
                                    mosaic_source_germline_allele = (
                                        MaxMosaicGT_for_pop_output[0][0]
                                    )
                                    # update_mosaic_gts_alleles = MaxMosaicGT_for_pop_output
                                    another_germline_allele = list(
                                        set(MaxMosaicGT_for_pop_output[0])
                                        & set(MaxMosaicGT_for_pop_output[1])
                                    )[0]
                                    update_mosaic_gts_alleles = tuple(
                                        [
                                            tuple(
                                                [
                                                    another_germline_allele,
                                                    mosaic_source_germline_allele,
                                                ]
                                            ),
                                            tuple(
                                                [
                                                    another_germline_allele,
                                                    mosaic_allele,
                                                ]
                                            ),
                                        ]
                                    )
                                    update_mosaic_fraction = (
                                        max_mosaic_mosaic_fraction
                                    )
                            else:  # HomHet or HetHet
                                mosaic_allele = list(
                                    set(MaxMosaicGT_for_pop_output[0])
                                    - set(MaxMosaicGT_for_pop_output[1])
                                )[0]
                                if (
                                    len(set(MaxMosaicGT_for_pop_output[1]))
                                    == 1
                                ):
                                    mosaic_source_germline_allele = (
                                        MaxMosaicGT_for_pop_output[1][0]
                                    )
                                else:
                                    mosaic_source_germline_allele = list(
                                        set(MaxMosaicGT_for_pop_output[1])
                                        - set(MaxMosaicGT_for_pop_output[0])
                                    )[0]
                                # update_mosaic_gts_alleles = tuple([MaxMosaicGT_for_pop_output[1], MaxMosaicGT_for_pop_output[0]])
                                another_germline_allele = list(
                                    set(MaxMosaicGT_for_pop_output[0])
                                    & set(MaxMosaicGT_for_pop_output[1])
                                )[0]
                                update_mosaic_gts_alleles = tuple(
                                    [
                                        tuple(
                                            [
                                                another_germline_allele,
                                                mosaic_source_germline_allele,
                                            ]
                                        ),
                                        tuple(
                                            [
                                                another_germline_allele,
                                                mosaic_allele,
                                            ]
                                        ),
                                    ]
                                )
                                update_mosaic_fraction = (
                                    1 - max_mosaic_mosaic_fraction
                                )
                        # if len(set(MaxMosaicGT_for_pop_output[0]) - set(MaxMosaicGT_for_pop_output[1])) == 1:
                        #     mosaic_allele = list(
                        #         set(MaxMosaicGT_for_pop_output[0]) - set(MaxMosaicGT_for_pop_output[1]))[0]
                        # else:
                        #     mosaic_allele = (
                        #         set(MaxMosaicGT_for_pop_output[1]) - set(MaxMosaicGT_for_pop_output[0]))[0]
                        if LIKELIHOOD_MODE == "length-based":
                            if (
                                all_alleles_list_for_gt_output[mosaic_allele]
                                == all_alleles_list_for_gt_output[
                                    mosaic_source_germline_allele
                                ]
                            ):
                                # if all_alleles_list_for_gt_output[mosaic_allele] == seg_locus.ref_allele_length:
                                samples_dict[bamname]["MUTP"] = "mismatch"
                                samples_dict[bamname]["FRAME"] = "zero"
                                samples_dict[bamname]["MBP"] = 0
                            elif (
                                all_alleles_list_for_gt_output[mosaic_allele]
                                > all_alleles_list_for_gt_output[
                                    mosaic_source_germline_allele
                                ]
                            ):
                                # elif all_alleles_list_for_gt_output[mosaic_allele] > seg_locus.ref_allele_length:
                                samples_dict[bamname]["MUTP"] = "ins"
                                if (
                                    abs(
                                        all_alleles_list_for_gt_output[
                                            mosaic_allele
                                        ]
                                        - all_alleles_list_for_gt_output[
                                            mosaic_source_germline_allele
                                        ]
                                    )
                                    % seg_locus.motif_length
                                    == 0
                                ):
                                    samples_dict[bamname]["FRAME"] = "inframe"
                                else:
                                    samples_dict[bamname]["FRAME"] = "outframe"
                                samples_dict[bamname]["MBP"] = (
                                    all_alleles_list_for_gt_output[
                                        mosaic_allele
                                    ]
                                    - all_alleles_list_for_gt_output[
                                        mosaic_source_germline_allele
                                    ]
                                )
                            else:
                                samples_dict[bamname]["MUTP"] = "del"
                                if (
                                    abs(
                                        all_alleles_list_for_gt_output[
                                            mosaic_allele
                                        ]
                                        - all_alleles_list_for_gt_output[
                                            mosaic_source_germline_allele
                                        ]
                                    )
                                    % seg_locus.motif_length
                                    == 0
                                ):
                                    samples_dict[bamname]["FRAME"] = "inframe"
                                else:
                                    samples_dict[bamname]["FRAME"] = "outframe"
                                samples_dict[bamname]["MBP"] = (
                                    all_alleles_list_for_gt_output[
                                        mosaic_allele
                                    ]
                                    - all_alleles_list_for_gt_output[
                                        mosaic_source_germline_allele
                                    ]
                                )
                        else:
                            if len(
                                all_alleles_list_for_gt_output[mosaic_allele][
                                    1
                                ]
                            ) == len(
                                all_alleles_list_for_gt_output[
                                    mosaic_source_germline_allele
                                ][1]
                            ):
                                # if len(all_alleles_list_for_gt_output[mosaic_allele][1]) == seg_locus.ref_allele_length:
                                samples_dict[bamname]["MUTP"] = "mismatch"
                                samples_dict[bamname]["FRAME"] = "zero"
                                samples_dict[bamname]["MBP"] = 0
                            # elif len(all_alleles_list_for_gt_output[mosaic_allele][1]) > seg_locus.ref_allele_length:
                            elif len(
                                all_alleles_list_for_gt_output[mosaic_allele][
                                    1
                                ]
                            ) > len(
                                all_alleles_list_for_gt_output[
                                    mosaic_source_germline_allele
                                ][1]
                            ):
                                samples_dict[bamname]["MUTP"] = "ins"
                                if (
                                    abs(
                                        len(
                                            all_alleles_list_for_gt_output[
                                                mosaic_allele
                                            ][1]
                                        )
                                        - len(
                                            all_alleles_list_for_gt_output[
                                                mosaic_source_germline_allele
                                            ][1]
                                        )
                                    )
                                    % seg_locus.motif_length
                                    == 0
                                ):
                                    samples_dict[bamname]["FRAME"] = "inframe"
                                else:
                                    samples_dict[bamname]["FRAME"] = "outframe"
                                samples_dict[bamname]["MBP"] = len(
                                    all_alleles_list_for_gt_output[
                                        mosaic_allele
                                    ][1]
                                ) - len(
                                    all_alleles_list_for_gt_output[
                                        mosaic_source_germline_allele
                                    ][1]
                                )  # 先前版本没括号并且 -减号 另起了一行造成 MBP 这个参数错误
                            else:
                                samples_dict[bamname]["MUTP"] = "del"
                                if (
                                    abs(
                                        len(
                                            all_alleles_list_for_gt_output[
                                                mosaic_allele
                                            ][1]
                                        )
                                        - len(
                                            all_alleles_list_for_gt_output[
                                                mosaic_source_germline_allele
                                            ][1]
                                        )
                                    )
                                    % seg_locus.motif_length
                                    == 0
                                ):
                                    samples_dict[bamname]["FRAME"] = "inframe"
                                else:
                                    samples_dict[bamname]["FRAME"] = "outframe"
                                samples_dict[bamname]["MBP"] = len(
                                    all_alleles_list_for_gt_output[
                                        mosaic_allele
                                    ][1]
                                ) - len(
                                    all_alleles_list_for_gt_output[
                                        mosaic_source_germline_allele
                                    ][1]
                                )  # 先前版本没括号并且 -减号 另起了一行造成 MBP 这个参数错误
                        # # (12) mut bp and mut motif
                        # mut_bp = mosaic_allele_str - mutated_allele_str
                        # mut_motif_steps = (mut_bp)/motif_length
                        # # 求余数
                        # remainder = (mut_bp)%motif_length
                        # # 求整数商
                        # integer_quotient = (mut_bp)//motif_length
                        # mutated_allele_is_perfect_tr = myutils.is_tandem_repeated_sequence(phase_mosaic_fraction_estimation.alleles_seq_used_for_gt[mutated_allele_str_index],motif_length)
                        # mosaic_allele_is_perfect_tr = myutils.is_tandem_repeated_sequence(phase_mosaic_fraction_estimation.alleles_seq_used_for_gt[mosaic_allele_str_index],motif_length)
                        # XXX: Here is observed GT depth and not expected depth (considering stutter error reads depth when use expected)
                        four_alleles_index_list = []
                        mle_allele_index_list = []
                        EAD = ""
                        EAF = ""
                        for mosaic_allele_pop_out_index in [
                            update_mosaic_gts_alleles[0][0],
                            update_mosaic_gts_alleles[0][1],
                            update_mosaic_gts_alleles[1][0],
                            update_mosaic_gts_alleles[1][1],
                        ]:
                            mosaic_allele = all_alleles_list_for_gt_output[
                                mosaic_allele_pop_out_index
                            ]
                            mosaic_allele_used_index = (
                                unphase_calling.alleles_used_for_gt.index(
                                    mosaic_allele
                                )
                            )
                            four_alleles_index_list.append(
                                mosaic_allele_used_index
                            )
                            if (
                                mosaic_allele_used_index
                                in mle_allele_index_list
                            ):
                                pass
                            else:
                                mle_allele_index_list.append(
                                    mosaic_allele_used_index
                                )
                            EAD += (
                                str(alleles_depth_dict.get(mosaic_allele, 0))
                                + ","
                            )
                            EAF += (
                                str(
                                    alleles_depth_dict.get(mosaic_allele, 0)
                                    / unphase_calling.reads_number
                                )
                                + ","
                            )
                        (
                            unphase_gt_hap_depth_list,
                            unphase_gt_unknown_assign_num,
                        ) = myutils.cal_mle_haplotype_depth_dict_given_mosaicgt_phase(
                            unphase_calling.all_reads_all_alleles_log_lik_array,
                            mle_allele_index_list,
                        )
                        EMLEAD = ""
                        EMLEAF = ""
                        for (
                            mosaic_allele_used_index
                        ) in four_alleles_index_list:
                            hap_depth_index = mle_allele_index_list.index(
                                mosaic_allele_used_index
                            )
                            mosaic_allele_used_depth = (
                                unphase_gt_hap_depth_list[hap_depth_index]
                            )
                            EMLEAD += str(mosaic_allele_used_depth) + ","
                            EMLEAF += (
                                str(
                                    mosaic_allele_used_depth
                                    / unphase_calling.reads_number
                                )
                                + ","
                            )
                        # for adp in hap_depth_list[all_mosaic_alleles]:
                        #     EAD = str(adp)+","
                        #     EAF = str(adp/unphase_calling.reads_number)+","
                        EAD = EAD[:-1]
                        EAF = EAF[:-1]
                        EMLEAD = EMLEAD[:-1]
                        EMLEAF = EMLEAF[:-1]
                        samples_dict[bamname]["GT"] = (
                            str(update_mosaic_gts_alleles[0][0])
                            + "/"
                            + str(update_mosaic_gts_alleles[0][1])
                        )
                        samples_dict[bamname]["MGT"] = (
                            str(update_mosaic_gts_alleles[1][0])
                            + "/"
                            + str(update_mosaic_gts_alleles[1][1])
                        )
                        samples_dict[bamname]["MF"] = update_mosaic_fraction
                    # HACK: MF may represent the Het2Hom mosaic fraction, and if only use hom2het and het2het mosaic fraction
                    # The het2hom mosaic fraction need transform to 1 - MF
                    if (
                        str(update_mosaic_gts_alleles[1][0])
                        == str(update_mosaic_gts_alleles[1][1])
                    ) and (
                        samples_dict[bamname]["GT"]
                        != samples_dict[bamname]["MGT"]
                    ):
                        samples_dict[bamname]["MF_hom2het_het2het"] = (
                            1 - samples_dict[bamname]["MF"]
                        )
                    else:
                        samples_dict[bamname][
                            "MF_hom2het_het2het"
                        ] = samples_dict[bamname]["MF"]
                    samples_dict[bamname]["EAD"] = EAD
                    samples_dict[bamname]["EAF"] = EAF
                    samples_dict[bamname]["EMLEAD"] = EMLEAD
                    samples_dict[bamname]["EMLEAF"] = EMLEAF
                    samples_dict[bamname]["MLEUAF"] = (
                        unphase_gt_unknown_assign_num
                        / unphase_calling.reads_number
                    )
                    # samples_dict[bamname]["GT"] = str(
                    #     MaxMosaicGT_for_pop_output[0][0])+"/"+str(MaxMosaicGT_for_pop_output[0][1])
                    # samples_dict[bamname]["MGT"] = str(
                    #     MaxMosaicGT_for_pop_output[1][0])+"/"+str(MaxMosaicGT_for_pop_output[1][1])
                    # samples_dict[bamname]["VARIANTTYPE"] = mosaic_type
                    samples_dict[bamname]["VARIANTTYPE"] = unphase_mosaic_type
                    samples_dict[bamname][
                        "REGION"
                    ] = (  # TODO: align mutated allele to reference and mutant allele to mutated allele to anchor mutation location and mutation region
                        "NA"
                    )
                    samples_dict[bamname]["MP"] = np.exp(MaxMosaicLogPosterior)
                    samples_dict[bamname]["CMP"] = np.exp(
                        ConsistentMaxMosaicLogPosterior
                    )
                    samples_dict[bamname]["AMP"] = np.exp(
                        Non_Germ2germLogPosterior
                    )
                    samples_dict[bamname][
                        "GQ"
                    ] = f"{mosaic_vs_second_or},{mosaic_vs_second_p}"
                    samples_dict[bamname][
                        "GG"
                    ] = f"{mosaic_vs_germline_or},{mosaic_vs_germline_p}"
                    samples_dict[bamname][
                        "PSTR"
                    ] = myutils.is_tandem_repeated_sequence(
                        seg_locus.ref_STR_sequence, seg_locus.motif_length
                    )
                    samples_dict[bamname]["DP"] = unphase_calling.reads_number
                    samples_dict[bamname]["FDP"] = filtered_out_reads_depth
                    AAD = ""
                    for allele in all_alleles_list_for_gt_output:
                        allele_depth = alleles_depth_dict.get(allele, 0)
                        AAD = AAD + str(allele_depth) + ","
                    AAD = AAD[:-1]
                    samples_dict[bamname]["AAD"] = AAD

                    # 添加ALLELE_BARCODE_UMI处理
                    ALLELE_BARCODE_UMI = ""
                    ALLELE_BARCODE_MOSAIC = ""
                    for i, allele in enumerate(all_alleles_list_for_gt_output):
                        allele_depth = alleles_depth_dict.get(allele, 0)
                        if int(allele_depth) > 0:  # 只处理有reads支持的allele
                            if allele in alleles_barcode_umi_dict:
                                barcode_umi_list = alleles_barcode_umi_dict[allele]
                                # 将所有barcode_umi组合起来
                                barcode_umi_str = "&".join([f"{info['read_name']}_{info['barcode']}" for info in barcode_umi_list])
                                if barcode_umi_str:  # 如果有barcode_umi信息
                                    ALLELE_BARCODE_UMI += f"{i}|{barcode_umi_str};"
                                if i == update_mosaic_gts_alleles[1][1]:
                                    ALLELE_BARCODE_MOSAIC += f"{i}|{barcode_umi_str}"
                    
                    # 去掉最后一个分号，如果没有信息则用"."表示
                    ALLELE_BARCODE_UMI = ALLELE_BARCODE_UMI[:-1] if ALLELE_BARCODE_UMI else "."
                    samples_dict[bamname]["ALLELE_BARCODE_UMI"] = ALLELE_BARCODE_UMI
                    ALLELE_BARCODE_MOSAIC = ALLELE_BARCODE_MOSAIC if ALLELE_BARCODE_MOSAIC else "."
                    samples_dict[bamname]["ALLELE_BARCODE_MOSAIC"] = ALLELE_BARCODE_MOSAIC

                    filter1 = int(
                        samples_dict[bamname]["MP"]
                        > MAX_MOSAIC_POSTERIOR_CUTOFF
                    )
                    filter2 = int(
                        samples_dict[bamname]["AMP"]
                        > ALL_MOSAIC_POSTERIOR_CUTOFF
                    )
                    filter3 = int(mosaic_vs_second_p < LRT_PVALUE_CUTOFF)
                    filter4 = int(mosaic_vs_germline_p < LRT_PVALUE_CUTOFF)
                    confidence = filter1 + filter2 + filter3 + filter4
                    samples_dict[bamname][
                        "FILTER"
                    ] = f"{confidence},{filter1},{filter2},{filter3},{filter4}"
                    # germline 的信息,修改 VCF header title 和 record
                    samples_dict[bamname]["GI"] = (
                        str(MaxGermGT_for_pop_output[0])
                        + "/"
                        + str(MaxGermGT_for_pop_output[1])
                    )
                    samples_dict[bamname]["GIP"] = np.exp(
                        MaxGermlineLogPosterior
                    )
                    samples_dict[bamname]["GIQ"] = best_germ_vs_second_germ_or
                    samples_dict[bamname][
                        "GIQP"
                    ] = best_germ_vs_second_germ_pvalue
                    mosaic_allele_list.append(
                        all_alleles_list_for_gt_output[
                            MaxMosaicGT_for_pop_output[0][0]
                        ]
                    )
                    mosaic_allele_list.append(
                        all_alleles_list_for_gt_output[
                            MaxMosaicGT_for_pop_output[0][1]
                        ]
                    )
                    mosaic_allele_list.append(
                        all_alleles_list_for_gt_output[
                            MaxMosaicGT_for_pop_output[1][0]
                        ]
                    )
                    mosaic_allele_list.append(
                        all_alleles_list_for_gt_output[
                            MaxMosaicGT_for_pop_output[1][1]
                        ]
                    )
                    mosaic_allele_list = list(set(mosaic_allele_list))
                else:
                    # samples_dict[bamname] = {}
                    uncallable_reason = unphase_calling.loci_calling
                    samples_dict[bamname]["FILTER"] = uncallable_reason
                    (
                        GI,
                        GIP,
                        GIQ,
                        GIQP,
                    ) = (
                        unphase_calling.still_output_germline_gt_and_post_when_uncallable()
                    )
                    if GI == ".":
                        samples_dict[bamname]["GI"] = GI
                    else:
                        MaxGermGT_for_pop_output = (
                            unphase_calling.get_final_gt_index(GI)
                        )
                        samples_dict[bamname]["GI"] = (
                            str(MaxGermGT_for_pop_output[0])
                            + "/"
                            + str(MaxGermGT_for_pop_output[1])
                        )
                    samples_dict[bamname]["GIP"] = GIP
                    samples_dict[bamname]["GIQ"] = GIQ
                    samples_dict[bamname]["GIQP"] = GIQP
                    logger_mosaic.info(
                        "Sample %s and Locus %s %s:%d-%d is not usable"
                        " because %s"
                        % (
                            bamname,
                            seg_locus.STR_id,
                            seg_locus.chrom,
                            seg_locus.start,
                            seg_locus.end,
                            uncallable_reason,
                        )
                    )
                    with open(fail_file, "a") as f:
                        if myutils.acquire_lock(f):
                            f.write(
                                "unphase\t%s\t%s\t%d\t%d\t%s\t%s\n"
                                % (
                                    bamname,
                                    seg_locus.chrom,
                                    seg_locus.start,
                                    seg_locus.end,
                                    seg_locus.STR_id,
                                    uncallable_reason,
                                )
                            )
                            myutils.release_lock(f)
                    # continue
                    # HACK: add extra infors for DP and AAD
                    samples_dict[bamname]["DP"] = unphase_calling.reads_number
                    AAD = ""
                    for allele in all_alleles_list_for_gt_output:
                        allele_depth = alleles_depth_dict.get(allele, 0)
                        AAD = AAD + str(allele_depth) + ","
                    AAD = AAD[:-1]
                    samples_dict[bamname]["AAD"] = AAD
                    # HACK: No check here

                    ALLELE_BARCODE_UMI = ""
                    for i, allele in enumerate(all_alleles_list_for_gt_output):
                        allele_depth = alleles_depth_dict.get(allele, 0)
                        if int(allele_depth) > 0:  # 只处理有reads支持的allele
                            if allele in alleles_barcode_umi_dict:
                                barcode_umi_list = alleles_barcode_umi_dict[allele]
                                # 将所有barcode_umi组合起来
                                barcode_umi_str = "&".join([f"{info['read_name']}_{info['barcode']}" for info in barcode_umi_list])
                                if barcode_umi_str:  # 如果有barcode_umi信息
                                    ALLELE_BARCODE_UMI += f"{i}|{barcode_umi_str};"
                    
                    # 去掉最后一个分号，如果没有信息则用"."表示
                    ALLELE_BARCODE_UMI = ALLELE_BARCODE_UMI[:-1] if ALLELE_BARCODE_UMI else "."
                    samples_dict[bamname]["ALLELE_BARCODE_UMI"] = ALLELE_BARCODE_UMI
                    samples_dict[bamname]["ALLELE_BARCODE_MOSAIC"] = "."
            else:
                # samples_dict[bamname] = {}
                MaxMosaicGT_for_output = False  # 为了下面 phase 的条件判断
                uncallable_reason = unphase_calling.loci_calling
                samples_dict[bamname]["FILTER"] = uncallable_reason
                (
                    GI,
                    GIP,
                    GIQ,
                    GIQP,
                ) = (
                    unphase_calling.still_output_germline_gt_and_post_when_uncallable()
                )
                if GI == ".":
                    samples_dict[bamname]["GI"] = GI
                else:
                    MaxGermGT_for_pop_output = (
                        unphase_calling.get_final_gt_index(GI)
                    )
                    samples_dict[bamname]["GI"] = (
                        str(MaxGermGT_for_pop_output[0])
                        + "/"
                        + str(MaxGermGT_for_pop_output[1])
                    )
                samples_dict[bamname]["GIP"] = GIP
                samples_dict[bamname]["GIQ"] = GIQ
                samples_dict[bamname]["GIQP"] = GIQP
                logger_mosaic.info(
                    "Sample %s and Locus %s %s:%d-%d is not usable because %s"
                    % (
                        bamname,
                        seg_locus.STR_id,
                        seg_locus.chrom,
                        seg_locus.start,
                        seg_locus.end,
                        uncallable_reason,
                    )
                )
                with open(fail_file, "a") as f:
                    if myutils.acquire_lock(f):
                        f.write(
                            "unphase\t%s\t%s\t%d\t%d\t%s\t%s\n"
                            % (
                                bamname,
                                seg_locus.chrom,
                                seg_locus.start,
                                seg_locus.end,
                                seg_locus.STR_id,
                                uncallable_reason,
                            )
                        )
                        myutils.release_lock(f)
                # continue
                # HACK: add extra infors for DP and AAD
                samples_dict[bamname]["DP"] = unphase_calling.reads_number
                AAD = ""
                for allele in all_alleles_list_for_gt_output:
                    allele_depth = alleles_depth_dict.get(allele, 0)
                    AAD = AAD + str(allele_depth) + ","
                AAD = AAD[:-1]
                samples_dict[bamname]["AAD"] = AAD
                # HACK: No check here
                
                ALLELE_BARCODE_UMI = ""
                for i, allele in enumerate(all_alleles_list_for_gt_output):
                    allele_depth = alleles_depth_dict.get(allele, 0)
                    if int(allele_depth) > 0:  # 只处理有reads支持的allele
                        if allele in alleles_barcode_umi_dict:
                            barcode_umi_list = alleles_barcode_umi_dict[allele]
                            # 将所有barcode_umi组合起来
                            barcode_umi_str = "&".join([f"{info['read_name']}_{info['barcode']}" for info in barcode_umi_list])
                            if barcode_umi_str:  # 如果有barcode_umi信息
                                ALLELE_BARCODE_UMI += f"{i}|{barcode_umi_str};"
                
                # 去掉最后一个分号，如果没有信息则用"."表示
                ALLELE_BARCODE_UMI = ALLELE_BARCODE_UMI[:-1] if ALLELE_BARCODE_UMI else "."
                samples_dict[bamname]["ALLELE_BARCODE_UMI"] = ALLELE_BARCODE_UMI
                samples_dict[bamname]["ALLELE_BARCODE_MOSAIC"] = "."

            if samples_dict[bamname]["GI"] == ".":
                GIOAF = "NA,NA"
                GIMLEAF = "NA,NA"
                GI_unknown_assign_fraction = "NA"
            else:
                if (
                    samples_dict[bamname]["GI"].split("/")[0]
                    == samples_dict[bamname]["GI"].split("/")[1]
                ):
                    GIOAF = "1,0"
                    GIMLEAF = "1,0"
                    GI_unknown_assign_fraction = "0"
                else:
                    GI1_allele_index = int(
                        samples_dict[bamname]["GI"].split("/")[0]
                    )
                    GI2_allele_index = int(
                        samples_dict[bamname]["GI"].split("/")[1]
                    )
                    GI1_allele = all_alleles_list_for_gt_output[
                        GI1_allele_index
                    ]
                    GI2_allele = all_alleles_list_for_gt_output[
                        GI2_allele_index
                    ]
                    GI1_allele_calling_index = (
                        unphase_calling.alleles_used_for_gt.index(GI1_allele)
                    )
                    GI2_allele_calling_index = (
                        unphase_calling.alleles_used_for_gt.index(GI2_allele)
                    )
                    GI1_allele_freq = (
                        alleles_depth_dict.get(GI1_allele, 0)
                        / unphase_calling.reads_number
                    )
                    GI2_allele_freq = (
                        alleles_depth_dict.get(GI2_allele, 0)
                        / unphase_calling.reads_number
                    )
                    GIOAF = f"{GI1_allele_freq:.3f},{GI2_allele_freq:.3f}"
                    all_expected_GI_haps = [
                        GI1_allele_calling_index,
                        GI2_allele_calling_index,
                    ]
                    (
                        mle_GI_depth_list,
                        GI_unknown_assign_num,
                    ) = myutils.cal_mle_haplotype_depth_dict_given_mosaicgt_phase(
                        unphase_calling.all_reads_all_alleles_log_lik_array,
                        all_expected_GI_haps,
                    )
                    GI1_mle_allele_freq = mle_GI_depth_list[0] / sum(
                        mle_GI_depth_list
                    )  # XXX: only assignable as all
                    GI2_mle_allele_freq = mle_GI_depth_list[1] / sum(
                        mle_GI_depth_list
                    )  # XXX: only assignable as all
                    GIMLEAF = (
                        f"{GI1_mle_allele_freq:.3f},{GI2_mle_allele_freq:.3f}"
                    )
                    GI_unknown_assign_fraction = (
                        GI_unknown_assign_num / unphase_calling.reads_number
                    )  # XXX: total depth as all
            samples_dict[bamname]["GIOAF"] = GIOAF
            samples_dict[bamname]["GIMLEAF"] = GIMLEAF
            samples_dict[bamname]["GIUAF"] = GI_unknown_assign_fraction
            # 开始写 phase 部分
            if unphase_calling.callable and MaxMosaicGT_for_output:
                # 如果不加这个条件限制，会导致出现 LLD decrease/mosaic fraction out of range/MaxIterReached 的时候，phase这边出现报错，所以需要加上此条判断
                # 因为上面用了 continue 所以不加这个限制也不会报错，会直接进入下一个样本，那现在把 continue 注释掉，只有 unphase_calling.callable 和 MaxMosaicGT_for_output 都为 True 时才进入 phase 模式哈
                if sample_seq_method == "illumina":
                    if GIAB:
                        (
                            best_var_id,
                            best_af1,
                            reads_snp_seq_dict,
                            reads_snp_seq_accuracy_dict,
                            sample_hSNP_num,
                            sample_hSNP_INDEL_num,
                        ) = select_high_confidence_nearby_snp(
                            records,
                            bamname,
                            spanning_reads_name,
                            spanning_pysam_read,
                            bamfile,
                            spanning_reads_dict,
                            mosaic_allele_list,
                            gn_bed,
                        )
                    else:
                        if (
                            bamname not in vcf_all_samples
                        ):  # HACK: MIX Pseudo-bulk and GIAB need revise this block
                            (
                                best_var_id,
                                best_af1,
                                reads_snp_seq_dict,
                                reads_snp_seq_accuracy_dict,
                                sample_hSNP_num,
                                sample_hSNP_INDEL_num,
                            ) = (False, False, False, False, "NA", "NA")
                        else:
                            (
                                best_var_id,
                                best_af1,
                                reads_snp_seq_dict,
                                reads_snp_seq_accuracy_dict,
                                sample_hSNP_num,
                                sample_hSNP_INDEL_num,
                            ) = select_high_confidence_nearby_snp(
                                records,
                                bamname,
                                spanning_reads_name,
                                spanning_pysam_read,
                                bamfile,
                                spanning_reads_dict,
                                mosaic_allele_list,
                                gn_bed,
                            )
                elif sample_seq_method == "pacbio":
                    (
                        best_var_id,
                        best_af1,
                        reads_snp_seq_dict,
                        reads_snp_seq_accuracy_dict,
                        sample_hSNP_num,
                        sample_hSNP_INDEL_num,
                    ) = pacbio_phasing(
                        HP_tag_dict,
                        pacbio_phasing_allele_list,
                        mosaic_allele_list,
                        PS_tag_dict,
                        chrom,
                    )
                else:
                    raise ValueError("Unknown sequencing method")
                samples_dict[bamname]["NHSNPN"] = sample_hSNP_num
                samples_dict[bamname]["NHSNPINDELN"] = sample_hSNP_INDEL_num
                if best_var_id:
                    phasable = True
                    phasing_reads_list = []
                    phasing_reads_accuracy_list = []
                    phasing_alleles_depth_dict = {}
                    phasing_alleles_depth_dict2 = {}
                    read_snp_dict = {}
                    for read_name, read_snp in reads_snp_seq_dict.items():
                        read_snp_dict[read_snp] = (
                            read_snp_dict.get(read_snp, 0) + 1
                        )
                        phasing_reads_list.append(
                            (spanning_reads_dict[read_name], read_snp)
                        )
                        phasing_reads_accuracy_list.append(
                            (
                                spanning_reads_accuracy_dict[read_name],
                                reads_snp_seq_accuracy_dict[read_name],
                            )
                        )
                        phasing_alleles_depth_dict[
                            (spanning_reads_dict[read_name], read_snp)
                        ] = (
                            phasing_alleles_depth_dict.get(
                                (spanning_reads_dict[read_name], read_snp), 0
                            )
                            + 1
                        )
                        phasing_alleles_depth_dict2[
                            spanning_reads_dict[read_name]
                        ] = (
                            phasing_alleles_depth_dict2.get(
                                spanning_reads_dict[read_name], 0
                            )
                            + 1
                        )
                    # phasable 条件更改为需要包含一条 spanning mutant reads/germ reads/source reads
                    # 和添加更严苛的条件来判断
                    nearby_snp = (
                        best_var_id.split("_")[-2],
                        best_var_id.split("_")[-1],
                    )
                    phase_calling = mosaic_fraction_estimation_phase.PhasedMosaicFractionPerSampleEstimator(
                        phasing_reads_list,
                        phasing_reads_accuracy_list,
                        alleles_STR_length_list,
                        alleles_depth_dict,  # XXX: phasing_alleles_depth_dict,
                        all_alleles_list_for_gt_output,
                        unphase_calling.alleles_used_for_gt,
                        nearby_snp,
                        stutter_model_params,
                        inframe_stutter_model,
                        outframe_stutter_model,
                        locus_infos,
                        other_params,
                    )

                    if phase_calling.callable:
                        phasable = True
                        samples_dict[bamname].update(UNPHASE_PHASE_OUTPUT)
                        samples_dict[bamname]["PHA"] = "Yes"
                        samples_dict[bamname]["UPR"] = "HasNearbySNP"
                        samples_dict[bamname]["PS"] = best_var_id.split("_")[1]
                        samples_dict[bamname]["NSNP"] = (
                            nearby_snp[0] + "|" + nearby_snp[1]
                        )
                        samples_dict[bamname]["HVAF"] = best_af1
                        allele1_hap = all_alleles_list_for_gt_output[
                            update_mosaic_gts_alleles[0][0]
                        ]
                        allele2_hap = all_alleles_list_for_gt_output[
                            update_mosaic_gts_alleles[0][1]
                        ]
                        allele3_hap = all_alleles_list_for_gt_output[
                            update_mosaic_gts_alleles[1][0]
                        ]
                        allele4_hap = all_alleles_list_for_gt_output[
                            update_mosaic_gts_alleles[1][1]
                        ]
                        allele1_index = (
                            phase_calling.alleles_used_for_gt.index(
                                allele1_hap
                            )
                        )
                        allele2_index = (
                            phase_calling.alleles_used_for_gt.index(
                                allele2_hap
                            )
                        )
                        allele3_index = (
                            phase_calling.alleles_used_for_gt.index(
                                allele3_hap
                            )
                        )
                        allele4_index = (
                            phase_calling.alleles_used_for_gt.index(
                                allele4_hap
                            )
                        )
                        Phase1_GT1 = phase_calling.genotypes.index(
                            tuple([allele1_index, allele2_index])
                        )
                        Phase1_GT2 = phase_calling.genotypes.index(
                            tuple([allele3_index, allele4_index])
                        )
                        # phase gt 的顺序只代表和哪一个 hSNP 进行 link
                        # 而 unphase gt 的顺序则代表了是哪一个 allele 发生了 mosaic mutations
                        allele_number = len(phase_calling.alleles_used_for_gt)
                        (
                            p1_log_lik,
                            p2_log_lik,
                            p1_log_posterior,
                            p2_log_posterior,
                        ) = phase_calling.mosaic_genotype_linking_log_lik_posteriors(
                            tuple([Phase1_GT1, Phase1_GT2]),
                            nearby_snp,
                            update_mosaic_fraction,
                        )
                        if p1_log_posterior >= p2_log_posterior:
                            MaxPhasedMosaicLogLik = p1_log_lik
                            best_phasing_mosaic_gts_alleles = (
                                update_mosaic_gts_alleles
                            )
                            mosaic_gts_alleles_index = [
                                allele1_index,
                                allele2_index + allele_number,
                                allele3_index,
                                allele4_index + allele_number,
                            ]
                            phase_posterior = np.exp(p1_log_posterior)
                            update_Phase1_GT1 = phase_calling.genotypes.index(
                                tuple([allele1_index, allele2_index])
                            )
                            update_Phase1_GT2 = phase_calling.genotypes.index(
                                tuple([allele3_index, allele4_index])
                            )
                        else:
                            MaxPhasedMosaicLogLik = p2_log_lik
                            best_phasing_mosaic_gts_alleles = tuple(
                                [
                                    tuple(
                                        [
                                            update_mosaic_gts_alleles[0][1],
                                            update_mosaic_gts_alleles[0][0],
                                        ]
                                    ),
                                    tuple(
                                        [
                                            update_mosaic_gts_alleles[1][1],
                                            update_mosaic_gts_alleles[1][0],
                                        ]
                                    ),
                                ]
                            )
                            mosaic_gts_alleles_index = [
                                allele2_index,
                                allele1_index + allele_number,
                                allele4_index,
                                allele3_index + allele_number,
                            ]
                            phase_posterior = np.exp(p2_log_posterior)
                            update_Phase1_GT1 = phase_calling.genotypes.index(
                                tuple([allele2_index, allele1_index])
                            )
                            update_Phase1_GT2 = phase_calling.genotypes.index(
                                tuple([allele4_index, allele3_index])
                            )
                        (
                            germ1_p1_log_lik,
                            germ1_p2_log_lik,
                            germ1_p1_log_posterior,
                            germ1_p2_log_posterior,
                        ) = phase_calling.mosaic_genotype_linking_log_lik_posteriors(
                            tuple([update_Phase1_GT1, update_Phase1_GT1]),
                            nearby_snp,
                            update_mosaic_fraction,
                        )
                        (
                            germ2_p1_log_lik,
                            germ2_p2_log_lik,
                            germ2_p1_log_posterior,
                            germ2_p2_log_posterior,
                        ) = phase_calling.mosaic_genotype_linking_log_lik_posteriors(
                            tuple([update_Phase1_GT2, update_Phase1_GT2]),
                            nearby_snp,
                            update_mosaic_fraction,
                        )
                        if (germ1_p1_log_lik >= germ1_p2_log_lik) and (
                            germ2_p1_log_lik >= germ2_p2_log_lik
                        ):
                            if germ1_p1_log_lik >= germ2_p1_log_lik:
                                phase_germ_log_lik = germ1_p1_log_lik
                            else:
                                phase_germ_log_lik = germ2_p1_log_lik
                            (
                                phase_mosaic_vs_germline_or,
                                phase_mosaic_vs_germline_p,
                            ) = mutation_filtering.ht_log_likelihood_ratio_test(
                                MaxPhasedMosaicLogLik, phase_germ_log_lik, 2
                            )
                        else:
                            (
                                phase_mosaic_vs_germline_or,
                                phase_mosaic_vs_germline_p,
                            ) = ("NA", "NA")
                        samples_dict[bamname][
                            "GERMLR"
                        ] = phase_mosaic_vs_germline_or
                        samples_dict[bamname][
                            "GERMLRTP"
                        ] = phase_mosaic_vs_germline_p
                        best_phasing_mosaic_gts_alleles_hap = tuple(
                            [
                                tuple(
                                    [
                                        (
                                            all_alleles_list_for_gt_output[
                                                best_phasing_mosaic_gts_alleles[
                                                    0
                                                ][
                                                    0
                                                ]
                                            ],
                                            nearby_snp[0],
                                        ),
                                        (
                                            all_alleles_list_for_gt_output[
                                                best_phasing_mosaic_gts_alleles[
                                                    0
                                                ][
                                                    1
                                                ]
                                            ],
                                            nearby_snp[1],
                                        ),
                                    ]
                                ),
                                tuple(
                                    [
                                        (
                                            all_alleles_list_for_gt_output[
                                                best_phasing_mosaic_gts_alleles[
                                                    1
                                                ][
                                                    0
                                                ]
                                            ],
                                            nearby_snp[0],
                                        ),
                                        (
                                            all_alleles_list_for_gt_output[
                                                best_phasing_mosaic_gts_alleles[
                                                    1
                                                ][
                                                    1
                                                ]
                                            ],
                                            nearby_snp[1],
                                        ),
                                    ]
                                ),
                            ]
                        )
                        best_phasing_mosaic_gts_alleles_str_hap = tuple(
                            [
                                tuple(
                                    [
                                        all_alleles_list_for_gt_output[
                                            best_phasing_mosaic_gts_alleles[0][
                                                0
                                            ]
                                        ],
                                        all_alleles_list_for_gt_output[
                                            best_phasing_mosaic_gts_alleles[0][
                                                1
                                            ]
                                        ],
                                    ]
                                ),
                                tuple(
                                    [
                                        all_alleles_list_for_gt_output[
                                            best_phasing_mosaic_gts_alleles[1][
                                                0
                                            ]
                                        ],
                                        all_alleles_list_for_gt_output[
                                            best_phasing_mosaic_gts_alleles[1][
                                                1
                                            ]
                                        ],
                                    ]
                                ),
                            ]
                        )
                        phase_proportion = (
                            len(phasing_reads_list)
                            / unphase_calling.reads_number
                        )
                        observed_concordant_reads_number = 0
                        observed_concordant_haps_number = 0
                        for cordant_hap in set(
                            [
                                best_phasing_mosaic_gts_alleles_hap[0][0],
                                best_phasing_mosaic_gts_alleles_hap[0][1],
                                best_phasing_mosaic_gts_alleles_hap[1][0],
                                best_phasing_mosaic_gts_alleles_hap[1][1],
                            ]
                        ):
                            observed_concordant_reads_number += phasing_alleles_depth_dict.get(
                                cordant_hap, 0
                            )  # XXX: 有时候 cordant_hap 不在 phasing_alleles_depth_dict
                            if (
                                phasing_alleles_depth_dict.get(cordant_hap, 0)
                                > 0
                            ):
                                observed_concordant_haps_number += (
                                    1  # TODO: Last Time OK!
                                )
                        observed_discordant_reads_number = (
                            len(phasing_reads_list)
                            - observed_concordant_reads_number
                        )
                        # mle_discordant_reads_number =
                        observed_discordant_reads_rate = (
                            observed_discordant_reads_number
                            / len(phasing_reads_list)
                        )
                        # mle_discordant_reads_rate =
                        observed_discordant_hap_number = (
                            len(phasing_alleles_depth_dict)
                            - observed_concordant_haps_number
                        )
                        observed_discordant_hap_rate = (
                            observed_discordant_hap_number
                            / len(phasing_alleles_depth_dict)
                        )
                        if p1_log_posterior >= p2_log_posterior:
                            if (allele1_hap != allele2_hap) and (
                                allele3_hap == allele4_hap
                            ):  # HACK: 有时候代表性的 allele 不在 观察的 alleles_depth_dict 里面
                                if (
                                    alleles_depth_dict.get(
                                        best_phasing_mosaic_gts_alleles_hap[0][
                                            1
                                        ][0],
                                        0,
                                    )
                                    == 0
                                ):
                                    phased_mosaic_allele_proportion = 0
                                else:
                                    phased_mosaic_allele_proportion = (
                                        phasing_alleles_depth_dict2.get(
                                            best_phasing_mosaic_gts_alleles_hap[
                                                0
                                            ][
                                                1
                                            ][
                                                0
                                            ],
                                            0,
                                        )
                                        / alleles_depth_dict[
                                            best_phasing_mosaic_gts_alleles_hap[
                                                0
                                            ][
                                                1
                                            ][
                                                0
                                            ]
                                        ]
                                    )
                            else:
                                if (
                                    alleles_depth_dict.get(
                                        best_phasing_mosaic_gts_alleles_hap[1][
                                            1
                                        ][0],
                                        0,
                                    )
                                    == 0
                                ):
                                    phased_mosaic_allele_proportion = 0
                                else:
                                    phased_mosaic_allele_proportion = (
                                        phasing_alleles_depth_dict2.get(
                                            best_phasing_mosaic_gts_alleles_hap[
                                                1
                                            ][
                                                1
                                            ][
                                                0
                                            ],
                                            0,
                                        )
                                        / alleles_depth_dict[
                                            best_phasing_mosaic_gts_alleles_hap[
                                                1
                                            ][
                                                1
                                            ][
                                                0
                                            ]
                                        ]
                                    )
                        else:
                            if (allele1_hap != allele2_hap) and (
                                allele3_hap == allele4_hap
                            ):
                                if (
                                    alleles_depth_dict.get(
                                        best_phasing_mosaic_gts_alleles_hap[0][
                                            0
                                        ][0],
                                        0,
                                    )
                                    == 0
                                ):
                                    phased_mosaic_allele_proportion = 0
                                else:
                                    phased_mosaic_allele_proportion = (
                                        phasing_alleles_depth_dict2.get(
                                            best_phasing_mosaic_gts_alleles_hap[
                                                0
                                            ][
                                                0
                                            ][
                                                0
                                            ],
                                            0,
                                        )
                                        / alleles_depth_dict[
                                            best_phasing_mosaic_gts_alleles_hap[
                                                0
                                            ][
                                                0
                                            ][
                                                0
                                            ]
                                        ]
                                    )
                            else:
                                if (
                                    alleles_depth_dict.get(
                                        best_phasing_mosaic_gts_alleles_hap[1][
                                            0
                                        ][0],
                                        0,
                                    )
                                    == 0
                                ):
                                    phased_mosaic_allele_proportion = 0
                                else:
                                    phased_mosaic_allele_proportion = (
                                        phasing_alleles_depth_dict2.get(
                                            best_phasing_mosaic_gts_alleles_hap[
                                                1
                                            ][
                                                0
                                            ][
                                                0
                                            ],
                                            0,
                                        )
                                        / alleles_depth_dict[
                                            best_phasing_mosaic_gts_alleles_hap[
                                                1
                                            ][
                                                0
                                            ][
                                                0
                                            ]
                                        ]
                                    )
                        all_expected_phase_haps = []
                        for alleleindex in set(
                            [
                                allele1_index,
                                allele2_index,
                                allele3_index,
                                allele4_index,
                            ]
                        ):
                            all_expected_phase_haps.append(alleleindex)
                            all_expected_phase_haps.append(
                                alleleindex + allele_number
                            )  # Here should be alleleindex
                        all_expected_phase_haps = list(
                            set(all_expected_phase_haps)
                        )
                        (
                            mle_hap_depth_list,
                            mosaic_unknown_assign_num,
                        ) = myutils.cal_mle_haplotype_depth_dict_given_mosaicgt_phase(
                            phase_calling.all_reads_all_alleles_linking_log_lik_array,
                            all_expected_phase_haps,
                        )
                        mle_all_hap_number = np.sum(
                            np.array(mle_hap_depth_list) > 0
                        )
                        mle_hap_number = np.sum(
                            np.array(mle_hap_depth_list)
                            > COUNT_MLE_HAP_NUM_DEPTH_CUTOFF  # 1
                        )
                        discordant_mle_reads = 0
                        concordant_mle_reads = 0
                        discordant_mle_hap_depth_list = []
                        for index, alleleindex in enumerate(
                            all_expected_phase_haps
                        ):
                            if alleleindex not in mosaic_gts_alleles_index:
                                discordant_mle_reads += mle_hap_depth_list[
                                    index
                                ]
                                discordant_mle_hap_depth_list.append(
                                    mle_hap_depth_list[index]
                                )
                            else:
                                concordant_mle_reads += mle_hap_depth_list[
                                    index
                                ]
                        discordant_mle_rate = discordant_mle_reads / (
                            discordant_mle_reads + concordant_mle_reads
                        )  # XXX: only assignable as all
                        # 添加 mle phase read counts for every hap number include for mosaic hap
                        mle_mosaic_gt_allele_depth_list = []
                        mle_mosaic_gt_allele_depth_frequency_list = []
                        for allele_index in mosaic_gts_alleles_index:
                            depth_index = all_expected_phase_haps.index(
                                allele_index
                            )
                            allele_depth = mle_hap_depth_list[depth_index]
                            mle_mosaic_gt_allele_depth_list.append(
                                allele_depth
                            )
                            depth_frequency = allele_depth / len(
                                phasing_reads_list
                            )  # all phase reads number as 分母
                            mle_mosaic_gt_allele_depth_frequency_list.append(
                                depth_frequency
                            )
                        mle_mosaic_gt_allele_depth_str_list = list(
                            map(
                                lambda x: str(x),
                                mle_mosaic_gt_allele_depth_list,
                            )
                        )
                        mle_mosaic_gt_allele_depth_frequency_str_list = list(
                            map(
                                lambda x: str(x),
                                mle_mosaic_gt_allele_depth_frequency_list,
                            )
                        )
                        discordant_mle_hap_depth_str_list = list(
                            map(
                                lambda x: str(x), discordant_mle_hap_depth_list
                            )
                        )
                        samples_dict[bamname]["MLEPEAD"] = ",".join(
                            mle_mosaic_gt_allele_depth_str_list
                        )
                        samples_dict[bamname]["MLEPEAF"] = ",".join(
                            mle_mosaic_gt_allele_depth_frequency_str_list
                        )
                        samples_dict[bamname]["MLEDISHD"] = ",".join(
                            discordant_mle_hap_depth_str_list
                        )
                        samples_dict[bamname][
                            "MLEPUAF"
                        ] = mosaic_unknown_assign_num / len(phasing_reads_list)
                        all_expected_mle_allele_depth_list = []
                        all_expected_mle_allele_mosaic_allele_logic_list = []
                        all_observed_allele_depth_list = []
                        obs_hap_depth_list = []
                        same_mle_hap_depth_list = []
                        used_obs_allele_index_list = []
                        # XXX: allele1_index 等跟随 GT MGT 领域的顺序，GT MGT 跟随概率最大为 Germ 顺序，所以可能出现杂合变纯合的情况出现
                        for enu_index, allele_index in enumerate(
                            [
                                allele1_index,
                                allele2_index,
                                allele3_index,
                                allele4_index,
                            ]
                        ):
                            depth_index = all_expected_phase_haps.index(
                                allele_index
                            )
                            allele_depth = mle_hap_depth_list[depth_index]
                            all_expected_mle_allele_depth_list.append(
                                str(allele_depth)
                            )
                            obs_allele_hap = (
                                phase_calling.alleles_used_for_gt[
                                    allele_index
                                ],
                                nearby_snp[0],
                            )
                            observed_allele_depth = (
                                phasing_alleles_depth_dict.get(
                                    obs_allele_hap, 0
                                )
                            )
                            all_observed_allele_depth_list.append(
                                str(observed_allele_depth)
                            )
                            depth_index2 = all_expected_phase_haps.index(
                                allele_index + allele_number
                            )
                            allele_depth2 = mle_hap_depth_list[depth_index2]
                            all_expected_mle_allele_depth_list.append(
                                str(allele_depth2)
                            )
                            obs_allele_hap2 = (
                                phase_calling.alleles_used_for_gt[
                                    allele_index
                                ],
                                nearby_snp[1],
                            )
                            observed_allele_depth2 = (
                                phasing_alleles_depth_dict.get(
                                    obs_allele_hap2, 0
                                )
                            )
                            all_observed_allele_depth_list.append(
                                str(observed_allele_depth2)
                            )
                            if allele_index not in mosaic_gts_alleles_index:
                                all_expected_mle_allele_mosaic_allele_logic_list.append(
                                    "0"
                                )
                            else:
                                all_expected_mle_allele_mosaic_allele_logic_list.append(
                                    "1"
                                )
                            if (
                                allele_index + allele_number
                                not in mosaic_gts_alleles_index
                            ):
                                all_expected_mle_allele_mosaic_allele_logic_list.append(
                                    "0"
                                )
                            else:
                                all_expected_mle_allele_mosaic_allele_logic_list.append(
                                    "1"
                                )
                            if allele_index in used_obs_allele_index_list:
                                pass
                            else:
                                used_obs_allele_index_list.append(allele_index)
                                obs_hap_depth_list.append(
                                    observed_allele_depth
                                )
                                obs_hap_depth_list.append(
                                    observed_allele_depth2
                                )
                                same_mle_hap_depth_list.append(allele_depth)
                                same_mle_hap_depth_list.append(allele_depth2)

                        all_expected_mle_allele_depth_str = ",".join(
                            all_expected_mle_allele_depth_list
                        )
                        all_expected_mle_allele_mosaic_allele_logic_str = (
                            ",".join(
                                all_expected_mle_allele_mosaic_allele_logic_list
                            )
                        )
                        all_observed_allele_depth_str = ",".join(
                            all_observed_allele_depth_list
                        )
                        samples_dict[bamname][
                            "MLEPALLAD"
                        ] = all_expected_mle_allele_depth_str
                        samples_dict[bamname][
                            "MLEPALLAL"
                        ] = all_expected_mle_allele_mosaic_allele_logic_str
                        samples_dict[bamname][
                            "OPALLAD"
                        ] = all_observed_allele_depth_str
                        # TODO all_observed phasing hap depth in mosaicGT allele pools Has Solved !
                        filter_hap_number = 0
                        obs_all_phasing_hap_count_dict = {}
                        mle_all_phasing_hap_count_dict = {}
                        # obs 和 mle 也跟随 GT 和 MGT 的顺序，但是 HetHom 会被颠倒以使 Allele2 一直为 mosaic allele 对于 HomHet 和 HetHet
                        for hap, depth in phasing_alleles_depth_dict.items():
                            if depth > COUNT_HAP_NUM_DEPTH_CUTOFF:  # 1
                                filter_hap_number += 1
                        if allele1_index == allele2_index:
                            if (allele1_index == allele2_index) and (
                                allele3_index == allele4_index
                            ):
                                update_minor_allele_phasing_index = "NA"
                                update_minor_allele_phasing_index_source = "NA"
                                update_minor_allele_phasing_index_germ = "NA"
                            else:
                                obs_all_phasing_hap_count_dict[
                                    "allele1_h1"
                                ] = obs_hap_depth_list[0]
                                obs_all_phasing_hap_count_dict[
                                    "allele1_h2"
                                ] = obs_hap_depth_list[1]
                                obs_all_phasing_hap_count_dict[
                                    "allele2_h1"
                                ] = obs_hap_depth_list[2]
                                obs_all_phasing_hap_count_dict[
                                    "allele2_h2"
                                ] = obs_hap_depth_list[3]
                                mle_all_phasing_hap_count_dict[
                                    "allele1_h1"
                                ] = same_mle_hap_depth_list[0]
                                mle_all_phasing_hap_count_dict[
                                    "allele1_h2"
                                ] = same_mle_hap_depth_list[1]
                                mle_all_phasing_hap_count_dict[
                                    "allele2_h1"
                                ] = same_mle_hap_depth_list[2]
                                mle_all_phasing_hap_count_dict[
                                    "allele2_h2"
                                ] = same_mle_hap_depth_list[3]
                                update_minor_allele_phasing_index = (
                                    allele4_index
                                )
                                update_minor_allele_phasing_index_source = "NA"
                                update_minor_allele_phasing_index_germ = "NA"
                        else:
                            if allele3_index == allele4_index:
                                if allele1_index == allele3_index:
                                    obs_all_phasing_hap_count_dict[
                                        "allele1_h1"
                                    ] = obs_hap_depth_list[0]
                                    obs_all_phasing_hap_count_dict[
                                        "allele1_h2"
                                    ] = obs_hap_depth_list[1]
                                    obs_all_phasing_hap_count_dict[
                                        "allele2_h1"
                                    ] = obs_hap_depth_list[2]
                                    obs_all_phasing_hap_count_dict[
                                        "allele2_h2"
                                    ] = obs_hap_depth_list[3]
                                    mle_all_phasing_hap_count_dict[
                                        "allele1_h1"
                                    ] = same_mle_hap_depth_list[0]
                                    mle_all_phasing_hap_count_dict[
                                        "allele1_h2"
                                    ] = same_mle_hap_depth_list[1]
                                    mle_all_phasing_hap_count_dict[
                                        "allele2_h1"
                                    ] = same_mle_hap_depth_list[2]
                                    mle_all_phasing_hap_count_dict[
                                        "allele2_h2"
                                    ] = same_mle_hap_depth_list[3]
                                else:
                                    obs_all_phasing_hap_count_dict[
                                        "allele1_h1"
                                    ] = obs_hap_depth_list[2]
                                    obs_all_phasing_hap_count_dict[
                                        "allele1_h2"
                                    ] = obs_hap_depth_list[3]
                                    obs_all_phasing_hap_count_dict[
                                        "allele2_h1"
                                    ] = obs_hap_depth_list[0]
                                    obs_all_phasing_hap_count_dict[
                                        "allele2_h2"
                                    ] = obs_hap_depth_list[1]
                                    mle_all_phasing_hap_count_dict[
                                        "allele1_h1"
                                    ] = same_mle_hap_depth_list[2]
                                    mle_all_phasing_hap_count_dict[
                                        "allele1_h2"
                                    ] = same_mle_hap_depth_list[3]
                                    mle_all_phasing_hap_count_dict[
                                        "allele2_h1"
                                    ] = same_mle_hap_depth_list[0]
                                    mle_all_phasing_hap_count_dict[
                                        "allele2_h2"
                                    ] = same_mle_hap_depth_list[1]
                                update_minor_allele_phasing_index = (
                                    allele2_index
                                )
                                update_minor_allele_phasing_index_source = "NA"
                                update_minor_allele_phasing_index_germ = "NA"
                            else:
                                if (
                                    len(
                                        set([allele1_index, allele2_index])
                                        & set([allele3_index, allele4_index])
                                    )
                                    == 2
                                ):
                                    obs_all_phasing_hap_count_dict[
                                        "allele1_h1"
                                    ] = obs_hap_depth_list[0]
                                    obs_all_phasing_hap_count_dict[
                                        "allele1_h2"
                                    ] = obs_hap_depth_list[1]
                                    obs_all_phasing_hap_count_dict[
                                        "allele2_h1"
                                    ] = obs_hap_depth_list[2]
                                    obs_all_phasing_hap_count_dict[
                                        "allele2_h2"
                                    ] = obs_hap_depth_list[3]
                                    mle_all_phasing_hap_count_dict[
                                        "allele1_h1"
                                    ] = same_mle_hap_depth_list[0]
                                    mle_all_phasing_hap_count_dict[
                                        "allele1_h2"
                                    ] = same_mle_hap_depth_list[1]
                                    mle_all_phasing_hap_count_dict[
                                        "allele2_h1"
                                    ] = same_mle_hap_depth_list[2]
                                    mle_all_phasing_hap_count_dict[
                                        "allele2_h2"
                                    ] = same_mle_hap_depth_list[3]
                                else:
                                    obs_all_phasing_hap_count_dict[
                                        "allele1_h1"
                                    ] = obs_hap_depth_list[0]
                                    obs_all_phasing_hap_count_dict[
                                        "allele1_h2"
                                    ] = obs_hap_depth_list[1]
                                    obs_all_phasing_hap_count_dict[
                                        "allele2_h1"
                                    ] = obs_hap_depth_list[2]
                                    obs_all_phasing_hap_count_dict[
                                        "allele2_h2"
                                    ] = obs_hap_depth_list[3]
                                    obs_all_phasing_hap_count_dict[
                                        "allele3_h1"
                                    ] = obs_hap_depth_list[4]
                                    obs_all_phasing_hap_count_dict[
                                        "allele3_h2"
                                    ] = obs_hap_depth_list[5]
                                    mle_all_phasing_hap_count_dict[
                                        "allele1_h1"
                                    ] = same_mle_hap_depth_list[0]
                                    mle_all_phasing_hap_count_dict[
                                        "allele1_h2"
                                    ] = same_mle_hap_depth_list[1]
                                    mle_all_phasing_hap_count_dict[
                                        "allele2_h1"
                                    ] = same_mle_hap_depth_list[2]
                                    mle_all_phasing_hap_count_dict[
                                        "allele2_h2"
                                    ] = same_mle_hap_depth_list[3]
                                    mle_all_phasing_hap_count_dict[
                                        "allele3_h1"
                                    ] = same_mle_hap_depth_list[4]
                                    mle_all_phasing_hap_count_dict[
                                        "allele3_h2"
                                    ] = same_mle_hap_depth_list[5]
                                # 低级 BUG ，修复 Het2Het Hap 数的时候存在问题
                                # HACK: 这边包括 germhet 和 hethet, germ het 可能需要进行 DEBUG
                                update_minor_allele_phasing_index = (
                                    allele4_index
                                )
                                update_minor_allele_phasing_index_source = (
                                    allele2_index
                                )
                                update_minor_allele_phasing_index_germ = (
                                    allele1_index
                                )
                        if update_minor_allele_phasing_index == "NA":
                            mle_mutant_discordant_rate = "NA"
                            update_minor_allele_phasing_index_mle_hap_dis_depth = (
                                "NA"
                            )
                        else:
                            if (
                                update_minor_allele_phasing_index
                                in mosaic_gts_alleles_index
                            ):
                                update_minor_allele_phasing_index_mle_hap_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index
                                )
                                update_minor_allele_phasing_index_mle_hap_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_mle_hap_index
                                ]
                                update_minor_allele_phasing_index_mle_hap_dis_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index
                                    + allele_number
                                )
                                update_minor_allele_phasing_index_mle_hap_dis_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_mle_hap_dis_index
                                ]
                            else:
                                update_minor_allele_phasing_index_mle_hap_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index
                                    + allele_number
                                )
                                update_minor_allele_phasing_index_mle_hap_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_mle_hap_index
                                ]
                                update_minor_allele_phasing_index_mle_hap_dis_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index
                                )
                                update_minor_allele_phasing_index_mle_hap_dis_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_mle_hap_dis_index
                                ]
                            if (
                                update_minor_allele_phasing_index_mle_hap_depth
                                + update_minor_allele_phasing_index_mle_hap_dis_depth
                            ) == 0:
                                mle_mutant_discordant_rate = "NA"
                            else:
                                mle_mutant_discordant_rate = (
                                    update_minor_allele_phasing_index_mle_hap_dis_depth
                                    / (
                                        update_minor_allele_phasing_index_mle_hap_depth
                                        + update_minor_allele_phasing_index_mle_hap_dis_depth
                                    )
                                )  # XXX: only assignable as all
                        if update_minor_allele_phasing_index_source != "NA":
                            if (
                                update_minor_allele_phasing_index_source
                                in mosaic_gts_alleles_index
                            ):
                                update_minor_allele_phasing_index_source_mle_hap_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index_source
                                )
                                update_minor_allele_phasing_index_source_mle_hap_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_source_mle_hap_index
                                ]
                                update_minor_allele_phasing_index_source_mle_hap_dis_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index_source
                                    + allele_number
                                )
                                update_minor_allele_phasing_index_source_mle_hap_dis_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_source_mle_hap_dis_index
                                ]
                                # update_minor_allele_phasing_index_source_mle_mutant_discordant_rate = update_minor_allele_phasing_index_source_mle_hap_dis_depth / (update_minor_allele_phasing_index_source_mle_hap_depth + update_minor_allele_phasing_index_source_mle_hap_dis_depth)  # XXX: only assignable as all
                            else:
                                update_minor_allele_phasing_index_source_mle_hap_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index_source
                                    + allele_number
                                )
                                update_minor_allele_phasing_index_source_mle_hap_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_source_mle_hap_index
                                ]
                                update_minor_allele_phasing_index_source_mle_hap_dis_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index_source
                                )
                                update_minor_allele_phasing_index_source_mle_hap_dis_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_source_mle_hap_dis_index
                                ]
                                # update_minor_allele_phasing_index_source_mle_mutant_discordant_rate = update_minor_allele_phasing_index_source_mle_hap_dis_depth / (update_minor_allele_phasing_index_source_mle_hap_depth + update_minor_allele_phasing_index_source_mle_hap_dis_depth)  # XXX: only assignable as all
                            if (
                                update_minor_allele_phasing_index_germ
                                in mosaic_gts_alleles_index
                            ):
                                update_minor_allele_phasing_index_germ_mle_hap_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index_germ
                                )
                                update_minor_allele_phasing_index_germ_mle_hap_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_germ_mle_hap_index
                                ]
                                update_minor_allele_phasing_index_germ_mle_hap_dis_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index_germ
                                    + allele_number
                                )
                                update_minor_allele_phasing_index_germ_mle_hap_dis_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_germ_mle_hap_dis_index
                                ]
                                # update_minor_allele_phasing_index_germ_mle_mutant_discordant_rate = update_minor_allele_phasing_index_germ_mle_hap_dis_depth / (update_minor_allele_phasing_index_germ_mle_hap_depth + update_minor_allele_phasing_index_germ_mle_hap_dis_depth)  # XXX: only assignable as all
                            else:
                                update_minor_allele_phasing_index_germ_mle_hap_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index_germ
                                    + allele_number
                                )
                                update_minor_allele_phasing_index_germ_mle_hap_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_germ_mle_hap_index
                                ]
                                update_minor_allele_phasing_index_germ_mle_hap_dis_index = all_expected_phase_haps.index(
                                    update_minor_allele_phasing_index_germ
                                )
                                update_minor_allele_phasing_index_germ_mle_hap_dis_depth = mle_hap_depth_list[
                                    update_minor_allele_phasing_index_germ_mle_hap_dis_index
                                ]
                                # update_minor_allele_phasing_index_germ_mle_mutant_discordant_rate = update_minor_allele_phasing_index_germ_mle_hap_dis_depth / (update_minor_allele_phasing_index_germ_mle_hap_depth + update_minor_allele_phasing_index_germ_mle_hap_dis_depth)  # XXX: only assignable as all
                            if (
                                update_minor_allele_phasing_index_source_mle_hap_depth
                                + update_minor_allele_phasing_index_source_mle_hap_dis_depth
                            ) == 0:
                                update_minor_allele_phasing_index_source_mle_mutant_discordant_rate = (
                                    "NA"
                                )
                            else:
                                update_minor_allele_phasing_index_source_mle_mutant_discordant_rate = (
                                    update_minor_allele_phasing_index_source_mle_hap_dis_depth
                                    / (
                                        update_minor_allele_phasing_index_source_mle_hap_depth
                                        + update_minor_allele_phasing_index_source_mle_hap_dis_depth
                                    )
                                )  # XXX: only assignable as all
                            if (
                                update_minor_allele_phasing_index_germ_mle_hap_depth
                                + update_minor_allele_phasing_index_germ_mle_hap_dis_depth
                            ) == 0:
                                update_minor_allele_phasing_index_germ_mle_mutant_discordant_rate = (
                                    "NA"
                                )
                            else:
                                update_minor_allele_phasing_index_germ_mle_mutant_discordant_rate = (
                                    update_minor_allele_phasing_index_germ_mle_hap_dis_depth
                                    / (
                                        update_minor_allele_phasing_index_germ_mle_hap_depth
                                        + update_minor_allele_phasing_index_germ_mle_hap_dis_depth
                                    )
                                )  # XXX: only assignable as all
                        else:
                            update_minor_allele_phasing_index_source_mle_hap_depth = (
                                "NA"
                            )
                            update_minor_allele_phasing_index_source_mle_hap_dis_depth = (
                                "NA"
                            )
                            update_minor_allele_phasing_index_source_mle_mutant_discordant_rate = (
                                "NA"
                            )
                            update_minor_allele_phasing_index_germ_mle_hap_depth = (
                                "NA"
                            )
                            update_minor_allele_phasing_index_germ_mle_hap_dis_depth = (
                                "NA"
                            )
                            update_minor_allele_phasing_index_germ_mle_mutant_discordant_rate = (
                                "NA"
                            )
                        samples_dict[bamname][
                            "PMMDD"
                        ] = update_minor_allele_phasing_index_mle_hap_dis_depth
                        samples_dict[bamname][
                            "PMMDR"
                        ] = mle_mutant_discordant_rate
                        samples_dict[bamname][
                            "PSMMDD"
                        ] = update_minor_allele_phasing_index_source_mle_hap_dis_depth
                        samples_dict[bamname][
                            "PSMMDR"
                        ] = update_minor_allele_phasing_index_source_mle_mutant_discordant_rate
                        samples_dict[bamname][
                            "PGMMDD"
                        ] = update_minor_allele_phasing_index_germ_mle_hap_dis_depth
                        samples_dict[bamname][
                            "PGMMDR"
                        ] = update_minor_allele_phasing_index_germ_mle_mutant_discordant_rate
                        samples_dict[bamname]["PPP"] = phase_proportion
                        samples_dict[bamname][
                            "PODD"
                        ] = observed_discordant_reads_number
                        samples_dict[bamname][
                            "PODR"
                        ] = observed_discordant_reads_rate
                        samples_dict[bamname][
                            "PODH"
                        ] = observed_discordant_hap_number
                        samples_dict[bamname]["POAH"] = len(
                            phasing_alleles_depth_dict
                        )
                        samples_dict[bamname][
                            "PODHR"
                        ] = observed_discordant_hap_rate
                        samples_dict[bamname][
                            "PMAP"
                        ] = phased_mosaic_allele_proportion
                        samples_dict[bamname]["PMLEAH"] = mle_all_hap_number
                        samples_dict[bamname]["PMLEEH"] = mle_hap_number
                        if mle_all_hap_number > 0:
                            samples_dict[bamname]["PMLEEHR"] = (
                                mle_hap_number / mle_all_hap_number
                            )
                        else:
                            samples_dict[bamname]["PMLEEHR"] = "NA"
                        samples_dict[bamname]["PMLEDD"] = discordant_mle_reads
                        samples_dict[bamname]["PMLEDR"] = discordant_mle_rate
                        samples_dict[bamname]["PFAH"] = filter_hap_number
                        EAD = ""
                        EAF = ""
                        DOAD = ""
                        EAD_allele_list = []
                        DOAD_allele_list = []
                        for index, mosaic_allele_pop_out_index in enumerate(
                            [
                                best_phasing_mosaic_gts_alleles[0][0],
                                best_phasing_mosaic_gts_alleles[0][1],
                                best_phasing_mosaic_gts_alleles[1][0],
                                best_phasing_mosaic_gts_alleles[1][1],
                            ]
                        ):
                            if index % 2 == 0:
                                nsnp = nearby_snp[0]
                                dsnp = nearby_snp[1]
                            else:
                                nsnp = nearby_snp[1]
                                dsnp = nearby_snp[0]
                            mosaic_allele = (
                                all_alleles_list_for_gt_output[
                                    mosaic_allele_pop_out_index
                                ],
                                nsnp,
                            )
                            EAD += (
                                str(
                                    phasing_alleles_depth_dict.get(
                                        mosaic_allele, 0
                                    )
                                )
                                + ","
                            )
                            EAF += (
                                str(
                                    phasing_alleles_depth_dict.get(
                                        mosaic_allele, 0
                                    )
                                    / phase_calling.reads_number
                                )
                                + ","
                            )
                            dis_mosaic_allele = (
                                all_alleles_list_for_gt_output[
                                    mosaic_allele_pop_out_index
                                ],
                                dsnp,
                            )
                            DOAD += (
                                str(
                                    phasing_alleles_depth_dict.get(
                                        dis_mosaic_allele, 0
                                    )
                                )
                                + ","
                            )
                            EAD_allele_list.append(mosaic_allele)
                            DOAD_allele_list.append(dis_mosaic_allele)
                        # for adp in hap_depth_list[all_mosaic_alleles]:
                        #     EAD = str(adp)+","
                        #     EAF = str(adp/unphase_calling.reads_number)+","
                        EAD = EAD[:-1]
                        EAF = EAF[:-1]
                        DOAD = EAD + "," + DOAD[:-1]
                        DOADL = "1,1,1,1"
                        for dis_allele in DOAD_allele_list:
                            if dis_allele in EAD_allele_list:
                                DOADL += ",1"
                            else:
                                DOADL += ",0"
                        samples_dict[bamname]["PEAD"] = EAD
                        samples_dict[bamname]["PEAF"] = EAF
                        samples_dict[bamname]["PDOAD"] = DOAD
                        samples_dict[bamname]["PDOADL"] = DOADL
                        # discordant rate vs stutter error rate 的 LRT 的 discordant rate 使用观察到的 discordant rate 而非 MLE 的 discordant rate
                        # 因为 mle 的已经将 stutter errors 考虑在内了，这样再比较不太好，观察的没考虑 stutter errors，比较起来比较合理哈
                        mosaic_observed_log_likelihood_using_discordant = myutils.cal_mosaic_likelihood_using_discordant_rate(
                            observed_discordant_reads_rate,
                            best_phasing_mosaic_gts_alleles_str_hap,
                            nearby_snp,
                            update_mosaic_fraction,
                            ALLELE_BALANCE,
                            phase_calling.reads_list,
                            phase_calling.reads_accuracy_list,
                        )
                        (
                            mapping_vs_stutter_observed_LR,
                            mapping_vs_stutter_observed_LR_pvalue,
                        ) = mutation_filtering.cal_stutter_errors_discordant_rate_LRT_for_phasing_confidence(
                            mosaic_observed_log_likelihood_using_discordant,
                            MaxPhasedMosaicLogLik,
                            LRT_FREEDOM[seg_locus.motif_length],
                        )
                        if (
                            observed_discordant_reads_rate == 0
                            or observed_discordant_reads_rate == 1
                        ):
                            # mosaic_observed_log_likelihood_using_discordant_only = np.log(1)
                            mapping_vs_stutter_observed_LR_simple = "NA"
                            mapping_vs_stutter_observed_LR_pvalue_simple = "NA"
                        else:
                            mosaic_observed_log_likelihood_using_discordant_only = (
                                observed_discordant_reads_number
                                * np.log(observed_discordant_reads_rate)
                            ) + (
                                (
                                    len(phasing_reads_list)
                                    - observed_discordant_reads_number
                                )
                                * np.log(1 - observed_discordant_reads_rate)
                            )
                            (
                                mapping_vs_stutter_observed_LR_simple,
                                mapping_vs_stutter_observed_LR_pvalue_simple,
                            ) = mutation_filtering.cal_stutter_errors_discordant_rate_LRT_for_phasing_confidence(
                                mosaic_observed_log_likelihood_using_discordant_only,
                                MaxPhasedMosaicLogLik,
                                LRT_FREEDOM[seg_locus.motif_length],
                            )
                        samples_dict[bamname][
                            "MAPLR"
                        ] = mapping_vs_stutter_observed_LR
                        samples_dict[bamname][
                            "MAPLRTP"
                        ] = mapping_vs_stutter_observed_LR_pvalue
                        samples_dict[bamname][
                            "MAPLRS"
                        ] = mapping_vs_stutter_observed_LR_simple
                        samples_dict[bamname][
                            "MAPLRTPS"
                        ] = mapping_vs_stutter_observed_LR_pvalue_simple
                        if (
                            update_mosaic_gts_alleles[0][0]
                            == update_mosaic_gts_alleles[0][1]
                        ):
                            if (
                                update_mosaic_gts_alleles[1][0]
                                == update_mosaic_gts_alleles[1][1]
                            ):
                                update_minor_allele_popout_index = "NA"
                                update_minor_allele_popout_index_source = "NA"
                                update_minor_allele_popout_index_germ = "NA"
                            else:
                                update_minor_allele_popout_index = (
                                    update_mosaic_gts_alleles[1][1]
                                )
                                update_minor_allele_popout_index_source = "NA"
                                update_minor_allele_popout_index_germ = "NA"
                        else:
                            if (
                                update_mosaic_gts_alleles[1][0]
                                == update_mosaic_gts_alleles[1][1]
                            ):
                                update_minor_allele_popout_index = (
                                    update_mosaic_gts_alleles[0][1]
                                )
                                update_minor_allele_popout_index_source = "NA"
                                update_minor_allele_popout_index_germ = "NA"
                            else:
                                update_minor_allele_popout_index = (
                                    update_mosaic_gts_alleles[1][1]
                                )
                                update_minor_allele_popout_index_source = (
                                    update_mosaic_gts_alleles[0][1]
                                )
                                update_minor_allele_popout_index_germ = (
                                    update_mosaic_gts_alleles[0][0]
                                )
                        if update_minor_allele_popout_index != "NA":
                            update_minor_allele_popout_str_allele_hap = (
                                all_alleles_list_for_gt_output[
                                    update_minor_allele_popout_index
                                ]
                            )
                            phase_minor_allele_str_allele_depth = (
                                phasing_alleles_depth_dict2.get(
                                    update_minor_allele_popout_str_allele_hap,
                                    0,
                                )
                            )
                        else:
                            update_minor_allele_popout_str_allele_hap = "NA"
                            phase_minor_allele_str_allele_depth = "NA"
                        if update_minor_allele_popout_index_source != "NA":
                            update_minor_allele_popout_str_allele_hap_source = all_alleles_list_for_gt_output[
                                update_minor_allele_popout_index_source
                            ]
                            phase_minor_allele_str_allele_depth_source = phasing_alleles_depth_dict2.get(
                                update_minor_allele_popout_str_allele_hap_source,
                                0,
                            )
                            update_minor_allele_popout_str_allele_hap_germ = (
                                all_alleles_list_for_gt_output[
                                    update_minor_allele_popout_index_germ
                                ]
                            )
                            phase_minor_allele_str_allele_depth_germ = phasing_alleles_depth_dict2.get(
                                update_minor_allele_popout_str_allele_hap_germ,
                                0,
                            )
                        else:
                            update_minor_allele_popout_str_allele_hap_source = (
                                "NA"
                            )
                            phase_minor_allele_str_allele_depth_source = "NA"
                            update_minor_allele_popout_str_allele_hap_germ = (
                                "NA"
                            )
                            phase_minor_allele_str_allele_depth_germ = "NA"
                        cordant_minor_allele_depth = "NA"
                        cordant_minor_allele_depth_source = "NA"
                        cordant_minor_allele_depth_germ = "NA"
                        for cordant_hap in set(
                            [
                                best_phasing_mosaic_gts_alleles_hap[0][0],
                                best_phasing_mosaic_gts_alleles_hap[0][1],
                                best_phasing_mosaic_gts_alleles_hap[1][0],
                                best_phasing_mosaic_gts_alleles_hap[1][1],
                            ]
                        ):
                            # if DEBUG:
                            #     print(cordant_hap[0])
                            #     print(update_minor_allele_popout_str_allele_hap)
                            if (
                                cordant_hap[0]
                                == update_minor_allele_popout_str_allele_hap
                            ):
                                cordant_minor_allele_depth = (
                                    phasing_alleles_depth_dict.get(
                                        cordant_hap, 0
                                    )
                                )
                            elif (
                                cordant_hap[0]
                                == update_minor_allele_popout_str_allele_hap_source
                            ):
                                cordant_minor_allele_depth_source = (
                                    phasing_alleles_depth_dict.get(
                                        cordant_hap, 0
                                    )
                                )
                            elif (
                                cordant_hap[0]
                                == update_minor_allele_popout_str_allele_hap_germ
                            ):
                                cordant_minor_allele_depth_germ = (
                                    phasing_alleles_depth_dict.get(
                                        cordant_hap, 0
                                    )
                                )
                            else:
                                pass
                        if cordant_minor_allele_depth != "NA":
                            if phase_minor_allele_str_allele_depth > 0:
                                observed_mutant_discordant_rate = (
                                    phase_minor_allele_str_allele_depth
                                    - cordant_minor_allele_depth
                                ) / phase_minor_allele_str_allele_depth
                            else:
                                observed_mutant_discordant_rate = "NA"
                            samples_dict[bamname]["POMDD"] = (
                                phase_minor_allele_str_allele_depth
                                - cordant_minor_allele_depth
                            )
                        else:
                            observed_mutant_discordant_rate = "NA"
                            samples_dict[bamname]["POMDD"] = "NA"
                        if cordant_minor_allele_depth_source != "NA":
                            if phase_minor_allele_str_allele_depth_source > 0:
                                observed_source_discordant_rate = (
                                    (
                                        phase_minor_allele_str_allele_depth_source
                                        - cordant_minor_allele_depth_source
                                    )
                                    / phase_minor_allele_str_allele_depth_source
                                )
                            else:
                                observed_source_discordant_rate = "NA"
                            samples_dict[bamname]["PSOMDD"] = (
                                phase_minor_allele_str_allele_depth_source
                                - cordant_minor_allele_depth_source
                            )
                        else:
                            observed_source_discordant_rate = "NA"
                            samples_dict[bamname]["PSOMDD"] = "NA"
                        if cordant_minor_allele_depth_germ != "NA":
                            if phase_minor_allele_str_allele_depth_germ > 0:
                                observed_germ_discordant_rate = (
                                    (
                                        phase_minor_allele_str_allele_depth_germ
                                        - cordant_minor_allele_depth_germ
                                    )
                                    / phase_minor_allele_str_allele_depth_germ
                                )
                            else:
                                observed_germ_discordant_rate = "NA"
                            samples_dict[bamname]["PGOMDD"] = (
                                phase_minor_allele_str_allele_depth_germ
                                - cordant_minor_allele_depth_germ
                            )
                        else:
                            observed_germ_discordant_rate = "NA"
                            samples_dict[bamname]["PGOMDD"] = "NA"

                        samples_dict[bamname][
                            "POMDR"
                        ] = observed_mutant_discordant_rate
                        samples_dict[bamname][
                            "PSOMDR"
                        ] = observed_source_discordant_rate
                        samples_dict[bamname][
                            "PGOMDR"
                        ] = observed_germ_discordant_rate

                        samples_dict[bamname]["PP"] = phase_posterior
                        samples_dict[bamname]["PGT"] = (
                            str(best_phasing_mosaic_gts_alleles[0][0])
                            + "|"
                            + str(best_phasing_mosaic_gts_alleles[0][1])
                        )
                        samples_dict[bamname]["PMGT"] = (
                            str(best_phasing_mosaic_gts_alleles[1][0])
                            + "|"
                            + str(best_phasing_mosaic_gts_alleles[1][1])
                        )
                        if unphase_mosaic_type == "mosaic":
                            if phase_posterior > PHASE_POSTERIOR_CUTOFF:
                                phase_mosaic_type = "mosaic"
                            else:
                                phase_mosaic_type = "artifact"
                        else:
                            phase_mosaic_type = unphase_mosaic_type

                        samples_dict[bamname][
                            "PVARIANTTYPE"
                        ] = phase_mosaic_type
                        samples_dict[bamname][
                            "PDP"
                        ] = phase_calling.reads_number
                        PAAD = ""
                        for allele in all_alleles_list_for_gt_output:
                            for snp_seq in nearby_snp:
                                allele_depth = phasing_alleles_depth_dict.get(
                                    (allele, snp_seq), 0
                                )
                                PAAD = PAAD + str(allele_depth) + ","
                        PAAD = PAAD[:-1]
                        samples_dict[bamname]["PAAD"] = PAAD
                        if phase_mosaic_vs_germline_p == "NA":
                            filter1 = 0
                        else:
                            filter1 = int(
                                phase_mosaic_vs_germline_p < LRT_PVALUE_CUTOFF
                            )
                        filter2 = int(phase_posterior > PHASE_POSTERIOR_CUTOFF)
                        filter3 = int(
                            mapping_vs_stutter_observed_LR_pvalue
                            < LRT_PVALUE_CUTOFF
                        )
                        filter4 = int(mle_hap_number == 3)
                        confidence = filter1 + filter2 + filter3 + filter4
                        samples_dict[bamname][
                            "PFILTER"
                        ] = f"{confidence},{filter1},{filter2},{filter3},{filter4}"

                        observed_mosaic_gt_allele_depth_list = [
                            int(i) for i in EAD.split(",")
                        ]
                        if (
                            mle_all_hap_number == 3
                        ):  # assign errors noise and other discordant hap noise
                            if (
                                np.sum(
                                    np.array(mle_mosaic_gt_allele_depth_list)
                                    > 0
                                )
                                == 4
                            ) or (
                                np.sum(
                                    np.array(
                                        observed_mosaic_gt_allele_depth_list
                                    )
                                    > 0
                                )
                                == 4
                            ):  # HACK: and change to or, allow no representative alleles
                                refine_hap_num = 3
                                if (
                                    np.sum(
                                        np.array(
                                            mle_mosaic_gt_allele_depth_list
                                        )
                                        >= HIGH_QUALITY_PHASE_DEPTH_CUTOFF
                                    )
                                    == 4
                                ) or (
                                    np.sum(
                                        np.array(
                                            observed_mosaic_gt_allele_depth_list
                                        )
                                        >= HIGH_QUALITY_PHASE_DEPTH_CUTOFF
                                    )
                                    == 4
                                ):  # HACK: and change to or, allow no representative alleles
                                    high_quality_hap_num = 3
                                else:
                                    high_quality_hap_num = 2
                            else:
                                # Don't refine to het2het because source allele and germ allele may be perversions
                                if REFINE_HAP_COUNT_HETHET:
                                    refine_hap_num = 2
                                    high_quality_hap_num = 2
                                else:
                                    if (
                                        len(
                                            set(
                                                [
                                                    str(
                                                        MaxMosaicGT_for_pop_output[
                                                            0
                                                        ][
                                                            0
                                                        ]
                                                    ),
                                                    str(
                                                        MaxMosaicGT_for_pop_output[
                                                            0
                                                        ][
                                                            1
                                                        ]
                                                    ),
                                                    str(
                                                        MaxMosaicGT_for_pop_output[
                                                            1
                                                        ][
                                                            0
                                                        ]
                                                    ),
                                                    str(
                                                        MaxMosaicGT_for_pop_output[
                                                            1
                                                        ][
                                                            1
                                                        ]
                                                    ),
                                                ]
                                            )
                                        )
                                        == 3
                                    ):
                                        refine_hap_num = 3
                                        if (
                                            np.sum(
                                                np.array(mle_hap_depth_list)
                                                >= HIGH_QUALITY_PHASE_DEPTH_CUTOFF
                                            )
                                            == 3
                                        ) or (
                                            np.sum(
                                                np.array(obs_hap_depth_list)
                                                >= HIGH_QUALITY_PHASE_DEPTH_CUTOFF
                                            )
                                            == 3
                                        ):
                                            high_quality_hap_num = 3
                                        else:
                                            high_quality_hap_num = np.sum(
                                                np.array(mle_hap_depth_list)
                                                >= HIGH_QUALITY_PHASE_DEPTH_CUTOFF
                                            )
                                    else:
                                        refine_hap_num = 2
                                        high_quality_hap_num = 2
                        else:
                            refine_hap_num = mle_all_hap_number
                            high_quality_hap_num = mle_all_hap_number
                        samples_dict[bamname]["PRHN"] = refine_hap_num
                        samples_dict[bamname]["HPRHN"] = high_quality_hap_num
                        # REFINE HAP Count According MosaicFocast and Spatial
                        mosaic_gt_allele_num = len(
                            set(
                                [
                                    allele1_index,
                                    allele2_index,
                                    allele3_index,
                                    allele4_index,
                                ]
                            )
                        )
                        if mosaic_gt_allele_num == 2:
                            (
                                obs_phase_state,
                                obs_hap_count,
                                obs_hap_state,
                                obs_mut_order,
                            ) = myutils.ym_hap_count_homhet_gpt(
                                obs_all_phasing_hap_count_dict,
                                scale_factor=PHASE_COUNT_SCALE_FACTOR,
                            )
                            (
                                mle_phase_state,
                                mle_hap_count,
                                mle_hap_state,
                                mle_mut_order,
                            ) = myutils.ym_hap_count_homhet_gpt(
                                mle_all_phasing_hap_count_dict,
                                scale_factor=PHASE_COUNT_SCALE_FACTOR,
                            )
                            obs_depth_string = (
                                str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele1_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele1_h2", 0
                                    )
                                )
                                + "_"
                                + str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele2_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele2_h2", 0
                                    )
                                )
                            )
                            mle_depth_string = (
                                str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele1_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele1_h2", 0
                                    )
                                )
                                + "_"
                                + str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele2_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele2_h2", 0
                                    )
                                )
                            )
                        elif mosaic_gt_allele_num == 3:
                            (
                                obs_phase_state,
                                obs_hap_count,
                                obs_hap_state,
                                obs_mut_order,
                            ) = myutils.ym_hap_count_hethet(
                                obs_all_phasing_hap_count_dict,
                                scale_factor=PHASE_COUNT_SCALE_FACTOR,
                            )
                            (
                                mle_phase_state,
                                mle_hap_count,
                                mle_hap_state,
                                mle_mut_order,
                            ) = myutils.ym_hap_count_hethet(
                                mle_all_phasing_hap_count_dict,
                                scale_factor=PHASE_COUNT_SCALE_FACTOR,
                            )
                            obs_depth_string = (
                                str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele1_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele1_h2", 0
                                    )
                                )
                                + "_"
                                + str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele2_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele2_h2", 0
                                    )
                                )
                                + "_"
                                + str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele3_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    obs_all_phasing_hap_count_dict.get(
                                        "allele3_h2", 0
                                    )
                                )
                            )
                            mle_depth_string = (
                                str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele1_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele1_h2", 0
                                    )
                                )
                                + "_"
                                + str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele2_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele2_h2", 0
                                    )
                                )
                                + "_"
                                + str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele3_h1", 0
                                    )
                                )
                                + "_"
                                + str(
                                    mle_all_phasing_hap_count_dict.get(
                                        "allele3_h2", 0
                                    )
                                )
                            )
                        else:
                            (
                                obs_phase_state,
                                obs_hap_count,
                                obs_hap_state,
                                obs_mut_order,
                            ) = ("unphase", "NA", "hap1", "NA")
                            (
                                mle_phase_state,
                                mle_hap_count,
                                mle_hap_state,
                                mle_mut_order,
                            ) = ("unphase", "NA", "hap1", "NA")
                            obs_depth_string = ""
                            mle_depth_string = ""
                        samples_dict[bamname][
                            "obs_phase_state"
                        ] = obs_phase_state
                        samples_dict[bamname]["obs_hap_state"] = obs_hap_state
                        samples_dict[bamname]["obs_hap_count"] = obs_hap_count
                        samples_dict[bamname]["obs_mut_order"] = obs_mut_order
                        samples_dict[bamname][
                            "obs_depth_string"
                        ] = obs_depth_string
                        samples_dict[bamname][
                            "mle_phase_state"
                        ] = mle_phase_state
                        samples_dict[bamname]["mle_hap_state"] = mle_hap_state
                        samples_dict[bamname]["mle_hap_count"] = mle_hap_count
                        samples_dict[bamname]["mle_mut_order"] = mle_mut_order
                        samples_dict[bamname][
                            "mle_depth_string"
                        ] = mle_depth_string
                        if (
                            samples_dict[bamname]["PGT"].split("|")[0]
                            == samples_dict[bamname]["PMGT"].split("|")[0]
                        ):  # 第一个 allele 相等
                            if samples_dict[bamname]["PGT"].split(
                                "|"
                            ) == samples_dict[bamname]["PMGT"].split(
                                "|"
                            ):  # germline
                                if (
                                    sum(observed_mosaic_gt_allele_depth_list)
                                    - observed_mosaic_gt_allele_depth_list[0]
                                    - observed_mosaic_gt_allele_depth_list[1]
                                ) > 0:
                                    phasing_hap_observe_balance = observed_mosaic_gt_allele_depth_list[
                                        0
                                    ] / (
                                        sum(
                                            observed_mosaic_gt_allele_depth_list
                                        )
                                        - observed_mosaic_gt_allele_depth_list[
                                            0
                                        ]
                                        - observed_mosaic_gt_allele_depth_list[
                                            1
                                        ]
                                    )
                                else:
                                    phasing_hap_observe_balance = "NA"
                                if (
                                    sum(mle_mosaic_gt_allele_depth_list)
                                    - mle_mosaic_gt_allele_depth_list[0]
                                    - mle_mosaic_gt_allele_depth_list[1]
                                ) > 0:
                                    phasing_hap_MLE_balance = (
                                        mle_mosaic_gt_allele_depth_list[0]
                                        / (
                                            sum(
                                                mle_mosaic_gt_allele_depth_list
                                            )
                                            - mle_mosaic_gt_allele_depth_list[
                                                0
                                            ]
                                            - mle_mosaic_gt_allele_depth_list[
                                                1
                                            ]
                                        )
                                    )
                                else:
                                    phasing_hap_MLE_balance = "NA"
                            else:  # mosaic
                                if (
                                    sum(observed_mosaic_gt_allele_depth_list)
                                    - observed_mosaic_gt_allele_depth_list[0]
                                ) > 0:
                                    phasing_hap_observe_balance = observed_mosaic_gt_allele_depth_list[
                                        0
                                    ] / (
                                        sum(
                                            observed_mosaic_gt_allele_depth_list
                                        )
                                        - observed_mosaic_gt_allele_depth_list[
                                            0
                                        ]
                                    )
                                else:
                                    phasing_hap_observe_balance = "NA"
                                if (
                                    sum(mle_mosaic_gt_allele_depth_list)
                                    - mle_mosaic_gt_allele_depth_list[0]
                                ) > 0:
                                    phasing_hap_MLE_balance = (
                                        mle_mosaic_gt_allele_depth_list[0]
                                        / (
                                            sum(
                                                mle_mosaic_gt_allele_depth_list
                                            )
                                            - mle_mosaic_gt_allele_depth_list[
                                                0
                                            ]
                                        )
                                    )
                                else:
                                    phasing_hap_MLE_balance = "NA"
                            germ_hsnp_af = read_snp_dict.get(
                                nearby_snp[0], 0
                            ) / sum(read_snp_dict.values())
                            mut_hsnp_af = read_snp_dict.get(
                                nearby_snp[1], 0
                            ) / sum(read_snp_dict.values())
                            hsnp3_af = 1 - germ_hsnp_af - mut_hsnp_af
                            if mle_mosaic_gt_allele_depth_list[0] > 0:
                                germ_dis = (
                                    observed_mosaic_gt_allele_depth_list[0]
                                    / mle_mosaic_gt_allele_depth_list[0]
                                )
                            else:
                                germ_dis = "NA"
                            if (
                                samples_dict[bamname]["PGT"].split("|")[0]
                            ) == (samples_dict[bamname]["PGT"].split("|")[1]):
                                if mle_mosaic_gt_allele_depth_list[1] > 0:
                                    source_dis = (
                                        observed_mosaic_gt_allele_depth_list[1]
                                        / mle_mosaic_gt_allele_depth_list[1]
                                    )
                                else:
                                    source_dis = "NA"
                                if mle_mosaic_gt_allele_depth_list[3] > 0:
                                    mut_dis = (
                                        observed_mosaic_gt_allele_depth_list[3]
                                        / mle_mosaic_gt_allele_depth_list[3]
                                    )
                                else:
                                    mut_dis = "NA"
                            else:
                                if (
                                    samples_dict[bamname]["PMGT"].split("|")[0]
                                ) == (
                                    samples_dict[bamname]["PMGT"].split("|")[1]
                                ):
                                    if mle_mosaic_gt_allele_depth_list[3] > 0:
                                        source_dis = (
                                            observed_mosaic_gt_allele_depth_list[
                                                3
                                            ]
                                            / mle_mosaic_gt_allele_depth_list[
                                                3
                                            ]
                                        )
                                    else:
                                        source_dis = "NA"
                                    if mle_mosaic_gt_allele_depth_list[1] > 0:
                                        mut_dis = (
                                            observed_mosaic_gt_allele_depth_list[
                                                1
                                            ]
                                            / mle_mosaic_gt_allele_depth_list[
                                                1
                                            ]
                                        )
                                    else:
                                        mut_dis = "NA"
                                else:
                                    source_dis = myutils.check_denominator(
                                        mle_mosaic_gt_allele_depth_list[1],
                                        observed_mosaic_gt_allele_depth_list[
                                            1
                                        ],
                                    )
                                    mut_dis = myutils.check_denominator(
                                        mle_mosaic_gt_allele_depth_list[3],
                                        observed_mosaic_gt_allele_depth_list[
                                            3
                                        ],
                                    )
                        else:  # 第二个 allele 相等
                            if samples_dict[bamname]["PGT"].split(
                                "|"
                            ) == samples_dict[bamname]["PMGT"].split(
                                "|"
                            ):  # germline
                                phasing_hap_observe_balance = myutils.check_denominator(
                                    (
                                        sum(
                                            observed_mosaic_gt_allele_depth_list
                                        )
                                        - observed_mosaic_gt_allele_depth_list[
                                            1
                                        ]
                                        - observed_mosaic_gt_allele_depth_list[
                                            0
                                        ]
                                    ),
                                    observed_mosaic_gt_allele_depth_list[1],
                                )
                                phasing_hap_MLE_balance = (
                                    myutils.check_denominator(
                                        (
                                            sum(
                                                mle_mosaic_gt_allele_depth_list
                                            )
                                            - mle_mosaic_gt_allele_depth_list[
                                                1
                                            ]
                                            - mle_mosaic_gt_allele_depth_list[
                                                0
                                            ]
                                        ),
                                        mle_mosaic_gt_allele_depth_list[1],
                                    )
                                )
                            else:  # mosaic
                                phasing_hap_observe_balance = myutils.check_denominator(
                                    (
                                        sum(
                                            observed_mosaic_gt_allele_depth_list
                                        )
                                        - observed_mosaic_gt_allele_depth_list[
                                            1
                                        ]
                                    ),
                                    observed_mosaic_gt_allele_depth_list[1],
                                )
                                phasing_hap_MLE_balance = (
                                    myutils.check_denominator(
                                        (
                                            sum(
                                                mle_mosaic_gt_allele_depth_list
                                            )
                                            - mle_mosaic_gt_allele_depth_list[
                                                1
                                            ]
                                        ),
                                        mle_mosaic_gt_allele_depth_list[1],
                                    )
                                )
                            germ_hsnp_af = read_snp_dict.get(
                                nearby_snp[1], 0
                            ) / sum(read_snp_dict.values())
                            mut_hsnp_af = read_snp_dict.get(
                                nearby_snp[0], 0
                            ) / sum(read_snp_dict.values())
                            hsnp3_af = 1 - germ_hsnp_af - mut_hsnp_af
                            germ_dis = myutils.check_denominator(
                                mle_mosaic_gt_allele_depth_list[1],
                                observed_mosaic_gt_allele_depth_list[1],
                            )
                            if (
                                samples_dict[bamname]["PGT"].split("|")[0]
                            ) == (samples_dict[bamname]["PGT"].split("|")[1]):
                                source_dis = myutils.check_denominator(
                                    mle_mosaic_gt_allele_depth_list[0],
                                    observed_mosaic_gt_allele_depth_list[0],
                                )
                                mut_dis = myutils.check_denominator(
                                    mle_mosaic_gt_allele_depth_list[2],
                                    observed_mosaic_gt_allele_depth_list[2],
                                )
                            else:
                                if (
                                    samples_dict[bamname]["PMGT"].split("|")[0]
                                ) == (
                                    samples_dict[bamname]["PMGT"].split("|")[1]
                                ):
                                    source_dis = myutils.check_denominator(
                                        mle_mosaic_gt_allele_depth_list[2],
                                        observed_mosaic_gt_allele_depth_list[
                                            2
                                        ],
                                    )
                                    mut_dis = myutils.check_denominator(
                                        mle_mosaic_gt_allele_depth_list[0],
                                        observed_mosaic_gt_allele_depth_list[
                                            0
                                        ],
                                    )
                                else:
                                    source_dis = myutils.check_denominator(
                                        mle_mosaic_gt_allele_depth_list[0],
                                        observed_mosaic_gt_allele_depth_list[
                                            0
                                        ],
                                    )
                                    mut_dis = myutils.check_denominator(
                                        mle_mosaic_gt_allele_depth_list[2],
                                        observed_mosaic_gt_allele_depth_list[
                                            2
                                        ],
                                    )
                        samples_dict[bamname]["PMB"] = phasing_hap_MLE_balance
                        samples_dict[bamname][
                            "POB"
                        ] = phasing_hap_observe_balance
                        samples_dict[bamname]["PGHAF"] = germ_hsnp_af
                        samples_dict[bamname]["PMHAF"] = mut_hsnp_af
                        samples_dict[bamname]["PH3AF"] = hsnp3_af
                        # HACK: PGD PSD PMD 有问题不使用
                        samples_dict[bamname]["PGD"] = germ_dis
                        samples_dict[bamname]["PSD"] = source_dis
                        samples_dict[bamname]["PMD"] = mut_dis
                    else:
                        phasable = True
                        samples_dict[bamname].update(UNPHASE_PHASE_OUTPUT)
                        uncallable_reason = phase_calling.loci_calling
                        samples_dict[bamname]["UPR"] = uncallable_reason
                        samples_dict[bamname]["PHA"] = "NO"
                        # 不输出到日志，但会输出到 vcf 文件
                else:
                    phasable = False
                    samples_dict[bamname].update(UNPHASE_PHASE_OUTPUT)
                    samples_dict[bamname]["PHA"] = "NO"
                    samples_dict[bamname]["UPR"] = "NoNearbySNP"
                    # 不输出到日志，但会输出到 vcf 文件
            else:
                if ANYWAY_LOOK_NSNP:
                    if sample_seq_method == "illumina":
                        if GIAB:
                            (
                                best_var_id,
                                best_af1,
                                reads_snp_seq_dict,
                                reads_snp_seq_accuracy_dict,
                                sample_hSNP_num,
                                sample_hSNP_INDEL_num,
                            ) = select_high_confidence_nearby_snp(
                                records,
                                bamname,
                                spanning_reads_name,
                                spanning_pysam_read,
                                bamfile,
                                spanning_reads_dict,
                                mosaic_allele_list,
                                gn_bed,
                            )
                        else:
                            if (
                                bamname not in vcf_all_samples
                            ):  # HACK: MIX Pseudo-bulk and GIAB need revise this block
                                (
                                    best_var_id,
                                    best_af1,
                                    reads_snp_seq_dict,
                                    reads_snp_seq_accuracy_dict,
                                    sample_hSNP_num,
                                    sample_hSNP_INDEL_num,
                                ) = (False, False, False, False, "NA", "NA")
                            else:
                                (
                                    best_var_id,
                                    best_af1,
                                    reads_snp_seq_dict,
                                    reads_snp_seq_accuracy_dict,
                                    sample_hSNP_num,
                                    sample_hSNP_INDEL_num,
                                ) = select_high_confidence_nearby_snp(
                                    records,
                                    bamname,
                                    spanning_reads_name,
                                    spanning_pysam_read,
                                    bamfile,
                                    spanning_reads_dict,
                                    mosaic_allele_list,
                                    gn_bed,
                                )
                    elif sample_seq_method == "pacbio":
                        (
                            best_var_id,
                            best_af1,
                            reads_snp_seq_dict,
                            reads_snp_seq_accuracy_dict,
                            sample_hSNP_num,
                            sample_hSNP_INDEL_num,
                        ) = pacbio_phasing(
                            HP_tag_dict,
                            pacbio_phasing_allele_list,
                            mosaic_allele_list,
                            PS_tag_dict,
                            chrom,
                        )
                    else:
                        raise ValueError("Unknown sequencing method")
                    samples_dict[bamname]["NHSNPN"] = sample_hSNP_num
                    samples_dict[bamname][
                        "NHSNPINDELN"
                    ] = sample_hSNP_INDEL_num
                    if best_var_id:
                        phasable = True
                        samples_dict[bamname].update(UNPHASE_PHASE_OUTPUT)
                        samples_dict[bamname]["PHA"] = "NO"
                        samples_dict[bamname]["UPR"] = "HasNearbySNP"
                        # 不输出到日志，但会输出到 vcf 文件
                    else:
                        phasable = False
                        samples_dict[bamname].update(UNPHASE_PHASE_OUTPUT)
                        samples_dict[bamname]["PHA"] = "NO"
                        samples_dict[bamname]["UPR"] = "NoNearbySNP"
                        # 不输出到日志，但会输出到 vcf 文件
                else:
                    phasable = "UnKnown"
                    samples_dict[bamname].update(UNPHASE_PHASE_OUTPUT)
                    samples_dict[bamname]["PHA"] = "NO"
                    samples_dict[bamname]["UPR"] = "UnKnownNearbySNP"
                    # 不输出到日志，但会输出到 vcf 文件
        variant_info["ref"] = all_alleles_list_for_gt_output[0]
        alt_string = ""
        alt_bp_diff_list = []
        alt_ru_diff_list = []
        if LIKELIHOOD_MODE == "length-based":
            for alt in all_alleles_list_for_gt_output[1:]:
                alt_string += f"{alt},"
                alt_bp_diff_list.append(alt - seg_locus.ref_allele_length)
                alt_ru_diff_list.append(
                    (alt - seg_locus.ref_allele_length)
                    / seg_locus.motif_length
                )
            alt_string = alt_string[:-1]
        else:
            for alt in all_alleles_list_for_gt_output[1:]:
                alt_str = "".join(list(alt))
                alt_string += f"{alt_str},"
                alt_bp_diff_list.append(
                    len(alt[1]) - seg_locus.ref_allele_length
                )
                alt_ru_diff_list.append(
                    (len(alt[1]) - seg_locus.ref_allele_length)
                    / seg_locus.motif_length
                )
            alt_string = alt_string[:-1]
        variant_info["alt"] = alt_string
        variant_info["filter"] = "PASS"
        variant_info["allele_num"] = len(all_alleles_list_for_gt_output)
        variant_info["altered_repeat_number"] = alt_ru_diff_list
        variant_info["altered_base_pairs"] = alt_bp_diff_list
        variant_info["all_samples_hSNP_INDEL_num"] = all_samples_hSNP_INDEL_num
        logger_mosaic.info(
            f"Finish mosaic calling for locus {seg_locus.STR_id}"
        )
        output_vcf.write_STR_vcf(
            samples_dict, locus_infos, variant_info, output_vcf_files
        )
    else:
        logger_mosaic.info(
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
                    "ALL\t%s\t%s\t%d\t%d\t%s\t%s\n"
                    % (
                        "ALL",
                        seg_locus.chrom,
                        seg_locus.start,
                        seg_locus.end,
                        seg_locus.STR_id,
                        seg_locus.unusable_reason,
                    )
                )
                myutils.release_lock(f)

def run(
    # Required arguments
    metadata,
    reference_genome,
    bed_panel,
    output_dir,
    # Optional input arguments
    gene_model=None,
    stutter_model="",
    gnomad_freq_in="",
    phasing="",
    allele_imbalance=None,
    # Optional output arguments
    loglevel="INFO",
    log_to_file=False,
    vcf_out=None,
    # Other optional arguments
    chrom="",
    start=0,
    end=1_000_000_000,
    threads=-1,
    verbose=0,
    debug=False,
    versions=False,
    # 控制执行模式
    single_thread=False,
):
    # 动态获取 CPU 核心数
    max_cores = os.cpu_count()
    effective_threads = threads if threads > 0 else max_cores
    if threads == -1:
        effective_threads = max_cores


    # 构造 argparse.Namespace 对象，模拟命令行解析结果
    args = argparse.Namespace(
        metadata=metadata,
        reference_genome=reference_genome,
        bed_panel=bed_panel,
        output_dir=output_dir,
        gene_model=gene_model,
        stutter_model=stutter_model,
        gnomad_freq_in=gnomad_freq_in,
        phasing=phasing,
        allele_imbalance=allele_imbalance,
        loglevel=loglevel,
        log_to_file=log_to_file,
        vcf_out=vcf_out,
        chrom=chrom,
        start=start,
        end=end,
        threads=effective_threads,
        verbose=verbose,
        debug=debug,
        versions=versions,
    )


    # 版本信息
    if versions:
        print(f"MosaicSTR version '{__VERSION__}'")
        return


    # 构建 Python 命令（用于日志记录）
    python_command = " ".join([sys.executable, __file__] + sys.argv[1:]) if hasattr(sys, "_getframe") else "unknown"


    # 准备全局参数
    try:
        genome_wide_info_dict, mosaic_fraction_all_params = mosaic_fraction_estimate_prepare(args)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare parameters: {e}")


    genome_wide_info_dict["commands"] = python_command


    # 创建输出 VCF 头文件
    try:
        output_vcf.create_vcf_header(genome_wide_info_dict, mosaic_fraction_all_params["vcf_output"])
    except Exception as e:
        raise RuntimeError(f"Failed to create VCF header: {e}")

    # 初始化日志
    logger_mosaic_fraction = logger_config.mosaic_fraction_logger(
        mosaic_fraction_all_params["loglevel"],
        mosaic_fraction_all_params["log_to_file"],
        mosaic_fraction_all_params["log_file"],
    )


    start_time = time.time()
    logger_mosaic_fraction.info("Start time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    logger_mosaic_fraction.info("Parameters: %s", args)
    logger_mosaic_fraction.info("Command: %s", python_command)


    # 打开 BED 注释文件
    try:
        STR_refpanel_bed = pysam.TabixFile(bed_panel)
    except Exception as e:
        logger_mosaic_fraction.error("Failed to open BED panel: %s", str(e))
        raise


    try:
        if chrom == "":
            # 全基因组模式
            total_tasks_count = sum(1 for _ in STR_refpanel_bed.fetch())
            STR_refpanel_bed.close()
            STR_refpanel_bed = pysam.TabixFile(bed_panel)  # 重新打开以重置迭代器
            task_iter = task_generator(STR_refpanel_bed, mosaic_fraction_all_params)
        else:
            # 区域模式
            total_tasks_count = sum(1 for _ in STR_refpanel_bed.fetch(chrom, start, end))
            STR_refpanel_bed.close()
            STR_refpanel_bed = pysam.TabixFile(bed_panel)
            task_iter = task_generator_given_region(
                STR_refpanel_bed, mosaic_fraction_all_params, chrom, start, end
            )


        if single_thread:
            # 单线程模式（调试）
            logger_mosaic_fraction.info("Running in single-threaded mode (debug).")
            for task in task_iter:
                main_per_locus_estimation(task[0], task[1])
        else:
            # 多进程模式
            logger_mosaic_fraction.info(f"Running with {effective_threads} processes.")
            with Manager() as manager:
                with Pool(processes=effective_threads) as pool:
                    progress_bar = tqdm(total=total_tasks_count)


                    def update_progress(_):
                        progress_bar.update(1)


                    # 提交任务
                    results = []
                    for task in task_iter:
                        r = pool.apply_async(
                            main_per_locus_estimation,
                            args=task,
                            callback=update_progress,
                            error_callback=lambda e: logger_mosaic_fraction.error("Task failed: %s", str(e))
                        )
                        results.append(r)


                    # 等待完成
                    for r in results:
                        r.wait()
                    pool.close()
                    pool.join()
                    progress_bar.close()


    except Exception as e:
        logger_mosaic_fraction.error("Pipeline failed: %s", str(e))
        traceback.print_exc()
        raise
    finally:
        STR_refpanel_bed.close()


    # 结束日志
    end_time = time.time()
    logger_mosaic_fraction.info("Finish all STR loci mosaic fraction estimation.")
    logger_mosaic_fraction.info("End time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    logger_mosaic_fraction.info("Total time spent: %.2f seconds", end_time - start_time)



if __name__ == "__main__":
    args = cmd_args()
    run(
        metadata=args.metadata,
        reference_genome=args.reference_genome,
        bed_panel=args.bed_panel,
        output_dir=args.output_dir,
        gene_model=args.gene_model,
        stutter_model=args.stutter_model,
        gnomad_freq_in=args.gnomad_freq_in,
        phasing=args.phasing,
        allele_imbalance=args.allele_imbalance,
        loglevel=args.loglevel,
        log_to_file=args.log_to_file,
        vcf_out=args.vcf_out,
        chrom=args.chrom,
        start=args.start,
        end=args.end,
        threads=args.threads,
        verbose=args.verbose,
        debug=args.debug,
        versions=args.versions,
        single_thread=False,  # 可通过环境变量或额外参数控制
    )