import argparse
import sys
import os
import textwrap
import hmmsegger
import coordinate_segger
import myutils
import logging
import time
import pysam
import pandas as pd
import extract_allele

# import concurrent
from collections import defaultdict

# from ..configs import config_params
# sys.path.append("/Users/lid/Github/MosaicSTR/configs")
# from ..configs import config_params
import config_params
import estimate_noise
from multiprocessing import Pool, Manager
from tqdm import tqdm
import traceback

__VERSION__ = "0.1"
DEBUG = False  # False
# GLOBAL_FOR_STR_AND_FLANKING = config_params.COORDINATE_SEG["GLOBAL_FOR_STR_AND_FLANKING"]
GLOBAL_FOR_STR_AND_FLANKING = True  # True


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
        description=(
            textwrap.dedent(
                """
------------------------------------------------------------------------------
StutterEstimator: Detecting PCR stutter errors from bulk short-read sequencing data
------------------------------------------------------------------------------
"""
            )
        ),
        epilog=(
            textwrap.dedent(
                """
------------------------------------------------------------------------------
Any bugs can be reported to
Github:"https://github.com/Lidweixiang/MosaicSTR" or
E-Mail: "18829352615@163.com"'
------------------------------------------------------------------------------
"""
            )
        ),
        prog="stutter_model_estimation",
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
        "-j",
        "--config_json",
        required=False,
        default=os.path.join(
            os.path.dirname(__file__), "configs", "config.json"
        ),
        help=(
            "Configuration(JSON file) for more details parameters enactment"
            " (A developer options)"
        ),
    )
    required_args.add_argument(
        "-m",
        "--mode",
        choices=["RNA-seq", "WGS", "WES"],
        default="WGS",
        help="Sequencing type of analyzed data",
    )
    # Optional output arguments
    optional_output_args = parser.add_argument_group(
        "Optional output arguments"
    )
    optional_output_args.add_argument(
        "-p", "--path", help="Path to output files"
    )
    # optional_output_args.add_argument(
    #     "-l", "--log", action="extend", nargs="*", help="Log files"
    # )
    optional_output_args.add_argument("-l", "--log", help="Log files")
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
        "-q",
        "--stutter_model_out",
        help="Output stutter model from EM-algorithm",
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
        default=1,
        help="Number of threads to use, set to -1 when using all threads)",
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


class stutter_estimate_parse_params:
    def __init__(
        self,
        metadata,
        reference_genome,
        bed_panel,
        config_json,
        mode,
        path,
        log,
        loglevel,
        log_to_file,
        stutter_model_out,
        chrom,
        start,
        end,
        threads,
        verbose,
        debug,
    ) -> None:
        self.metadata = metadata
        self.reference_genome = reference_genome
        self.bed_panel = bed_panel
        self.config_json = config_json
        self.mode = mode
        self.path = path
        self.log = log
        self.loglevel = loglevel
        self.log_to_file = log_to_file
        self.stutter_model_out = stutter_model_out
        self.chrom = chrom
        self.start = start
        self.end = end
        self.threads = threads
        self.verbose = verbose
        self.debug = debug


# class stutter_estimate_prepare:
#     def __init__(self,parse_params) -> None:
#         self.parse_params = parse_params
#         log_dir = parse_params.path  + "/stutter_log"
#         result_dir = parse_params.path + "/stutter_results"
#         os.system("mkdir -p " + parse_params.path)
#         os.system("mkdir -p " + log_dir)
#         os.system("mkdir -p " + result_dir)
#         bed_name = parse_params.bed_panel.split(".")[0]
#         if parse_params.chrom:
#             uid = f"{bed_name}_{parse_params.chrom}_{parse_params.start}_{parse_params.end}"
#         else:
#             uid = bed_name
#         self.fail_file = log_dir + "/" + parse_params.log + "_" + uid + "_stutter_fail_loci.log"
#         self.logfile = log_dir + "/" + parse_params.log + "_" + uid + ".log"
#         self.stutter_file = result_dir + "/" + parse_params.stutter_model_out + "_" + uid + "_stutter_model.txt"
#         fasta_chromosome_name = myutils.check_reference_fasta_name(parse_params.reference_genome)
#         bed_chromosome_name = myutils.check_reference_bed_name(parse_params.bed_panel)
#         if "chr" in fasta_chromosome_name and "chr" not in bed_chromosome_name:
#             self.add_chr_character = True
#             self.remove_chr_character = False
#         elif "chr" not in fasta_chromosome_name and "chr" in bed_chromosome_name:
#             self.remove_chr_character = True
#             self.add_chr_character = False
#         else:
#             self.add_chr_character = False
#             self.remove_chr_character = False
#         log_level = getattr(logging, parse_params.loglevel.upper(), None)
#         if not isinstance(log_level, int):
#             raise ValueError(f"Invalid log level: {parse_params.loglevel}")
#         logger_stutter = logging.getLogger('StutterEstimator')
#         logger_stutter.setLevel(log_level)
#         if parse_params.log_to_file:
#             logger_stutter_file_handler = myutils.LockedFileHandler(self.logfile)
#         else:
#             logger_stutter_file_handler = logging.StreamHandler()
#         # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         logger_stutter_file_handler.setFormatter(formatter)
#         logger_stutter.addHandler(logger_stutter_file_handler)
#         self.logger_stutter = logger_stutter
#         logger_stutter.info("StutterEstimator: Stutter Model Estimation Using Population Data")
#         logger_stutter.info("Version: 1.0")
#         logger_stutter.info(
#             "Start time: %s",
#             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
#         )
#         self.STR_refpanel_bed = pysam.TabixFile(parse_params.bed_panel)
#         self.metadata = pd.read_csv(parse_params.metadata, sep=",")
#         self.bam_files = self.metadata.iloc[:, 4]
#         self.pysam_bam = [pysam.AlignmentFile(bam,"rb") for bam in self.bam_files]
#         self.sex = self.metadata.iloc[0,1]
#         self.bam_name = [bam.split(".")[0] for bam in self.bam_files]
#         self.hmm_seg_init = hmmsegger.hmm_seg_init

# def main_per_locus_estimation(stutter_estimate_prepare_objects,
#                               pysam_bed_row):
#     try:
#         all_samples_all_alleles_dict = {}
#         all_samples_fail_reads_dict = defaultdict(int)
#         all_samples_dp = 0
#         seg_locus = hmmsegger.HMMSeggerLocus(
#                 stutter_estimate_prepare_objects.hmm_seg_init,
#                 pysam_bed_row
#             )
#         if seg_locus.is_usable:
#             if stutter_estimate_prepare_objects.add_chr_character:
#                 chrom = "chr" + seg_locus.chrom
#             elif stutter_estimate_prepare_objects.remove_chr_character:
#                 chrom = seg_locus.chrom.replace("chr", "")
#             for bamname, bamfile in zip(stutter_estimate_prepare_objects.bam_name,stutter_estimate_prepare_objects.pysam_bam):
#                 all_samples_all_alleles_dict[bamname] = {}
#                 for read in bamfile.fetch(chrom, seg_locus.start, seg_locus.end):
#                     all_samples_dp += 1
#                     per_read_feature = extract_allele.PerReadFeature(
#                         read,
#                         seg_locus.start,
#                         seg_locus.end
#                     )
#                     per_read_feature.read_is_usable_for_bwa()
#                     if per_read_feature.is_usable:
#                         readsegger = hmmsegger.SegRead(stutter_estimate_prepare_objects.hmm_seg_init, seg_locus, read)
#                         if readsegger.is_usable:
#                             all_samples_all_alleles_dict[bamname][readsegger.STR_length] = all_samples_all_alleles_dict[bamname].get(readsegger.STR_length, 0) + 1
#                         else:
#                             all_samples_fail_reads_dict[readsegger.unusable_reason] += 1
#                     else:
#                         all_samples_fail_reads_dict[per_read_feature.unusable_reason] += 1
#             # noise class prep
#             other_params={}
#             locus_infos ={}
#             other_params["logger_object"] = stutter_estimate_prepare_objects.logger_stutter
#             other_params["total_unfiltered_depth"] = all_samples_dp
#             other_params["fail_reason_reads_number"] = {}
#             other_params["fail_reason_reads_number"]["unmap_issue"] = all_samples_fail_reads_dict.get("unmap_issue",0)
#             other_params["fail_reason_reads_number"]["library_issue"] = all_samples_fail_reads_dict.get("library_issue",0)
#             other_params["fail_reason_reads_number"]["spanning_issue"] = all_samples_fail_reads_dict.get("spanning_issue",0)
#             other_params["fail_reason_reads_number"]["map_issue"] = all_samples_fail_reads_dict.get("map_issue",0)
#             other_params["fail_reason_reads_number"]["SegmentConditionFail"] = all_samples_fail_reads_dict.get("SegmentConditionFail",0)
#             other_params["fail_reason_reads_number"]["ReadN"] = all_samples_fail_reads_dict.get("ReadN",0)
#             other_params["fail_reason_reads_number"]["SegmentResultFail"] = all_samples_fail_reads_dict.get("SegmentResultFail",0)
#             locus_infos["motif"] = seg_locus.motif
#             if stutter_estimate_prepare_objects.sex = "male":
#                 if ("X" in chrom and seg_locus.end <= config_params.X_PAR1_start_zero_based) or ("X" in chrom and seg_locus.end <= config_params.X_PAR2_start_zero_based and seg_locus.start >=config_params.X_PAR1_end_zero_based)\
#             or ("X" in chrom and seg_locus.start >= config_params.X_PAR2_end_zero_based)\
#                 or ("Y" in chrom and seg_locus.end <= config_params.Y_PAR1_start_zero_based) or ("Y" in chrom and seg_locus.end <= config_params.Y_PAR2_start_zero_based and seg_locus.start >=config_params.Y_PAR1_end_zero_based)\
#                     or ("Y" in chrom and seg_locus.start >= config_params.Y_PAR2_end_zero_based):
#                 locus_infos["ploidy"] = 1
#             else:
#                 locus_infos["ploidy"] = 2
#             locus_infos["motif_length"] = seg_locus.motif_length
#             locus_infos["STR_id"] = seg_locus.STR_id
#             locus_infos['str_zero_based_start_included'] = seg_locus.start
#             locus_infos['str_zero_based_end_excluded'] = seg_locus.end
#             locus_infos['chr'] = chrom
#             locus_infos['period'] = seg_locus.period
#             locus_infos["ref_allele_length"] = seg_locus.ref_allele_length
#             stutter_estimator = estimate_noise.NoiseEstimator(
#             all_samples_all_alleles_dict,
#             locus_infos,
#             other_params,
#         )
#             stutter_estimator.train()
#             stutter_estimator.stutter_output(
#                 stutter_estimate_prepare_objects.stutter_file
#             )
#             stutter_estimate_prepare_objects.logger_stutter.info(f"Finish stutter model estimation for locus {seg_locus.STR_id}")
#         else:
#             stutter_estimate_prepare_objects.logger_stutter.info("Locus %s %s:%d-%d is not usable because %s"%(seg_locus.STR_id ,seg_locus.chrom, seg_locus.start, seg_locus.end, seg_locus.unusable_reason))
#             with open(stutter_estimate_prepare_objects.fail_file, "a") as f:
#                 if myutils.acquire_lock(f):
#                     f.write("%s\t%d\t%d\t%s\n" % (seg_locus.chrom, seg_locus.start, seg_locus.end, seg_locus.STR_id))
#                     myutils.release_lock(f)
#     except:
#         stutter_estimate_prepare_objects.logger_stutter.info("Locus %s %s:%d-%d failed because error is raised"%(seg_locus.STR_id ,seg_locus.chrom, seg_locus.start, seg_locus.end))
#         with open(stutter_estimate_prepare_objects.fail_file, "a") as f:
#             if myutils.acquire_lock(f):
#                 f.write("%s\t%d\t%d\t%s\n" % (seg_locus.chrom, seg_locus.start, seg_locus.end, seg_locus.STR_id))
#                 myutils.release_lock(f)

# def main():
#     parse_params = cmd_args(args=sys.argv[1:])
#     stutter_estimate_prep = stutter_estimate_prepare(parse_params)
#     start_time = time.time()
#     stutter_estimate_prep.logger_stutter.info("Start time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
#     stutter_estimate_prep.logger_stutter.info("Parameters: %s", parse_params)
#     python_command = " ".join([sys.executable] + sys.argv)
#     stutter_estimate_prep.logger_stutter.info("Command: %s", python_command)
#     MAX_THREADS = (
#         stutter_estimate_prep.parse_params.threads  # Adjust this value based on your system's capabilities
#     )
#     if stutter_estimate_prep.parse_params.chrom == "":
#         with concurrent.futures.ProcessPoolExecutor(
#             max_workers=MAX_THREADS
#         ) as executor:
#             futures = [executor.submit(main_per_locus_estimation,stutter_estimate_prep ,arg) for arg in stutter_estimate_prep.STR_refpanel_bed.fetch()]
#             for future in concurrent.futures.as_completed(futures):
#                 pass
#             # results = [future.result() for future in futures]
#             # list(executor.map(main_per_locus_estimation,
#             #                   stutter_estimate_prep.STR_refpanel_bed.fetch()))
#     else:
#         with concurrent.futures.ProcessPoolExecutor(
#             max_workers=MAX_THREADS
#         ) as executor:
#             futures = [executor.submit(main_per_locus_estimation,stutter_estimate_prep ,arg) for arg in stutter_estimate_prep.STR_refpanel_bed.fetch(stutter_estimate_prep.parse_params.chrom, stutter_estimate_prep.parse_params.start, stutter_estimate_prep.parse_params.end)]
#             for future in concurrent.futures.as_completed(futures):
#                 pass
#             # list(
#             #     executor.map(
#             #         main_per_locus_estimation,
#             #         stutter_estimate_prep.STR_refpanel_bed.fetch(stutter_estimate_prep.parse_params.chrom, stutter_estimate_prep.parse_params.start, stutter_estimate_prep.parse_params.end),
#             #     )
#             # )
#     stutter_estimate_prep.logger_stutter.info("Finish all STR loci stutter model estimation.")
#     end_time = time.time()
#     stutter_estimate_prep.logger_stutter.info("End time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
#     total_time = end_time - start_time
#     stutter_estimate_prep.logger_stutter.info("Total time spent: %.2f seconds", total_time)
#     # stutter_estimate_prep.logger_stutter.info(
#     #     "End time: %s",
#     #     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
#     # )


class stutter_estimate_prepare:
    def __init__(self, parse_params) -> None:
        self.parse_params = parse_params
        log_dir = parse_params.path + "/stutter_log"
        result_dir = parse_params.path + "/stutter_results"
        os.system("mkdir -p " + parse_params.path)
        os.system("mkdir -p " + log_dir)
        os.system("mkdir -p " + result_dir)
        bed_name = parse_params.bed_panel.split("/")[-1].split(".")[0]
        if parse_params.chrom:
            uid = f"{bed_name}_{parse_params.chrom}_{parse_params.start}_{parse_params.end}"
        else:
            uid = bed_name
        self.fail_file = (
            log_dir
            + "/"
            + parse_params.log
            + "_"
            + uid
            + "_stutter_fail_loci.log"
        )
        self.logfile = log_dir + "/" + parse_params.log + "_" + uid + ".log"
        self.stutter_file = (
            result_dir
            + "/"
            + parse_params.stutter_model_out
            + "_"
            + uid
            + "_stutter_model.txt"
        )
        fasta_chromosome_name = myutils.check_reference_fasta_name(
            parse_params.reference_genome
        )
        bed_chromosome_name = myutils.check_reference_bed_name(
            parse_params.bed_panel
        )
        if "chr" in fasta_chromosome_name and "chr" not in bed_chromosome_name:
            self.add_chr_character = True
            self.remove_chr_character = False
        elif (
            "chr" not in fasta_chromosome_name and "chr" in bed_chromosome_name
        ):
            self.remove_chr_character = True
            self.add_chr_character = False
        else:
            self.add_chr_character = False
            self.remove_chr_character = False
        log_level = getattr(logging, parse_params.loglevel.upper(), None)
        if not isinstance(log_level, int):
            raise ValueError(f"Invalid log level: {parse_params.loglevel}")
        logger_stutter = logging.getLogger("StutterEstimator")
        logger_stutter.setLevel(log_level)
        if parse_params.log_to_file:
            logger_stutter_file_handler = myutils.LockedFileHandler(
                self.logfile
            )
        else:
            logger_stutter_file_handler = logging.StreamHandler()
        # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger_stutter_file_handler.setFormatter(formatter)
        if not logger_stutter.handlers:
            logger_stutter.addHandler(logger_stutter_file_handler)
        self.logger_stutter = logger_stutter
        logger_stutter.info(
            "StutterEstimator: Stutter Model Estimation Using Population Data"
        )
        logger_stutter.info("Version: 1.0")
        logger_stutter.info(
            "Start time: %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        )
        self.STR_refpanel_bed = pysam.TabixFile(parse_params.bed_panel)
        self.metadata = pd.read_csv(
            parse_params.metadata, sep=",", index_col=0
        )
        self.bam_files = self.metadata.iloc[:, 4]
        # self.pysam_bam = [pysam.AlignmentFile(bam,"rb") for bam in self.bam_files]
        self.sex = self.metadata.iloc[0, 1]  # HACK don't consider sex chromosome currently, TODO list: add sex chromosome estimation
        # self.bam_name = [
        #     bam.split("/")[-1].split(".")[0] for bam in self.bam_files
        # ]
        self.bam_name = self.metadata.iloc[:, 0]
        # self.hmm_seg_init = hmmsegger.hmm_seg_init


def main_per_locus_estimation(parse_params, pysam_bed_row):
    fasta_file = parse_params["fasta_file"]
    pysam_fasta = pysam.FastaFile(fasta_file)
    bam_name = parse_params["bam_name"]
    add_chr_character = parse_params["add_chr_character"]
    remove_chr_character = parse_params["remove_chr_character"]
    bam_files = parse_params["bam_files"]
    pysam_bam = [bam for bam in bam_files]
    sex = parse_params["sex"]
    stutter_file = parse_params["stutter_file"]
    fail_file = parse_params["fail_file"]
    log_level = getattr(logging, parse_params["loglevel"].upper(), None)
    logfile = parse_params["log_file"]
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {parse_params['loglevel']}")
    logger_stutter = logging.getLogger("StutterEstimator")
    logger_stutter.setLevel(log_level)
    if parse_params["log_to_file"]:
        logger_stutter_file_handler = myutils.LockedFileHandler(logfile)
    else:
        logger_stutter_file_handler = logging.StreamHandler()
    # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger_stutter_file_handler.setFormatter(formatter)
    if not logger_stutter.handlers:  # 避免重复添加handler
        logger_stutter.addHandler(logger_stutter_file_handler)
    # XXX try:
    all_samples_all_alleles_dict = {}
    all_samples_fail_reads_dict = defaultdict(int)
    all_samples_dp = 0
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
        for bamname, bamfile_nopysam in zip(bam_name, pysam_bam):
            if "bam" in bamfile_nopysam:
                bamfile = pysam.AlignmentFile(bamfile_nopysam, "rb")
            else:
                bamfile = pysam.AlignmentFile(bamfile_nopysam, "rc", reference_filename=fasta_file)
            all_samples_all_alleles_dict[bamname] = {}
            for read in bamfile.fetch(chrom, seg_locus.start, seg_locus.end):
                all_samples_dp += 1
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
                        if GLOBAL_FOR_STR_AND_FLANKING:
                            all_samples_all_alleles_dict[bamname][
                                readsegger.STR_flanking_length
                            ] = (
                                all_samples_all_alleles_dict[bamname].get(
                                    readsegger.STR_flanking_length, 0
                                )
                                + 1
                            )
                        else:
                            all_samples_all_alleles_dict[bamname][
                                readsegger.STR_length
                            ] = (
                                all_samples_all_alleles_dict[bamname].get(
                                    readsegger.STR_length, 0
                                )
                                + 1
                            )
                    else:
                        all_samples_fail_reads_dict[
                            readsegger.unusable_reason
                        ] += 1
                else:
                    all_samples_fail_reads_dict[
                        per_read_feature.unusable_reason
                    ] += 1
        # noise class prep
        other_params = {}
        locus_infos = {}
        # other_params["log_file"] = parse_params["log_file"]
        # other_params["loglevel"] = parse_params["loglevel"]
        # other_params["log_to_file"] = parse_params["log_to_file"]
        other_params["logger_object"] = logger_stutter
        other_params["total_unfiltered_depth"] = all_samples_dp
        other_params["fail_reason_reads_number"] = {}
        other_params["fail_reason_reads_number"][
            "unmap_issue"
        ] = all_samples_fail_reads_dict.get("unmap_issue", 0)
        other_params["fail_reason_reads_number"][
            "library_issue"
        ] = all_samples_fail_reads_dict.get("library_issue", 0)
        other_params["fail_reason_reads_number"][
            "spanning_issue"
        ] = all_samples_fail_reads_dict.get("spanning_issue", 0)
        other_params["fail_reason_reads_number"][
            "map_issue"
        ] = all_samples_fail_reads_dict.get("map_issue", 0)
        other_params["fail_reason_reads_number"][
            "SegmentConditionFail"
        ] = all_samples_fail_reads_dict.get("SegmentConditionFail", 0)
        other_params["fail_reason_reads_number"][
            "ReadN"
        ] = all_samples_fail_reads_dict.get("ReadN", 0)
        other_params["fail_reason_reads_number"][
            "SegmentResultFail"
        ] = all_samples_fail_reads_dict.get("SegmentResultFail", 0)
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
        if DEBUG:
            print(locus_infos)
        locus_infos["motif_length"] = seg_locus.motif_length
        locus_infos["STR_id"] = seg_locus.STR_id
        locus_infos["str_zero_based_start_included"] = seg_locus.start
        locus_infos["str_zero_based_end_excluded"] = seg_locus.end
        locus_infos["chr"] = chrom
        locus_infos["period"] = seg_locus.period
        locus_infos["ref_allele_length"] = seg_locus.ref_allele_length
        stutter_estimator = estimate_noise.NoiseEstimator(
            all_samples_all_alleles_dict,
            locus_infos,
            other_params,
        )
        stutter_estimator.train()
        stutter_estimator.stutter_output(stutter_file)
        logger_stutter.info(
            f"Finish stutter model estimation for locus {seg_locus.STR_id}"
        )
    else:
        logger_stutter.info(
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
    # XXX except:
    #     logger_stutter.info(
    #         "Locus %s %s:%d-%d failed because error is raised"
    #         % (
    #             seg_locus.STR_id,
    #             seg_locus.chrom,
    #             seg_locus.start,
    #             seg_locus.end,


#                 seg_locus.unusable_reason,
#         )
#     )
#     with open(fail_file, "a") as f:
#         if myutils.acquire_lock(f):
#             f.write(
#                 "%s\t%d\t%d\t%s\t%s\n"
#                 % (
#                     seg_locus.chrom,
#                     seg_locus.start,
#                     seg_locus.end,
#                     seg_locus.STR_id,
#                     seg_locus.unusable_reason,
#                 )
#             )
#             myutils.release_lock(f)


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


if DEBUG:
    # single process for debug
    def main():
        parse_params = cmd_args(args=sys.argv[1:])
        stutter_estimate_prep = stutter_estimate_prepare(parse_params)
        start_time = time.time()
        stutter_estimate_prep.logger_stutter.info(
            "Start time: %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        )
        stutter_estimate_prep.logger_stutter.info(
            "Parameters: %s", parse_params
        )
        python_command = " ".join([sys.executable] + sys.argv)
        stutter_estimate_prep.logger_stutter.info(
            "Command: %s", python_command
        )
        MAX_THREADS = (
            stutter_estimate_prep.parse_params.threads  # Adjust this value based on your system's capabilities
        )
        per_locus_parse_params = {}
        per_locus_parse_params["bam_name"] = stutter_estimate_prep.bam_name
        per_locus_parse_params[
            "add_chr_character"
        ] = stutter_estimate_prep.add_chr_character
        per_locus_parse_params[
            "remove_chr_character"
        ] = stutter_estimate_prep.remove_chr_character
        per_locus_parse_params["bam_files"] = stutter_estimate_prep.bam_files
        per_locus_parse_params["sex"] = stutter_estimate_prep.sex
        per_locus_parse_params[
            "stutter_file"
        ] = stutter_estimate_prep.stutter_file
        per_locus_parse_params["fail_file"] = stutter_estimate_prep.fail_file
        per_locus_parse_params["loglevel"] = parse_params.loglevel
        per_locus_parse_params["log_file"] = stutter_estimate_prep.logfile
        per_locus_parse_params["fasta_file"] = parse_params.reference_genome
        per_locus_parse_params[
            "log_to_file"
        ] = stutter_estimate_prep.parse_params.log_to_file
        if stutter_estimate_prep.parse_params.chrom == "":
            for task in task_generator(
                stutter_estimate_prep.STR_refpanel_bed,
                per_locus_parse_params,
            ):
                main_per_locus_estimation(task[0], task[1])
        else:
            for task in task_generator_given_region(
                stutter_estimate_prep.STR_refpanel_bed,
                per_locus_parse_params,
                parse_params.chrom,
                parse_params.start,
                parse_params.end,
            ):
                main_per_locus_estimation(task[0], task[1])
        stutter_estimate_prep.logger_stutter.info(
            "Finish all STR loci stutter model estimation."
        )
        end_time = time.time()
        stutter_estimate_prep.logger_stutter.info(
            "End time: %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
        )
        total_time = end_time - start_time
        stutter_estimate_prep.logger_stutter.info(
            "Total time spent: %.2f seconds", total_time
        )

else:
    # test multipleprocesses
    def main():
        parse_params = cmd_args(args=sys.argv[1:])
        stutter_estimate_prep = stutter_estimate_prepare(parse_params)
        start_time = time.time()
        stutter_estimate_prep.logger_stutter.info(
            "Start time: %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        )
        stutter_estimate_prep.logger_stutter.info(
            "Parameters: %s", parse_params
        )
        python_command = " ".join([sys.executable] + sys.argv)
        stutter_estimate_prep.logger_stutter.info(
            "Command: %s", python_command
        )
        MAX_THREADS = (
            stutter_estimate_prep.parse_params.threads  # Adjust this value based on your system's capabilities
        )
        per_locus_parse_params = {}
        per_locus_parse_params["bam_name"] = stutter_estimate_prep.bam_name
        per_locus_parse_params[
            "add_chr_character"
        ] = stutter_estimate_prep.add_chr_character
        per_locus_parse_params[
            "remove_chr_character"
        ] = stutter_estimate_prep.remove_chr_character
        per_locus_parse_params["bam_files"] = stutter_estimate_prep.bam_files
        per_locus_parse_params["sex"] = stutter_estimate_prep.sex
        per_locus_parse_params[
            "stutter_file"
        ] = stutter_estimate_prep.stutter_file
        per_locus_parse_params["fail_file"] = stutter_estimate_prep.fail_file
        per_locus_parse_params["loglevel"] = parse_params.loglevel
        per_locus_parse_params["log_file"] = stutter_estimate_prep.logfile
        per_locus_parse_params["fasta_file"] = parse_params.reference_genome
        per_locus_parse_params[
            "log_to_file"
        ] = stutter_estimate_prep.parse_params.log_to_file
        if stutter_estimate_prep.parse_params.chrom == "":
            total_tasks_count = len(
                list(stutter_estimate_prep.STR_refpanel_bed.fetch())
            )
            stutter_estimate_prep.STR_refpanel_bed = pysam.TabixFile(
                parse_params.bed_panel
            )
            # for task in task_generator(
            #                 stutter_estimate_prep.STR_refpanel_bed,
            #                 per_locus_parse_params,
            #             ):
            #     main_per_locus_estimation(task[0], task[1])
            # 除非给定了 fetch 的坐标，否则调用了 fetch() 以后，再重新调用 fetch 后，fetch 的起始位置是上一次 fetch 的结束位置
            with Manager() as manager:
                pbar = manager.list([0])  # 使用Manager创建一个可共享的列表用于跟踪进度
                #     total = 1700000  # 总任务数

                def update_progress_bar(_):
                    # 更新进度条的回调函数
                    pbar[0] += 1
                    progress_bar.update(1)

                with Pool(processes=MAX_THREADS) as p:
                    progress_bar = tqdm(total=total_tasks_count)
                    try:
                        for task in task_generator(
                            stutter_estimate_prep.STR_refpanel_bed,
                            per_locus_parse_params,
                        ):
                            p.apply_async(
                                main_per_locus_estimation,
                                args=task,
                                callback=update_progress_bar,
                            )

                        p.close()
                        p.join()
                        progress_bar.close()  # 确保在所有任务完成后关闭进度条
                    except Exception as e:
                        traceback.print_exc()
            # with concurrent.futures.ProcessPoolExecutor(
            #     max_workers=MAX_THREADS
            # ) as executor:
            #     futures = [executor.submit(main_per_locus_estimation,stutter_estimate_prep ,arg) for arg in stutter_estimate_prep.STR_refpanel_bed.fetch()]
            #     for future in concurrent.futures.as_completed(futures):
            #         pass
            # results = [future.result() for future in futures]
            # list(executor.map(main_per_locus_estimation,
            #                   stutter_estimate_prep.STR_refpanel_bed.fetch()))
        else:
            # for task in task_generator_given_region(
            #                 stutter_estimate_prep.STR_refpanel_bed,
            #                 per_locus_parse_params,
            #                 parse_params.chrom,
            #                 parse_params.start,
            #                 parse_params.end,
            #             ):
            #     main_per_locus_estimation(task[0], task[1])
            total_tasks_count = len(
                list(
                    stutter_estimate_prep.STR_refpanel_bed.fetch(
                        parse_params.chrom,
                        parse_params.start,
                        parse_params.end,
                    )
                )
            )
            stutter_estimate_prep.STR_refpanel_bed = pysam.TabixFile(
                parse_params.bed_panel
            )
            with Manager() as manager:
                pbar = manager.list([0])  # 使用Manager创建一个可共享的列表用于跟踪进度
                #     total = 1700000  # 总任务数

                def update_progress_bar(_):
                    # 更新进度条的回调函数
                    pbar[0] += 1
                    progress_bar.update(1)

                with Pool(processes=MAX_THREADS) as p:
                    progress_bar = tqdm(total=total_tasks_count)
                    try:
                        for task in task_generator_given_region(
                            stutter_estimate_prep.STR_refpanel_bed,
                            per_locus_parse_params,
                            parse_params.chrom,
                            parse_params.start,
                            parse_params.end,
                        ):
                            p.apply_async(
                                main_per_locus_estimation,
                                args=task,
                                callback=update_progress_bar,
                            )

                        p.close()
                        p.join()
                        progress_bar.close()  # 确保在所有任务完成后关闭进度条
                    except Exception as e:
                        traceback.print_exc()
            # with concurrent.futures.ProcessPoolExecutor(
            #     max_workers=MAX_THREADS
            # ) as executor:
            #     futures = [executor.submit(main_per_locus_estimation,stutter_estimate_prep ,arg) for arg in stutter_estimate_prep.STR_refpanel_bed.fetch(stutter_estimate_prep.parse_params.chrom, stutter_estimate_prep.parse_params.start, stutter_estimate_prep.parse_params.end)]
            #     for future in concurrent.futures.as_completed(futures):
            #         pass
            # list(
            #     executor.map(
            #         main_per_locus_estimation,
            #         stutter_estimate_prep.STR_refpanel_bed.fetch(stutter_estimate_prep.parse_params.chrom, stutter_estimate_prep.parse_params.start, stutter_estimate_prep.parse_params.end),
            #     )
            # )
        stutter_estimate_prep.logger_stutter.info(
            "Finish all STR loci stutter model estimation."
        )
        end_time = time.time()
        stutter_estimate_prep.logger_stutter.info(
            "End time: %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
        )
        total_time = end_time - start_time
        stutter_estimate_prep.logger_stutter.info(
            "Total time spent: %.2f seconds", total_time
        )
        # stutter_estimate_prep.logger_stutter.info(
        #     "End time: %s",
        #     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        # )


if __name__ == "__main__":
    main()

## example:
# sort -k1,1 -k2,2n /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/hg38.hipstr_reference_zero_based_chr22.bed > /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/hg38.hipstr_reference_zero_based_chr22_sorted.bed
# bgzip /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/hg38.hipstr_reference_zero_based_chr22_sorted.bed
# tabix -p bed /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/hg38.hipstr_reference_zero_based_chr22_sorted.bed.gz
# python3 stutter_model_estimation.py -i /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/metadata.csv -r /Users/lid/Desktop/Project_20230218/STR_reference_bed_from_HMM/Homo_sapiens_assembly38.fasta -b /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/hg38.hipstr_reference_zero_based_chr22_sorted.bed.gz -p /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation -l stutter_log -q stutter_result -c chr22 -t 4
# python3 stutter_model_estimation.py -i /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/metadata.csv -r /Users/lid/Desktop/Project_20230218/STR_reference_bed_from_HMM/Homo_sapiens_assembly38.fasta -b /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/hg38.hipstr_reference_zero_based_chr22_sorted.bed.gz -p /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation -l stutter_log2 -lf -q stutter_result2 -c chr22 -t 4
# python3 stutter_model_estimation.py -i /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/metadata.csv -r /Users/lid/Desktop/Project_20230218/STR_reference_bed_from_HMM/Homo_sapiens_assembly38.fasta -b /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation/hg38.hipstr_reference_zero_based_chr22_sorted.bed.gz -p /Users/lid/Github/MosaicSTR/test/test_pop_data_stutter_estimation -l stutter_log2 -lf -q stutter_result2 -c chr22 -t 6
# sort -k1,1 -k2,2n /Users/lid/Desktop/Project_20230218/STR_reference_bed_from_HMM/5bp/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised.bed > /Users/lid/Desktop/Project_20230218/STR_reference_bed_from_HMM/5bp/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed
# bgzip /Users/lid/Desktop/Project_20230218/STR_reference_bed_from_HMM/5bp/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed
# tabix -p bed human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed.gz
# GLOBAL_FOR_STR_AND_FLANKING = True
# python3 stutter_model_estimation2.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_male_hasHipSTR_data.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/software/HipSTR-references-master/human/GRCh37.hipstr_reference_startminusone_zerobased_leftcloserightopen.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l missing_loci -lf -q missing_loci -c 10 -s 191073 -e 191211 > ../debug/missing_loci.txt
# python3 stutter_model_estimation2.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_male_hasHipSTR_data.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/software/HipSTR-references-master/human/GRCh37.hipstr_reference_startminusone_zerobased_leftcloserightopen.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l diff_HipSTR_indel -lf -q diff_HipSTR_indel -c 12 -s 8092306 -e 8092329 > ../debug/diff_HipSTR_indel.txt
# 22      50000000        51000000
# python3 stutter_model_estimation2.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_male_hasHipSTR_data.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/software/HipSTR-references-master/human/GRCh37.hipstr_reference_startminusone_zerobased_leftcloserightopen.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l log_divide_zero -lf -q log_divide_zero -c 22 -s 50000000 -e 51000000 > ../debug/log_divide_zero_error.txt
