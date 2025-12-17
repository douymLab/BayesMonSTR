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
    required_args.add_argument(
        "-b",
        "--bed_panel",
        required=True,
        help="STR genome annotation(BED file)",
    )
    # Optional output arguments
    optional_output_args = parser.add_argument_group(
        "Optional output arguments"
    )
    optional_output_args.add_argument(
        "-o", "--output_dir", help="Path to output files"
    )
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
        output_dir,
        loglevel,
        log_to_file,
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
        self.output_dir = output_dir
        self.loglevel = loglevel
        self.log_to_file = log_to_file
        self.chrom = chrom
        self.start = start
        self.end = end
        self.threads = threads
        self.verbose = verbose
        self.debug = debug


class stutter_estimate_prepare:
    def __init__(self, parse_params) -> None:
        self.parse_params = parse_params
        log_dir = parse_params.output_dir + "/log"
        result_dir = parse_params.output_dir + "/results"
        os.system("mkdir -p " + parse_params.output_dir)
        os.system("mkdir -p " + log_dir)
        os.system("mkdir -p " + result_dir)
        bed_name = parse_params.bed_panel.split("/")[-1].split(".")[0]
        if parse_params.chrom:
            uid = f"{bed_name}_{parse_params.chrom}_{parse_params.start}_{parse_params.end}"
        else:
            uid = bed_name
        self.logfile = log_dir + "/" + uid + ".log"
        self.fail_file = (
            log_dir
            + "/"
            + uid
            + "_fail_loci.log"
        )
        self.stutter_file = (
            result_dir
            + "/"
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


def run(
    metadata: str,
    reference_genome: str,
    bed_panel: str,
    output_dir: str = "./stutter_output",
    loglevel: str = "INFO",
    log_to_file: bool = True,
    chrom: str = "",
    start: int = 0,
    end: int = 1000000000,
    threads: int = 1,
    verbose: int = 0,
    debug: bool = False,
):
    global DEBUG
    DEBUG = debug

    class Args:
        def __init__(self):
            self.metadata = metadata
            self.reference_genome = reference_genome
            self.bed_panel = bed_panel
            self.output_dir = output_dir
            self.loglevel = loglevel
            self.log_to_file = log_to_file
            self.chrom = chrom
            self.start = start
            self.end = end
            self.threads = threads if threads > 0 else os.cpu_count()
            self.verbose = verbose
            self.debug = debug


    options = Args()


    # 执行主流程
    _main_with_params(options)


def _main_with_params(parse_params):
    stutter_estimate_prep = stutter_estimate_prepare(parse_params)
    start_time = time.time()


    logger = stutter_estimate_prep.logger_stutter
    logger.info("Start stutter model estimation.")
    logger.info("Parameters: %s", parse_params)
    logger.info("Command: Python API call (stutter_estimator.run_stutter_estimation)")


    MAX_THREADS = parse_params.threads


    # 构建 per_locus 参数字典
    per_locus_parse_params = {
        "bam_name": stutter_estimate_prep.bam_name,
        "add_chr_character": stutter_estimate_prep.add_chr_character,
        "remove_chr_character": stutter_estimate_prep.remove_chr_character,
        "bam_files": stutter_estimate_prep.bam_files,
        "sex": stutter_estimate_prep.sex,
        "stutter_file": stutter_estimate_prep.stutter_file,
        "fail_file": stutter_estimate_prep.fail_file,
        "loglevel": parse_params.loglevel,
        "log_file": stutter_estimate_prep.logfile,
        "fasta_file": parse_params.reference_genome,
        "log_to_file": parse_params.log_to_file,
    }


    # 判断是否指定区域
    is_region_given = bool(parse_params.chrom)


    # 重新打开 Tabix（避免多进程共享问题）
    bed_file = parse_params.bed_panel
    pysam_bed = pysam.TabixFile(bed_file)


    try:
        if is_region_given:
            total_tasks_count = len(list(pysam_bed.fetch(parse_params.chrom, parse_params.start, parse_params.end)))
            task_iter = task_generator_given_region(pysam_bed, per_locus_parse_params, parse_params.chrom, parse_params.start, parse_params.end)
        else:
            total_tasks_count = len(list(pysam_bed.fetch()))
            task_iter = task_generator(pysam_bed, per_locus_parse_params)


        if DEBUG:
            # 单进程调试
            for task in task_iter:
                main_per_locus_estimation(task[0], task[1])
        else:
            # 多进程
            with Manager() as manager, Pool(processes=MAX_THREADS) as pool:
                pbar_counter = manager.list([0])
                progress_bar = tqdm(total=total_tasks_count, desc="Processing STR loci")


                def update_progress(_):
                    pbar_counter[0] += 1
                    progress_bar.update(1)


                for task in task_iter:
                    pool.apply_async(
                        main_per_locus_estimation,
                        args=task,
                        callback=update_progress,
                        error_callback=lambda e: logger.error(f"Error in worker: {e}")
                    )
                pool.close()
                pool.join()
                progress_bar.close()


    except Exception as e:
        logger.error(f"Error during processing: {e}")
        traceback.print_exc()
        raise
    finally:
        pysam_bed.close()


    # 完成
    logger.info("Finish all STR loci stutter model estimation.")
    end_time = time.time()
    logger.info("End time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    logger.info("Total time spent: %.2f seconds", end_time - start_time)


def main():
    args = cmd_args()
    _main_with_params(args)


if __name__ == "__main__":
    main()