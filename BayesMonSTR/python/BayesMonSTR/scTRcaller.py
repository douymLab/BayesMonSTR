# -*- coding: utf-8 -*-

import argparse
import csv
import gc
import math
import os
import random
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pysam
import scipy.stats as stats

# from memory_profiler import profile
from tqdm import tqdm

from . import __VERSION__, stutter_dict
from .modeling.align import align_flk, align_ms
from .modeling.stutter import HapStutterEM, LengthStutterEM
from .processing import bam_processor, bed_processor, segmenter, vcf_processor
from .utils import FLK_BP_SEG, tools
from .utils.tools import console, logger


def init_args():
    """Initialization of Arguments"""

    parser = argparse.ArgumentParser(description="BayesMonSTR")
    parser.add_argument(
        "-D",
        "--debug",
        action="store_true",
        help="Use debug mode (default: False)",
    )
    parser.add_argument(
        "--nobulk",
        action="store_true",
        help="Ignore bulk data (default: False)",
    )
    parser.add_argument(
        "--ra",
        action="store_true",
        help="Not adding reference allele automatically (default: False)",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers used for processing, use all workers if -1 (default: 1)",
    )
    parser.add_argument(
        "-r",
        "--ref",
        type=str,
        required=True,
        help="Reference genome FASTA/FA file (required)",
    )
    parser.add_argument(
        "-i",
        "--info",
        type=str,
        required=True,
        help="Metadata for BAM/CRAM/SAM files and sample infomation (required)",
    )
    parser.add_argument(
        "-a",
        "--ab_info",
        type=str,
        required=False,
        help="Allelic imbalance information for WGA samples for read-based phasing",
    )
    parser.add_argument(
        "-p",
        "--phasing",
        nargs="+",
        type=str,
        default=None,
        help="Nearby germline hSNP information for bulk samples",
    )
    parser.add_argument(
        "-g",
        "--sc_info",
        nargs="+",
        type=str,
        required=False,
        help="Nearby germline hSNP information for WGA samples",
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=str,
        default=None,
        help="STR population allele frequency database.",
    )
    parser.add_argument(
        "-b",
        "--region",
        type=str,
        required=True,
        help="STR reference region BED file (required)",
    )
    parser.add_argument(
        "-S",
        "--stutter",
        type=str,
        default=None,
        help="STR stutter error model output filename (required)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="se",
        help="STR mutation calling mode, do not consider mosaic allele in bulk samples (sp) or consider mosaic allele in bulk samples (se)",
    )
    parser.add_argument(
        "-C",
        "--coord",
        action="store_true",
        help="Do not use HMM segmentation",
    )
    parser.add_argument(
        "-Q",
        "--quiet",
        action="store_true",
        help="Avoid verbose output",
    )
    parser.add_argument(
        "-U",
        "--unphase",
        type=str,
        default=None,
        help="Unphase STR variant call output filename",
    )
    parser.add_argument(
        "-O",
        "--vcf",
        type=str,
        default=None,
        help="STR variant call output filename",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s" + " version '" + __VERSION__ + "'",
    )

    args = parser.parse_args()
    return args


# @tools.timing
# @profile
def scTRcaller(args, info, region_dict, phase_info=None, is_denovo_stutter=True):
    """A per-locus MS mutation caller.

    Detect mosaic MS mutations with paired bulk
    and single cell WGA data.

    """
    # region DEBUG
    # M locus: nan, float
    if "/" in str(region_dict.get("motif")) or "nan" in str(region_dict.get("motif")):
        logger.warning("%s: SkippedMotifLocus", region_dict)
        return [region_dict["name"], "SkippedMotifLocus"], [], None
    # endregion DEBUG

    # region Processing
    logger.info(
        "Processing %s %s:%s%s",
        region_dict["name"],
        region_dict["chr"],
        region_dict["start"],
        "" if phase_info else " unphase",
    )
    # load bam file path
    bam_path_list = []
    sc_prot_list = []
    for ind in info.individual.unique():
        ind_data = info.loc[info["individual"] == ind]

        ind_bulk_data = ind_data.loc[ind_data["type"] == "bulk"]
        ind_sc_data = ind_data.loc[ind_data["type"] != "bulk"]

        bam_ind_bulk_list = []
        if not args.nobulk:
            for bam_path in ind_bulk_data.path:
                bam_ind_bulk_list.append(bam_path)

        bam_ind_sc_list = []
        for bam_path in ind_sc_data.path:
            bam_ind_sc_list.append(bam_path)

        bam_path_list.append([bam_ind_bulk_list, bam_ind_sc_list])
        sc_prot_list.append([v for v in ind_sc_data.type])

    # length-based params
    if not is_denovo_stutter:
        stutter = bed_processor.get_stutter(args, region_dict)
        stutter_data = (region_dict["name"], stutter)
    # endregion Processing

    # region length-based modeling
    else:
        # ================ preprocessing length-based reads ================
        try:
            # get read lens
            reads_data_len = bam_processor.get_lens(args, region_dict, bam_path_list)

            # ================ model ================
            # per-locus length-based stutter error model parameter estimation
            lengthStutterEM = LengthStutterEM(
                args, reads_data_len, region_dict["motif_len"]
            )
            # ================ Not Enough Allele (special case) ================
            if lengthStutterEM.num_total_allele <= 1:
                logger.warning(f"{region_dict}: NotEnoughAllele")
                return (region_dict["name"], "NEA"), (), None
            else:
                logger.info("Estimating stutter %s", region_dict["name"])
                lengthStutterEM.train()

            stutter = lengthStutterEM.stutter
            stutter_data = (region_dict["name"], stutter)
            if not args.vcf:
                return stutter_data, [], None
        except Exception as e:
            logger.exception(f"{region_dict} lengthStutterEM calculation failed: {e}")
            return [region_dict["name"], "SEMCF"], [], None
    # endregion length-based modeling

    # Seg reads
    ref_allele = bam_processor.fa_reader(args.ref, region_dict, flank_size=5)
    if "N" in ref_allele or not ref_allele[0] or not ref_allele[2]:
        logger.warning("%s: SegNSeqInRefError", region_dict)
        return stutter_data, [], None

    # region sequence-based modeling
    try:
        # pre check
        if bam_processor.check_locus(args, bam_path_list, region_dict):
            logger.warning("%s: ExtremeLocus", region_dict)
            return [region_dict["name"], "ExtremeLocus"], [], None

        # read bam file
        bam_obj_list = tools.map_nested_list(
            bam_processor.bam_reader, bam_path_list, args
        )

        # extract and filter reads
        filtered_reads_list = [
            [
                [
                    [
                        read
                        for read in bam
                        if not read.is_duplicate
                        and not read.is_qcfail
                        and read.is_mapped
                        and read.reference_start
                        <= region_dict["start"] - FLK_BP_SEG - 1  # 11 6
                        and read.reference_end
                        >= region_dict["end"] + FLK_BP_SEG  # 10 5
                        # and read.mapping_quality >= 20
                        # and np.mean(read.query_alignment_qualities) >= 20
                    ]
                    for bam in bams_types
                ]
                for bams_types in bams_ind
            ]
            for bams_ind in tools.map_nested_list(
                bam_processor.extract_reads, bam_obj_list, region_dict
            )
        ]

        # region nearby germline hSNP
        if phase_info and args.sc_info and phase_info != ["auto"]:
            phase_vcfs = [pysam.VariantFile(vcf) for vcf in args.phasing]
            sc_vcfs = [pysam.VariantFile(vcf) for vcf in args.sc_info]
            best_distances = []
            best_records = []
            best_genotypes = []
            best_sc_gts = []
            best_spn_reads_lists = []
            sample_names = []
            for ind in info["individual"].unique():
                df = info.loc[(info["individual"] == ind) & (info["type"] != "bulk")]
                sample_names.append([Path(row.path).stem for idx, row in df.iterrows()])

            for idx_ind, ind in enumerate(info["individual"].unique()):
                best_distance = float("inf")
                best_record = None
                best_genotype = None
                best_sc_gt = None
                best_spn_reads_list = []
                bulk_file_names = []

                df_bulk = info.loc[
                    (info["individual"] == ind) & (info["type"] == "bulk")
                ]
                if df_bulk.empty:
                    best_distances.append(best_distance)
                    best_records.append(best_record)
                    best_genotypes.append(best_genotype)
                    best_sc_gts.append(best_sc_gt)
                    best_spn_reads_lists.append(best_spn_reads_list)
                    continue
                bulk_file_name = tools.filename_wo_ext(
                    df_bulk.loc[df_bulk.index[0], "path"]
                )
                bulk_file_names.append(bulk_file_name)

                matching_phase_vcf = None
                for vcf in phase_vcfs:
                    if bulk_file_name in vcf.header.samples:
                        matching_phase_vcf = vcf
                        break

                if matching_phase_vcf:
                    pos_MS = round((region_dict["start"] + region_dict["end"]) / 2)
                    records = list(
                        matching_phase_vcf.fetch(
                            contig=region_dict["chr"],
                            start=(
                                region_dict["start"] - 1000
                                if region_dict["start"] > 1000
                                else 0
                            ),
                            stop=region_dict["start"] - FLK_BP_SEG - 1,  # 10
                        )
                    ) + list(
                        matching_phase_vcf.fetch(
                            contig=region_dict["chr"],
                            start=region_dict["end"] + FLK_BP_SEG,  # 10
                            stop=region_dict["end"] + 1000,
                        )
                    )
                    records = sorted(records, key=lambda x: abs(x.pos - pos_MS))
                    for record in records:
                        # print(record.ref, record.alts, record.alleles_variant_types)
                        # if record.alleles_variant_types != ("REF", "SNP"):
                        if (
                            len(record.ref) != 1
                            or len(record.alts) != 1
                            or len(record.alts[0]) != 1
                            or record.ref not in ("A", "C", "T", "G")
                            or record.alts[0] not in ("A", "C", "T", "G")
                        ):
                            continue
                        try:
                            genotype = record.samples[bulk_file_names[0]]["GT"]
                            dp = record.samples[bulk_file_names[0]]["DP"]
                            dp = dp if isinstance(dp, int) else 0
                            ad = record.samples[bulk_file_names[0]]["AD"][1]
                            ad = ad if isinstance(ad, int) else 0
                            if 0 in (ad, dp):
                                continue
                            if (
                                (genotype[0] != genotype[1])
                                and (
                                    (0.4 <= ad / dp <= 0.6)
                                    or (stats.binomtest(ad, dp, 0.5).pvalue >= 0.01)
                                )
                                and dp >= 10
                            ):
                                distance = abs(pos_MS - record.pos)
                                if distance < best_distance:
                                    spn_reads_list = [
                                        [
                                            [
                                                read
                                                for read in bam
                                                if tools.check_spn(
                                                    read,
                                                    record.start,
                                                    bam_obj_list[idx_ind][
                                                        idx_bams_types
                                                    ][idx_bam],
                                                )
                                            ]
                                            for idx_bam, bam in enumerate(bams_types)
                                        ]
                                        for idx_bams_types, bams_types in enumerate(
                                            filtered_reads_list[idx_ind]
                                        )
                                    ]
                                    if len(
                                        list(tools.flatten_list(spn_reads_list))
                                    ) >= len(spn_reads_list[1]):
                                        best_distance = distance
                                        best_record = record
                                        best_genotype = genotype
                                        best_spn_reads_list = spn_reads_list
                        except Exception:
                            pass
                            # logger.exception(
                            #     "%s one hSNP extraction failed: %s", region_dict, e
                            # )
                else:
                    best_distances_temp, best_records_temp, best_hSNPs_temp = (
                        vcf_processor.find_hSNP_around(
                            args,
                            info.loc[(info["individual"] == ind)],
                            region_dict,
                        )
                    )
                    best_distance = best_distances_temp[0]
                    best_record = best_records_temp[0]
                    best_hSNP = best_hSNPs_temp[0]
                    best_genotype = best_hSNPs_temp[0]
                    best_spn_reads_list = (
                        [
                            [
                                [
                                    read
                                    for read in bam
                                    if tools.check_spn(
                                        read,
                                        best_records_temp[0].start,
                                        bam_obj_list[idx_ind][idx_bams_types][idx_bam],
                                    )
                                ]
                                for idx_bam, bam in enumerate(bams_types)
                            ]
                            for idx_bams_types, bams_types in enumerate(
                                filtered_reads_list[idx_ind]
                            )
                        ]
                        if best_genotype
                        else []
                    )

                sc_gt = []
                is_mda = "mda" in list(
                    info.loc[
                        (info["individual"] == ind) & (info["type"] != "bulk")
                    ].type
                )
                if best_record and is_mda:
                    matching_sc_vcf = None
                    for vcf in sc_vcfs:
                        if any(
                            sample in vcf.header.samples
                            for sample in sample_names[idx_ind]
                        ):
                            matching_sc_vcf = vcf
                            break

                    if matching_sc_vcf:
                        for sc_record in matching_sc_vcf.fetch(
                            best_record.chrom, best_record.start, best_record.stop
                        ):
                            if sc_record.alleles == best_record.alleles:
                                for sample in sample_names[idx_ind]:
                                    if sc_record.samples.get(sample):
                                        sc_gt.append(sc_record.samples[sample]["GT"])
                                    else:
                                        sc_gt.append(None)
                                break
                    else:
                        sc_gt = None
                else:
                    sc_gt = None
                best_distances.append(best_distance)
                best_records.append(best_record)
                best_genotypes.append(best_genotype)
                best_sc_gts.append(sc_gt)
                best_spn_reads_lists.append(best_spn_reads_list)
            for phase_vcf in phase_vcfs:
                phase_vcf.close()
            for sc_vcf in sc_vcfs:
                sc_vcf.close()
        elif phase_info == ["auto"]:
            best_distances, best_records, best_hSNPs = vcf_processor.find_hSNP_around(
                args, info, region_dict
            )
            best_spn_reads_lists = [
                (
                    [
                        [
                            [
                                read
                                for read in bam
                                if tools.check_spn(
                                    read,
                                    best_records[idx_ind].start,
                                    bam_obj_list[idx_ind][idx_bams_types][idx_bam],
                                )
                            ]
                            for idx_bam, bam in enumerate(bams_types)
                        ]
                        for idx_bams_types, bams_types in enumerate(
                            filtered_reads_list[idx_ind]
                        )
                    ]
                    if best_records[idx_ind]
                    else []
                )
                for idx_ind in range(len(bam_obj_list))
            ]
            best_genotypes = None
            best_sc_gts = None
        else:
            best_distances = None
            best_records = None
            best_genotypes = None
            best_sc_gts = None
            best_spn_reads_lists = None
        # breakpoint()
        # endregion nearby germline hSNP

        # region per-locus sequence-based genotyping
        logger.info("Genotyping %s", region_dict["name"])
        segmodel = segmenter.SegModel(
            "\t".join(
                [
                    str(region_dict["chr"]),
                    str(int(region_dict["start"] - 1)),
                    str(region_dict["end"]),
                    str(region_dict["motif_len"]),
                    str(region_dict["len"]),
                    str(region_dict["name"]),
                    str(region_dict["motif"]),
                    ref_allele[2:5],
                    ref_allele[-5:-2],
                    ref_allele[5:-5],
                ]
            ),
            region_dict,
        )
        is_tr_coor = (
            not tools.is_tr_perf(ref_allele[5:-5], int(region_dict["motif_len"]))
        ) and int(region_dict["motif_len"]) >= 4

        # region get reads seq
        # ################## all MS reads ##################
        reads_data_anchor_seg = tools.map_nested_list(segmodel.seg, filtered_reads_list)
        _, has_coor_reads = tools.get_seq_from_reads_src(
            filtered_reads_list, reads_data_anchor_seg
        )
        if args.coord or is_tr_coor:
            reads_data_anchor = [
                [
                    [
                        [tools.extract_hap(read, region_dict) for read in bam]
                        for bam in bams_types
                    ]
                    for bams_types in bams_ind
                ]
                for bams_ind in filtered_reads_list
            ]
            reads_data_seq = tools.get_seq_from_reads_bwa(
                filtered_reads_list, reads_data_anchor, filters=True
            )
            reads_data_qual = tools.get_qual_from_reads_bwa(
                filtered_reads_list, reads_data_anchor, filters=True
            )
            reads_data_seq_all = tools.get_seq_from_reads_bwa(
                filtered_reads_list, reads_data_anchor
            )
            reads_data_qual_all = tools.get_qual_from_reads_bwa(
                filtered_reads_list, reads_data_anchor
            )
        else:
            # reads_data_anchor = tools.map_nested_list(segmodel.seg, filtered_reads_list)
            reads_data_anchor = reads_data_anchor_seg
            reads_data_seq = tools.get_seq_from_reads(
                filtered_reads_list, reads_data_anchor_seg, filters=True
            )
            reads_data_qual = tools.get_qual_from_reads(
                filtered_reads_list, reads_data_anchor_seg, filters=True
            )
            reads_data_seq_all = tools.get_seq_from_reads(
                filtered_reads_list, reads_data_anchor_seg
            )
            reads_data_qual_all = tools.get_qual_from_reads(
                filtered_reads_list, reads_data_anchor_seg
            )
        # ################## MS hSNP spanning reads ##################
        reads_data_spn_anchor = None
        reads_data_spn_seq = None
        reads_data_spn_qual = None
        reads_data_spn_seq_all = None
        reads_data_spn_qual_all = None
        if best_records and any(best_records):
            if args.coord or is_tr_coor:
                reads_data_spn_anchor = [
                    [
                        [
                            [tools.extract_hap(read, region_dict) for read in bam]
                            for bam in bams_types
                        ]
                        for bams_types in bams_ind
                    ]
                    for bams_ind in best_spn_reads_lists
                ]
                reads_data_spn_seq = tools.get_seq_from_reads_bwa(
                    best_spn_reads_lists,
                    reads_data_spn_anchor,
                    best_records=best_records,
                    bams=bam_obj_list,
                    filters=True,
                )
                reads_data_spn_qual = tools.get_qual_from_reads_bwa(
                    best_spn_reads_lists,
                    reads_data_spn_anchor,
                    best_records=best_records,
                    bams=bam_obj_list,
                    filters=True,
                )
                reads_data_spn_seq_all = tools.get_seq_from_reads_bwa(
                    best_spn_reads_lists,
                    reads_data_spn_anchor,
                    best_records=best_records,
                    bams=bam_obj_list,
                )
                reads_data_spn_qual_all = tools.get_qual_from_reads_bwa(
                    best_spn_reads_lists,
                    reads_data_spn_anchor,
                    best_records=best_records,
                    bams=bam_obj_list,
                )
            else:
                reads_data_spn_anchor = tools.map_nested_list(
                    segmodel.seg, best_spn_reads_lists
                )
                reads_data_spn_seq = tools.get_seq_from_reads(
                    best_spn_reads_lists,
                    reads_data_spn_anchor,
                    best_records=best_records,
                    bams=bam_obj_list,
                    filters=True,
                )
                reads_data_spn_qual = tools.get_qual_from_reads(
                    best_spn_reads_lists,
                    reads_data_spn_anchor,
                    best_records=best_records,
                    bams=bam_obj_list,
                    filters=True,
                )
                reads_data_spn_seq_all = tools.get_seq_from_reads(
                    best_spn_reads_lists,
                    reads_data_spn_anchor,
                    best_records=best_records,
                    bams=bam_obj_list,
                )
                reads_data_spn_qual_all = tools.get_qual_from_reads(
                    best_spn_reads_lists,
                    reads_data_spn_anchor,
                    best_records=best_records,
                    bams=bam_obj_list,
                )
        # ################## merge bulk read ##################
        for bams_ind in reads_data_seq:
            bams_ind[0] = list(tools.flatten_list(bams_ind[0]))
        for bams_ind in reads_data_qual:
            bams_ind[0] = list(tools.flatten_list(bams_ind[0]))
        for bams_ind in reads_data_seq_all:
            bams_ind[0] = list(tools.flatten_list(bams_ind[0]))
        for bams_ind in reads_data_qual_all:
            bams_ind[0] = list(tools.flatten_list(bams_ind[0]))
        # merge bulk spn read
        if best_records and any(best_records):
            if reads_data_spn_seq:
                for bams_ind in reads_data_spn_seq:
                    bams_ind[0] = list(tools.flatten_list(bams_ind[0]))
                for bams_ind in reads_data_spn_qual:
                    bams_ind[0] = list(tools.flatten_list(bams_ind[0]))
            # merge bulk spn read
            if reads_data_spn_seq_all:
                for bams_ind in reads_data_spn_seq_all:
                    bams_ind[0] = list(tools.flatten_list(bams_ind[0]))
                for bams_ind in reads_data_spn_qual_all:
                    bams_ind[0] = list(tools.flatten_list(bams_ind[0]))
        # endregion get reads seq

        if not (is_tr_coor or has_coor_reads):
            logger.info("%s: NotLongMotifInter and NoCoorReads", region_dict)
            # return stutter_data, [], None
        if has_coor_reads:
            logger.info("%s: CoorReads", region_dict)
        if is_tr_coor:
            logger.info("%s: LongMotifInter", region_dict)

        if args.freq:
            hap_freq = bed_processor.get_freq_db(args.freq, region_dict)
        else:
            hap_freq = None

        hapStutterEM = HapStutterEM(
            args,
            region_dict,
            stutter,
            reads_data_seq,
            reads_data_qual,
            sc_prot_list,
            hSNP_info=(
                (phase_info, best_records, best_genotypes, best_distances)
                if phase_info != ["auto"]
                else (phase_info, best_records, best_hSNPs, best_distances)
            ),
            reads_data_spn_seq=reads_data_spn_seq,
            reads_data_spn_qual=reads_data_spn_qual,
            hap_freq=hap_freq,
        )

        if hapStutterEM.num_total_allele <= 1 or hapStutterEM.total_num_reads <= sum(
            hapStutterEM.num_sc_sample
        ):
            logger.warning("%s: NoSeqAllele or NoSeqReads", region_dict)
            return stutter_data, [], None

        # region get alpha beta values
        if "alpha" in info.columns:
            ab_data = []
            for individual in info["individual"].unique():
                individual_rows = info[info["individual"] == individual]
                filtered_rows = individual_rows[individual_rows["type"] != "bulk"]
                tuples = tuple(map(tuple, filtered_rows[["alpha", "beta"]].values))
                if any(math.isnan(x) for tuple_pair in tuples for x in tuple_pair):
                    ab_data.append(None)
                else:
                    ab_data.append(tuples)
            hapStutterEM.ab_data = tuple(ab_data)
        else:
            hapStutterEM.ab_data = None
        # endregion get alpha beta values

        # region Gaussian Process Regression-AF
        if (
            phase_info
            # and args.sc_info
            # and args.ab_info
            and reads_data_spn_seq
            and reads_data_spn_qual
        ):
            # get phasable sample file names
            file_names = []
            for idx_ind, ind in enumerate(info["individual"].unique()):
                if not hapStutterEM.phasable[idx_ind]:
                    file_names.append(None)
                    continue
                ind_file_names = []

                df_sc = info.loc[(info["individual"] == ind) & (info["type"] != "bulk")]
                ind_file_names.extend(
                    [
                        (
                            tools.filename_wo_ext(df_sc["path"][idx]),
                            df_sc["type"][idx],
                        )
                        for idx in df_sc.index
                    ]
                )
                file_names.append(ind_file_names)
            af_list = bed_processor.get_afs(args, region_dict, file_names)
        else:
            af_list = None
        if af_list and best_genotypes and best_sc_gts:
            af = []
            for idx_ind, ind in enumerate(af_list):
                if ind and best_sc_gts[idx_ind]:
                    af_ind = []
                    for af_h1, phased_gt in zip(ind, best_sc_gts[idx_ind]):
                        if af_h1 is None or phased_gt is None:
                            af_ind.append(af_h1)
                        elif phased_gt == best_genotypes[idx_ind]:
                            af_ind.append(af_h1)
                        elif phased_gt[::-1] == best_genotypes[idx_ind]:
                            af_ind.append(1 - af_h1)
                        elif phased_gt == (0, 0):
                            if af_h1 >= 0.5:
                                af_ind.append(
                                    1 - af_h1 if best_genotypes[idx_ind][0] else af_h1
                                )
                            else:
                                af_ind.append(
                                    af_h1 if best_genotypes[idx_ind][0] else 1 - af_h1
                                )
                        elif phased_gt == (1, 1):
                            if af_h1 >= 0.5:
                                af_ind.append(
                                    af_h1 if best_genotypes[idx_ind][0] else 1 - af_h1
                                )
                            else:
                                af_ind.append(
                                    1 - af_h1 if best_genotypes[idx_ind][0] else af_h1
                                )
                        else:
                            af_ind.append(None)
                    af.append(af_ind)
                else:
                    af.append(ind)
        else:
            af = af_list

        hapStutterEM.af = af
        # endregion Gaussian Process Regression-AF
        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            hapStutterEM.train()
        try:
            hapStutterEM.mosaic_posteriors()
        except Exception as e:
            logger.exception("%s mosaic posteriors failed: %s", region_dict, e)
        hapStutterEM.train_refine(reads_data_seq_all, reads_data_qual_all)
        hapStutterEM.mosaic_posteriors_refine()
        # endregion per-locus sequence-based genotyping
        logger.info("Analyzing %s", region_dict["name"])
        # region feat extraction
        hapStutterEM.supp = []
        hapStutterEM.phy = []
        # hapStutterEM.reads_data_anchor = reads_data_anchor
        # hapStutterEM.reads_data_spn_anchor = reads_data_spn_anchor
        feat_vec, hap_freq = bam_processor.feat_extract(
            args,
            info,
            filtered_reads_list,
            best_spn_reads_lists,
            hapStutterEM,
            bam_obj_list,
            reads_data_anchor,
            reads_data_spn_anchor,
            reads_data_seq_all,
            reads_data_qual_all,
            reads_data_spn_seq_all,
            reads_data_spn_qual_all,
        )
        # endregion feat extraction

        # region collate output data
        # careful when debug
        # hapStutterEM.total_allele_pool.remove(hapStutterEM.ref_allele)
        vcf_data = [
            region_dict["chr"],
            region_dict["start"],
            region_dict["name"],
            tools.desegmentation(hapStutterEM.ref_allele),
            (
                ",".join(
                    map(
                        tools.desegmentation,
                        [
                            i
                            for i in hapStutterEM.total_allele_pool
                            if i != hapStutterEM.ref_allele
                        ],
                    )
                )
                if hapStutterEM.num_total_allele != 1
                else "."
            ),
            ".",
            "PASS",
            "DP={};DPSC={};RHO_BULK_IN={:.3f};UP_BULK_IN={:.3f};DOWN_BULK_IN={:.3f};RHO_BULK_OUT={:.3f};UP_BULK_OUT={:.3f};DOWN_BULK_OUT={:.3f};MA={};MC={};MU={};MP={};TYPE={}".format(
                hapStutterEM.total_num_reads,
                hapStutterEM.total_num_sc_reads,
                stutter["rho_bulk_in"],
                stutter["up_bulk_in"],
                stutter["down_bulk_in"],
                # stutter["rho_sc_in"],
                # stutter["up_sc_in"],
                # stutter["down_sc_in"],
                stutter["rho_bulk_out"],
                stutter["up_bulk_out"],
                stutter["down_bulk_out"],
                # stutter["rho_sc_out"],
                # stutter["up_sc_out"],
                # stutter["down_sc_out"],
                ",".join(
                    (
                        f"{hapStutterEM.total_allele_pool.index(j)}->{hapStutterEM.total_allele_pool.index(k)}"
                        if mu
                        else "NA"
                    )
                    for j, k, mu, _, _, _ in hapStutterEM.mu_mosaic
                ),
                ",".join(
                    f"{len(j[1])}->{len(k[1])}" if mu else "NA"
                    for j, k, mu, _, _, _ in hapStutterEM.mu_mosaic
                ),
                ",".join(
                    f"{mu:.3f}" if mu else "NA"
                    for _, _, mu, _, _, _ in hapStutterEM.mu_mosaic
                ),
                ",".join(
                    f"{mo:.3f}" if mo else "0" for mo in hapStutterEM.mo_posteriors
                ),
                ",".join(t for t in hapStutterEM.site_type),
            ),
            "GT:DP:L:P",
        ]
        for gt, dp, gl, gp in zip(
            [f"{gt[0]}/{gt[1]}" for gt in tools.flatten_list(hapStutterEM.gts)],
            list(tools.flatten_list(hapStutterEM.samples_depth)),
            list(tools.flatten_list(hapStutterEM.gt_lls)),
            list(tools.flatten_list(hapStutterEM.gt_posts)),
        ):
            # one for each sample
            vcf_data.append("{}:{}:{:.3f}:{:.3f}".format(gt, dp, gl, gp))
        # endregion collate output data

        # region debug_output
        hapCounts = []
        for idx_ind, ind_haps in enumerate(hapStutterEM.allele_pool):
            ind = []
            ind.append(
                {
                    str(
                        hapStutterEM.total_allele_pool.index(hap)
                    ): hapStutterEM.reads_data_seq[idx_ind][0].count(hap)
                    for hap in ind_haps
                }
            )
            for sc in hapStutterEM.reads_data_seq[idx_ind][1]:
                ind.append(
                    {
                        str(hapStutterEM.total_allele_pool.index(hap)): sc.count(hap)
                        for hap in ind_haps
                    }
                )
            hapCounts.append(ind)
        lenCounts = tools.map_nested_list(
            tools.hapToLen, hapCounts, hapStutterEM.total_allele_pool
        )
        ooo = vcf_data + [
            hapStutterEM.phasable,
            hapStutterEM.phasing_check,
            feat_vec,
            hapStutterEM.con,
            hapStutterEM.supp,
            region_dict,
            lenCounts,
            hapCounts,
            hapStutterEM.phy,
        ]
        vcf_data.extend(
            hapStutterEM.phasable
            if hapStutterEM.phasable
            else [None] * hapStutterEM.num_ind
        )
        vcf_data.extend(
            hapStutterEM.phasing_check
            if hapStutterEM.phasing_check
            else [None] * hapStutterEM.num_ind
        )
        vcf_data.extend(tools.flatten([i.values() for i in feat_vec]))
        vcf_data.extend(
            hapStutterEM.con if hapStutterEM.con else [None] * hapStutterEM.num_ind
        )
        vcf_data.extend(
            tools.flatten_list([list(i.values()) for i in hapStutterEM.supp])
        )
        vcf_data.append(f"{region_dict}")
        vcf_data.extend(region_dict.values())
        vcf_data.append(f"{dict(hap_freq)}")
        vcf_data.extend(lenCounts)
        vcf_data.extend(hapCounts)
        vcf_data.extend(hapStutterEM.phy)
        # endregion debug_output

        if args.debug:
            console.print(ooo)
            if hapStutterEM.phasable:
                console.print(tools.check_spn_phasing(hapStutterEM))
            breakpoint()
        # endregion misc
        return stutter_data, vcf_data, hapStutterEM.phasable
    # endregion sequence-based modeling
    except Exception as e:
        logger.exception("%s: hapStutterEM modeling failed: %s", region_dict, e)
        return stutter_data, [], None


# @tools.timing
# @profile
# @tools.profile_time
def main():
    """The main entry point of program"""

    # gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
    logger.info("Started: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    args = init_args()
    random.seed(42)
    np.random.seed(42)
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if args.quiet:
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    # load sites file
    ref_panel = bed_processor.load_ref_panel(args.region)
    ref_panel_dicts = ref_panel.to_dict("records")
    len_ref_panel = len(ref_panel_dicts)
    info = bed_processor.load_metadata(args)
    # stutter header
    is_denovo_stutter = not os.path.exists(args.stutter)
    if is_denovo_stutter:
        with open(args.stutter, "w", encoding="utf-8") as f:
            csv_out = csv.writer(f)
            csv_out.writerow(
                [
                    "name",
                    "rho_bulk_in",
                    "up_bulk_in",
                    "down_bulk_in",
                    "rho_sc_in",
                    "up_sc_in",
                    "down_sc_in",
                    "rho_bulk_out",
                    "up_bulk_out",
                    "down_bulk_out",
                    "rho_sc_out",
                    "up_sc_out",
                    "down_sc_out",
                ]
            )

    # parallel settings
    if args.num_workers == 1:
        # vcf header
        if args.vcf:
            vcf_processor.generate_vcf(args, info, [])
        # running
        init_time = time.time()
        for idx_rec, record in enumerate(ref_panel_dicts, start=1):
            start_time = time.time()
            stutter_data, vcf_data, phasable = scTRcaller(
                args, info, record, args.phasing, is_denovo_stutter
            )
            vcf_data_un = None
            if args.unphase and phasable:
                _, vcf_data_un, _ = scTRcaller(
                    args, info, record, None, is_denovo_stutter
                )
            # write stutter and vcf
            if is_denovo_stutter and stutter_data:
                with open(args.stutter, "a", encoding="utf-8") as f:
                    csv_out = csv.writer(f)
                    try:
                        row = [
                            stutter_data[0],
                            stutter_data[1]["rho_bulk_in"],
                            stutter_data[1]["up_bulk_in"],
                            stutter_data[1]["down_bulk_in"],
                            stutter_data[1]["rho_sc_in"],
                            stutter_data[1]["up_sc_in"],
                            stutter_data[1]["down_sc_in"],
                            stutter_data[1]["rho_bulk_out"],
                            stutter_data[1]["up_bulk_out"],
                            stutter_data[1]["down_bulk_out"],
                            stutter_data[1]["rho_sc_out"],
                            stutter_data[1]["up_sc_out"],
                            stutter_data[1]["down_sc_out"],
                        ]
                    except:
                        row = [stutter_data[0], stutter_data[1]]
                    csv_out.writerow(row)
            if args.vcf and vcf_data:
                with open(args.vcf + ".vcf", "at", encoding="utf-8") as f:
                    csv_out = csv.writer(f, delimiter="\t")
                    csv_out.writerow(vcf_data)
            if args.unphase and vcf_data_un:
                with open(args.unphase + ".vcf", "at", encoding="utf-8") as f:
                    csv_out = csv.writer(f, delimiter="\t")
                    csv_out.writerow(vcf_data_un)
            del stutter_data, vcf_data, vcf_data_un
            align_flk.cache_clear()
            align_ms.cache_clear()
            gc.collect()
            logger.info(
                f"{idx_rec}/{len_ref_panel}: {int((time.time() - start_time) / 60)}min {tools.get_memory_usage():.2f}MB {(len_ref_panel - idx_rec) * ((time.time() - init_time) / idx_rec) / 60:.2f}left {(time.time() - init_time) / 60:.2f}passed"
                + "\n"
                + "-" * 24
            )
    else:
        if args.num_workers == -1:
            args.num_workers = os.cpu_count()

        with Pool(processes=args.num_workers) as p, tqdm(total=len_ref_panel) as pbar:
            res = [
                p.apply_async(
                    scTRcaller,
                    args=(args, info, record, args.phasing, is_denovo_stutter),
                    callback=lambda _: pbar.update(1),
                )
                for record in ref_panel_dicts
            ]
            results = [r.get() for r in res]
        stutter_data, vcf_data, phasable = zip(*results)  # re-write

        # gen stutter error model parameters
        if not os.path.exists(args.stutter):
            bed_processor.generate_stutter(args, stutter_data)

        # gen vcf outputs
        sample_names = []
        for ind in info["individual"].unique():
            sample_names.append(ind + "_" + "bulk")
            df = info.loc[(info["individual"] == ind) & (info["type"] != "bulk")]
            sample_names.extend(
                [df["individual"][idx] + "_" + df["cell"][idx] for idx in df.index]
            )
        if args.vcf:
            vcf_processor.generate_vcf(args, info, vcf_data)

    logger.info("Ended: %s", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
