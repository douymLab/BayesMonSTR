# -*- coding: utf-8 -*-

import math
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pysam
from scipy.stats import fisher_exact, ranksums
from statsmodels.stats.multitest import fdrcorrection

from ..utils import FLK_BP, FLK_BP_SEG, tools


class AutoBAMHandler:
    """
    Wrapper class for pysam.AlignmentFile that automatically handles file opening and closing.
    Ensures files are closed after list comprehensions and generator operations.
    """

    def __init__(self, bam_path: str, reference_filename: Optional[str] = None):
        self.bam_path = bam_path
        self.reference_filename = reference_filename

    @contextmanager
    def _get_handle(self):
        """Internal context manager for file handle management"""
        handle = self._open_file()
        try:
            yield handle
        finally:
            handle.close()

    def _open_file(self) -> pysam.AlignmentFile:
        """Open the BAM/SAM/CRAM file with appropriate mode"""
        file_type = os.path.splitext(self.bam_path)[1].lower()

        try:
            if file_type == ".bam":
                return pysam.AlignmentFile(self.bam_path, "rb")
            elif file_type == ".sam":
                return pysam.AlignmentFile(self.bam_path, "r")
            elif file_type == ".cram":
                if not self.reference_filename:
                    raise ValueError("Reference filename is required for CRAM files")
                return pysam.AlignmentFile(
                    self.bam_path, "rc", reference_filename=self.reference_filename
                )
            else:
                raise ValueError(
                    f"Unsupported file type: {file_type}. Expected .bam, .sam, or .cram"
                )
        except Exception as e:
            raise IOError(f"Failed to open {self.bam_path}: {str(e)}")

    def fetch(self, *args, **kwargs) -> List[pysam.AlignedSegment]:
        """
        Wrapper for pysam.AlignmentFile.fetch()
        Returns a list of reads instead of a generator to ensure immediate file closing
        """
        with self._get_handle() as handle:
            return list(handle.fetch(*args, **kwargs))

    def mate(self, read: pysam.AlignedSegment) -> Optional[pysam.AlignedSegment]:
        """
        Wrapper for pysam.AlignmentFile.mate()
        Opens file, gets mate read, then closes automatically
        """
        with self._get_handle() as handle:
            return handle.mate(read)

    def count_coverage(
        self,
        contig: str,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        quality_threshold: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrapper for pysam.AlignmentFile.count_coverage()
        Returns coverage for each base in the specified region as a tuple of arrays (A, C, G, T)

        Args:
            contig: Name of the contig/chromosome
            start: Start position (0-based)
            stop: Stop position (exclusive)
            quality_threshold: Minimum base quality to consider (default: 15)

        Returns:
            Tuple of four arrays containing the coverage for A, C, G, T bases respectively
        """
        with self._get_handle() as handle:
            return handle.count_coverage(
                contig,
                start=start,
                stop=stop,
                quality_threshold=quality_threshold,
            )


def bam_reader_new(
    bam_path: str, reference_filename: Optional[str] = None
) -> AutoBAMHandler:
    """
    Create an AutoBAMHandler instance for the given BAM/SAM/CRAM file.

    Args:
        bam_path: Path to the BAM/SAM/CRAM file
        reference_filename: Path to reference genome (required for CRAM files)

    Returns:
        AutoBAMHandler instance
    """
    return AutoBAMHandler(bam_path, reference_filename)


def bam_reader(bam_path, args=None):
    """Read aligned reads file."""
    # ignore_truncation = True
    file_type = os.path.splitext(bam_path)[1]
    try:
        if file_type == ".bam":
            return pysam.AlignmentFile(bam_path, "rb")
        elif file_type == ".sam":
            return pysam.AlignmentFile(bam_path, "r")
        elif file_type == ".cram":
            return pysam.AlignmentFile(bam_path, "rc", reference_filename=args.ref)
        raise ValueError(
            f"file should be BAM/CRAM/SAM, but got {file_type} for {bam_path}"
        )
    except:
        tools.logger.error(
            "Aligned reads loading failed: type %s for %s.",
            file_type,
            bam_path,
        )


def extract_reads(bam, region_dict):
    """extract all reads from a bam file given a specific region"""
    return bam.fetch(
        region_dict["chr"],
        region_dict["start"] - 1,
        region_dict["end"],
        multiple_iterators=False,
    )


def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def check_locus(args, bam_path_list, region_dict):
    for individual_bams in bam_path_list:
        individual_total_reads = 0
        individual_soft_clips = 0
        total_valid_reads = 0
        total_files = 0

        for bam_path in flatten(individual_bams):
            try:
                file_type = os.path.splitext(bam_path)[1]
                if file_type == ".bam":
                    bam_file = pysam.AlignmentFile(bam_path, "rb")
                elif file_type == ".sam":
                    bam_file = pysam.AlignmentFile(bam_path, "r")
                elif file_type == ".cram":
                    bam_file = pysam.AlignmentFile(
                        bam_path, "rc", reference_filename=args.ref
                    )
                else:
                    raise ValueError(
                        f"file should be BAM/CRAM/SAM, but got {file_type} for {bam_path}"
                    )

                reads = bam_file.fetch(
                    region_dict["chr"], region_dict["start"] - 1, region_dict["end"]
                )

                for read in reads:
                    if read.is_duplicate or read.is_qcfail or not read.is_mapped:
                        continue

                    individual_total_reads += 1

                    if read.get_cigar_stats()[0][4] >= 10:
                        individual_soft_clips += 1

                    if (
                        read.reference_start <= region_dict["start"] - 6
                        and read.reference_end >= region_dict["end"] + 5
                        and read.mapping_quality >= 20
                        and np.mean(read.query_alignment_qualities) >= 20
                    ):
                        total_valid_reads += 1

                total_files += 1
                bam_file.close()

            except Exception as e:
                tools.logger.warning(
                    "%s %s %s count_reads failed: %s",
                    region_dict,
                    file_type,
                    bam_path,
                    e,
                )
                continue

        if total_files == 0:
            continue

        individual_avg_depth = total_valid_reads / total_files
        soft_clip_proportion = (
            individual_soft_clips / individual_total_reads
            if individual_total_reads > 0
            else 0
        )

        if individual_avg_depth >= 600 or (
            soft_clip_proportion >= 0.3 and individual_avg_depth >= 40
        ):
            return True
    return False


def get_lens(args, region_dict, bam_path_list):
    reads_data_len = []
    for bams_ind in bam_path_list:
        bams_types_len = []
        for bams_types in bams_ind:
            bam_len = []
            for bam in bams_types:
                bam_obj = bam_reader(bam, args)
                reads_list = [
                    tools.extract_allele(read, region_dict)
                    for read in extract_reads(bam_obj, region_dict)
                    if not read.is_duplicate
                    and not read.is_qcfail
                    and read.is_mapped
                    and read.reference_start
                    <= region_dict["start"] - FLK_BP - 1  # 11 6
                    and read.reference_end >= region_dict["end"] + FLK_BP  # 10 5
                    and read.mapping_quality >= 20
                    and np.mean(read.query_alignment_qualities) >= 20
                    # and "N" not in read.cigarstring # RNA
                ]
                bam_len.append(reads_list)
            bams_types_len.append(bam_len)
        reads_data_len.append(bams_types_len)

    for bams_ind in reads_data_len:
        bams_ind[0] = list(tools.flatten(bams_ind[0]))

    return reads_data_len


def find_interruption(seq, motif):
    for idx_base_motif, base_motif in enumerate(motif):
        if motif[idx_base_motif] == seq[0]:
            for idx_base_seq, base_seq in enumerate(seq):
                if base_seq != motif[idx_base_motif % len(motif)]:
                    break
                idx_base_motif += 1
                if idx_base_seq + 1 == len(seq):
                    return 0
    return 1


def fa_reader(fa_path, region_dict, flank_size=0):
    f = pysam.Fastafile(fa_path)
    seq = f.fetch(
        region_dict["chr"],
        region_dict["start"] - 1 - flank_size,
        region_dict["end"] + flank_size,
    )
    f.close()
    return seq


def allele_gen_old(reads_cnt_dict):
    if not reads_cnt_dict:
        return []
    reads_cnt = sorted(reads_cnt_dict.items(), key=lambda x: x[1], reverse=True)
    haps, counts = list(zip(*reads_cnt))
    if len(haps) <= 2:
        return list(haps)
    pvals = [
        fisher_exact(
            [
                [int(counts[idx_allele]), int(sum(counts) - counts[idx_allele])],
                [int(counts[-1]), int(sum(counts) - counts[-1])],
            ]
        )[1]
        for idx_allele in range(0, len(haps) - 1)
    ]
    rej, _ = fdrcorrection(pvals=pvals, alpha=0.1, is_sorted=True)
    res = [haps[i] for i in range(len(rej)) if rej[i]]
    return res


def allele_gen(reads_cnt_dict):
    if not reads_cnt_dict:
        return []
    reads_cnt = sorted(reads_cnt_dict.items(), key=lambda x: x[1], reverse=True)
    haps, counts = list(zip(*reads_cnt))
    if len(haps) <= 2:
        if sum(counts) <= 5:
            return list(haps)
        return [hap for hap, count in reads_cnt if count >= 3]
    significant_alleles = []
    pvals = []
    for idx_allele in range(0, len(haps) - 1):
        if counts[idx_allele] >= 3:
            significant_alleles.append(haps[idx_allele])
        elif counts[idx_allele] >= 2:
            contingency_table = [
                [int(counts[idx_allele]), int(sum(counts) - counts[idx_allele])],
                [int(counts[-1]), int(sum(counts) - counts[-1])],
            ]
            pval = fisher_exact(contingency_table)[1]
            pvals.append(pval)
            if pval <= 0.1:
                significant_alleles.append(haps[idx_allele])
    return significant_alleles


def rm_snp(read):
    r"""Convert read w/ nearby g hSNP to one w/o"""
    return (read[0], read[1], read[2])


def feat_extract(
    args,
    info,
    filtered_reads_list,
    spn_reads_list,
    model,
    bam_obj_list,
    reads_data_anchor,
    reads_data_spn_anchor,
    reads_data_seq,
    reads_data_qual,
    reads_data_spn_seq,
    reads_data_spn_qual,
):
    r"""feature extration"""
    feat = []
    hap_freq = defaultdict(int)

    hanging_reads_list = [
        [
            [
                [
                    read
                    for read in bam
                    if not read.is_duplicate
                    and not read.is_qcfail
                    and read.is_mapped
                    and (
                        (
                            model.region["start"] - FLK_BP_SEG - 1
                            < read.reference_start
                            < model.region["start"] - 1
                            and read.reference_end > model.region["end"] + FLK_BP_SEG
                        )
                        or (
                            model.region["end"] + FLK_BP_SEG
                            > read.reference_end
                            > model.region["end"]
                            and read.reference_start
                            < model.region["start"] - FLK_BP_SEG - 1
                        )
                    )
                ]
                for bam in bams_types
            ]
            for bams_types in bams_ind
        ]
        for bams_ind in tools.map_nested_list(extract_reads, bam_obj_list, model.region)
    ]
    hanging_reads_data_anchor = [
        [
            [
                [tools.extract_hanging_hap(read, model.region) for read in bam]
                for bam in bams_types
            ]
            for bams_types in bams_ind
        ]
        for bams_ind in hanging_reads_list
    ]
    hanging_reads_data_seq = tools.get_seq_from_reads_bwa(
        hanging_reads_list, hanging_reads_data_anchor
    )

    raw_reads_list = [
        [
            [
                [
                    read
                    for read in bam
                    if not read.is_duplicate and not read.is_qcfail and read.is_mapped
                ]
                for bam in bams_types
            ]
            for bams_types in bams_ind
        ]
        for bams_ind in tools.map_nested_list(extract_reads, bam_obj_list, model.region)
    ]
    raw_reads = tuple(tools.flatten(raw_reads_list))

    for idx_ind in range(model.num_ind):
        (
            mu_j_allele,
            mu_k_allele,
            mu_jk,
            gt_germline,
            gt_mosaic,
            ref_g_allele,
        ) = model.mu_mosaic[idx_ind]

        for hap in gt_germline:
            hap_freq[hap] += 1
        haps = (
            (mu_k_allele, mu_j_allele)
            if mu_j_allele == ref_g_allele
            else (mu_k_allele, mu_j_allele, ref_g_allele)
        )
        # region check hap consistence
        if model.phasable and model.phasable[idx_ind]:
            # discordant
            con = "NA"
            if not model.mu_mosaic[idx_ind][1]:
                model.dis_prop.append(None)
                model.phasing_check.append([])
            else:
                if (ref_g_allele, mu_k_allele) in model.genotypes[idx_ind]:
                    p1 = (
                        np.exp(model.phase_probs[idx_ind][gt_mosaic])
                        if gt_mosaic
                        else np.exp(model.phase_probs[idx_ind][gt_germline])
                    )
                    if p1 >= 0.5:
                        con = "j1k2"
                    else:
                        con = "j2k1"
                elif (mu_k_allele, ref_g_allele) in model.genotypes[idx_ind]:
                    p1 = (
                        np.exp(model.phase_probs[idx_ind][gt_mosaic])
                        if gt_mosaic
                        else np.exp(model.phase_probs[idx_ind][gt_germline])
                    )
                    if p1 >= 0.5:
                        con = "j2k1"
                    else:
                        con = "j1k2"
                else:
                    model.dis_prop.append(None)
                    continue
                cnt_j1 = 0
                cnt_k2 = 0
                cnt_j2 = 0
                cnt_k1 = 0
                cnt_j1_bulk = 0
                cnt_k2_bulk = 0
                cnt_j2_bulk = 0
                cnt_k1_bulk = 0
                phasing_check_ind = []
                hSNP_1, hSNP_2 = model.hSNP[idx_ind]
                # bulk
                r_spn_seq = reads_data_spn_seq[idx_ind][0]
                if r_spn_seq:
                    count_j1 = 0
                    count_j2 = 0
                    count_k1 = 0
                    count_k2 = 0
                    for idx_r, r in enumerate(r_spn_seq):
                        assign = model.read_aln(
                            r,
                            reads_data_spn_qual[idx_ind][0][idx_r],
                            haps,
                            "bulk",
                        )
                        if assign:
                            if r[3] == hSNP_1:
                                count_j1 += 1
                            elif r[3] == hSNP_2:
                                count_j2 += 1
                        else:
                            if r[3] == hSNP_1:
                                count_k1 += 1
                            elif r[3] == hSNP_2:
                                count_k2 += 1
                    phasing_check_ind.append(
                        [
                            count_j1,
                            count_k2,
                            count_j2,
                            count_k1,
                        ]
                    )
                    cnt_j1_bulk += count_j1
                    cnt_k2_bulk += count_k2
                    cnt_j2_bulk += count_j2
                    cnt_k1_bulk += count_k1
                else:
                    phasing_check_ind.append([])
                # sc
                for idx_sample in range(model.num_sc_sample[idx_ind]):
                    r_spn_seq = reads_data_spn_seq[idx_ind][1][idx_sample]
                    if r_spn_seq:
                        count_j1 = 0
                        count_j2 = 0
                        count_k1 = 0
                        count_k2 = 0
                        for idx_r, r in enumerate(r_spn_seq):
                            assign = model.read_aln(
                                r,
                                reads_data_spn_qual[idx_ind][1][idx_sample][idx_r],
                                haps,
                                model.sc_prot_list[idx_ind][idx_sample],
                            )
                            if assign:
                                if r[3] == hSNP_1:
                                    count_j1 += 1
                                elif r[3] == hSNP_2:
                                    count_j2 += 1
                            else:
                                if r[3] == hSNP_1:
                                    count_k1 += 1
                                elif r[3] == hSNP_2:
                                    count_k2 += 1
                        phasing_check_ind.append(
                            [
                                count_j1,
                                count_k2,
                                count_j2,
                                count_k1,
                            ]
                        )
                        cnt_j1 += count_j1
                        cnt_k2 += count_k2
                        cnt_j2 += count_j2
                        cnt_k1 += count_k1
                    else:
                        phasing_check_ind.append([])
                model.phasing_check.append(phasing_check_ind)
                # discordant rate per ind
                dis_rate_bulk = 0  # None
                dis_rate = 0  # None
                if con == "j1k2":
                    if cnt_k1 + cnt_j1:
                        dis_rate = cnt_k1 / (cnt_k1 + cnt_j1)
                    if cnt_k1_bulk + cnt_j1_bulk:
                        dis_rate_bulk = cnt_k1_bulk / (cnt_k1_bulk + cnt_j1_bulk)
                elif con == "j2k1":
                    if cnt_k2 + cnt_j2:
                        dis_rate = cnt_k2 / (cnt_k2 + cnt_j2)
                    if cnt_k2_bulk + cnt_j2_bulk:
                        dis_rate_bulk = cnt_k2_bulk / (cnt_k2_bulk + cnt_j2_bulk)
                else:
                    model.dis_prop.append(None)
                    continue
                model.dis_prop.append((dis_rate_bulk, dis_rate))
            model.con.append(con)

            if (
                model.dis_prop[idx_ind]
                and None not in model.dis_prop[idx_ind]
                and model.site_type[idx_ind] == "mosaic"
            ):
                try:
                    model.lr_phase.append(model.lr_test(idx_ind))
                except Exception as e:
                    tools.logger.exception("%s lrt failed: %s", model.region, e)
                    model.lr_phase.append(None)
            else:
                model.lr_phase.append(None)
        else:
            if model.dis_prop is not None:
                model.dis_prop.append(None)
            if model.phasing_check is not None:
                model.phasing_check.append([])
            if model.lr_phase is not None:
                model.lr_phase.append(None)
            if model.con is not None:
                model.con.append(None)
        # endregion check hap consistence

        # ms_ref_len = model.region["len"]
        ms_motif_len = model.region["motif_len"]
        ref_seq = fa_reader(args.ref, model.region, flank_size=40)
        cntxt_40bp = ref_seq[:20] + ref_seq[-20:]
        gc_content = (cntxt_40bp.count("G") + cntxt_40bp.count("C")) / len(cntxt_40bp)
        mappability = model.region.get("mappability", 0)
        mo_posterior = model.mo_posteriors[idx_ind]
        if model.mode == "sp":
            germline_ratio = tools.max_div_sec(model.germ_priors[idx_ind])
        else:
            germline_ratio = tools.max_div_sec(model.mu_germ_priors[idx_ind])
        # region init NOTE: p values are log-scaled, log(1) == 0
        bulk_pos_p = 0
        bulk_mapq_p = 0
        bulk_baseq_p = 0
        bulk_strand_p = 0
        bulk_read12_p = 0
        bulk_pos_stats = 0
        bulk_mapq_stats = 0
        bulk_baseq_stats = 0
        bulk_strand_stats = 0
        bulk_read12_stats = 0
        bulk_baseq_ref = []
        bulk_baseq_alt = []
        bulk_mismatch_flk_ref = []
        bulk_mismatch_flk_alt = []
        bulk_mismatch_flk_ref_mean = 0
        bulk_mismatch_flk_alt_mean = 0
        bulk_mismatch_flk_p = 0
        bulk_mismatch_flk_stats = 0
        bulk_mismatch_ref = []
        bulk_mismatch_alt = []
        bulk_mismatch_ref_mean = 0
        bulk_mismatch_alt_mean = 0
        bulk_mismatch_p = 0
        bulk_mismatch_stats = 0
        bulk_mismatch_ref_a = 0
        bulk_mismatch_alt_a = 0
        bulk_indel_ref = []
        bulk_indel_alt = []
        bulk_indel_ref_mean = 0
        bulk_indel_alt_mean = 0
        bulk_indel_p = 0
        bulk_indel_stats = 0
        bulk_indel_flk_ref = []
        bulk_indel_flk_alt = []
        bulk_indel_flk_ref_mean = 0
        bulk_indel_flk_alt_mean = 0
        bulk_indel_flk_p = 0
        bulk_indel_flk_stats = 0
        bulk_soft_clip_ref = 0
        bulk_soft_clip_alt = 0
        bulk_improp_ref_prop = 0
        bulk_improp_alt_prop = 0
        bulk_improp_p = 0
        bulk_improp_stats = 0
        bulk_sec_ref_prop = 0
        bulk_sec_alt_prop = 0
        bulk_sec_p = 0
        bulk_sec_stats = 0
        bulk_supp_ref_prop = 0
        bulk_supp_alt_prop = 0
        bulk_supp_p = 0
        bulk_supp_stats = 0
        sc_pos_p = 0
        sc_mapq_p = 0
        sc_baseq_p = 0
        sc_strand_p = 0
        sc_read12_p = 0
        sc_pos_stats = 0
        sc_mapq_stats = 0
        sc_baseq_stats = 0
        sc_strand_stats = 0
        sc_read12_stats = 0
        sc_baseq_ref = []
        sc_baseq_alt = []
        sc_mismatch_flk_ref = []
        sc_mismatch_flk_alt = []
        sc_mismatch_flk_ref_mean = 0
        sc_mismatch_flk_alt_mean = 0
        sc_mismatch_flk_p = 0
        sc_mismatch_flk_stats = 0
        sc_mismatch_ref = []
        sc_mismatch_alt = []
        sc_mismatch_ref_mean = 0
        sc_mismatch_alt_mean = 0
        sc_mismatch_p = 0
        sc_mismatch_stats = 0
        sc_mismatch_ref_a = 0
        sc_mismatch_alt_a = 0
        sc_indel_ref = []
        sc_indel_alt = []
        sc_indel_ref_mean = 0
        sc_indel_alt_mean = 0
        sc_indel_p = 0
        sc_indel_stats = 0
        sc_indel_flk_ref = []
        sc_indel_flk_alt = []
        sc_indel_flk_ref_mean = 0
        sc_indel_flk_alt_mean = 0
        sc_indel_flk_p = 0
        sc_indel_flk_stats = 0
        sc_soft_clip_ref = 0
        sc_soft_clip_alt = 0
        sc_improp_ref_prop = 0
        sc_improp_alt_prop = 0
        sc_improp_p = 0
        sc_improp_stats = 0
        sc_sec_ref_prop = 0
        sc_sec_alt_prop = 0
        sc_sec_p = 0
        sc_sec_stats = 0
        sc_supp_ref_prop = 0
        sc_supp_alt_prop = 0
        sc_supp_p = 0
        sc_supp_stats = 0
        ms_mut_size = 0
        is_ms_indel = 0
        j_ms_size = 0
        k_ms_size = 0
        prop_mut_cell_prob = 0
        prop_mut_cell_allele = 0
        mo_pop_freq = 0
        vaf_bulk = 0
        vaf_sc = 0
        vaf_all = 0
        vaf_mut_sc_mean_prob = 0
        vaf_mut_sc_max_prob = 0
        vaf_mut_sc_min_prob = 0
        vaf_nmut_sc_mean_prob = 0
        vaf_nmut_sc_max_prob = 0
        vaf_nmut_sc_min_prob = 0
        vaf_mut_sc_mean_allele = 0
        vaf_mut_sc_max_allele = 0
        vaf_mut_sc_min_allele = 0
        vaf_list_prob = []
        vaf_list_allele = []
        vaf_list_eq = []
        vaf_list_count = []
        vaf_list_count_eq = []
        vaf_list_spn_count = []
        ref_list_count_eq = []
        hanging_counts = []
        prop_exact = 0
        exact_p = 0
        exact_stats = 0
        num_ref_exact = 0
        num_alt_exact = 0
        num_ref_non_exact = 0
        num_alt_non_exact = 0
        num_hanging_ref = 0
        num_hanging_alt = 0
        num_hanging_ref_mut = 0
        num_hanging_alt_mut = 0
        prop_mismap = 0
        prop_mismap_ref = 0
        prop_mismap_alt = 0
        raw_soft_clip = 0
        raw_mapQ = 0
        mean_mut_sc_counts = 0
        max_mut_sc_counts = 0
        min_mut_sc_counts = 0
        mean_nmut_sc_counts = 0
        max_nmut_sc_counts = 0
        probs_mut_sc_mean = 0
        probs_mut_sc_max = 0
        probs_mut_sc_min = 0
        probs_nmut_sc_mean = 0
        probs_nmut_sc_max = 0
        probs_nmut_sc_min = 0
        germ_hom = -1
        avg_af = 0.5
        avg_af_mut = 0.5
        max_dev_af = 0.5
        max_dev_af_mut = 0.5
        ll_ratio = None  # 1e3  # 1e7
        ll_pval = None  # 100  # -np.log10(1e-16)
        dis_prop_bulk = 0
        dis_prop_sc = 0
        dis_prop_mut = 0
        dis_prop_j = 0
        dis_prop_k = 0
        dis_prop_k_avg = 0
        dis_amp = 0
        dis_amp_avg = 0
        phasing_s = 0.5
        phasing_b = 0
        phasing_j = 0
        phasing_k = 0
        prop_sc_spn_k_read = 0
        prop_bulk_spn_k_read = 0
        prop_all_spn_k_read = 0
        phased_k_vaf = 0
        p1 = 0.5
        phy = []
        gi_lfs = []
        gi_rfs = []
        gm_lfs = []
        gm_rfs = []
        germ_coords = []
        ref_mm_baseQ = []
        alt_mm_baseQ = []
        mm_baseQ_p = 0
        mm_baseQ_stats = 0
        # endregion init
        supp = {
            "af_list": model.af[idx_ind] if model.af and model.af[idx_ind] else None,
            "allele_flk_size": model.allele_flk_size,
            "alt_counts": None,
            "alt_mm_baseQ_dict": None,
            "bulk_mismatch_alt_a": None,
            "bulk_mismatch_ref_a": None,
            "dis_prop_j": None,
            "dis_prop_mut": None,
            "dp_list": None,
            "dp_list_norm": None,
            "gi_lfseq": None,
            "gi_rfseq": None,
            "gm_lfseq": None,
            "gm_rfseq": None,
            "gt_lls": None,
            "gt_mosaic": None,
            "gt_posts": None,
            "gts": None,
            "hSNP": model.hSNP[idx_ind] if model.hSNP else None,
            "hSNP_distance": (
                model.best_distances[idx_ind] if model.best_distances else None
            ),
            "hSNP_start": model.hSNP_start[idx_ind] if model.hSNP_start else None,
            "hanging_counts": None,
            "is_cmplx": None,
            "is_end_mm": None,
            "is_homo_mm_int": None,
            "is_indel_ms_region": None,
            "is_mm_ms_region": None,
            "is_ms_indel_seg": None,
            "is_mut_region": None,
            "is_refhom": None,
            "max_mut_sc_counts": None,
            "max_nmut_sc_counts": None,
            "mean_mut_sc_counts": None,
            "mean_nmut_sc_counts": None,
            "min_mut_sc_counts": None,
            "ms_mut_size_seg": None,
            "mu_mosaic": None,
            "mut_cell_list": None,
            "mut_cell_list_final": None,
            "mut_cell_prob": None,
            "mut_cell_prob_i": None,
            "mut_contxt": None,
            "mut_info": None,
            "mut_refalt": None,
            "mut_res_all": None,
            "mut_type": None,
            "num_hap_pool": model.num_hap_pool[idx_ind] if model.num_hap_pool else 0,
            "num_haps_phased": model.num_haps_phased[idx_ind]
            if model.num_haps_phased
            else 0,
            "num_mut_cell": 0,
            "num_mut_cell_dup": 0,
            "phasable_k": model.phasable[idx_ind] if model.phasable else False,
            "phasing_b": None,
            "phasing_block": None,
            "phasing_counts": None,
            "phasing_j": None,
            "phasing_k": None,
            "phasing_k_counts": None,
            "phasing_s": None,
            "prop_all_spn_k_read_c": None,
            "prop_all_spn_k_read_eq": None,
            "prop_bulk_spn_k_read_c": None,
            "prop_bulk_spn_k_read_eq": None,
            "prop_nonseg_reads": None,
            "prop_sc_spn_k_read_c": None,
            "prop_sc_spn_k_read_eq": None,
            "ref_allele": model.ref_allele,
            "ref_list_count_eq": None,
            "ref_mm_baseQ_dict": None,
            "sc_mismatch_alt_a": None,
            "sc_mismatch_ref_a": None,
            "sc_posteriors_final": None,
            "spn_list": model.num_spn[idx_ind] if model.num_spn else None,  # bulk + sc
            "spn_prop": None,
            "tot_bulk_dp": None,
            "tot_bulk_spn_dp": None,
            "tot_sc_dp": None,
            "tot_sc_spn_dp": None,
            "vaf_hanging": None,
            "vaf_hanging_mut": None,
            "vaf_list_allele": None,
            "vaf_list_count_eq": None,
            "vaf_list_eq": None,
            "vaf_list_prob": None,
            "vaf_list_spn_count": None,
        }
        # total depth
        tot_bulk_dp = len(reads_data_seq[idx_ind][0])
        tot_bulk_spn_dp = (
            len(list(tools.flatten_list(reads_data_spn_seq[idx_ind][0])))
            if reads_data_spn_seq and any(reads_data_spn_seq[idx_ind])
            else 0
        )
        tot_sc_dp = len(list(tools.flatten_list(reads_data_seq[idx_ind][1])))
        tot_sc_spn_dp = (
            len(list(tools.flatten_list(reads_data_spn_seq[idx_ind][1])))
            if reads_data_spn_seq and any(reads_data_spn_seq[idx_ind])
            else 0
        )
        dp_mut_p = 0
        dp_mut_stats = 0
        hanging_counts.append(sum(len(r) for r in raw_reads_list[idx_ind][0]))
        hanging_counts.extend([len(r) for r in raw_reads_list[idx_ind][1]])
        supp["prop_nonseg_reads"] = tools.get_nonseg_prop(reads_data_anchor)
        if "dp" in info.columns:
            ind_data = info.loc[info["individual"] == info.individual.unique()[idx_ind]]
            ind_bulk_data = ind_data.loc[ind_data["type"] == "bulk"]
            ind_sc_data = ind_data.loc[ind_data["type"] != "bulk"]
            if ind_bulk_data.empty:
                dp_bulk = 0
            else:
                dp_bulk = (
                    tot_bulk_dp
                    if ind_bulk_data.dp.isna().any()
                    else (
                        tot_bulk_dp / sum(ind_bulk_data.dp)
                        if sum(ind_bulk_data.dp)
                        else 0
                    )
                )
            if ind_sc_data.dp.isna().any():
                dp_sc = (
                    tot_sc_dp / model.num_sc_sample[idx_ind]
                    if model.num_sc_sample[idx_ind]
                    else 0
                )
            else:
                dp_sc = np.mean(
                    [
                        r / dp if dp else 0
                        for r, dp in zip(
                            [len(r) for r in reads_data_seq[idx_ind][1]],
                            ind_sc_data.dp,
                        )
                    ]
                )
            if ind_bulk_data.empty:
                dp_list_avg = [1] + list(ind_sc_data.dp)
            else:
                dp_list_avg = [sum(ind_bulk_data.dp)] + list(ind_sc_data.dp)
        else:
            dp_bulk = tot_bulk_dp
            dp_sc = (
                tot_sc_dp / model.num_sc_sample[idx_ind]
                if model.num_sc_sample[idx_ind]
                else 0
            )
        if tot_bulk_dp:
            prop_bulk_spn_read = tot_bulk_spn_dp / tot_bulk_dp
        else:
            prop_bulk_spn_read = 0
        if tot_sc_dp:
            prop_sc_spn_read = tot_sc_spn_dp / tot_sc_dp
        else:
            prop_sc_spn_read = 0
        if tot_bulk_dp + tot_sc_dp:
            prop_all_spn_read = (tot_bulk_spn_dp + tot_sc_spn_dp) / (
                tot_bulk_dp + tot_sc_dp
            )
        else:
            prop_all_spn_read = 0

        j_allele_interruption = (
            find_interruption(mu_j_allele[1], model.region["motif"])
            if mu_j_allele[1]
            else 0
        )
        haps = (
            (mu_k_allele, mu_j_allele)
            if mu_j_allele == ref_g_allele
            else (mu_k_allele, mu_j_allele, ref_g_allele)
        )
        # int(bool(len(mu_j_allele[1]) % ms_motif_len))
        germ_hom = int(len(set(model.gts[idx_ind][0])) == 1)
        supp["mu_mosaic"] = model.mu_mosaic[idx_ind]
        if None not in model.mu_mosaic[idx_ind]:  # NOTE: mosaic or het
            try:
                supp["mut_type"] = model.site_type[idx_ind]
                supp["gt_mosaic"] = gt_mosaic
                supp["is_refhom"] = (
                    int(model.gts[idx_ind][0] == (0, 0))
                    if model.gts and model.gts[idx_ind] and model.gts[idx_ind][0]
                    else None
                )
                if model.phasable and model.phasable[idx_ind]:
                    p1 = np.exp(model.phase_probs[idx_ind][gt_mosaic])

                if None not in gt_mosaic:
                    j_ms_size = len(mu_j_allele[1])
                    k_ms_size = len(mu_k_allele[1])
                    mut_1 = model.total_allele_pool.index(gt_mosaic[0])
                    mut_2 = model.total_allele_pool.index(gt_mosaic[1])
                    if model.gts[idx_ind]:
                        supp["mut_cell_prob_i"] = [
                            int(mut_1 in sc and mut_2 in sc)
                            for sc in model.gts[idx_ind][1:]
                        ]
                    ms_mut_size = len(mu_k_allele[1]) - len(mu_j_allele[1])
                    is_ms_indel = int(bool(ms_mut_size))
                    supp["ms_mut_size_seg"] = ms_mut_size
                    supp["is_ms_indel_seg"] = is_ms_indel
                    # region phy
                    phy.append(tools.conv(model.mo_posteriors[idx_ind]))
                    # sc_gt_probs = [
                    #     tools.conv(
                    #         np.exp(cell[model.genotypes[idx_ind].index(gt_mosaic)])
                    #     )
                    #     for cell in model.sample_posteriors[idx_ind]
                    # ]
                    sc_gt_probs = [
                        tools.conv(np.exp(cell[1]))
                        for cell in model.sc_posteriors_final[idx_ind]
                    ]
                    phy.append(str(sc_gt_probs))
                    # endregion
                else:
                    phy = [0, str([0] * model.num_sc_sample[idx_ind])]
                    supp["mut_cell_prob_i"] = [0] * (model.num_sc_sample[idx_ind])
                    sc_gt_probs = [0] * (model.num_sc_sample[idx_ind])

                # region get bulk ref/alt reads
                bulk_ref_reads = []
                bulk_alt_reads = []
                bulk_ref_reads_s = []
                bulk_alt_reads_s = []
                bulk_ref_pos = []
                bulk_alt_pos = []
                bulk_alt_spn_reads = []
                bulk_alt_spn_reads_eq = []
                vaf_bulk_alt = 0
                vaf_bulk_alt_eq = 0
                vaf_bulk_ref_eq = 0
                for idx_bulk_sample in range(len(filtered_reads_list[idx_ind][0])):
                    for idx_r, r in enumerate(
                        filtered_reads_list[idx_ind][0][idx_bulk_sample]
                    ):
                        anchor = reads_data_anchor[idx_ind][0][idx_bulk_sample][idx_r]
                        seg_allele = tools.segmentation(
                            r.query_alignment_sequence, anchor
                        )
                        seg_allele = tools.trim_flk(seg_allele, model.allele_flk_size)
                        seg_allele_qual = tools.segmentation(
                            r.query_alignment_qualities, anchor
                        )
                        seg_allele_qual = tools.trim_flk(
                            seg_allele_qual, model.allele_flk_size
                        )
                        if anchor and seg_allele and seg_allele_qual:
                            seg_allele_prob = tools.map_nested(
                                tools.phred_score_q, seg_allele_qual
                            )
                            assign = model.read_aln(
                                seg_allele, seg_allele_prob, haps, "bulk"
                            )
                            if assign:
                                bulk_ref_reads.append(r)
                                bulk_ref_pos.append(anchor[0])
                                if seg_allele_qual:
                                    bulk_baseq_ref.append(
                                        np.mean(tools.desegmentation(seg_allele_qual))
                                    )
                                gi_lf, gi_rf = tools.ext_bp(
                                    r.query_alignment_sequence,
                                    anchor[0] + 1 - model.allele_flk_size[0],
                                    anchor[1] - 1 + model.allele_flk_size[1],
                                )
                                gi_lfs.append(gi_lf)
                                gi_rfs.append(gi_rf)
                                if seg_allele in (mu_j_allele, ref_g_allele):
                                    num_ref_exact += 1
                                else:
                                    num_ref_non_exact += 1
                            else:
                                bulk_alt_reads.append(r)
                                bulk_alt_pos.append(anchor[0])
                                vaf_bulk_alt += 1
                                if seg_allele_qual:
                                    bulk_baseq_alt.append(
                                        np.mean(tools.desegmentation(seg_allele_qual))
                                    )
                                gm_lf, gm_rf = tools.ext_bp(
                                    r.query_alignment_sequence,
                                    anchor[0] + 1 - model.allele_flk_size[0],
                                    anchor[1] - 1 + model.allele_flk_size[1],
                                )
                                gm_lfs.append(gm_lf)
                                gm_rfs.append(gm_rf)
                                if seg_allele == mu_k_allele:
                                    num_alt_exact += 1
                                else:
                                    num_alt_non_exact += 1
                            if seg_allele in (mu_j_allele, ref_g_allele):
                                bulk_ref_reads_s.append(r)
                                vaf_bulk_ref_eq += 1
                            elif seg_allele == mu_k_allele:
                                bulk_alt_reads_s.append(r)
                                vaf_bulk_alt_eq += 1
                vaf_list_count_eq.append(vaf_bulk_alt_eq)
                ref_list_count_eq.append(vaf_bulk_ref_eq)
                altdp_list = []
                # altdp_list.append(f"{len(bulk_alt_reads)}/{tot_bulk_dp}")
                if tot_bulk_dp:
                    vaf_bulk = len(bulk_alt_reads) / tot_bulk_dp
                else:
                    vaf_bulk = 0
                # pos_p
                bulk_pos_stats, bulk_pos_p = my_ranksums(bulk_ref_pos, bulk_alt_pos)
                # spn reads
                if reads_data_spn_anchor and any(reads_data_spn_anchor[idx_ind]):
                    for idx_bulk_sample in range(len(spn_reads_list[idx_ind][0])):
                        for idx_r, r in enumerate(
                            spn_reads_list[idx_ind][0][idx_bulk_sample]
                        ):
                            anchor = reads_data_spn_anchor[idx_ind][0][idx_bulk_sample][
                                idx_r
                            ]
                            seg_allele = tools.segmentation(
                                r.query_alignment_sequence, anchor
                            )
                            seg_allele = tools.trim_flk(
                                seg_allele, model.allele_flk_size
                            )
                            seg_allele_qual = tools.segmentation(
                                r.query_alignment_qualities, anchor
                            )
                            seg_allele_qual = tools.trim_flk(
                                seg_allele_qual, model.allele_flk_size
                            )
                            if anchor and seg_allele and seg_allele_qual:
                                seg_allele_prob = tools.map_nested(
                                    tools.phred_score_q, seg_allele_qual
                                )
                                assign = model.read_aln(
                                    seg_allele[:3], seg_allele_prob, haps, "bulk"
                                )
                                if seg_allele[:3] == mu_k_allele:
                                    bulk_alt_spn_reads_eq.append(r)
                                if assign == 0:
                                    bulk_alt_spn_reads.append(r)
                    if bulk_alt_reads:
                        prop_bulk_spn_k_read = len(bulk_alt_spn_reads) / len(
                            bulk_alt_reads
                        )
                        supp["prop_bulk_spn_k_read_c"] = (
                            len(bulk_alt_spn_reads),
                            len(bulk_alt_reads),
                        )
                    if bulk_alt_reads_s:
                        supp["prop_bulk_spn_k_read_eq"] = len(
                            bulk_alt_spn_reads_eq
                        ) / len(bulk_alt_reads_s)
                # all soft_clip
                if raw_reads:
                    raw_soft_clip = sum(
                        [1 for r in raw_reads if r.get_cigar_stats()[0][4] >= 10]
                    ) / len(raw_reads)
                    raw_mapQ = np.mean([r.mapping_quality for r in raw_reads])
                # BaseQ
                bulk_baseq_stats, bulk_baseq_p = my_ranksums(
                    bulk_baseq_ref, bulk_baseq_alt, "greater"
                )
                # MapQ
                bulk_mapq_ref = [r.mapping_quality for r in bulk_ref_reads]
                bulk_mapq_alt = [r.mapping_quality for r in bulk_alt_reads]
                bulk_mapq_stats, bulk_mapq_p = my_ranksums(
                    bulk_mapq_ref, bulk_mapq_alt, "greater"
                )
                # read12 bias
                bulk_read_1_ref = sum([1 for r in bulk_ref_reads if r.is_read1])
                bulk_read_2_ref = sum([1 for r in bulk_ref_reads if r.is_read2])
                bulk_read_1_alt = sum([1 for r in bulk_alt_reads if r.is_read1])
                bulk_read_2_alt = sum([1 for r in bulk_alt_reads if r.is_read2])
                bulk_read12_stats, bulk_read12_p = my_fisher(
                    bulk_read_1_ref, bulk_read_2_ref, bulk_read_1_alt, bulk_read_2_alt
                )
                # strand bias
                bulk_fwd_ref = sum([1 for r in bulk_ref_reads if r.is_forward])
                bulk_rev_ref = sum([1 for r in bulk_ref_reads if r.is_reverse])
                bulk_fwd_alt = sum([1 for r in bulk_alt_reads if r.is_forward])
                bulk_rev_alt = sum([1 for r in bulk_alt_reads if r.is_reverse])
                bulk_strand_stats, bulk_strand_p = my_fisher(
                    bulk_fwd_ref, bulk_rev_ref, bulk_fwd_alt, bulk_rev_alt
                )
                # mismatches/indels
                bulk_err_ref = [
                    tools.calc_err_ms(
                        r,
                        model.region["start"] - 1 - model.allele_flk_size[0],
                        model.region["end"] + model.allele_flk_size[1],
                    )
                    for r in bulk_ref_reads
                ]
                bulk_err_alt = [
                    tools.calc_err_ms(
                        r,
                        model.region["start"] - 1 - model.allele_flk_size[0],
                        model.region["end"] + model.allele_flk_size[1],
                    )
                    for r in bulk_alt_reads
                ]
                if bulk_err_ref:
                    bulk_mismatch_ref, bulk_indel_ref = zip(*bulk_err_ref)
                    bulk_mismatch_ref_mean = np.mean(bulk_mismatch_ref)
                    bulk_indel_ref_mean = np.mean(bulk_indel_ref)
                if bulk_err_alt:
                    bulk_mismatch_alt, bulk_indel_alt = zip(*bulk_err_alt)
                    bulk_mismatch_alt_mean = np.mean(bulk_mismatch_alt)
                    bulk_indel_alt_mean = np.mean(bulk_indel_alt)
                bulk_mismatch_stats, bulk_mismatch_p = my_ranksums(
                    bulk_mismatch_ref, bulk_mismatch_alt, "less"
                )
                bulk_indel_stats, bulk_indel_p = my_ranksums(
                    bulk_indel_ref, bulk_indel_alt, "less"
                )
                bulk_err_flk_ref = [
                    tools.calc_err_flk(
                        r,
                        model.region["start"] - 1 - model.allele_flk_size[0],
                        model.region["start"] - 1,
                    )
                    for r in bulk_ref_reads
                ]
                bulk_err_flk_ref.extend(
                    [
                        tools.calc_err_flk(
                            r,
                            model.region["end"],
                            model.region["end"] + model.allele_flk_size[1],
                        )
                        for r in bulk_ref_reads
                    ]
                )
                bulk_err_flk_alt = [
                    tools.calc_err_flk(
                        r,
                        model.region["start"] - 1 - model.allele_flk_size[0],
                        model.region["start"] - 1,
                    )
                    for r in bulk_alt_reads
                ]
                bulk_err_flk_alt.extend(
                    [
                        tools.calc_err_flk(
                            r,
                            model.region["end"],
                            model.region["end"] + model.allele_flk_size[1],
                        )
                        for r in bulk_alt_reads
                    ]
                )
                if bulk_err_flk_ref:
                    bulk_mismatch_flk_ref, bulk_indel_flk_ref = zip(*bulk_err_flk_ref)
                    bulk_mismatch_flk_ref_mean = np.mean(bulk_mismatch_flk_ref)
                    bulk_indel_flk_ref_mean = np.mean(bulk_indel_flk_ref)
                if bulk_err_flk_alt:
                    bulk_mismatch_flk_alt, bulk_indel_flk_alt = zip(*bulk_err_flk_alt)
                    bulk_mismatch_flk_alt_mean = np.mean(bulk_mismatch_flk_alt)
                    bulk_indel_flk_alt_mean = np.mean(bulk_indel_flk_alt)
                bulk_mismatch_flk_stats, bulk_mismatch_flk_p = my_ranksums(
                    bulk_mismatch_flk_ref, bulk_mismatch_flk_alt, "less"
                )
                bulk_indel_flk_stats, bulk_indel_flk_p = my_ranksums(
                    bulk_indel_flk_ref, bulk_indel_flk_alt, "less"
                )
                # soft_clip
                if bulk_ref_reads:
                    bulk_soft_clip_ref = sum(
                        [1 for r in bulk_ref_reads if r.get_cigar_stats()[0][4] >= 10]
                    ) / len(bulk_ref_reads)
                    bulk_mismatch_ref_a = sum(
                        [
                            int(r.get_tag("NM")) / r.infer_query_length()
                            for r in bulk_ref_reads
                        ]
                    ) / len(bulk_ref_reads)
                if bulk_alt_reads:
                    bulk_soft_clip_alt = sum(
                        [1 for r in bulk_alt_reads if r.get_cigar_stats()[0][4] >= 10]
                    ) / len(bulk_alt_reads)
                    bulk_mismatch_alt_a = sum(
                        [
                            int(r.get_tag("NM")) / r.infer_query_length()
                            for r in bulk_alt_reads
                        ]
                    ) / len(bulk_alt_reads)
                # mapping
                if bulk_ref_reads:
                    bulk_prop_ref = len([r for r in bulk_ref_reads if r.is_proper_pair])
                    bulk_improp_ref = len(bulk_ref_reads) - bulk_prop_ref
                    bulk_improp_ref_prop = bulk_improp_ref / len(bulk_ref_reads)
                    bulk_sec_ref = len([r for r in bulk_ref_reads if r.is_secondary])
                    bulk_prim_ref = len(bulk_ref_reads) - bulk_sec_ref
                    bulk_sec_ref_prop = bulk_sec_ref / len(bulk_ref_reads)
                    bulk_supp_ref = len(
                        [r for r in bulk_ref_reads if r.is_supplementary]
                    )
                    bulk_notsupp_ref = len(bulk_ref_reads) - bulk_supp_ref
                    bulk_supp_ref_prop = bulk_supp_ref / len(bulk_ref_reads)
                if bulk_alt_reads:
                    bulk_prop_alt = len([r for r in bulk_alt_reads if r.is_proper_pair])
                    bulk_improp_alt = len(bulk_alt_reads) - bulk_prop_alt
                    bulk_improp_alt_prop = bulk_improp_alt / len(bulk_alt_reads)
                    bulk_sec_alt = len([r for r in bulk_alt_reads if r.is_secondary])
                    bulk_prim_alt = len(bulk_alt_reads) - bulk_sec_alt
                    bulk_sec_alt_prop = bulk_sec_alt / len(bulk_alt_reads)
                    bulk_supp_alt = len(
                        [r for r in bulk_alt_reads if r.is_supplementary]
                    )
                    bulk_notsupp_alt = len(bulk_alt_reads) - bulk_supp_alt
                    bulk_supp_alt_prop = bulk_supp_alt / len(bulk_alt_reads)
                if bulk_ref_reads and bulk_alt_reads:
                    bulk_improp_stats, bulk_improp_p = my_fisher(
                        bulk_prop_ref, bulk_improp_ref, bulk_prop_alt, bulk_improp_alt
                    )
                    bulk_sec_stats, bulk_sec_p = my_fisher(
                        bulk_prim_ref, bulk_sec_ref, bulk_prim_alt, bulk_sec_alt
                    )
                    bulk_supp_stats, bulk_supp_p = my_fisher(
                        bulk_supp_ref, bulk_notsupp_ref, bulk_supp_alt, bulk_notsupp_alt
                    )
                # endregion

                # region get sc ref/alt reads
                sc_ref_reads = []
                sc_alt_reads = []
                sc_ref_reads_s = []
                sc_alt_reads_s = []
                sc_ref_pos = []
                sc_alt_pos = []
                sc_alt_spn_reads = []
                sc_alt_spn_reads_eq = []
                supp["dp_list"] = [tot_bulk_dp] + [
                    len(filtered_reads_list[idx_ind][1][idx_sc_sample])
                    for idx_sc_sample in range(model.num_sc_sample[idx_ind])
                ]  # bulk + sc
                if "dp" in info.columns:
                    supp["dp_list_norm"] = [
                        dp / avg for dp, avg in zip(supp["dp_list"], dp_list_avg)
                    ]
                else:
                    supp["dp_list_norm"] = supp["dp_list"]
                supp["spn_prop"] = tuple(
                    s / t if t else 0 for t, s in zip(hanging_counts, supp["dp_list"])
                )
                for idx_sc_sample in range(model.num_sc_sample[idx_ind]):
                    vaf_sc_alt = 0
                    vaf_sc_alt_eq = 0
                    vaf_sc_ref_eq = 0
                    for idx_r, r in enumerate(
                        filtered_reads_list[idx_ind][1][idx_sc_sample]
                    ):
                        anchor = reads_data_anchor[idx_ind][1][idx_sc_sample][idx_r]
                        seg_allele = tools.segmentation(
                            r.query_alignment_sequence, anchor
                        )
                        seg_allele = tools.trim_flk(seg_allele, model.allele_flk_size)
                        seg_allele_qual = tools.segmentation(
                            r.query_alignment_qualities, anchor
                        )
                        seg_allele_qual = tools.trim_flk(
                            seg_allele_qual, model.allele_flk_size
                        )
                        if anchor and seg_allele and seg_allele_qual:
                            seg_allele_prob = tools.map_nested(
                                tools.phred_score_q, seg_allele_qual
                            )
                            assign = model.read_aln(
                                seg_allele,
                                seg_allele_prob,
                                haps,
                                model.sc_prot_list[idx_ind][idx_sc_sample],
                            )
                            if assign:
                                sc_ref_reads.append(r)
                                sc_ref_pos.append(anchor[0])
                                if seg_allele_qual:
                                    sc_baseq_ref.append(
                                        np.mean(tools.desegmentation(seg_allele_qual))
                                    )

                                gi_lf, gi_rf = tools.ext_bp(
                                    r.query_alignment_sequence,
                                    anchor[0] + 1 - model.allele_flk_size[0],
                                    anchor[1] - 1 + model.allele_flk_size[1],
                                )
                                gi_lfs.append(gi_lf)
                                gi_rfs.append(gi_rf)
                                if seg_allele in (mu_j_allele, ref_g_allele):
                                    num_ref_exact += 1
                                else:
                                    num_ref_non_exact += 1
                            else:
                                sc_alt_reads.append(r)
                                sc_alt_pos.append(anchor[0])
                                vaf_sc_alt += 1
                                if seg_allele_qual:
                                    sc_baseq_alt.append(
                                        np.mean(tools.desegmentation(seg_allele_qual))
                                    )

                                gm_lf, gm_rf = tools.ext_bp(
                                    r.query_alignment_sequence,
                                    anchor[0] + 1 - model.allele_flk_size[0],
                                    anchor[1] - 1 + model.allele_flk_size[1],
                                )
                                gm_lfs.append(gm_lf)
                                gm_rfs.append(gm_rf)
                                if seg_allele == mu_k_allele:
                                    num_alt_exact += 1
                                else:
                                    num_alt_non_exact += 1
                            if seg_allele in (mu_j_allele, ref_g_allele):
                                sc_ref_reads_s.append(r)
                                vaf_sc_ref_eq += 1
                            elif seg_allele == mu_k_allele:
                                sc_alt_reads_s.append(r)
                                vaf_sc_alt_eq += 1
                            if seg_allele == mu_j_allele:
                                germ_coords.append(
                                    r.reference_start
                                    + anchor[0]
                                    + 1
                                    - model.allele_flk_size[0]
                                    if r.reference_start
                                    else None
                                )
                    vaf_list_allele.append(
                        vaf_sc_alt / len(filtered_reads_list[idx_ind][1][idx_sc_sample])
                        if len(filtered_reads_list[idx_ind][1][idx_sc_sample])
                        else 0
                    )
                    vaf_list_count.append(vaf_sc_alt)
                    vaf_list_count_eq.append(vaf_sc_alt_eq)
                    ref_list_count_eq.append(vaf_sc_ref_eq)
                    if vaf_sc_alt:
                        prop_mut_cell_allele += 1
                    altdp_list.append(
                        f"{vaf_sc_alt}/{len(filtered_reads_list[idx_ind][1][idx_sc_sample])}"
                    )
                vaf_list_eq = [
                    m / d if d else 0
                    for m, d in zip(vaf_list_count_eq, supp["dp_list"])
                ]
                if tot_sc_dp:
                    vaf_sc = sum(vaf_list_count) / tot_sc_dp
                else:
                    vaf_sc = 0
                if tot_bulk_dp or tot_sc_dp:
                    vaf_all = (len(bulk_alt_reads) + sum(vaf_list_count)) / (
                        tot_bulk_dp + tot_sc_dp
                    )
                else:
                    vaf_all = 0
                # sum(k * v for k, v in d.items()) / sum(d.values())

                # region refine seg flk indel
                gi_lfseq = tools.get_ext(gi_lfs)
                gi_rfseq = tools.get_ext(gi_rfs)
                supp["gi_lfseq"] = gi_lfseq
                supp["gi_rfseq"] = gi_rfseq
                gm_lfseq = tools.get_ext(gm_lfs)
                gm_rfseq = tools.get_ext(gm_rfs)
                supp["gm_lfseq"] = gm_lfseq
                supp["gm_rfseq"] = gm_rfseq
                (
                    supp["mut_info"],
                    supp["mut_contxt"],
                    supp["mut_refalt"],
                    supp["mut_res_all"],
                ) = tools.analyze_aln(
                    (gi_lfseq,) + mu_j_allele + (gi_rfseq,),
                    (gm_lfseq,) + mu_k_allele + (gm_rfseq,),
                    model.region,
                    germ_coords,
                )
                if supp["mut_res_all"]:
                    supp["is_cmplx"] = int(len(supp["mut_res_all"]) > 1)
                if supp["mut_info"] and supp["mut_info"][2] == "mismatch":
                    ms_mut_size = 0
                    is_ms_indel = 0
                    if supp["mut_info"][0] in (
                        len(mu_j_allele[0]),
                        len(mu_j_allele[0]) + len(mu_j_allele[1]) + 1,
                    ):
                        supp["is_end_mm"] = 1
                    else:
                        supp["is_end_mm"] = 0
                    if (
                        model.allele_flk_size[0]
                        < supp["mut_info"][0]
                        <= len(mu_j_allele[1]) + model.allele_flk_size[1]
                    ):
                        supp["is_mm_ms_region"] = 1
                        supp["is_mut_region"] = 1
                        if model.region["motif_len"] == 1:
                            if supp["mut_info"][4] == model.region["motif"]:
                                supp["is_homo_mm_int"] = 1
                    else:
                        supp["is_mm_ms_region"] = 0
                        supp["is_mut_region"] = 0
                    for idx_bulk_sample in range(len(filtered_reads_list[idx_ind][0])):
                        for idx_r, r in enumerate(
                            filtered_reads_list[idx_ind][0][idx_bulk_sample]
                        ):
                            anchor = reads_data_anchor[idx_ind][0][idx_bulk_sample][
                                idx_r
                            ]
                            seg_allele = tools.segmentation(
                                r.query_alignment_sequence, anchor
                            )
                            seg_allele = tools.trim_flk(
                                seg_allele, model.allele_flk_size
                            )
                            seg_allele_qual = tools.segmentation(
                                r.query_alignment_qualities, anchor
                            )
                            seg_allele_qual = tools.trim_flk(
                                seg_allele_qual, model.allele_flk_size
                            )
                            try:
                                if anchor and seg_allele and seg_allele_qual:
                                    seg_allele_prob = tools.map_nested(
                                        tools.phred_score_q, seg_allele_qual
                                    )
                                    assign = model.read_aln(
                                        seg_allele, seg_allele_prob, haps, "bulk"
                                    )
                                    if assign:
                                        if seg_allele_qual:
                                            ref_mm_baseQ.append(
                                                tools.desegmentation(seg_allele_qual)[
                                                    supp["mut_info"][0]
                                                ]
                                            )
                                    else:
                                        if seg_allele_qual:
                                            alt_mm_baseQ.append(
                                                tools.desegmentation(seg_allele_qual)[
                                                    supp["mut_info"][5]
                                                ]
                                            )
                            except:
                                pass
                    for idx_sc_sample in range(model.num_sc_sample[idx_ind]):
                        for idx_r, r in enumerate(
                            filtered_reads_list[idx_ind][1][idx_sc_sample]
                        ):
                            anchor = reads_data_anchor[idx_ind][1][idx_sc_sample][idx_r]
                            seg_allele = tools.segmentation(
                                r.query_alignment_sequence, anchor
                            )
                            seg_allele = tools.trim_flk(
                                seg_allele, model.allele_flk_size
                            )
                            seg_allele_qual = tools.segmentation(
                                r.query_alignment_qualities, anchor
                            )
                            seg_allele_qual = tools.trim_flk(
                                seg_allele_qual, model.allele_flk_size
                            )
                            try:
                                if anchor and seg_allele and seg_allele_qual:
                                    seg_allele_prob = tools.map_nested(
                                        tools.phred_score_q, seg_allele_qual
                                    )
                                    assign = model.read_aln(
                                        seg_allele,
                                        seg_allele_prob,
                                        haps,
                                        model.sc_prot_list[idx_ind][idx_sc_sample],
                                    )
                                    if assign:
                                        if seg_allele_qual:
                                            ref_mm_baseQ.append(
                                                tools.desegmentation(seg_allele_qual)[
                                                    supp["mut_info"][0]
                                                ]
                                            )
                                    else:
                                        if seg_allele_qual:
                                            alt_mm_baseQ.append(
                                                tools.desegmentation(seg_allele_qual)[
                                                    supp["mut_info"][5]
                                                ]
                                            )
                            except:
                                pass
                elif supp["mut_info"] and supp["mut_info"][2] == "insertion":
                    ms_mut_size = supp["mut_info"][1]
                    is_ms_indel = 1
                    if (
                        model.allele_flk_size[0]
                        <= supp["mut_info"][0]
                        <= len(mu_j_allele[1]) + model.allele_flk_size[1]
                    ):
                        supp["is_indel_ms_region"] = 1
                        supp["is_mut_region"] = 1
                    else:
                        supp["is_indel_ms_region"] = 0
                        supp["is_mut_region"] = 0
                elif supp["mut_info"] and supp["mut_info"][2] == "deletion":
                    ms_mut_size = -supp["mut_info"][1]
                    is_ms_indel = 1
                    if (
                        model.allele_flk_size[0]
                        < supp["mut_info"][0]
                        <= len(mu_j_allele[1]) + model.allele_flk_size[1]
                    ):
                        supp["is_indel_ms_region"] = 1
                        supp["is_mut_region"] = 1
                    else:
                        supp["is_indel_ms_region"] = 0
                        supp["is_mut_region"] = 0
                supp["ref_mm_baseQ_dict"] = dict(Counter(ref_mm_baseQ))
                supp["alt_mm_baseQ_dict"] = dict(Counter(alt_mm_baseQ))
                mm_baseQ_stats, mm_baseQ_p = my_ranksums(
                    ref_mm_baseQ, alt_mm_baseQ, "greater"
                )
                ref_mm_baseQ = np.mean(ref_mm_baseQ) if ref_mm_baseQ else None
                alt_mm_baseQ = np.mean(alt_mm_baseQ) if alt_mm_baseQ else None
                if supp.get("mut_contxt", None):
                    contxt_l, contxt_r = supp["mut_contxt"]
                    if len(contxt_l) < 3 and gi_lfseq:
                        supp["mut_contxt"][0] = (
                            gi_lfseq[(len(contxt_l) - 3) :] + contxt_l
                        )
                    if len(contxt_r) < 3 and gi_rfseq:
                        supp["mut_contxt"][1] = contxt_r + gi_rfseq[: 3 - len(contxt_r)]
                # endregion refine seg flk indel
                mut_cell_prob = (
                    tuple(
                        x if y else 0
                        for x, y in zip(
                            model.mut_cell_list_final[idx_ind], vaf_list_allele
                        )
                    )
                    if model.mut_cell_list_final[idx_ind]
                    else model.mut_cell_list_final[idx_ind]
                )
                supp["mut_cell_prob"] = mut_cell_prob
                prop_mut_cell_prob = sum(mut_cell_prob) / len(mut_cell_prob)
                supp["alt_counts"] = [len(bulk_alt_reads)] + vaf_list_count  # bulk + sc
                vaf_list_prob = [
                    ele1 * ele2 for ele1, ele2 in zip(mut_cell_prob, vaf_list_allele)
                ]
                vaf_mut_prob_list = [
                    y for x, y in zip(mut_cell_prob, vaf_list_allele) if x and y
                ]
                vaf_nmut_prob_list = [
                    y for x, y in zip(mut_cell_prob, vaf_list_allele) if not (x and y)
                ]
                vaf_mut_sc_mean_prob = (
                    np.mean(vaf_mut_prob_list) if vaf_mut_prob_list else 0
                )
                vaf_mut_sc_max_prob = max(vaf_mut_prob_list) if vaf_mut_prob_list else 0
                vaf_mut_sc_min_prob = min(vaf_mut_prob_list) if vaf_mut_prob_list else 0
                vaf_nmut_sc_mean_prob = (
                    np.mean(vaf_nmut_prob_list) if vaf_nmut_prob_list else 0
                )
                vaf_nmut_sc_max_prob = (
                    max(vaf_nmut_prob_list) if vaf_nmut_prob_list else 0
                )
                vaf_nmut_sc_min_prob = (
                    min(vaf_nmut_prob_list) if vaf_nmut_prob_list else 0
                )
                supp["vaf_list_allele"] = vaf_list_allele  # at least one read sc
                supp["vaf_list_prob"] = vaf_list_prob  # prob and at least one read sc
                supp["mut_cell_list"] = [int(bool(v)) for v in vaf_list_prob]
                supp["num_mut_cell"] = sum(supp["mut_cell_list"])
                dp_mut_stats, dp_mut_p = my_ranksums(
                    [
                        x * y
                        for x, y in zip(supp["mut_cell_list"], supp["dp_list_norm"][1:])
                        if x
                    ],
                    [
                        int(1 - x) * y
                        for x, y in zip(supp["mut_cell_list"], supp["dp_list_norm"][1:])
                        if 1 - x
                    ],
                )
                if "note" in info.columns:
                    ind_data = info.loc[
                        info["individual"] == info.individual.unique()[idx_ind]
                    ]
                    ind_sc_data = ind_data.loc[ind_data["type"] != "bulk"].reset_index(
                        drop=True
                    )
                    ind_sc_data["muts"] = pd.Series(supp["mut_cell_list"])
                    ind_sc_data["note"] = ind_sc_data["note"].fillna(
                        ind_sc_data.index.to_series().astype(str)
                    )
                    ind_sc_data = ind_sc_data.groupby("note", group_keys=False).apply(
                        tools.process_group
                    )
                    final_muts_list = ind_sc_data["muts"].tolist()
                    supp["num_mut_cell_dup"] = sum(final_muts_list)
                vaf_mut_allele_list = [v for v in vaf_list_allele if v]
                vaf_mut_sc_mean_allele = (
                    np.mean(vaf_mut_allele_list) if vaf_mut_allele_list else 0
                )
                vaf_mut_sc_max_allele = (
                    max(vaf_mut_allele_list) if vaf_mut_allele_list else 0
                )
                vaf_mut_sc_min_allele = (
                    min(vaf_mut_allele_list) if vaf_mut_allele_list else 0
                )

                vaf_list_count_mut = [
                    v for v, c in zip(vaf_list_count, vaf_list_prob) if c
                ]
                vaf_list_count_nmut = [
                    v for v, c in zip(vaf_list_count, vaf_list_prob) if not c
                ]
                mean_mut_sc_counts = (
                    np.mean(vaf_list_count_mut) if vaf_list_count_mut else 0
                )
                max_mut_sc_counts = max(vaf_list_count_mut) if vaf_list_count_mut else 0
                min_mut_sc_counts = min(vaf_list_count_mut) if vaf_list_count_mut else 0
                mean_nmut_sc_counts = (
                    np.mean(vaf_list_count_nmut) if vaf_list_count_nmut else 0
                )
                max_nmut_sc_counts = (
                    max(vaf_list_count_nmut) if vaf_list_count_nmut else 0
                )

                sc_gt_probs_mut = [v for v, c in zip(sc_gt_probs, vaf_list_prob) if c]
                sc_gt_probs_nmut = [
                    v for v, c in zip(sc_gt_probs, vaf_list_prob) if not c
                ]
                probs_mut_sc_mean = np.mean(sc_gt_probs_mut) if sc_gt_probs_mut else 0
                probs_mut_sc_max = max(sc_gt_probs_mut) if sc_gt_probs_mut else 0
                probs_mut_sc_min = min(sc_gt_probs_mut) if sc_gt_probs_mut else 0
                probs_nmut_sc_mean = (
                    np.mean(sc_gt_probs_nmut) if sc_gt_probs_nmut else 0
                )
                probs_nmut_sc_max = max(sc_gt_probs_nmut) if sc_gt_probs_nmut else 0
                probs_nmut_sc_min = min(sc_gt_probs_nmut) if sc_gt_probs_nmut else 0
                if model.af and model.af[idx_ind]:
                    af_mut = [
                        x
                        for x, y in zip(model.af[idx_ind], vaf_list_prob)
                        if y and x is not None
                    ]
                    avg_af = np.mean([af for af in model.af[idx_ind] if af is not None])
                    dev_af_list = [
                        abs(x - 0.5) for x in model.af[idx_ind] if x is not None
                    ]
                    max_dev_af = max(dev_af_list) if dev_af_list else 0
                    avg_af_mut = np.mean(af_mut) if af_mut else 0
                    max_dev_af_mut = (
                        max((abs(x - 0.5) for x in af_mut)) if af_mut else 0
                    )
                # hanging reads ref/alt
                if supp["is_mut_region"]:
                    for idx_bulk_sample in range(
                        len(hanging_reads_data_seq[idx_ind][0])
                    ):
                        for idx_r, r in enumerate(
                            hanging_reads_data_seq[idx_ind][0][idx_bulk_sample]
                        ):
                            if r[1] == mu_k_allele[1]:
                                num_hanging_alt += 1
                            elif r[1] in (mu_j_allele[1], ref_g_allele[1]):
                                num_hanging_ref += 1
                    for idx_sc_sample in range(model.num_sc_sample[idx_ind]):
                        for idx_r, r in enumerate(
                            hanging_reads_data_seq[idx_ind][1][idx_sc_sample]
                        ):
                            if r[1] == mu_k_allele[1]:
                                num_hanging_alt += 1
                            elif r[1] in (mu_j_allele[1], ref_g_allele[1]):
                                num_hanging_ref += 1
                    for idx_sc_sample in range(model.num_sc_sample[idx_ind]):
                        if not mut_cell_prob[idx_sc_sample]:
                            continue
                        for idx_r, r in enumerate(
                            hanging_reads_data_seq[idx_ind][1][idx_sc_sample]
                        ):
                            if r[1] == mu_k_allele[1]:
                                num_hanging_alt_mut += 1
                            elif r[1] in (mu_j_allele[1], ref_g_allele[1]):
                                num_hanging_ref_mut += 1
                    supp["vaf_hanging"] = (
                        num_hanging_alt / (num_hanging_alt + num_hanging_ref)
                        if num_hanging_alt + num_hanging_ref
                        else 0
                    )
                    supp["vaf_hanging_mut"] = (
                        num_hanging_alt_mut
                        / (num_hanging_alt_mut + num_hanging_ref_mut)
                        if num_hanging_alt_mut + num_hanging_ref_mut
                        else 0
                    )
                # pos_p
                sc_pos_stats, sc_pos_p = my_ranksums(sc_ref_pos, sc_alt_pos)
                # spn reads
                if reads_data_spn_anchor and any(reads_data_spn_anchor[idx_ind]):
                    for idx_sc_sample in range(model.num_sc_sample[idx_ind]):
                        vaf_sc_spn_alt = 0
                        for idx_r, r in enumerate(
                            spn_reads_list[idx_ind][1][idx_sc_sample]
                        ):
                            anchor = reads_data_spn_anchor[idx_ind][1][idx_sc_sample][
                                idx_r
                            ]
                            seg_allele = tools.segmentation(
                                r.query_alignment_sequence, anchor
                            )
                            seg_allele = tools.trim_flk(
                                seg_allele, model.allele_flk_size
                            )
                            seg_allele_qual = tools.segmentation(
                                r.query_alignment_qualities, anchor
                            )
                            seg_allele_qual = tools.trim_flk(
                                seg_allele_qual, model.allele_flk_size
                            )
                            if anchor and seg_allele and seg_allele_qual:
                                seg_allele_prob = tools.map_nested(
                                    tools.phred_score_q, seg_allele_qual
                                )
                                assign = model.read_aln(
                                    seg_allele[:3],
                                    seg_allele_prob,
                                    haps,
                                    model.sc_prot_list[idx_ind][idx_sc_sample],
                                )
                                if seg_allele[:3] == mu_k_allele:
                                    sc_alt_spn_reads_eq.append(r)
                                if assign == 0:
                                    sc_alt_spn_reads.append(r)
                                    vaf_sc_spn_alt += 1
                        vaf_list_spn_count.append(vaf_sc_spn_alt)
                    if sc_alt_reads:
                        prop_sc_spn_k_read = len(sc_alt_spn_reads) / len(sc_alt_reads)
                        supp["prop_sc_spn_k_read_c"] = (
                            len(sc_alt_spn_reads),
                            len(sc_alt_reads),
                        )
                    if sc_alt_reads_s:
                        supp["prop_sc_spn_k_read_eq"] = len(sc_alt_spn_reads_eq) / len(
                            sc_alt_reads_s
                        )
                if bulk_alt_reads or sc_alt_reads:
                    prop_all_spn_k_read = (
                        len(bulk_alt_spn_reads) + len(sc_alt_spn_reads)
                    ) / (len(bulk_alt_reads) + len(sc_alt_reads))
                    supp["prop_all_spn_k_read_c"] = (
                        len(bulk_alt_spn_reads) + len(sc_alt_spn_reads),
                        len(bulk_alt_reads) + len(sc_alt_reads),
                    )
                if bulk_alt_reads_s or sc_alt_reads_s:
                    supp["prop_all_spn_k_read_eq"] = (
                        len(bulk_alt_spn_reads_eq) + len(sc_alt_spn_reads_eq)
                    ) / (len(bulk_alt_reads_s) + len(sc_alt_reads_s))
                supp["vaf_list_spn_count"] = (  # bulk + sc
                    [len(bulk_alt_spn_reads)] + vaf_list_spn_count
                    if vaf_list_spn_count
                    else None
                )
                # BaseQ
                sc_baseq_stats, sc_baseq_p = my_ranksums(
                    sc_baseq_ref, sc_baseq_alt, "greater"
                )
                # MapQ
                sc_mapq_ref = [r.mapping_quality for r in sc_ref_reads]
                sc_mapq_alt = [r.mapping_quality for r in sc_alt_reads]
                sc_mapq_stats, sc_mapq_p = my_ranksums(
                    sc_mapq_ref, sc_mapq_alt, "greater"
                )
                # read12 bias
                sc_read_1_ref = sum([1 for r in sc_ref_reads if r.is_read1])
                sc_read_2_ref = sum([1 for r in sc_ref_reads if r.is_read2])
                sc_read_1_alt = sum([1 for r in sc_alt_reads if r.is_read1])
                sc_read_2_alt = sum([1 for r in sc_alt_reads if r.is_read2])
                sc_read12_stats, sc_read12_p = my_fisher(
                    sc_read_1_ref, sc_read_2_ref, sc_read_1_alt, sc_read_2_alt
                )
                # strand bias
                sc_fwd_ref = sum([1 for r in sc_ref_reads if r.is_forward])
                sc_rev_ref = sum([1 for r in sc_ref_reads if r.is_reverse])
                sc_fwd_alt = sum([1 for r in sc_alt_reads if r.is_forward])
                sc_rev_alt = sum([1 for r in sc_alt_reads if r.is_reverse])
                sc_strand_stats, sc_strand_p = my_fisher(
                    sc_fwd_ref, sc_rev_ref, sc_fwd_alt, sc_rev_alt
                )
                # mismatch
                sc_err_ref = [
                    tools.calc_err_ms(
                        r,
                        model.region["start"] - 1 - model.allele_flk_size[0],
                        model.region["end"] + model.allele_flk_size[1],
                    )
                    for r in sc_ref_reads
                ]
                sc_err_alt = [
                    tools.calc_err_ms(
                        r,
                        model.region["start"] - 1 - model.allele_flk_size[0],
                        model.region["end"] + model.allele_flk_size[1],
                    )
                    for r in sc_alt_reads
                ]
                if sc_err_ref:
                    sc_mismatch_ref, sc_indel_ref = zip(*sc_err_ref)
                    sc_mismatch_ref_mean = np.mean(sc_mismatch_ref)
                    sc_indel_ref_mean = np.mean(sc_indel_ref)
                if sc_err_alt:
                    sc_mismatch_alt, sc_indel_alt = zip(*sc_err_alt)
                    sc_mismatch_alt_mean = np.mean(sc_mismatch_alt)
                    sc_indel_alt_mean = np.mean(sc_indel_alt)
                if sc_mismatch_ref:
                    sc_mismatch_ref_mean = np.mean(sc_mismatch_ref)
                if sc_mismatch_alt:
                    sc_mismatch_alt_mean = np.mean(sc_mismatch_alt)
                sc_mismatch_stats, sc_mismatch_p = my_ranksums(
                    sc_mismatch_ref, sc_mismatch_alt, "less"
                )
                sc_indel_stats, sc_indel_p = my_ranksums(
                    sc_indel_ref, sc_indel_alt, "less"
                )

                sc_err_flk_ref = [
                    tools.calc_err_flk(
                        r,
                        model.region["start"] - 1 - model.allele_flk_size[0],
                        model.region["start"] - 1,
                    )
                    for r in sc_ref_reads
                ]
                sc_err_flk_ref.extend(
                    [
                        tools.calc_err_flk(
                            r,
                            model.region["end"],
                            model.region["end"] + model.allele_flk_size[1],
                        )
                        for r in sc_ref_reads
                    ]
                )
                sc_err_flk_alt = [
                    tools.calc_err_flk(
                        r,
                        model.region["start"] - 1 - model.allele_flk_size[0],
                        model.region["start"] - 1,
                    )
                    for r in sc_alt_reads
                ]
                sc_err_flk_alt.extend(
                    [
                        tools.calc_err_flk(
                            r,
                            model.region["end"],
                            model.region["end"] + model.allele_flk_size[1],
                        )
                        for r in sc_alt_reads
                    ]
                )
                if sc_err_flk_ref:
                    sc_mismatch_flk_ref, sc_indel_flk_ref = zip(*sc_err_flk_ref)
                    sc_mismatch_flk_ref_mean = np.mean(sc_mismatch_flk_ref)
                    sc_indel_flk_ref_mean = np.mean(sc_indel_flk_ref)
                if sc_err_flk_alt:
                    sc_mismatch_flk_alt, sc_indel_flk_alt = zip(*sc_err_flk_alt)
                    sc_mismatch_flk_alt_mean = np.mean(sc_mismatch_flk_alt)
                    sc_indel_flk_alt_mean = np.mean(sc_indel_flk_alt)
                sc_mismatch_flk_stats, sc_mismatch_flk_p = my_ranksums(
                    sc_mismatch_flk_ref, sc_mismatch_flk_alt, "less"
                )
                sc_indel_flk_stats, sc_indel_flk_p = my_ranksums(
                    sc_indel_flk_ref, sc_indel_flk_alt, "less"
                )
                # soft_clip
                if sc_ref_reads:
                    sc_soft_clip_ref = sum(
                        [1 for r in sc_ref_reads if r.get_cigar_stats()[0][4] >= 10]
                    ) / len(sc_ref_reads)
                    sc_mismatch_ref_a = sum(
                        [
                            int(r.get_tag("NM")) / r.infer_query_length()
                            for r in sc_ref_reads
                        ]
                    ) / len(sc_ref_reads)
                if sc_alt_reads:
                    sc_soft_clip_alt = sum(
                        [1 for r in sc_alt_reads if r.get_cigar_stats()[0][4] >= 10]
                    ) / len(sc_alt_reads)
                    sc_mismatch_alt_a = sum(
                        [
                            int(r.get_tag("NM")) / r.infer_query_length()
                            for r in sc_alt_reads
                        ]
                    ) / len(sc_alt_reads)
                # mapping
                if sc_ref_reads:
                    sc_prop_ref = len([r for r in sc_ref_reads if r.is_proper_pair])
                    sc_improp_ref = len(sc_ref_reads) - sc_prop_ref
                    sc_improp_ref_prop = sc_improp_ref / len(sc_ref_reads)
                    sc_sec_ref = len([r for r in sc_ref_reads if r.is_secondary])
                    sc_prim_ref = len(sc_ref_reads) - sc_sec_ref
                    sc_sec_ref_prop = sc_sec_ref / len(sc_ref_reads)
                    sc_supp_ref = len([r for r in sc_ref_reads if r.is_supplementary])
                    sc_notsupp_ref = len(sc_ref_reads) - sc_supp_ref
                    sc_supp_ref_prop = sc_supp_ref / len(sc_ref_reads)
                if sc_alt_reads:
                    sc_prop_alt = len([r for r in sc_alt_reads if r.is_proper_pair])
                    sc_improp_alt = len(sc_alt_reads) - sc_prop_alt
                    sc_improp_alt_prop = sc_improp_alt / len(sc_alt_reads)
                    sc_sec_alt = len([r for r in sc_alt_reads if r.is_secondary])
                    sc_prim_alt = len(sc_alt_reads) - sc_sec_alt
                    sc_sec_alt_prop = sc_sec_alt / len(sc_alt_reads)
                    sc_supp_alt = len([r for r in sc_alt_reads if r.is_supplementary])
                    sc_notsupp_alt = len(sc_alt_reads) - sc_supp_alt
                    sc_supp_alt_prop = sc_supp_alt / len(sc_alt_reads)
                if sc_ref_reads and sc_alt_reads:
                    sc_improp_stats, sc_improp_p = my_fisher(
                        sc_prop_ref, sc_improp_ref, sc_prop_alt, sc_improp_alt
                    )
                    sc_sec_stats, sc_sec_p = my_fisher(
                        sc_prim_ref, sc_sec_ref, sc_prim_alt, sc_sec_alt
                    )
                    sc_supp_stats, sc_supp_p = my_fisher(
                        sc_supp_ref, sc_notsupp_ref, sc_supp_alt, sc_notsupp_alt
                    )
                # mismapping
                if sc_ref_reads + sc_alt_reads:
                    prop_mismap = sum(
                        [1 for r in sc_ref_reads + sc_alt_reads if not r.is_proper_pair]
                    ) / len(sc_ref_reads + sc_alt_reads)
                if sc_ref_reads:
                    prop_mismap_ref = sum(
                        [1 for r in sc_ref_reads if not r.is_proper_pair]
                    ) / len(sc_ref_reads)
                if sc_alt_reads:
                    prop_mismap_alt = sum(
                        [1 for r in sc_alt_reads if not r.is_proper_pair]
                    ) / len(sc_alt_reads)
                # endregion

                mo_pop_freq = model.freq.get(mu_k_allele, 0)
                if tot_bulk_dp + tot_sc_dp:
                    prop_exact = (
                        len(bulk_ref_reads_s)
                        + len(bulk_alt_reads_s)
                        + len(sc_ref_reads_s)
                        + len(sc_alt_reads_s)
                    ) / (tot_bulk_dp + tot_sc_dp)
                    exact_stats, exact_p = my_fisher(
                        num_ref_exact,
                        num_alt_exact,
                        num_ref_non_exact,
                        num_alt_non_exact,
                    )
                if tot_sc_spn_dp + tot_bulk_spn_dp:
                    phased_k_vaf = (len(sc_alt_spn_reads) + len(bulk_alt_spn_reads)) / (
                        tot_sc_spn_dp + tot_bulk_spn_dp
                    )
                else:
                    phased_k_vaf = 0

                phy = (
                    model.region["name"],
                    model.region["chr"],
                    model.region["start"],
                    model.region["end"],
                    mu_j_allele,
                    mu_k_allele,
                ) + tuple(phy)

                if (
                    model.phasable
                    and model.phasable[idx_ind]
                    and model.lr_phase
                    and model.lr_phase[idx_ind]
                ):
                    ll_ratio, ll_pval = model.lr_phase[idx_ind]
                if model.phasing_check and model.phasing_check[idx_ind]:
                    phasing_check_mut = [
                        model.phasing_check[idx_ind][idx + 1]
                        for idx, i in enumerate(mut_cell_prob)  # vaf_list_prob
                        if i and model.phasing_check[idx_ind][idx + 1]
                    ]
                    phasing_check_mut_all = [
                        sum(column) for column in zip(*phasing_check_mut)
                    ]
                    supp["phasing_block"] = phasing_check_mut_all
                    supp["phasing_counts"] = sum(phasing_check_mut_all)
                    supp["phasing_k_counts"] = (
                        (phasing_check_mut_all[1] + phasing_check_mut_all[3])
                        if phasing_check_mut_all
                        else 0
                    )
                    if model.dis_prop and model.dis_prop[idx_ind]:
                        dis_prop_bulk, dis_prop_sc = model.dis_prop[idx_ind]
                        if dis_prop_sc:
                            if (
                                phasing_check_mut_all
                                and model.con[idx_ind] == "j1k2"
                                and phasing_check_mut_all[3] + phasing_check_mut_all[0]
                            ):
                                dis_prop_mut = phasing_check_mut_all[3] / (
                                    phasing_check_mut_all[3] + phasing_check_mut_all[0]
                                )
                            elif (
                                phasing_check_mut_all
                                and model.con[idx_ind] == "j2k1"
                                and phasing_check_mut_all[1] + phasing_check_mut_all[2]
                            ):
                                dis_prop_mut = phasing_check_mut_all[1] / (
                                    phasing_check_mut_all[1] + phasing_check_mut_all[2]
                                )
                            else:
                                dis_prop_mut = 0
                        else:
                            dis_prop_mut = 0
                    if phasing_check_mut_all and model.con[idx_ind] == "j1k2":
                        dis_amp = (
                            phasing_check_mut_all[2]
                            / (phasing_check_mut_all[1] + phasing_check_mut_all[2])
                            if phasing_check_mut_all[1] + phasing_check_mut_all[2]
                            else None
                        )
                        dis_prop_j = (
                            phasing_check_mut_all[2]
                            / (phasing_check_mut_all[0] + phasing_check_mut_all[2])
                            if phasing_check_mut_all[0] + phasing_check_mut_all[2]
                            else 0
                        )
                        dis_prop_k = (
                            phasing_check_mut_all[3]
                            / (phasing_check_mut_all[1] + phasing_check_mut_all[3])
                            if phasing_check_mut_all[1] + phasing_check_mut_all[3]
                            else 0
                        )
                    elif phasing_check_mut_all and model.con[idx_ind] == "j2k1":
                        dis_amp = (
                            phasing_check_mut_all[0]
                            / (phasing_check_mut_all[0] + phasing_check_mut_all[3])
                            if phasing_check_mut_all[0] + phasing_check_mut_all[3]
                            else None
                        )
                        dis_prop_j = (
                            phasing_check_mut_all[0]
                            / (phasing_check_mut_all[0] + phasing_check_mut_all[2])
                            if phasing_check_mut_all[0] + phasing_check_mut_all[2]
                            else 0
                        )
                        dis_prop_k = (
                            phasing_check_mut_all[1]
                            / (phasing_check_mut_all[1] + phasing_check_mut_all[3])
                            if phasing_check_mut_all[1] + phasing_check_mut_all[3]
                            else 0
                        )
                    else:
                        dis_amp = None
                        dis_prop_j = None
                        dis_prop_k = None
                        supp["phasable_k"] = False
                    if phasing_check_mut_all and model.con[idx_ind]:
                        if model.con[idx_ind] == "j1k2":
                            dis_amp_list = [
                                cell[2] / (cell[1] + cell[2])
                                for cell in phasing_check_mut
                                if cell and cell[1] + cell[2]
                            ]
                            dis_prop_k_list = [
                                cell[3] / (cell[1] + cell[3])
                                for cell in phasing_check_mut
                                if cell and cell[1] + cell[3]
                            ]
                        elif model.con[idx_ind] == "j2k1":
                            dis_amp_list = [
                                cell[0] / (cell[0] + cell[3])
                                for cell in phasing_check_mut
                                if cell and cell[0] + cell[3]
                            ]
                            dis_prop_k_list = [
                                cell[1] / (cell[1] + cell[3])
                                for cell in phasing_check_mut
                                if cell and cell[1] + cell[3]
                            ]

                        dis_amp_avg = np.mean(dis_amp_list) if dis_amp_list else None
                        dis_prop_k_avg = (
                            np.mean(dis_prop_k_list) if dis_prop_k_list else None
                        )
                    else:
                        dis_amp_avg = 0
                        dis_prop_k_avg = 0
                        supp["phasable_k"] = False
                    if sum(phasing_check_mut_all):
                        phasing_s = sum(phasing_check_mut_all[:2]) / sum(
                            phasing_check_mut_all
                        )
                        if sum(phasing_check_mut_all[:2]) >= sum(
                            phasing_check_mut_all[2:]
                        ):
                            phasing_b = (
                                phasing_check_mut_all[0]
                                / (phasing_check_mut_all[0] + phasing_check_mut_all[1])
                                if phasing_check_mut_all[0] + phasing_check_mut_all[1]
                                else 0
                            )
                            phasing_j = (
                                phasing_check_mut_all[0]
                                / (phasing_check_mut_all[0] + phasing_check_mut_all[2])
                                if phasing_check_mut_all[0] + phasing_check_mut_all[2]
                                else 0.5
                            )
                            phasing_k = (
                                phasing_check_mut_all[1]
                                / (phasing_check_mut_all[1] + phasing_check_mut_all[3])
                                if phasing_check_mut_all[1] + phasing_check_mut_all[3]
                                else 0.5
                            )
                        else:
                            phasing_b = (
                                phasing_check_mut_all[2]
                                / (phasing_check_mut_all[2] + phasing_check_mut_all[3])
                                if phasing_check_mut_all[2] + phasing_check_mut_all[3]
                                else 0
                            )
                            phasing_j = (
                                phasing_check_mut_all[2]
                                / (phasing_check_mut_all[0] + phasing_check_mut_all[2])
                                if phasing_check_mut_all[0] + phasing_check_mut_all[2]
                                else 0.5
                            )
                            phasing_k = (
                                phasing_check_mut_all[3]
                                / (phasing_check_mut_all[1] + phasing_check_mut_all[3])
                                if phasing_check_mut_all[1] + phasing_check_mut_all[3]
                                else 0.5
                            )
                    else:
                        phasing_b = 0
                        phasing_j = 0.5
                        phasing_k = 0.5
                        phasing_s = 0.5
                        supp["phasable_k"] = False

            except Exception as e:
                tools.logger.exception("%s feat extract failed: %s", model.region, e)
                # breakpoint()

            # region BB
            try:
                mo_posteriors_bb = model.bb_train(
                    idx_ind, supp, reads_data_seq[idx_ind], reads_data_qual[idx_ind]
                )
            except Exception as e:
                mo_posteriors_bb = None
                model.bb_likelihoods_p.append([])
                model.prod_likelihoods.append([])
                model.mo_posterior_bbs_p.append(None)
                model.phy_llbb_p.append([])
                tools.logger.exception("%s Beta-binomial failed: %s", model.region, e)

            if model.phasable and model.phasable[idx_ind]:
                try:
                    mo_posteriors_bb_p = model.bb_train_p(
                        idx_ind, supp, reads_data_seq[idx_ind], reads_data_qual[idx_ind]
                    )
                except Exception as e:
                    mo_posteriors_bb_p = None
                    model.bb_likelihoods_p.append([])
                    model.prod_likelihoods_p.append([])
                    model.mo_posterior_bbs_p.append(None)
                    model.phy_llbb_p.append([])
                    tools.logger.exception(
                        "%s Beta-binomial w/ phased failed: %s", model.region, e
                    )
            else:
                mo_posteriors_bb_p = None
                model.bb_likelihoods_p.append([])
                model.prod_likelihoods_p.append([])
                model.mo_posterior_bbs_p.append(None)
                model.phy_llbb_p.append([])
            try:
                if model.phy_llbb_p and model.phy_llbb_p[idx_ind]:
                    phy = phy + tuple(str(v) for v in model.phy_llbb_p[idx_ind])
                elif model.phy_llbb and model.phy_llbb[idx_ind]:
                    phy = phy + tuple(str(v) for v in model.phy_llbb[idx_ind])
                else:
                    phy = phy + (
                        (
                            str(
                                [
                                    cell[
                                        model.genotypes_refine[idx_ind].index(
                                            gt_germline
                                        )
                                    ]
                                    for cell in model.sample_likelihoods_refine[
                                        idx_ind
                                    ][1:]
                                ]
                            ),
                            str(
                                [
                                    cell[
                                        model.genotypes_refine[idx_ind].index(gt_mosaic)
                                    ]
                                    for cell in model.sample_likelihoods_refine[
                                        idx_ind
                                    ][1:]
                                ]
                            ),
                            str(
                                [
                                    cell[
                                        model.genotypes_refine[idx_ind].index(gt_mosaic)
                                    ]
                                    for cell in model.sample_likelihoods_refine[
                                        idx_ind
                                    ][1:]
                                ]
                            ),
                        )
                    )
            except Exception as e:
                phy = phy + (
                    (
                        str([0] * model.num_sc_sample[idx_ind]),
                        str([0] * model.num_sc_sample[idx_ind]),
                        str([0] * model.num_sc_sample[idx_ind]),
                    )
                )
                tools.logger.exception("%s phy failed: %s", model.region, e)
            phy = phy + (str(altdp_list),)
            phy = phy + (gt_germline, gt_mosaic, f"{len(bulk_alt_reads)}/{tot_bulk_dp}")
            # endregion BB
        else:
            mo_posteriors_bb = None
            mo_posteriors_bb_p = None
            model.bb_likelihoods.append([])
            model.bb_likelihoods_p.append([])
            model.prod_likelihoods.append([])
            model.prod_likelihoods_p.append([])
            model.mo_posterior_bbs.append(None)
            model.mo_posterior_bbs_p.append(None)
            model.phy_llbb.append([])
            model.phy_llbb_p.append([])
        supp.update(
            zip(
                [
                    "mean_mut_sc_counts",
                    "max_mut_sc_counts",
                    "min_mut_sc_counts",
                    "mean_nmut_sc_counts",
                    "max_nmut_sc_counts",
                    "sc_mismatch_ref_a",
                    "sc_mismatch_alt_a",
                    "bulk_mismatch_ref_a",
                    "bulk_mismatch_alt_a",
                    "dis_prop_mut",
                    "phasing_s",
                    "phasing_b",
                    "phasing_j",
                    "phasing_k",
                    "dis_prop_j",
                    "tot_bulk_dp",
                    "tot_sc_dp",
                    "tot_bulk_spn_dp",
                    "tot_sc_spn_dp",
                    "sc_posteriors_final",
                    "mut_cell_list_final",
                    "gts",
                    "gt_lls",
                    "gt_posts",
                    "hanging_counts",
                    "vaf_list_count_eq",
                    "vaf_list_eq",
                    "ref_list_count_eq",
                ],
                [
                    mean_mut_sc_counts,
                    max_mut_sc_counts,
                    min_mut_sc_counts,
                    mean_nmut_sc_counts,
                    max_nmut_sc_counts,
                    sc_mismatch_ref_a,
                    sc_mismatch_alt_a,
                    bulk_mismatch_ref_a,
                    bulk_mismatch_alt_a,
                    dis_prop_mut,
                    phasing_s,
                    phasing_b,
                    phasing_j,
                    phasing_k,
                    dis_prop_j,
                    tot_bulk_dp,
                    tot_sc_dp,
                    tot_bulk_spn_dp,
                    tot_sc_spn_dp,
                    model.sc_posteriors_final[idx_ind],
                    supp["mut_cell_prob"],
                    model.gts[idx_ind],
                    model.gt_lls[idx_ind],
                    model.gt_posts[idx_ind],
                    hanging_counts,
                    vaf_list_count_eq,
                    vaf_list_eq,
                    ref_list_count_eq,
                ],
            )
        )

        keys = [
            "rho_bulk_in",
            "up_bulk_in",
            "down_bulk_in",
            "rho_bulk_out",
            "up_bulk_out",
            "down_bulk_out",
            "rho_sc_in",
            "up_sc_in",
            "down_sc_in",
            "rho_sc_out",
            "up_sc_out",
            "down_sc_out",
            "rho_mda_in",
            "up_mda_in",
            "down_mda_in",
            "rho_mda_out",
            "up_mda_out",
            "down_mda_out",
            "rho_pta_in",
            "up_pta_in",
            "down_pta_in",
            "rho_pta_out",
            "up_pta_out",
            "down_pta_out",
            "rho_scc_in",
            "up_scc_in",
            "down_scc_in",
            "rho_scc_out",
            "up_scc_out",
            "down_scc_out",
            "dp_bulk",
            "dp_sc",
            "dp_mut_p",
            "dp_mut_stats",
            "num_allele_pool",
            "ms_motif_len",
            "mappability",
            "j_ms_size",
            "k_ms_size",
            "ms_mut_size",
            "ms_mut_size_div_len",
            "ms_type",
            "is_ms_indel",
            "is_truncated_mut",
            "j_allele_interruption",
            "gc_content",
            "vaf_bulk",
            "vaf_sc",
            "vaf_all",
            "vaf_mut_sc_mean_prob",
            "vaf_mut_sc_max_prob",
            "vaf_mut_sc_min_prob",
            "vaf_nmut_sc_mean_prob",
            "vaf_nmut_sc_max_prob",
            "vaf_nmut_sc_min_prob",
            "vaf_mut_sc_mean_allele",
            "vaf_mut_sc_max_allele",
            "vaf_mut_sc_min_allele",
            "probs_mut_sc_mean",
            "probs_mut_sc_max",
            "probs_mut_sc_min",
            "probs_nmut_sc_mean",
            "probs_nmut_sc_max",
            "probs_nmut_sc_min",
            "mu_jk",
            "mo_pop_freq",
            "raw_mapQ",
            "raw_soft_clip",
            "prop_exact",
            "exact_p",
            "exact_stats",
            "prop_mismap",
            "prop_mismap_ref",
            "prop_mismap_alt",
            "prop_mut_cell_prob",
            "prop_mut_cell_allele",
            "germline_ratio",
            "mo_posterior",
            "sc_pos_p",
            "sc_pos_stats",
            "sc_mapq_p",
            "sc_mapq_stats",
            "sc_baseq_p",
            "sc_baseq_stats",
            "sc_strand_p",
            "sc_strand_stats",
            "sc_read12_p",
            "sc_read12_stats",
            "sc_mismatch_ref_mean",
            "sc_mismatch_alt_mean",
            "sc_mismatch_p",
            "sc_mismatch_stats",
            "sc_mismatch_flk_ref_mean",
            "sc_mismatch_flk_alt_mean",
            "sc_mismatch_flk_p",
            "sc_mismatch_flk_stats",
            "sc_indel_ref_mean",
            "sc_indel_alt_mean",
            "sc_indel_p",
            "sc_indel_stats",
            "sc_indel_flk_ref_mean",
            "sc_indel_flk_alt_mean",
            "sc_indel_flk_p",
            "sc_indel_flk_stats",
            "sc_soft_clip_ref",
            "sc_soft_clip_alt",
            "sc_improp_ref_prop",
            "sc_improp_alt_prop",
            "sc_improp_p",
            "sc_improp_stats",
            "sc_sec_ref_prop",
            "sc_sec_alt_prop",
            "sc_sec_p",
            "sc_sec_stats",
            "sc_supp_ref_prop",
            "sc_supp_alt_prop",
            "sc_supp_p",
            "sc_supp_stats",
            "bulk_pos_p",
            "bulk_pos_stats",
            "bulk_mapq_p",
            "bulk_mapq_stats",
            "bulk_baseq_p",
            "bulk_baseq_stats",
            "bulk_strand_p",
            "bulk_strand_stats",
            "bulk_read12_p",
            "bulk_read12_stats",
            "bulk_mismatch_ref_mean",
            "bulk_mismatch_alt_mean",
            "bulk_mismatch_p",
            "bulk_mismatch_stats",
            "bulk_mismatch_flk_ref_mean",
            "bulk_mismatch_flk_alt_mean",
            "bulk_mismatch_flk_p",
            "bulk_mismatch_flk_stats",
            "bulk_indel_flk_ref_mean",
            "bulk_indel_flk_alt_mean",
            "bulk_indel_flk_p",
            "bulk_indel_flk_stats",
            "bulk_indel_ref_mean",
            "bulk_indel_alt_mean",
            "bulk_indel_p",
            "bulk_indel_stats",
            "bulk_soft_clip_ref",
            "bulk_soft_clip_alt",
            "bulk_improp_ref_prop",
            "bulk_improp_alt_prop",
            "bulk_improp_p",
            "bulk_improp_stats",
            "bulk_sec_ref_prop",
            "bulk_sec_alt_prop",
            "bulk_sec_p",
            "bulk_sec_stats",
            "bulk_supp_ref_prop",
            "bulk_supp_alt_prop",
            "bulk_supp_p",
            "bulk_supp_stats",
            "is_germ_hom",
            "ref_mm_baseQ",
            "alt_mm_baseQ",
            "end_mm_p",
            "mm_baseQ_stats",
            "mo_posteriors_bb",
            "mo_posteriors_bb_p",
            "num_haps_phased",
            "p1",
            "ll_ratio",
            "ll_pval",
            "dis_prop_bulk",
            "dis_prop_sc",
            "dis_amp",
            "dis_amp_avg",
            "dis_prop_k",
            "dis_prop_k_avg",
            "prop_all_spn_read",
            "prop_bulk_spn_read",
            "prop_sc_spn_read",
            "prop_all_spn_k_read",
            "prop_bulk_spn_k_read",
            "prop_sc_spn_k_read",
            "phased_k_vaf",
            "avg_af",
            "avg_af_mut",
            "max_dev_af",
            "max_dev_af_mut",
        ]
        vals = [
            model.stutter.get("rho_bulk_in", 0.9),
            model.stutter.get("up_bulk_in", 0.01),
            model.stutter.get("down_bulk_in", 0.01),
            model.stutter.get("rho_bulk_out", 0.9),
            model.stutter.get("up_bulk_out", 0.01),
            model.stutter.get("down_bulk_out", 0.01),
            model.stutter.get("rho_sc_in", 0.9),
            model.stutter.get("up_sc_in", 0.01),
            model.stutter.get("down_sc_in", 0.01),
            model.stutter.get("rho_sc_out", 0.9),
            model.stutter.get("up_sc_out", 0.01),
            model.stutter.get("down_sc_out", 0.01),
            model.stutter.get("rho_mda_in", 0.9),
            model.stutter.get("up_mda_in", 0.01),
            model.stutter.get("down_mda_in", 0.01),
            model.stutter.get("rho_mda_out", 0.9),
            model.stutter.get("up_mda_out", 0.01),
            model.stutter.get("down_mda_out", 0.01),
            model.stutter.get("rho_pta_in", 0.9),
            model.stutter.get("up_pta_in", 0.01),
            model.stutter.get("down_pta_in", 0.01),
            model.stutter.get("rho_pta_out", 0.9),
            model.stutter.get("up_pta_out", 0.01),
            model.stutter.get("down_pta_out", 0.01),
            model.stutter.get("rho_scc_in", 0.9),
            model.stutter.get("up_scc_in", 0.01),
            model.stutter.get("down_scc_in", 0.01),
            model.stutter.get("rho_scc_out", 0.9),
            model.stutter.get("up_scc_out", 0.01),
            model.stutter.get("down_scc_out", 0.01),
            dp_bulk,
            dp_sc,
            dp_mut_p,
            dp_mut_stats,
            model.num_allele_pool[idx_ind],
            ms_motif_len,
            mappability,
            j_ms_size,
            k_ms_size,
            ms_mut_size,
            ms_mut_size / ms_motif_len,
            1 if ms_mut_size > 0 else 0 if ms_mut_size == 0 else -1,
            is_ms_indel,
            int(bool(ms_mut_size % ms_motif_len)),
            j_allele_interruption,
            gc_content,
            vaf_bulk,
            vaf_sc,
            vaf_all,
            vaf_mut_sc_mean_prob,
            vaf_mut_sc_max_prob,
            vaf_mut_sc_min_prob,
            vaf_nmut_sc_mean_prob,
            vaf_nmut_sc_max_prob,
            vaf_nmut_sc_min_prob,
            vaf_mut_sc_mean_allele,
            vaf_mut_sc_max_allele,
            vaf_mut_sc_min_allele,
            probs_mut_sc_mean,
            probs_mut_sc_max,
            probs_mut_sc_min,
            probs_nmut_sc_mean,
            probs_nmut_sc_max,
            probs_nmut_sc_min,
            mu_jk if mu_jk else 0,
            mo_pop_freq,
            raw_mapQ,
            raw_soft_clip,
            prop_exact,
            exact_p,
            exact_stats,
            prop_mismap,
            prop_mismap_ref,
            prop_mismap_alt,
            prop_mut_cell_prob,
            (
                prop_mut_cell_allele / model.num_sc_sample[idx_ind]
                if model.num_sc_sample[idx_ind]
                else 0
            ),
            germline_ratio,
            mo_posterior,
            sc_pos_p,
            sc_pos_stats,
            sc_mapq_p,
            sc_mapq_stats,
            sc_baseq_p,
            sc_baseq_stats,
            sc_strand_p,
            sc_strand_stats,
            sc_read12_p,
            sc_read12_stats,
            sc_mismatch_ref_mean,
            sc_mismatch_alt_mean,
            sc_mismatch_p,
            sc_mismatch_stats,
            sc_mismatch_flk_ref_mean,
            sc_mismatch_flk_alt_mean,
            sc_mismatch_flk_p,
            sc_mismatch_flk_stats,
            sc_indel_ref_mean,
            sc_indel_alt_mean,
            sc_indel_p,
            sc_indel_stats,
            sc_indel_flk_ref_mean,
            sc_indel_flk_alt_mean,
            sc_indel_flk_p,
            sc_indel_flk_stats,
            sc_soft_clip_ref,
            sc_soft_clip_alt,
            sc_improp_ref_prop,
            sc_improp_alt_prop,
            sc_improp_p,
            sc_improp_stats,
            sc_sec_ref_prop,
            sc_sec_alt_prop,
            sc_sec_p,
            sc_sec_stats,
            sc_supp_ref_prop,
            sc_supp_alt_prop,
            sc_supp_p,
            sc_supp_stats,
            bulk_pos_p,
            bulk_pos_stats,
            bulk_mapq_p,
            bulk_mapq_stats,
            bulk_baseq_p,
            bulk_baseq_stats,
            bulk_strand_p,
            bulk_strand_stats,
            bulk_read12_p,
            bulk_read12_stats,
            bulk_mismatch_ref_mean,
            bulk_mismatch_alt_mean,
            bulk_mismatch_p,
            bulk_mismatch_stats,
            bulk_mismatch_flk_ref_mean,
            bulk_mismatch_flk_alt_mean,
            bulk_mismatch_flk_p,
            bulk_mismatch_flk_stats,
            bulk_indel_flk_ref_mean,
            bulk_indel_flk_alt_mean,
            bulk_indel_flk_p,
            bulk_indel_flk_stats,
            bulk_indel_ref_mean,
            bulk_indel_alt_mean,
            bulk_indel_p,
            bulk_indel_stats,
            bulk_soft_clip_ref,
            bulk_soft_clip_alt,
            bulk_improp_ref_prop,
            bulk_improp_alt_prop,
            bulk_improp_p,
            bulk_improp_stats,
            bulk_sec_ref_prop,
            bulk_sec_alt_prop,
            bulk_sec_p,
            bulk_sec_stats,
            bulk_supp_ref_prop,
            bulk_supp_alt_prop,
            bulk_supp_p,
            bulk_supp_stats,
            germ_hom,
            ref_mm_baseQ if ref_mm_baseQ != [] else None,
            alt_mm_baseQ if alt_mm_baseQ != [] else None,
            mm_baseQ_p,
            mm_baseQ_stats,
            mo_posteriors_bb,
            mo_posteriors_bb_p,
            model.num_haps_phased[idx_ind] if model.num_haps_phased else 0,
            p1,
            ll_ratio,
            None if ll_pval is None else 100 if ll_pval <= 0 else -np.log10(ll_pval),
            dis_prop_bulk,
            dis_prop_sc,
            dis_amp,
            dis_amp_avg,
            dis_prop_k,
            dis_prop_k_avg,
            prop_all_spn_read,
            prop_bulk_spn_read,
            prop_sc_spn_read,
            prop_all_spn_k_read,
            prop_bulk_spn_k_read,
            prop_sc_spn_k_read,
            phased_k_vaf,
            avg_af,
            avg_af_mut,
            max_dev_af,
            max_dev_af_mut,
        ]

        model.supp.append(
            dict(
                sorted(
                    {
                        k: tuple(v) if isinstance(v, list) else v
                        for k, v in supp.items()
                    }.items()
                )
            )
        )
        model.phy.append(phy)
        feat.append(dict(zip(keys, vals)))
        if args.debug:
            # tools.console.print(feat)
            # tools.console.print(supp)
            breakpoint()
    return feat, hap_freq


def my_ranksums(a, b, alternative="two-sided"):
    if a and b:
        statistic, pval = ranksums(a, b, alternative)
        return statistic if statistic < 1e7 else 1e7, (
            -np.log10(pval) if pval > 1e-16 else -np.log10(1e-16)
        )
    else:
        return 0, 0


def my_fisher(a, b, c, d):
    try:
        statistic, pval = fisher_exact([[a, b], [c, d]])
        return (
            statistic if statistic < 1e7 else 1e7 if not math.isnan(statistic) else 0
        ), (-np.log10(pval) if pval > 1e-16 else -np.log10(1e-16))
    except Exception as e:
        tools.logger.warning("%s Fisher's failed: %s", (a, b, c, d), e)
        return 0, 1
