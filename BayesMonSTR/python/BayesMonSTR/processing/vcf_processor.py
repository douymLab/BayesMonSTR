# -*- coding: utf-8 -*-

import csv
import os
import sys
from datetime import date

import pysam
import scipy.stats as stats

from BayesMonSTR import __VERSION__

from ..processing import bam_processor
from ..utils import FLK_BP_SEG


def generate_vcf(args, info, results):
    """Generate VCF file

    Parameters
    ----------
    args : `argparse.Namespace`
        Input arguments.
    sample_names : `list`
        Sample names.
    results : `list` of `tuple`
        Variant calling information.
    """
    metadata = (
        "##fileformat=VCFv4.2"
        + os.linesep
        + "##fileDate="
        + date.today().strftime("%Y%m%d")
        + os.linesep
        + "##source=BayesMonSTRV"
        + __VERSION__
        + os.linesep
        + "##command="
        + " ".join(sys.argv)
        + os.linesep
        + "##reference="
        + args.ref
        # + os.linesep
        # + "##contig=<ID=chr1,length=248956422>"  # to-do
        + os.linesep
        + r"""##phasing=partial
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=DPSC,Number=1,Type=Integer,Description="Total Depth For Single Cell Samples">
##INFO=<ID=PR,Number=0,Type=Flag,Description="Provisional reference allele, may not be based on real reference genome">
##INFO=<ID=RHO_BULK_IN,Number=1,Type=Float,Description="Stutter Error Model for Inframe Bulk Data">
##INFO=<ID=UP_BULK_IN,Number=1,Type=Float,Description="Stutter Error Model for Inframe Bulk Data">
##INFO=<ID=DOWN_BULK_IN,Number=1,Type=Float,Description="Stutter Error Model for Inframe Bulk Data">
##INFO=<ID=RHO_BULK_OUT,Number=1,Type=Float,Description="Stutter Error Model for Outframe Bulk Data">
##INFO=<ID=UP_BULK_OUT,Number=1,Type=Float,Description="Stutter Error Model for Outframe Bulk Data">
##INFO=<ID=DOWN_BULK_OUT,Number=1,Type=Float,Description="Stutter Error Model for Outframe Bulk Data">
##INFO=<ID=MA,Number=.,Type=String,Description="Mosaic Alleles">
##INFO=<ID=MC,Number=.,Type=String,Description="Mosaic Allele Length Size Changes">
##INFO=<ID=TYPE,Number=.,Type=String,Description="Mutation Type">
##INFO=<ID=MP,Number=.,Type=Float,Description="Mosaic Posteriors">
##INFO=<ID=MU,Number=.,Type=Float,Description="Mosaic Mutation Rate">
##FILTER=<ID=ML0,Description="Candidate Mosaic MS mutaion failed by the ML model">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=L,Number=1,Type=Float,Description="Genotype Likelihoods">
##FORMAT=<ID=P,Number=1,Type=Float,Description="Genotype Posteriors">
"""
    )
    n = len(info.individual.unique())
    sample_names = []
    for ind in info["individual"].unique():
        sample_names.append(ind + "_" + "Bulk")
        df = info.loc[(info["individual"] == ind) & (info["type"] != "bulk")]
        sample_names.extend(
            [df["individual"][idx] + "_" + df["cell"][idx] for idx in df.index]
        )
    header = (
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	"
        + "\t".join(sample_names)
        + os.linesep
    )
    feat = (
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
    )
    supp = (
        "af_list",
        "allele_flk_size",
        "alt_counts",
        "alt_mm_baseQ_dict",
        "bulk_mismatch_alt_a",
        "bulk_mismatch_ref_a",
        "dis_prop_j",
        "dis_prop_mut",
        "dp_list",
        "dp_list_norm",
        "gi_lfseq",
        "gi_rfseq",
        "gm_lfseq",
        "gm_rfseq",
        "gt_lls",
        "gt_mosaic",
        "gt_posts",
        "gts",
        "hSNP",
        "hSNP_distance",
        "hSNP_start",
        "hanging_counts",
        "is_cmplx",
        "is_end_mm",
        "is_homo_mm_int",
        "is_indel_ms_region",
        "is_mm_ms_region",
        "is_ms_indel_seg",
        "is_mut_region",
        "is_refhom",
        "max_mut_sc_counts",
        "max_nmut_sc_counts",
        "mean_mut_sc_counts",
        "mean_nmut_sc_counts",
        "min_mut_sc_counts",
        "ms_mut_size_seg",
        "mu_mosaic",
        "mut_cell_list",
        "mut_cell_list_final",
        "mut_cell_prob",
        "mut_cell_prob_i",
        "mut_contxt",
        "mut_info",
        "mut_refalt",
        "mut_res_all",
        "mut_type",
        "num_hap_pool_c",
        "num_haps_phased_c",
        "num_mut_cell",
        "num_mut_cell_dup",
        "phasable_k",
        "phasing_b",
        "phasing_block",
        "phasing_counts",
        "phasing_j",
        "phasing_k",
        "phasing_k_counts",
        "phasing_s",
        "prop_all_spn_k_read_c",
        "prop_all_spn_k_read_eq",
        "prop_bulk_spn_k_read_c",
        "prop_bulk_spn_k_read_eq",
        "prop_nonseg_reads",
        "prop_sc_spn_k_read_c",
        "prop_sc_spn_k_read_eq",
        "ref_allele",
        "ref_list_count_eq",
        "ref_mm_baseQ_dict",
        "sc_mismatch_alt_a",
        "sc_mismatch_ref_a",
        "sc_posteriors_final",
        "spn_list",
        "spn_prop",
        "tot_bulk_dp",
        "tot_bulk_spn_dp",
        "tot_sc_dp",
        "tot_sc_spn_dp",
        "vaf_hanging",
        "vaf_hanging_mut",
        "vaf_list_allele",
        "vaf_list_count_eq",
        "vaf_list_eq",
        "vaf_list_prob",
        "vaf_list_spn_count",
    )
    csv_info = (
        [
            "chr",
            "pos",
            "id",
            "ref",
            "alt",
            "qual",
            "filter",
            "info",
            "format",
        ]
        + sample_names
        + [f"phasable_{i}" for i in range(1, n + 1)]
        + [f"phasing_check_{i}" for i in range(1, n + 1)]
        + [v + "_" + str(i) for i in range(1, n + 1) for v in feat]
        + [f"con_{i}" for i in range(1, n + 1)]
        + [v + "_" + str(i) for i in range(1, n + 1) for v in supp]
        + [
            "region_dict",
            "chr_region",
            "start",
            "end",
            "motif_len",
            "len",
            "name",
            "motif",
            "mappability",
            "hap_freq",
        ]
        + [f"lenCounts_{i}" for i in range(1, n + 1)]
        + [f"hapCounts_{i}" for i in range(1, n + 1)]
        + [f"phy_{i}" for i in range(1, n + 1)]
    )

    dir_vcf = os.path.dirname(args.vcf)
    if dir_vcf:
        os.makedirs(dir_vcf, exist_ok=True)
    with open(args.vcf + ".vcf", "wt", encoding="utf-8") as f:
        f.write(metadata)
        f.write(header)
        csv_out = csv.writer(f, delimiter="\t")
        csv_out.writerow(csv_info)
        csv_out.writerows([res for res in results if res])
    if args.unphase:
        with open(args.unphase + ".vcf", "wt", encoding="utf-8") as f:
            f.write(metadata)
            f.write(header)
            csv_out = csv.writer(f, delimiter="\t")
            csv_out.writerow(csv_info)
            csv_out.writerows([res for res in results if res])


def get_vcf_header(info):
    n = len(info.individual.unique())
    sample_names = []
    for ind in info["individual"].unique():
        sample_names.append(ind + "_" + "Bulk")
        df = info.loc[(info["individual"] == ind) & (info["type"] != "bulk")]
        sample_names.extend(
            [df["individual"][idx] + "_" + df["cell"][idx] for idx in df.index]
        )
    feat = (
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
    )
    supp = (
        "af_list",
        "allele_flk_size",
        "alt_counts",
        "alt_mm_baseQ_dict",
        "bulk_mismatch_alt_a",
        "bulk_mismatch_ref_a",
        "dis_prop_j",
        "dis_prop_mut",
        "dp_list",
        "dp_list_norm",
        "gi_lfseq",
        "gi_rfseq",
        "gm_lfseq",
        "gm_rfseq",
        "gt_lls",
        "gt_mosaic",
        "gt_posts",
        "gts",
        "hSNP",
        "hSNP_distance",
        "hSNP_start",
        "hanging_counts",
        "is_cmplx",
        "is_end_mm",
        "is_homo_mm_int",
        "is_indel_ms_region",
        "is_mm_ms_region",
        "is_ms_indel_seg",
        "is_mut_region",
        "is_refhom",
        "max_mut_sc_counts",
        "max_nmut_sc_counts",
        "mean_mut_sc_counts",
        "mean_nmut_sc_counts",
        "min_mut_sc_counts",
        "ms_mut_size_seg",
        "mu_mosaic",
        "mut_cell_list",
        "mut_cell_list_final",
        "mut_cell_prob",
        "mut_cell_prob_i",
        "mut_contxt",
        "mut_info",
        "mut_refalt",
        "mut_res_all",
        "mut_type",
        "num_hap_pool_c",
        "num_haps_phased_c",
        "num_mut_cell",
        "num_mut_cell_dup",
        "phasable_k",
        "phasing_b",
        "phasing_block",
        "phasing_counts",
        "phasing_j",
        "phasing_k",
        "phasing_k_counts",
        "phasing_s",
        "prop_all_spn_k_read_c",
        "prop_all_spn_k_read_eq",
        "prop_bulk_spn_k_read_c",
        "prop_bulk_spn_k_read_eq",
        "prop_nonseg_reads",
        "prop_sc_spn_k_read_c",
        "prop_sc_spn_k_read_eq",
        "ref_allele",
        "ref_list_count_eq",
        "ref_mm_baseQ_dict",
        "sc_mismatch_alt_a",
        "sc_mismatch_ref_a",
        "sc_posteriors_final",
        "spn_list",
        "spn_prop",
        "tot_bulk_dp",
        "tot_bulk_spn_dp",
        "tot_sc_dp",
        "tot_sc_spn_dp",
        "vaf_hanging",
        "vaf_hanging_mut",
        "vaf_list_allele",
        "vaf_list_count_eq",
        "vaf_list_eq",
        "vaf_list_prob",
        "vaf_list_spn_count",
    )
    csv_info = (
        [
            "chr",
            "pos",
            "id",
            "ref",
            "alt",
            "qual",
            "filter",
            "info",
            "format",
        ]
        + sample_names
        + [f"phasable_{i}" for i in range(1, n + 1)]
        + [f"phasing_check_{i}" for i in range(1, n + 1)]
        + [v + "_" + str(i) for i in range(1, n + 1) for v in feat]
        + [f"con_{i}" for i in range(1, n + 1)]
        + [v + "_" + str(i) for i in range(1, n + 1) for v in supp]
        + [
            "region_dict",
            "chr_region",
            "start",
            "end",
            "motif_len",
            "len",
            "name",
            "motif",
            "mappability",
            "hap_freq",
        ]
        + [f"lenCounts_{i}" for i in range(1, n + 1)]
        + [f"hapCounts_{i}" for i in range(1, n + 1)]
        + [f"phy_{i}" for i in range(1, n + 1)]
    )
    return csv_info


def pos_generator_around(start, end, limit):
    # yield center
    for offset in range(1, limit + 1):
        yield int(start - offset)
        yield int(end + offset)


# @tools.timing
def find_hSNP_around(args, info, region_dict, qual=20):
    # info_bulk = info[info["type"] == "bulk"]
    # grouped_info = info_bulk.groupby("individual")
    # grouped_info = info.groupby("individual")
    best_distances = []
    best_hSNPs = []
    best_records = []

    def is_valid_hSNP(alleles_counts):
        total = sum(alleles_counts.values())
        if total == 0:
            return False
        sorted_alleles = sorted(
            alleles_counts.items(), key=lambda item: item[1], reverse=True
        )
        major, minor = sorted_alleles[0][1], sorted_alleles[1][1]
        allele_freq = minor / total
        return (
            (
                0.4 <= allele_freq <= 0.6
                or stats.binomtest(minor, total, 0.5).pvalue >= 0.01
            )
            and minor >= 3
            and major >= 3
        )

    for ind in info.individual.unique():
        df = info[(info["type"] == "bulk") & (info["individual"] == ind)]
        if df.empty:
            df = info[(info["individual"] == ind)].sample(
                n=len(info[(info["type"] != "bulk") & (info["individual"] == ind)]),
                random_state=42,
            )
        best_distance = None
        best_hSNP = None
        best_record = BestRecord(None)
        bam_files = {
            row.path: bam_processor.bam_reader(row.path, args)
            for _, row in df.iterrows()
        }
        for pos in pos_generator_around(
            region_dict["start"] - FLK_BP_SEG - 1, region_dict["end"] + FLK_BP_SEG, 1000
        ):
            a_count_all, c_count_all, g_count_all, t_count_all = 0, 0, 0, 0
            for _, row in df.iterrows():
                bam_file = bam_files[row.path]
                counts = bam_file.count_coverage(
                    region_dict["chr"], pos, pos + 1, quality_threshold=qual
                )
                a_count_all += counts[0][0]
                c_count_all += counts[1][0]
                g_count_all += counts[2][0]
                t_count_all += counts[3][0]

            alleles_counts = {
                "A": a_count_all,
                "C": c_count_all,
                "G": g_count_all,
                "T": t_count_all,
            }
            if is_valid_hSNP(alleles_counts):
                best_distance = abs(
                    int((region_dict["start"] + region_dict["end"]) / 2 - pos)
                )
                best_record.start = pos
                sorted_alleles = sorted(
                    alleles_counts.items(), key=lambda item: item[1], reverse=True
                )
                best_hSNP = (sorted_alleles[0][0], sorted_alleles[1][0])
                best_hSNP = tuple(sorted(best_hSNP))
                break

        for bam_file in bam_files.values():
            bam_file.close()

        best_distances.append(best_distance)
        best_hSNPs.append(best_hSNP)
        best_records.append(best_record)
    return best_distances, best_records, best_hSNPs


def read_gnomad_positions(gnomad_file, region_dict):
    left_positions = []
    right_positions = []

    try:
        with pysam.TabixFile(gnomad_file) as tbx:

            region = f"{region_dict['chr']}:{int(region_dict['start']-FLK_BP_SEG-1000)}-{int(region_dict['end']+FLK_BP_SEG+1000)}"

            try:
                for row in tbx.fetch(region=region):
                    # Parse the row (tab-separated)
                    fields = row.split("\t")
                    pos = int(fields[1])  # Start position (1-based)

                    # Sort into left or right flanking regions
                    if pos < region_dict["start"] - FLK_BP_SEG:
                        left_positions.append(pos)
                    elif pos > region_dict["end"] + FLK_BP_SEG:
                        right_positions.append(pos)

            except Exception as e:
                # Handle case where region is not found in the index
                print(f"Warning: Region {region} not found in index: {e}")
                return [], []

    except Exception as e:
        print(f"{e}")
    return (sorted(left_positions, reverse=True), sorted(right_positions))


def pos_generator(left_positions, right_positions):
    """
    Yield positions from inward to outward.
    Converts 1-based positions to 0-based.
    """
    left_idx = 0
    right_idx = 0

    while left_idx < len(left_positions) or right_idx < len(right_positions):
        if left_idx < len(left_positions):
            # Convert 1-based to 0-based coordinate
            yield int(left_positions[left_idx] - 1)
            left_idx += 1

        if right_idx < len(right_positions):
            # Convert 1-based to 0-based coordinate
            yield int(right_positions[right_idx] - 1)
            right_idx += 1


def find_hSNP(
    args,
    info,
    region_dict,
    qual=20,
):
    # Get filtered positions from gnomad file (1-based coordinates)

    left_positions, right_positions = read_gnomad_positions(args.hSNP_file, region_dict)

    best_distances = []
    best_hSNPs = []
    best_records = []

    def is_valid_hSNP(alleles_counts):
        total = sum(alleles_counts.values())
        if total == 0:
            return False
        sorted_alleles = sorted(
            alleles_counts.items(), key=lambda item: item[1], reverse=True
        )
        major, minor = sorted_alleles[0][1], sorted_alleles[1][1]
        allele_freq = minor / total
        return (
            (
                0.4 <= allele_freq <= 0.6
                or stats.binomtest(minor, total, 0.5).pvalue >= 0.01
            )
            and minor >= 3
            and major >= 3
        )

    for ind in info.individual.unique():
        df = info[(info["type"] == "bulk") & (info["individual"] == ind)]
        if df.empty:
            df = info[(info["individual"] == ind)].sample(
                n=len(info[(info["type"] != "bulk") & (info["individual"] == ind)]),
                random_state=42,
            )

        best_distance = None
        best_hSNP = None
        best_record = BestRecord(None)
        bam_files = {
            row.path: bam_processor.bam_reader(row.path, args)
            for _, row in df.iterrows()
        }

        # Use pos_generator with filtered positions (converts 1-based to 0-based)
        for pos in pos_generator(left_positions, right_positions):
            a_count_all, c_count_all, g_count_all, t_count_all = 0, 0, 0, 0
            for _, row in df.iterrows():
                bam_file = bam_files[row.path]
                counts = bam_file.count_coverage(
                    region_dict["chr"], pos, pos + 1, quality_threshold=qual
                )
                a_count_all += counts[0][0]
                c_count_all += counts[1][0]
                g_count_all += counts[2][0]
                t_count_all += counts[3][0]

            alleles_counts = {
                "A": a_count_all,
                "C": c_count_all,
                "G": g_count_all,
                "T": t_count_all,
            }
            if is_valid_hSNP(alleles_counts):
                best_distance = abs(
                    int((region_dict["start"] + region_dict["end"]) / 2 - pos)
                )
                best_record.start = pos
                sorted_alleles = sorted(
                    alleles_counts.items(), key=lambda item: item[1], reverse=True
                )
                best_hSNP = (sorted_alleles[0][0], sorted_alleles[1][0])
                best_hSNP = tuple(sorted(best_hSNP))
                break

        for bam_file in bam_files.values():
            bam_file.close()

        best_distances.append(best_distance)
        best_hSNPs.append(best_hSNP)
        best_records.append(best_record)

    return best_distances, best_records, best_hSNPs


class BestRecord:
    def __init__(self, start):
        self.start = start

    def __bool__(self):
        return self.start is not None
