import argparse

# import myutils
import pysam
import os
import config_params
import time
import numpy as np
import subprocess
import collections

ALLELE_EXTRACT = config_params.ALLELE_EXTRACT
PADDING_BPS = ALLELE_EXTRACT["PADDING_BPS"]  # 5 # 10 # revise_padding
# import warnings
# warnings.filterwarnings("ignore")
# TODO: DEBUG VCF Format Some Headers Are Missing DONE #


def initial_hard_filter(
    vcf_based_features,
    posterior_filter,
    mutant_dp_filter,
    lower_vaf_filter,
    upper_vaf_filter,
    min_depth_filter,
    max_mutation_length,
    het_no_filter,
):
    # five condition leads to GT "." include mosaic fraction estimation fail (LD/MaxIter/MFOutOfRange) and single allele germ and coverage < 5 but GI "." only when no alleles or no reads
    # mosaic/somatic calling no consider for mosaic fraction estimation fail loci and single allele germ and low coverage loci
    # so don't consider the GT == ".,."
    if vcf_based_features["GT"] == ".,.":
        return "Fail"
    if het_no_filter:
        if vcf_based_features["GT"] == vcf_based_features["MGT"]:
            if (
                vcf_based_features["GT"].split(",")[0]
                == vcf_based_features["GT"].split(",")[1]
            ):
                return "Fail"
    else:
        if vcf_based_features["GT"] == vcf_based_features["MGT"]:
            return "Fail"
    depth = int(vcf_based_features["depth"])
    em_vaf = float(vcf_based_features["mosaic_fraction"]) / 2
    if vcf_based_features["GT"] == vcf_based_features["MGT"]:
        if (
            float(
                vcf_based_features["observed_mosaic_allele_vaf_single_locus"]
            )
            >= 0.5
        ):
            mutant_vaf = 1 - float(
                vcf_based_features["observed_mosaic_allele_vaf_single_locus"]
            )
            mutant_dp = mutant_vaf * depth
        else:
            mutant_vaf = float(
                vcf_based_features["observed_mosaic_allele_vaf_single_locus"]
            )
            mutant_dp = mutant_vaf * depth
        if float(vcf_based_features["mosaic_fraction"]) >= 0.5:
            em_vaf = float(vcf_based_features["mosaic_fraction"]) / 2
        else:
            em_vaf = (1 - float(vcf_based_features["mosaic_fraction"])) / 2
    else:
        mutant_vaf = float(
            vcf_based_features["observed_mosaic_allele_vaf_single_locus"]
        )
        mutant_dp = mutant_vaf * depth
    if float(vcf_based_features["CMP"]) <= posterior_filter:
        return "Fail"
    if depth < min_depth_filter:
        return "Fail"
    if mutant_dp < (
        mutant_dp_filter - 0.2
    ):  # HACK: avoid abnormal value, so minus 0.2 TODO HAS DONE!
        return "Fail"
    if mutant_vaf <= lower_vaf_filter:
        return "Fail"
    if mutant_vaf >= upper_vaf_filter:
        return "Fail"
    if em_vaf <= lower_vaf_filter:
        return "Fail"
    if em_vaf >= upper_vaf_filter:
        return "Fail"
    if abs(float(vcf_based_features["MBP"])) > max_mutation_length:
        return "Fail"
    return "Pass"


def bed_region_filter(vcf_based_features, bed_pysam):
    chrom = str(vcf_based_features["chrom"])
    start = int(vcf_based_features["STRSTART"])
    end = int(vcf_based_features["STREND"])
    for _ in bed_pysam.fetch(chrom, start - PADDING_BPS, end + PADDING_BPS):
        return "Fail"
    return "Pass"


def popaf_filter(vcf_based_features, pop_gi_pysam):
    chrom = vcf_based_features["chrom"]
    start = vcf_based_features["STRSTART"]
    end = vcf_based_features["STREND"]
    str_id = vcf_based_features["str_id"]
    germ_allele = vcf_based_features["germ_allele"]
    source_allele = vcf_based_features["source_allele"]
    mosaic_allele = vcf_based_features["mosaic_allele"]
    if germ_allele == ".":
        return 0, 0, 0
    for bed_row in pop_gi_pysam.fetch(chrom, start, end):
        bed_row_list = bed_row.split("\t")
        if bed_row_list[5] == str_id:
            sample_gi = bed_row_list[7:]
            sample_gi = [x for x in sample_gi if x != "."]
            sample_gi_string = "_".join(sample_gi)
            all_allele_list = sample_gi_string.split("_")
            germ_allele_popAF = all_allele_list.count(germ_allele) / len(
                all_allele_list
            )
            source_allele_popAF = all_allele_list.count(source_allele) / len(
                all_allele_list
            )
            mosaic_allele_popAF = all_allele_list.count(mosaic_allele) / len(
                all_allele_list
            )
            return germ_allele_popAF, source_allele_popAF, mosaic_allele_popAF
    return 0, 0, 0


def get_mosaic_gt_list(recurrent_mosaic_gt):
    mosaic_gt_list = []
    for i in recurrent_mosaic_gt:
        mosaic_gt_string = transform_mosaic_gt(i)
        mosaic_gt_list.append(mosaic_gt_string)
    return mosaic_gt_list


def transform_mosaic_gt(mosaic_gt):
    return "_".join(sorted(list(set(mosaic_gt.split("_")))))


def recurrent_mosaic_gt_cal(vcf_based_features, recurrent_filter_pysam):
    chrom = vcf_based_features["chrom"]
    start = vcf_based_features["STRSTART"]
    end = vcf_based_features["STREND"]
    str_id = vcf_based_features["str_id"]
    mosaic_gt_raw = vcf_based_features["GTMGT"]
    mosaic_gt = transform_mosaic_gt(mosaic_gt_raw)
    uncallable_num = 0
    uncallable_frac = 0
    recurrent_num = 0
    recurrent_fraction = 0
    sample_num = 0
    for bed_row in recurrent_filter_pysam.fetch(chrom, start, end):
        bed_row_list = bed_row.split("\t")
        if bed_row_list[5] == str_id:
            recurrent_mosaic_gt_raw = bed_row_list[7:-2]
            uncallable_num = bed_row_list[-2]
            uncallable_frac = bed_row_list[-1]
            recurrent_mosaic_gt = get_mosaic_gt_list(recurrent_mosaic_gt_raw)
            recurrent_num = recurrent_mosaic_gt.count(mosaic_gt)
            sample_num = len(recurrent_mosaic_gt)
            recurrent_fraction = recurrent_num / sample_num
            break
    return (
        uncallable_num,
        uncallable_frac,
        recurrent_num,
        recurrent_fraction,
        sample_num,
    )


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


def get_popAF(
    population_panel_pysam, reference_fasta_pysam, vcf_based_features
):
    external_germ_popAF = "NA"
    external_source_popAF = "NA"
    external_mutant_popAF = "NA"
    STRSTART = vcf_based_features["STRSTART"]
    STREND = vcf_based_features["STREND"]
    chrom = vcf_based_features["chrom"]
    germ_seq = vcf_based_features["germ_allele_seq"]
    source_seq = vcf_based_features["source_allele_seq"]
    mut_seq = vcf_based_features["mosaic_allele_seq"]
    for j in population_panel_pysam.fetch(chrom, STRSTART, STREND):
        row_list = j.split("\t")
        allele_start = int(row_list[1])
        allele_end = int(row_list[2])
        allele_af = row_list[9:]
        allele_af_dict = {}
        if (
            (allele_start >= STRSTART - PADDING_BPS)
            and (allele_start <= STRSTART + PADDING_BPS)
            and (allele_end >= STREND - PADDING_BPS)
            and (allele_end <= STREND + PADDING_BPS)
        ):
            start_offset = allele_start - (STRSTART - PADDING_BPS)
            end_offset = (STREND + PADDING_BPS) - allele_end
            for k in allele_af:
                i_list = k.split(":")
                allele_seq = i_list[0]
                allele_frq = float(i_list[1])
                if start_offset > 0:
                    left_padding_seq = reference_fasta_pysam.fetch(
                        chrom, STRSTART - PADDING_BPS, allele_start
                    )
                else:
                    left_padding_seq = ""
                if end_offset > 0:
                    right_padding_seq = reference_fasta_pysam.fetch(
                        chrom, allele_end, STREND + PADDING_BPS
                    )
                else:
                    right_padding_seq = ""
                allele_seq_padding = (
                    left_padding_seq + allele_seq + right_padding_seq
                )
                allele_af_dict[allele_seq_padding.upper()] = allele_frq
            germ_allele_af = allele_af_dict.get(germ_seq.upper(), 0)
            source_allele_af = allele_af_dict.get(source_seq.upper(), 0)
            mosaic_allele_af = allele_af_dict.get(mut_seq.upper(), 0)
            external_germ_popAF = germ_allele_af
            external_source_popAF = source_allele_af
            external_mutant_popAF = mosaic_allele_af
            if mosaic_allele_af != 0:
                break
        else:
            if external_mutant_popAF != "NA":
                pass
            else:
                external_germ_popAF = "NoItem"
                external_source_popAF = "NoItem"
                external_mutant_popAF = "NoItem"
    return external_germ_popAF, external_source_popAF, external_mutant_popAF


def fill_na_popAF(pop_af):
    if (pop_af == "NoItem") or (pop_af == "NA"):
        pop_af = 0
    return pop_af


def get_features_from_vcf(vcf_record, sample_name):
    all_alleles = vcf_record.alleles
    vcf_based_features = {}
    chrom = vcf_record.chrom
    STRSTART = int(vcf_record.info["STRSTART"])
    STREND = int(vcf_record.info["STREND"])
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
    sample_var = vcf_record.samples[sample_name]
    if sample_var["GT"][0] is None:
        GT = ".,."
    else:
        GT = str(sample_var["GT"][0]) + "," + str(sample_var["GT"][1])
    if sample_var["MGT"][0] == ".":
        MGT = ".,."
    else:
        MGT = (
            sample_var["MGT"][0].split("/")[0]
            + ","
            + sample_var["MGT"][0].split("/")[1]
        )
    GT_list = GT.split(",")
    MGT_list = MGT.split(",")
    if len(set(GT_list)) == 2 and len(set(MGT_list)) == 1:
        mosaic_allele = GT_list[1]
        germ_allele = GT_list[0]
        source_allele = MGT_list[1]
        muttype = "hethom"
    elif len(set(GT_list)) == 2 and len(set(MGT_list)) == 2:
        if GT == MGT:
            mosaic_allele = GT_list[1]
            germ_allele = GT_list[0]
            source_allele = GT_list[0]
            muttype = "clonalhet"
        else:
            mosaic_allele = MGT_list[1]
            germ_allele = GT_list[0]
            source_allele = GT_list[1]
            muttype = "hethet"
    elif len(set(GT_list)) == 1 and len(set(MGT_list)) == 2:
        mosaic_allele = MGT_list[1]
        germ_allele = GT_list[0]
        source_allele = GT_list[1]
        muttype = "homhet"
    else:
        mosaic_allele = "."
        germ_allele = "."
        source_allele = "."
        muttype = "unknown"
    if germ_allele != ".":
        germ_allele_seq = all_alleles[int(germ_allele)]
        source_allele_seq = all_alleles[int(source_allele)]
        mosaic_allele_seq = all_alleles[int(mosaic_allele)]
    else:
        germ_allele_seq = "."
        source_allele_seq = "."
        mosaic_allele_seq = "."
    vcf_based_features["germ_allele"] = germ_allele
    vcf_based_features["source_allele"] = source_allele
    vcf_based_features["mosaic_allele"] = mosaic_allele
    vcf_based_features["germ_allele_seq"] = germ_allele_seq
    vcf_based_features["source_allele_seq"] = source_allele_seq
    vcf_based_features["mosaic_allele_seq"] = mosaic_allele_seq
    vcf_based_features["muttype"] = muttype
    
    MP = sample_var["MP"]
    CMP = sample_var["CMP"]
    variant_type = sample_var["MUTP"]
    frame = sample_var["FRAME"]
    mosaic_fraction = sample_var["RMF"]
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
    vcf_based_features["GT"] = GT
    vcf_based_features["MGT"] = MGT
    vcf_based_features["MP"] = MP
    vcf_based_features["CMP"] = CMP
    vcf_based_features["variant_type"] = variant_type
    vcf_based_features["frame"] = frame
    vcf_based_features["mosaic_fraction"] = mosaic_fraction
    vcf_based_features["perfect_ref_str"] = perfect_ref_str
    vcf_based_features["depth"] = depth
    vcf_based_features["filtered_depth"] = filtered_depth
    if filtered_depth_category == ".":
        vcf_based_features["filtered_depth_category"] = "NA,NA,NA,NA,NA,NA,NA"
    else:
        vcf_based_features["filtered_depth_category"] = filtered_depth_category
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
    if (len(set(GT_list)) == 2) and (len(set(MGT_list)) == 2):
        if GT == MGT:
            MBP = len(mosaic_allele_seq) - len(source_allele_seq)
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
        vcf_based_features["GIOAF"] = "NA,NA"
    else:
        vcf_based_features["GIOAF"] = GIOAF
    if GIMLEAF == ".":
        vcf_based_features["GIMLEAF"] = "NA,NA"
    else:
        vcf_based_features["GIMLEAF"] = GIMLEAF
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
    GI = sample_var["GI"]
    if GI[0] == ".":
        vcf_based_features["GI"] = "./."
    else:
        vcf_based_features["GI"] = GI[0]
    if GI[0] == ".":
        GermGT2_string = "."
    else:
        GermGT = GI[0].split("/")
        GermGT1 = [int(allele) for allele in GermGT]
        GermGT1.sort()
        GermGT2 = [str(allele) for allele in GermGT1]
        GermGT2_string = "_".join(GermGT2)
    if germ_index[0] is None:
        if GermGT2_string == ".":
            mosaicGT = [".", ".", ".", "."]
        else:
            mosaicGT = [GermGT2[0], GermGT2[0], GermGT2[1], GermGT2[1]]
        mosaicGT = "_".join(mosaicGT)
    else:
        mosaicGT = [
            int(germ_index[0]),
            int(germ_index[1]),
            int(mosaic_index[0]),
            int(mosaic_index[1]),
        ]
        mosaicGT.sort()
        mosaicGT = [str(allele) for allele in mosaicGT]
        mosaicGT = "_".join(mosaicGT)
    vcf_based_features["Filter"] = sample_var["FILTER"]
    vcf_based_features["sample_name"] = sample_name
    vcf_based_features["GTMGT"] = mosaicGT
    if sample_var["AAD"][0] != ".":
        all_alleles_dp = sample_var["AAD"][0]  # HACK: For new version of mosaic calling output, add [0] to get the allele_dp_string
        if "|" in all_alleles_dp:
            allele_length_dp_dict = {}
            vcf_based_features["AAD"] = all_alleles_dp
            for allele_idx_dp in all_alleles_dp.split(";"):
                allele_idx, allele_dp = allele_idx_dp.split("|")
                if int(allele_dp) != 0:
                    allele_length = len(all_alleles[int(allele_idx)])
                    allele_length_dp_dict[
                        allele_length
                    ] = allele_length_dp_dict.get(allele_length, 0) + int(
                        allele_dp
                    )
        else:
            allele_length_dp_dict = {}
            allele_dp_dict = {}
            allele_idx = 0
            for allele, allele_dp in zip(
                all_alleles[: len(all_alleles_dp)], all_alleles_dp
            ):
                if int(allele_dp) != 0:
                    allele_length = len(allele)
                    allele_length_dp_dict[
                        allele_length
                    ] = allele_length_dp_dict.get(allele_length, 0) + int(
                        allele_dp
                    )
                    allele_dp_dict[allele_idx] = allele_dp
                allele_idx += 1
            allele_dp_string = ""
            for allele_idx, allele_dp in allele_dp_dict.items():
                allele_dp_string = (
                    allele_dp_string
                    + str(allele_idx)
                    + "|"
                    + str(allele_dp)
                    + ";"
                )
            if len(allele_dp_string) > 0:
                allele_dp_string = allele_dp_string[:-1]
            else:
                allele_dp_string = "NA"
            vcf_based_features["AAD"] = allele_dp_string
        allele_length_dp = ""
        for allele_length, allele_dp in allele_length_dp_dict.items():
            allele_length_dp = (
                allele_length_dp
                + str(allele_length)
                + "|"
                + str(allele_dp)
                + ";"
            )
        if len(allele_length_dp) > 0:
            allele_length_dp = allele_length_dp[:-1]
        else:
            allele_length_dp = "NA"
        vcf_based_features["allele_length_dp"] = allele_length_dp
    else:
        vcf_based_features["AAD"] = "."
        vcf_based_features["allele_length_dp"] = "."
    ref_allele_length_padding_flanking = (STREND - STRSTART) + (
        2 * PADDING_BPS
    )
    vcf_based_features[
        "ref_allele_length_padding_flanking"
    ] = ref_allele_length_padding_flanking

    vcf_based_features["ALLELE_BARCODE_UMI"] = sample_var["ALLELE_BARCODE_UMI"][0]
    return vcf_based_features


def process_vcf(
    vcf_file,
    sample_name,
    output_file,
    posterior_filter,
    mutant_dp_filter,
    lower_vaf_filter,
    upper_vaf_filter,
    min_depth_filter,
    max_mutation_length,
    pop_gi_filter,
    recurrent_filter,
    pop_gi_file,
    recurrent_filter_file,
    callable_filter,
    external_population_panel,
    bed_filter_out,
    reference_fasta,
    het_no_filter,
):
    vcf_in = pysam.VariantFile(vcf_file)
    result_features = [
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
        "CMP", #11
        "variant_type",
        "frame",
        "mosaic_fraction",
        "perfect_ref_str",
        "depth", #16
        "filtered_depth",
        "filtered_depth_category",
        "EAF",
        "observed_mosaic_allele_vaf_single_locus", #20
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
        "muttype",
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
        "ALLELE_BARCODE_UMI"
    ]
    f = open(output_file, "a")
    fail_output_file = (
        ".".join(args.output_file.split(".")[:-1]) + "_fail.bed"
    )
    f_fail = open(fail_output_file, "a")
    finish_str_id_list = []
    if bed_filter_out:
        bed_pysam = pysam.TabixFile(bed_filter_out)
    if pop_gi_file:
        pop_gi_pysam = pysam.TabixFile(pop_gi_file)
    if recurrent_filter_file:
        recurrent_filter_pysam = pysam.TabixFile(recurrent_filter_file)
    if external_population_panel:
        external_population_panel_pysam = pysam.TabixFile(
            external_population_panel
        )
        reference_fasta_pysam = pysam.FastaFile(reference_fasta)
    genotyping_dp_list = []

    for vcf_record in vcf_in.fetch():
        vcf_based_features = get_features_from_vcf(vcf_record, sample_name)
        line = ""
        for feature in result_features:
            line += str(vcf_based_features.get(feature, ".")) + "\t"


        if vcf_based_features["depth"] != ".":
            genotyping_dp_list.append(int(vcf_based_features["depth"]))
        muttype = vcf_based_features["muttype"]
        str_id = vcf_based_features["str_id"]
        if str_id in finish_str_id_list:
            line += str("1")
            line = line.strip()
            line += "\n"
            f_fail.write(line)
            continue
        finish_str_id_list.append(str_id)
        if bed_filter_out:
            if bed_region_filter(vcf_based_features, bed_pysam) == "Fail":
                line += str("2")
                line = line.strip()
                line += "\n"
                f_fail.write(line)
                continue
        if (
            initial_hard_filter(
                vcf_based_features,
                posterior_filter,
                mutant_dp_filter,
                lower_vaf_filter,
                upper_vaf_filter,
                min_depth_filter,
                max_mutation_length,
                het_no_filter,
            )
            == "Fail"
        ):
            line += str("3")
            line = line.strip()
            line += "\n"
            f_fail.write(line)
            continue
        if recurrent_filter_file:
            (
                uncallable_num,
                uncallable_frac,
                recurrent_num,
                recurrent_fraction,
                sample_num,
            ) = recurrent_mosaic_gt_cal(
                vcf_based_features, recurrent_filter_pysam
            )
            vcf_based_features["uncallable_num"] = uncallable_num
            vcf_based_features["uncallable_frac"] = uncallable_frac
            vcf_based_features["recurrent_num"] = recurrent_num
            vcf_based_features["recurrent_fraction"] = recurrent_fraction
            vcf_based_features["sample_num"] = sample_num
            callable_frac = 1 - float(uncallable_frac)
            # callable_num = sample_num - uncallable_num
            if (recurrent_fraction > recurrent_filter) and (
                recurrent_num > 1
            ):  # and (callable_num > 20)
                line += str("4")
                line = line.strip()
                line += "\n"
                f_fail.write(line)
                continue
            if callable_frac < callable_filter:
                line += str("5")
                line = line.strip()
                line += "\n"
                f_fail.write(line)
                continue
        if pop_gi_file:
            germ_popAF, source_popAF, mosaic_popAF = popaf_filter(
                vcf_based_features, pop_gi_pysam
            )
            vcf_based_features["germ_popAF"] = germ_popAF
            vcf_based_features["source_popAF"] = source_popAF
            vcf_based_features["mosaic_popAF"] = mosaic_popAF
            if muttype == "hethet":
                if (source_popAF > pop_gi_filter) and (
                    mosaic_popAF > pop_gi_filter
                ):
                    line += str("6")
                    line = line.strip()
                    line += "\n"
                    f_fail.write(line)
                    continue
            elif muttype == "clonalhet":
                if (source_popAF > pop_gi_filter) and (
                    mosaic_popAF > pop_gi_filter
                ):
                    line += str("7")
                    line = line.strip()
                    line += "\n"
                    f_fail.write(line)
                    continue
            else:
                if mosaic_popAF > pop_gi_filter:
                    line += str("7")
                    line = line.strip()
                    line += "\n"
                    f_fail.write(line)
                    continue
        if external_population_panel:
            (
                external_germ_popAF,
                external_source_popAF,
                external_mutant_popAF,
            ) = get_popAF(
                external_population_panel_pysam,
                reference_fasta_pysam,
                vcf_based_features,
            )
            vcf_based_features["external_germ_popAF"] = external_germ_popAF
            vcf_based_features["external_source_popAF"] = external_source_popAF
            vcf_based_features["external_mosaic_popAF"] = external_mutant_popAF
            external_source_popAF_r = fill_na_popAF(external_source_popAF)
            external_mutant_popAF_r = fill_na_popAF(external_mutant_popAF)
            if muttype == "hethet":
                if (external_source_popAF_r > pop_gi_filter) and (
                    external_mutant_popAF_r > pop_gi_filter
                ):
                    line += str("8")
                    line = line.strip()
                    line += "\n"
                    f_fail.write(line)
                    continue
            elif muttype == "clonalhet":
                if (external_source_popAF_r > pop_gi_filter) and (
                    external_mutant_popAF_r > pop_gi_filter
                ):
                    line += str("9")
                    line = line.strip()
                    line += "\n"
                    f_fail.write(line)
                    continue
            else:
                if external_mutant_popAF_r > pop_gi_filter:
                    line += str("10")
                    line = line.strip()
                    line += "\n"
                    f_fail.write(line)
                    continue
        else:
            vcf_based_features["external_germ_popAF"] = "NA"
            vcf_based_features["external_source_popAF"] = "NA"
            vcf_based_features["external_mosaic_popAF"] = "NA"
            
        line = ""
        for feature in result_features:
            line += str(vcf_based_features.get(feature, ".")) + "\t"
        line = line.strip()
        line += "\n"
        # if myutils.acquire_lock(f):
        f.write(line)
        # myutils.release_lock(f)
    # print(
    #     "Finish "
    #     + vcf_record.chrom
    #     + ":"
    #     + str(vcf_record.pos)
    #     + " "
    #     + vcf_record.id
    # )
    f.close()
    f_fail.close()
    median_genotyping_dp = np.median(genotyping_dp_list)
    return median_genotyping_dp


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Extract key information from vcf"
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Mosaic variant filtering parameters"
    )

    parser.add_argument("-v", "--vcf_file", type=str, help="VCF file path")
    parser.add_argument("-s", "--sample_name", type=str, help="Sample name")
    parser.add_argument(
        "-o", "--output_file", type=str, help="Output file path"
    )
    parser.add_argument(
        "-p",
        "--posterior_filter",
        default=0.9,
        type=float,
        help=(
            "Posterior probability filter (must be > this value, lower cutoff"
            " for low-depth samples)"
        ),
    )
    parser.add_argument(
        "-m",
        "--mutant_dp_filter",
        default=3,
        type=int,
        help="Mutant depth (DP) filter (must be ≥ this value)",
    )  # HACK: avoid abnormal value, so should minus 0.1 after
    parser.add_argument(
        "-u",
        "--upper_vaf_filter",
        default=0.8,
        type=float,
        help="Upper VAF filter (must be < this value)",
    )
    parser.add_argument(
        "-l",
        "--lower_vaf_filter",
        default=0.01,
        type=float,
        help="Lower VAF filter (must be > this value)",
    )
    parser.add_argument(
        "-d",
        "--min_depth_filter",
        default=10,
        type=int,
        help="Minimum read depth filter (must be ≥ this value)",
    )
    parser.add_argument(
        "-ms",
        "--max_mutation_length",
        default=18,
        type=int,
        help="Maximum allowed mutation length",
    )
    parser.add_argument(
        "-pgi",
        "--pop_gi_filter",
        default=0.3,
        type=float,
        help="Population allele frequency filter threshold",
    )
    parser.add_argument(
        "-r",
        "--recurrent_filter",
        default=0.1,
        type=float,
        help="Recurrent mutation filter threshold",
    )
    parser.add_argument(
        "-bf",
        "--pop_gi_file",
        type=str,
        help="Path to the population allele frequency file",
    )
    parser.add_argument(
        "-rf",
        "--recurrent_filter_file",
        type=str,
        help="Path to recurrent mutation filter file",
    )
    parser.add_argument(
        "-cf",
        "--callable_filter",
        default=0.2,
        type=float,
        help="Callable fraction filter threshold",
    )
    parser.add_argument(
        "-ep",
        "--external_population_panel",
        type=str,
        help="External population panel file",
    )
    parser.add_argument(
        "-b",
        "--bed_filter_out",
        type=str,
        help="BED file listing loci to filter out",
    )
    parser.add_argument(
        "-re",
        "--reference_fasta",
        type=str,
        help="Reference fasta file",
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
    args = parser.parse_args()
    # 调用函数提取关键词并输出文件名
    median_genotyping_dp = process_vcf(
        args.vcf_file,
        args.sample_name,
        args.output_file,
        args.posterior_filter,
        args.mutant_dp_filter,
        args.lower_vaf_filter,
        args.upper_vaf_filter,
        args.min_depth_filter,
        args.max_mutation_length,
        args.pop_gi_filter,
        args.recurrent_filter,
        args.pop_gi_file,
        args.recurrent_filter_file,
        args.callable_filter,
        args.external_population_panel,
        args.bed_filter_out,
        args.reference_fasta,
        args.het_no_filter,
    )
    sort_output_file = (
        ".".join(args.output_file.split(".")[:-1]) + "_sorted.bed"
    )
    time.sleep(1)
    input_file = args.output_file
    cmd = f"""awk -v val={median_genotyping_dp} 'BEGIN{{OFS="\\t"}} {{print $0, val}}' {input_file} | sort -k1,1 -k2,2n > {sort_output_file}"""
    subprocess.run(cmd, shell=True, check=True)
    time.sleep(1)
    subprocess.run(f"bgzip -f {sort_output_file}", shell=True, check=True)
    time.sleep(1)
    subprocess.run(
        f"tabix -p bed {sort_output_file}.gz", shell=True, check=True
    )

# python3 initial_hard_filter.py -v /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/merge_vcf_20241205/mosaic_calling_result_seq_based_GRCh37_2897_header_sorted.vcf.gz -s SRR13989893 -o /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/extract_features_1485/test_initial_filter.bed -p 0.9 -m 3 -u 0.8 -l 0.01 -d 10 -ms 18 -pgi 0.3 -r 0.3 -bf /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/extract_features_1485/popGT_32inds/phs1485v3_32inds_str_gi_test2.txt.gz -rf /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/extract_features_1485/popGT_32inds/phs1485v3_32inds_str_gi_test2_mosaic_recurrent_info.txt.gz -cf 0.2 -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/filter_loci_regions/GRCh37_alllowmapandsegdupregions_addT2T_liftover_SD_regions_HipSTR_GRCh37_zerobased_leftcloserightopen.bed.gz -re /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta
# python3 initial_hard_filter.py -v /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/hg002_mosaic_calling/result_hg005_sampling/mosaic_calling_result_seq_based_GRCh38_2887_header_sorted.vcf.gz -s HG005_GRCh38_100x -o /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/extract_features_1485/hg005_test_initial_filter.bed -p 0.9 -m 3 -u 0.8 -l 0.01 -d 10 -ms 18 -pgi 0.3 -r 0.3 -cf 0.2 -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/filter_loci_regions/GRCh38_alllowmapandsegdupregions_addT2T_liftover_SD_regions_HipSTR_GRCh38_zerobased_leftcloserightopen.bed.gz -re /storage/douyanmeiLab/wangweixiang/data/GIAB/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta -ep /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/hg002_mosaic_calling/extract_features/features_hg005_sampling/pop_AF_allchr_noXYM_final_sorted.bed.gz

# python3 initial_hard_filter.py -v /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/BulkMonSTR_Code_Test/test/mosaic_genotyping/mosaic_genotyping_result/mosaic_fraction_estimation_results/mosaic_result_hg38_chr1_30216700_30216750_mosaic_calling.sorted.vcf.gz -s Human_STR_21293_chr1_30216713_T_TAC_INS_sorted_1 -o /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/BulkMonSTR_Code_Test/test/initial_hard_cutoff/Human_STR_21293_chr1_30216713_T_TAC_INS_sorted_initial_hard_filter.bed -p 0.9 -m 3 -u 0.8 -l 0.01 -d 10 -ms 18 -pgi 1 -r 1 -cf 0.2 -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/filter_loci_regions/GRCh38_alllowmapandsegdupregions_addT2T_liftover_SD_regions_HipSTR_GRCh38_zerobased_leftcloserightopen.bed.gz -re /storage/douyanmeiLab/wangweixiang/data/GIAB/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta -ep /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/hg002_mosaic_calling/extract_features/features_hg005_sampling/pop_AF_allchr_noXYM_final_sorted.bed.gz -bf /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/BulkMonSTR_Code_Test/test/extract_pop_infors/pop_infors_output/pop_infors_output.txt.gz -rf /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/BulkMonSTR_Code_Test/test/extract_pop_infors/pop_infors_output/pop_infors_output_mosaic_recurrent_info.txt.gz
