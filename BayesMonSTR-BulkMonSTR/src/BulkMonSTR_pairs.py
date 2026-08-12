import config_params
import adjust_vcf
import pandas as pd
import pairmode_params
import numpy as np
from scipy.stats import binomtest
import argparse
import pysam

GIAB = False
ALLELE_EXTRACT = config_params.ALLELE_EXTRACT
PADDING_BPS = ALLELE_EXTRACT["PADDING_BPS"]  # 5 # 10 # revise_padding
PAIR_PARAMS = pairmode_params.pair_mode_params_dict
NORMAL_MAX_VAF = PAIR_PARAMS["NORMAL_MAX_VAF"]
NORMAL_DP_CUTOFF = PAIR_PARAMS["NORMAL_DP_CUTOFF"]
CASE_DP_CUTOFF = PAIR_PARAMS["CASE_DP_CUTOFF"]
RECURRENT_CASE_VAF_CUTOFF = PAIR_PARAMS["RECURRENT_CASE_VAF_CUTOFF"]
CASE_MUTATION = PAIR_PARAMS["CASE_MUTATION"]
STRICT_MODE = PAIR_PARAMS["STRICT_MODE"]


def transform_mosaic_gt(s):
    return s.str.split("_").apply(set)


def get_reference_chr_pos_ref_alt(
    chrom,
    STRSTART,
    STREND,
    muttype,
    germ_allele,
    source_allele,
    mut_allele,
    pysam_fasta,
):
    if PADDING_BPS > 0:
        refseq_left_padding_1bp = pysam_fasta.fetch(
            str(chrom),
            STRSTART - 1 - PADDING_BPS,
            STRSTART - PADDING_BPS,
        )
        reference_seq = pysam_fasta.fetch(
            str(chrom),
            STRSTART - PADDING_BPS,
            STREND + PADDING_BPS,
        )
    else:
        refseq_left_padding_1bp = pysam_fasta.fetch(
            str(chrom), STRSTART - 1, STRSTART
        )
        reference_seq = pysam_fasta.fetch(str(chrom), STRSTART, STREND)

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
        # seq0_length = len(germ_allele)
        # seq1_length = len(germ_allele)
        # seq2_length = len(mut_allele)
        # if seq0_length == seq2_length:
        #     (
        #         _,
        #         mis_location,
        #     ) = myutils.cal_edit_distance_based_on_mismatch(
        #         germ_allele, mut_allele
        #     )
        # else:
        #     mis_location = []
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
        # seq0_length = len(germ_allele)
        # seq1_length = len(source_allele)
        # seq2_length = len(mut_allele)
        # if seq1_length == seq2_length:
        #     (
        #         _,
        #         mis_location,
        #     ) = myutils.cal_edit_distance_based_on_mismatch(
        #         source_allele, mut_allele
        #     )
        # else:
        #     mis_location = []
    mut_source_seq = source_seq
    mut_target_seq = target_seq
    mut_source_seq_pos_tuple_0_based_leftrightclose = mutation_site_seq1
    mut_target_seq_pos_tuple_0_based_leftrightclose = mutation_site_seq2
    reference_start_index_0_based_include = refseq_left_index
    reference_end_index_0_based_include = refseq_right_index
    reference_start_coordinate_1_based_include = (
        STRSTART - PADDING_BPS + reference_start_index_0_based_include + 1
    )
    reference_end_coordinate_1_based_include = (
        STRSTART - PADDING_BPS + reference_end_index_0_based_include + 1
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
        alleles_mut_type,
        mutation_site_type,
        mutation_location_type,
    )


def get_sample_allele_dp(all_alleles, sample_var):
    if "AAD" not in sample_var or sample_var["AAD"][0] == ".":
        # Handle case where AAD is not present or is a placeholder
        return ".", "."

    # HACK: For new version of mosaic calling output, add [0]
    all_alleles_dp = sample_var["AAD"][0]

    allele_length_dp_dict = {}

    try:
        if "|" in all_alleles_dp:
            # Process case where all_alleles_dp contains '|'
            for allele_idx_dp in all_alleles_dp.split(";"):
                allele_idx, allele_dp = allele_idx_dp.split("|")
                allele_idx = int(allele_idx)
                allele_dp = int(allele_dp)
                if allele_dp != 0:
                    allele_length = len(all_alleles[allele_idx])
                    allele_length_dp_dict[allele_length] = (
                        allele_length_dp_dict.get(allele_length, 0) + allele_dp
                    )
            used_all_alleles_dp = all_alleles_dp
        else:
            # Process case where all_alleles_dp does not contain '|'
            allele_dp_dict = {}
            allele_idx = 0
            for allele, allele_dp in zip(
                all_alleles[: len(all_alleles_dp)], all_alleles_dp
            ):
                allele_dp = int(allele_dp)
                if allele_dp != 0:
                    allele_length = len(allele)
                    allele_length_dp_dict[allele_length] = (
                        allele_length_dp_dict.get(allele_length, 0) + allele_dp
                    )
                    allele_dp_dict[allele_idx] = allele_dp
                allele_idx += 1

            # Construct the allele_dp_string
            allele_dp_string = ";".join(
                [
                    f"{allele_idx}|{allele_dp}"
                    for allele_idx, allele_dp in allele_dp_dict.items()
                ]
            )
            used_all_alleles_dp = allele_dp_string if allele_dp_string else "."

    except Exception as e:
        # Log or raise error depending on preference
        raise ValueError(f"Error processing allele depth data: {e}")

    # Construct allele_length_dp string
    allele_length_dp = ";".join(
        [
            f"{allele_length}|{allele_dp}"
            for allele_length, allele_dp in allele_length_dp_dict.items()
        ]
    )
    allele_length_dp = allele_length_dp if allele_length_dp else "."

    return used_all_alleles_dp, allele_length_dp


def get_vcf_infors(row, vcf_in, case_sample_name, normal_sample_name):
    chrom = str(row["chrom"])
    start = row["STRSTART"]
    end = row["STREND"]
    str_id = row["str_id.1"]

    # Fetch the VCF records in the region with a padding of 10 base pairs on either side
    for vcf_record in vcf_in.fetch(chrom, start - 10, end + 10):
        if vcf_record.info.get("STRID") == str_id:  # Safer check for STRID
            all_alleles = vcf_record.alleles

            # Ensure that the sample names are valid and exist in the VCF record
            case_sample_var = vcf_record.samples.get(case_sample_name, None)
            normal_sample_var = vcf_record.samples.get(
                normal_sample_name, None
            )

            # If either sample is missing, return missing data
            if case_sample_var is None or normal_sample_var is None:
                return pd.Series(
                    {
                        "all_alleles": ".",
                        "case_used_all_alleles_dp": ".",
                        "case_allele_length_dp": ".",
                        "normal_used_all_alleles_dp": ".",
                        "normal_allele_length_dp": ".",
                    }
                )

            # Get allele depths and lengths for both case and normal samples
            (
                case_used_all_alleles_dp,
                case_allele_length_dp,
            ) = get_sample_allele_dp(all_alleles, case_sample_var)
            (
                normal_used_all_alleles_dp,
                normal_allele_length_dp,
            ) = get_sample_allele_dp(all_alleles, normal_sample_var)

            # Return results as a pandas Series
            return pd.Series(
                {
                    "all_alleles": all_alleles,
                    "case_used_all_alleles_dp": case_used_all_alleles_dp,
                    "case_allele_length_dp": case_allele_length_dp,
                    "normal_used_all_alleles_dp": normal_used_all_alleles_dp,
                    "normal_allele_length_dp": normal_allele_length_dp,
                }
            )

    # If no matching VCF record is found, return missing data
    return pd.Series(
        {
            "all_alleles": ".",
            "case_used_all_alleles_dp": ".",
            "case_allele_length_dp": ".",
            "normal_used_all_alleles_dp": ".",
            "normal_allele_length_dp": ".",
        }
    )


def get_dp_according_aad(AAD):
    if AAD == ".":
        return "NoInforsAvailable"
    all_dp = 0
    for i in AAD.split(";"):
        dp = int(i.split("|")[1])
        all_dp += dp
    return all_dp


def get_allele_dp_dict(AAD):
    if AAD == ".":
        return "NoInforsAvailable"
    all_allele_dp_dict = {}
    for i in AAD.split(";"):
        allele, dp = int(i.split("|")[0]), int(i.split("|")[1])
        all_allele_dp_dict[allele] = dp
    return all_allele_dp_dict


def calculate_vaf(
    case_dp_dict, normal_dp_dict, mutation_allele, case_dp, normal_dp
):
    obs_mut_dp_case = case_dp_dict.get(int(mutation_allele), 0)
    obs_mut_dp_normal = normal_dp_dict.get(int(mutation_allele), 0)
    obs_mut_vaf_new_case = obs_mut_dp_case / case_dp
    obs_mut_vaf_new_normal = obs_mut_dp_normal / normal_dp
    return obs_mut_vaf_new_case, obs_mut_vaf_new_normal, obs_mut_dp_normal


def apply_filters(
    obs_mut_vaf_new_case,
    obs_mut_vaf_new_normal,
    obs_mut_dp_normal,
    pvalue,
    diff_allele_norm_recurrent_num,
):
    filter1 = (
        "PASS" if obs_mut_vaf_new_normal <= NORMAL_MAX_VAF else "NormalVAF0p2"
    )
    filter2 = (
        "PASS"
        if obs_mut_vaf_new_normal < obs_mut_vaf_new_case
        else "HigherNormalVAF"
    )
    filter3 = (
        "PASS" if ((obs_mut_dp_normal == 0) or (
            pvalue < 0.05)) else "NoSigHigherVAF"
    )
    filter4 = (
        "PASS"
        if diff_allele_norm_recurrent_num > 0
        else "PASS"
        if ((obs_mut_dp_normal == 0) or (pvalue < 0.05))
        else "Recurrent"
    )
    return filter1, filter2, filter3, filter4


def handle_homhet_case_onediff(case_mosaic, normal_GI):
    """
    Handle the 'homhet' mutation type case (hom/het).

    Args:
    - case_mosaic (str): The case mosaic alleles in a string format (e.g., '1_1_1_2').
    - normal_GI (str): The normal germline alleles in a string format (e.g., '1_3').
    - muttype (str): The mutation type ('homhet' or 'hethet').

    Returns:
    - germ_allele (str): The allele representing the germline.
    - source_allele (str): The source allele used in mutation analysis.
    - muttype (str): The mutation type used in further processing.
    - case_muttype (str): The specific mutation type for case analysis.
    """
    # Case with homhet: two different alleles between normal and case
    germ_allele = list(
        set(case_mosaic.split("_")) & set(normal_GI.split("_"))
    )[0]
    if len(set(normal_GI.split("_"))) == 1:
        # Single allele in normal (e.g., hom)
        muttype = "homhet"
        case_muttype = "homhet_normhom_onediff"
        source_allele = germ_allele
    elif len(set(normal_GI.split("_"))) == 2:
        # Both alleles in normal (e.g., het)
        case_muttype = "homhet_normhet_onediff"
        muttype = "hethet"
        source_allele = list(
            set(normal_GI.split("_")) - set(case_mosaic.split("_"))
        )[0]

    return germ_allele, source_allele, muttype, case_muttype


def handle_hethet_case_onediff(case_mosaic, normal_GI, case_dp_dict):
    """
    Handle the 'hethet' mutation type case (heterozygous mutation) where the case and normal
    samples share some alleles, but the case has a mutation. This function determines the
    germline allele, source allele, and mutation allele from the case and normal alleles.

    Args:
    - case_mosaic (str): The case mosaic alleles as a string (e.g., '1_1_2_3').
    - normal_GI (str): The normal germline alleles as a string (e.g., '1_2').
    - case_dp_dict (dict): A dictionary with case allele depth information for each allele.

    Returns:
    - germ_allele (int): The allele representing the germline (the overlapping allele).
    - source_allele (int): The source allele used in mutation analysis (the non-overlapping allele).
    """

    # Identify the overlap between case mosaic alleles and normal germline alleles
    overlap_allele = list(
        set(case_mosaic.split("_")) & set(normal_GI.split("_"))
    )

    # If there are less than two overlapping alleles, we need to handle the case as a special situation
    if len(overlap_allele) < 2:
        raise ValueError(
            "Insufficient overlap between case mosaic and normal germline"
            " alleles."
        )

    # Convert the overlap alleles into integers (assuming they are represented as indices)
    germ_allele1 = int(overlap_allele[0])
    germ_allele2 = int(overlap_allele[1])

    # Get the depth for both germ alleles in the case
    germ_allele1_dp = case_dp_dict.get(germ_allele1, 0)
    germ_allele2_dp = case_dp_dict.get(germ_allele2, 0)

    # Choose the germ_allele and source_allele based on depth information
    if germ_allele2_dp > germ_allele1_dp:
        germ_allele = germ_allele2
        source_allele = germ_allele1
    else:
        germ_allele = germ_allele1
        source_allele = germ_allele2

    # Return the results
    return germ_allele, source_allele


def get_pair_return_value(status):
    return pd.Series(
        {
            "status_pair": status,
            "filter1_pair": ".",
            "filter2_pair": ".",
            "filter3_pair": ".",
            "filter4_pair": ".",
            "normal_dp_pair": ".",
            "case_dp_pair": ".",
            "germ_allele_pair": ".",
            "source_allele_pair": ".",
            "mutation_allele_pair": ".",
            "case_muttype_pair": ".",
            "mut_source_seq_pair": ".",
            "mut_target_seq_pair": ".",
            "mut_source_seq_pos_tuple_0_based_leftrightclose_pair": ".",
            "mut_target_seq_pos_tuple_0_based_leftrightclose_pair": ".",
            "reference_start_index_0_based_include_pair": ".",
            "reference_end_index_0_based_include_pair": ".",
            "reference_start_coordinate_1_based_include_pair": ".",
            "reference_end_coordinate_1_based_include_pair": ".",
            "mutated_reference_seq_pair": ".",
            "relative_to_reference_germ_seq_pair": ".",
            "relative_to_reference_source_seq_pair": ".",
            "relative_to_reference_mut_seq_pair": ".",
            "alleles_mut_type_pair": ".",
            "mutation_site_type_pair": ".",
            "mutation_location_type_pair": ".",
            "germ_allele_seq_pair": ".",
            "source_allele_seq_pair": ".",
            "mutation_allele_seq_pair": ".",
            "obs_mut_vaf_new_case_pair": ".",
            "obs_mut_vaf_new_normal_pair": ".",
            "pvalue_pair": ".",
        }
    )


def case_pair_mosaic_calling(row, pysam_fasta):
    # HACK: 缺了一种突变类型，就是两个 allele 是一样的突变类型，可能是来源两个 allele 都突变，也可能来自两个不一样的 allele，其中一个 germline allele 突变成另外一个 germline allele 导致两个不一样的 alleles 变成两个一样的 alleles
    # 但是这种情况发生的概率非常小
    # types
    # single allele diff (homhet and hom/het and have one overlap) (hethet and het and have two overlap),exclude artifacts calling
    # two allele diff (homhet and hom/het and no overlap) (hethet and hom/het and have one overlap)
    # three allele diff (hethet and hom/het and no overlap)
    # ok!
    # whether mutation or not，mutation type from allele1 to allele2
    all_allele_seq = row["all_alleles"]
    normal_GI = row["Normal_GI"]
    case_mosaic = row["Case_Moasic"]
    case_used_all_alleles_dp = row["case_used_all_alleles_dp"]
    normal_used_all_alleles_dp = row["normal_used_all_alleles_dp"]
    chrom, STRSTART, STREND = row["chrom"], row["STRSTART"], row["STREND"]
    normal_dp = get_dp_according_aad(normal_used_all_alleles_dp)
    case_dp = get_dp_according_aad(case_used_all_alleles_dp)
    normal_dp_dict = get_allele_dp_dict(normal_used_all_alleles_dp)
    case_dp_dict = get_allele_dp_dict(case_used_all_alleles_dp)
    if case_mosaic == "._._._.":
        return get_pair_return_value("case_uncallable_mosaic")
    if normal_GI == ".":
        return get_pair_return_value("normal_uncallable_germline")
    if normal_dp == "NoInforsAvailable" or case_dp == "NoInforsAvailable":
        return get_pair_return_value("NoAlleleInformationAvailable")
    if (normal_dp < NORMAL_DP_CUTOFF) or (case_dp < CASE_DP_CUTOFF):
        return get_pair_return_value("NoEnoughDPForSamples")
    # Mutation detection
    diff_allele_norm_GI_num = int(row["case_minus_normal"])
    diff_allele_norm_recurrent_num = int(row["case_minus_normal_mosaic"])
    if diff_allele_norm_GI_num == 1:
        if len(set(case_mosaic.split("_"))) == 2:
            # Case 1: homhet, mutation 1 difference
            (
                germ_allele,
                source_allele,
                muttype,
                case_muttype,
            ) = handle_homhet_case_onediff(case_mosaic, normal_GI)
        elif len(set(case_mosaic.split("_"))) == 3:
            # Case 2: hethet, mutation 1 difference
            muttype = "hethet"
            case_muttype = "hethet_normhet_onediff"
            germ_allele, source_allele = handle_hethet_case_onediff(
                case_mosaic, normal_GI, case_dp_dict
            )
        mutation_allele = list(
            set(case_mosaic.split("_")) - set(normal_GI.split("_"))
        )[0]

    elif diff_allele_norm_GI_num == 2:
        if len(set(case_mosaic.split("_"))) == 2:
            # Case 3: homhet, mutation 2 differences
            muttype = "homhet"
            if len(set(normal_GI.split("_"))) == 1:
                case_muttype = "homhet_normhom_twodiff"
            elif len(set(normal_GI.split("_"))) == 2:
                case_muttype = "homhet_normhet_twodiff"
            case_allele_list = list(set(case_mosaic.split("_")))
            case_allele_dp_list = [
                case_dp_dict[int(i)] for i in case_allele_list]
            norm_allele_list = list(set(normal_GI.split("_")))
            norm_allele_dp_list = [
                normal_dp_dict[int(i)] for i in norm_allele_list]
            mutation_allele = case_allele_list[np.argmax(case_allele_dp_list)]
            source_allele = norm_allele_list[np.argmax(norm_allele_dp_list)]
            germ_allele = source_allele
        elif len(set(case_mosaic.split("_"))) == 3:
            # Case 4: hethet, mutation 2 differences
            if len(set(normal_GI.split("_"))) == 1:
                case_muttype = "hethet_normhom_twodiff"
            elif len(set(normal_GI.split("_"))) == 2:
                case_muttype = "hethet_normhet_twodiff"
            overlap_allele = list(
                set(case_mosaic.split("_")) & set(normal_GI.split("_"))
            )[0]
            case_allele_list = list(
                set(case_mosaic.split("_")) - set(normal_GI.split("_"))
            )
            case_allele_dp_list = [
                case_dp_dict[int(i)] for i in case_allele_list]
            norm_allele_list = list(
                set(normal_GI.split("_")) - set(case_mosaic.split("_"))
            )
            norm_allele_dp_list = [
                normal_dp_dict[int(i)] for i in norm_allele_list]
            max_case_dp_allele = case_allele_list[
                np.argmax(case_allele_dp_list)
            ]
            if len(norm_allele_list) == 0:
                source_allele = overlap_allele
                germ_allele = overlap_allele
                muttype = (  # infact hethet but for get_reference_chr_pos_ref_alt
                    "homhet"
                )
            else:
                source_allele = norm_allele_list[
                    np.argmax(norm_allele_dp_list)
                ]
                germ_allele = overlap_allele
                muttype = "hethet"
            mutation_allele = max_case_dp_allele
    elif diff_allele_norm_GI_num == 3:
        # Case 5: hethet, mutation 3 differences
        muttype = (  # infact hethet but for get_reference_chr_pos_ref_alt
            "homhet"
        )
        if len(set(normal_GI.split("_"))) == 1:
            case_muttype = "hethet_normhom_threediff"
        elif len(set(normal_GI.split("_"))) == 2:
            case_muttype = "hethet_normhet_threediff"
        case_allele_list = list(set(case_mosaic.split("_")))
        case_allele_dp_list = [case_dp_dict[int(i)] for i in case_allele_list]
        norm_allele_list = list(set(normal_GI.split("_")))
        norm_allele_dp_list = [
            normal_dp_dict[int(i)] for i in norm_allele_list]
        mutation_allele = case_allele_list[np.argmax(case_allele_dp_list)]
        source_allele = norm_allele_list[np.argmax(norm_allele_dp_list)]
        germ_allele = source_allele
    # Get reference and alternative sequences
    (
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
        alleles_mut_type,
        mutation_site_type,
        mutation_location_type,
    ) = get_reference_chr_pos_ref_alt(
        chrom,
        STRSTART,
        STREND,
        muttype,
        all_allele_seq[int(germ_allele)],
        all_allele_seq[int(source_allele)],
        all_allele_seq[int(mutation_allele)],
        pysam_fasta,
    )

    # Calculate VAFs

    (
        obs_mut_vaf_new_case,
        obs_mut_vaf_new_normal,
        obs_mut_dp_normal,
    ) = calculate_vaf(
        case_dp_dict, normal_dp_dict, mutation_allele, case_dp, normal_dp
    )

    # Apply filters based on VAF and other conditions
    # res = binomtest(
    #     obs_mut_dp_normal,
    #     n=normal_dp,
    #     p=obs_mut_vaf_new_case,
    #     alternative="less",
    # )
    case_mut_dp = case_dp*obs_mut_vaf_new_case
    res = binomtest(
        round(case_mut_dp),
        n=case_dp,
        p=obs_mut_vaf_new_normal,
        alternative="greater",
    )
    pvalue = res.pvalue
    filter1, filter2, filter3, filter4 = apply_filters(
        obs_mut_vaf_new_case,
        obs_mut_vaf_new_normal,
        obs_mut_dp_normal,
        pvalue,
        diff_allele_norm_recurrent_num,
    )

    return pd.Series(
        {
            "status_pair": "Callable",
            "filter1_pair": filter1,
            "filter2_pair": filter2,
            "filter3_pair": filter3,
            "filter4_pair": filter4,
            "normal_dp_pair": normal_dp,
            "case_dp_pair": case_dp,
            "germ_allele_pair": germ_allele,
            "source_allele_pair": source_allele,
            "mutation_allele_pair": mutation_allele,
            "case_muttype_pair": case_muttype,
            "mut_source_seq_pair": mut_source_seq,
            "mut_target_seq_pair": mut_target_seq,
            "mut_source_seq_pos_tuple_0_based_leftrightclose_pair": mut_source_seq_pos_tuple_0_based_leftrightclose,
            "mut_target_seq_pos_tuple_0_based_leftrightclose_pair": mut_target_seq_pos_tuple_0_based_leftrightclose,
            "reference_start_index_0_based_include_pair": reference_start_index_0_based_include,
            "reference_end_index_0_based_include_pair": reference_end_index_0_based_include,
            "reference_start_coordinate_1_based_include_pair": reference_start_coordinate_1_based_include,
            "reference_end_coordinate_1_based_include_pair": reference_end_coordinate_1_based_include,
            "mutated_reference_seq_pair": mutated_reference_seq,
            "relative_to_reference_germ_seq_pair": relative_to_reference_germ_seq,
            "relative_to_reference_source_seq_pair": relative_to_reference_source_seq,
            "relative_to_reference_mut_seq_pair": relative_to_reference_mut_seq,
            "alleles_mut_type_pair": alleles_mut_type,
            "mutation_site_type_pair": mutation_site_type,
            "mutation_location_type_pair": mutation_location_type,
            "germ_allele_seq_pair": all_allele_seq[int(germ_allele)],
            "source_allele_seq_pair": all_allele_seq[int(source_allele)],
            "mutation_allele_seq_pair": all_allele_seq[int(mutation_allele)],
            "obs_mut_vaf_new_case_pair": obs_mut_vaf_new_case,
            "obs_mut_vaf_new_normal_pair": obs_mut_vaf_new_normal,
            "pvalue_pair": pvalue,
        }
    )


def mutation_decision(
    row
):
    status_pair = row["status_pair"]
    if status_pair == "Callable":
        filter1_pair = row["filter1_pair"]
        filter2_pair = row["filter2_pair"]
        filter3_pair = row["filter3_pair"]
        filter4_pair = row["filter4_pair"]
        case_muttype_pair = row["case_muttype_pair"]
        obs_mut_vaf_new_case_pair = row["obs_mut_vaf_new_case_pair"]
        RF_Prediction = row["RF_Prediction"]
        hard_filter_decision = row["hard_filter_decision"]
        hard_filter_category = row["hard_filter_category"]
        if pd.isna(hard_filter_category):
            hard_filter_category = ""
        unknown_hap_read_fraction = float(row["unknown_hap_read_fraction"])
        case_minus_normal_mosaic = int(row["case_minus_normal_mosaic"])
        case_dp = row["case_dp_pair"]
        if case_muttype_pair in ["hethet_normhet_onediff", "homhet_normhet_onediff", "homhet_normhom_onediff"]:
            if RF_Prediction != "Artifacts":
                filter5 = "PASS"
            else:
                filter5 = "RFArtifacts"
            if hard_filter_decision == "Mosaic mutations":
                filter6 = "PASS"
            else:
                filter6 = hard_filter_decision
            if (("LowMutantArtifacts" in hard_filter_category) or ("LowBaseQArtifacts" in hard_filter_category)):
                filter7 = hard_filter_category
            else:
                filter7 = "PASS"
        else:
            if STRICT_MODE:
                if RF_Prediction != "Artifacts":
                    filter5 = "PASS"
                else:
                    filter5 = "RFArtifacts"
                if hard_filter_decision == "Mosaic mutations":
                    filter6 = "PASS"
                else:
                    filter6 = hard_filter_decision
                filter7 = "PASS"
            else:
                filter5 = "PASS"
                filter6 = "PASS"
                filter7 = "PASS"
        unknown_hap_read_number = round(unknown_hap_read_fraction*case_dp)
        # res = binomtest(
        #     unknown_hap_read_number,
        #     n=case_dp,
        #     p=obs_mut_vaf_new_case_pair,
        #     alternative="less",
        # )
        res = binomtest(
            round(obs_mut_vaf_new_case_pair*case_dp),
            n=case_dp,
            p=unknown_hap_read_fraction,
            alternative="greater",
        )
        pvalue = res.pvalue
        if ((pvalue < 0.05) or (unknown_hap_read_number == 0)):
            filter8 = "PASS"
        else:
            filter8 = "HighUnknownReads"
        if ((case_minus_normal_mosaic == 0) & (obs_mut_vaf_new_case_pair < RECURRENT_CASE_VAF_CUTOFF)):
            filter9 = "RecurrentLowCaseVAF"
        else:
            filter9 = "PASS"
        if case_muttype_pair in CASE_MUTATION:
            filter10 = "PASS"
        else:
            filter10 = case_muttype_pair
        filter_pair = filter1_pair + "_" + filter2_pair + "_" + filter3_pair + "_" + filter4_pair + \
            "_" + filter5 + "_" + filter6 + "_" + filter7 + \
            "_" + filter8 + "_" + filter9 + "_" + filter10
        if filter_pair == "PASS_PASS_PASS_PASS_PASS_PASS_PASS_PASS_PASS_PASS":
            decision = "Mosaic mutations"
        else:
            decision = filter_pair
    else:
        decision = status_pair
    return pd.Series({
        "paired_mode_prediction": decision
    })


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="BulkMonSTR pair-samples mode prediction")

    # Define the command-line arguments
    parser.add_argument(
        "-rf", "--recurrent_file", help="Recurrent file for analysis"
    )
    parser.add_argument(
        "-bf", "--pop_gi_file", help="Population allele frequency file"
    )
    parser.add_argument(
        "-i", "--metadata", help="Sample metadata file"
    )
    parser.add_argument(
        "-v", "--vcf_file", help="VCF file containing variant information"
    )
    parser.add_argument(
        "-n", "--normal_name", help="Normal sample name"
    )
    parser.add_argument(
        "-c", "--case_name", help="Case sample name"
    )
    parser.add_argument(
        "-bo", "--BulkMonSTR_output", help="BulkMonSTR output file"
    )
    parser.add_argument(
        "-re", "--reference_fasta", help="Reference genome FASTA file"
    )
    parser.add_argument(
        "-o", "--output", help="BulkMonSTR paired-samples mode output"
    )

    # Parse the arguments
    args = parser.parse_args()

    recurrent_file = args.recurrent_file
    pop_gi_file = args.pop_gi_file
    sample_meta = args.metadata
    vcf_file = args.vcf_file
    normal_name = args.normal_name
    case_name = args.case_name
    BulkMonSTR_output = args.BulkMonSTR_output
    reference_fasta = args.reference_fasta
    output = args.output
    metadata = pd.read_csv(sample_meta, sep=",", index_col=0)
    bam_files = metadata.iloc[:, 4]

    if GIAB:
        sample_name_list = [
            "_".join(bam.split("/")[-1].split(".")[0:-1]) for bam in bam_files
        ]
    else:
        sample_name_list = [
            bam.split("/")[-1].split(".")[0] for bam in bam_files
        ]
    bulkpre = pd.read_csv(BulkMonSTR_output, header=0, index_col=0)
    if "chrom" in bulkpre.columns:
        bulkpre["chrom"] = bulkpre["chrom"].astype(str)
    vcf_in = pysam.VariantFile(vcf_file)
    pysam_fasta = pysam.FastaFile(reference_fasta)
    recurrent_gt = pd.read_csv(recurrent_file,
                               sep="\t",
                               header=None,
                               index_col=None)
    recurrent_gt.columns = ["chr", "start", "end", "motif_length", "period",
                            "str_id", "motif"] + sample_name_list + ["uncallable_num", "uncallable_frac"]
    pop_gi = pd.read_csv(pop_gi_file, sep="\t", header=None, index_col=None)
    pop_gi.columns = ["chr", "start", "end", "motif_length",
                      "period", "str_id", "motif"] + sample_name_list
    recurrent_gt.index = recurrent_gt["str_id"]
    pop_gi.index = pop_gi["str_id"]

    germline_filter_per_sample = pd.DataFrame(
        index=pop_gi.index,
        columns=["Normal_GI", "Case_GI", "Normal_Mosaic", "Case_Moasic", "case_minus_normal",
                 "normal_minus_case", "case_minus_normal_mosaic", "normal_minus_case_mosaic"]
    )
    germline = transform_mosaic_gt(pop_gi[normal_name])
    case_germline = transform_mosaic_gt(pop_gi[case_name])
    case_mosaic = transform_mosaic_gt(
        recurrent_gt.loc[pop_gi.index, case_name])
    normal_mosaic = transform_mosaic_gt(
        recurrent_gt.loc[pop_gi.index, normal_name])
    GI_diff = (case_mosaic - germline).str.len()
    GI_diff2 = (germline-case_mosaic).str.len()
    recurrent_diff = (case_mosaic - normal_mosaic).str.len()
    recurrent_diff2 = (normal_mosaic - case_mosaic).str.len()
    germline_filter_per_sample["Normal_GI"] = pop_gi.loc[germline_filter_per_sample.index, normal_name]
    germline_filter_per_sample["Case_GI"] = pop_gi.loc[germline_filter_per_sample.index, case_name]
    germline_filter_per_sample["Normal_Mosaic"] = recurrent_gt.loc[germline_filter_per_sample.index, normal_name]
    germline_filter_per_sample["Case_Moasic"] = recurrent_gt.loc[germline_filter_per_sample.index, case_name]
    germline_filter_per_sample["case_minus_normal"] = GI_diff
    germline_filter_per_sample["normal_minus_case"] = GI_diff2
    germline_filter_per_sample["case_minus_normal_mosaic"] = recurrent_diff
    germline_filter_per_sample["normal_minus_case_mosaic"] = recurrent_diff2
    germline_filter_per_sample_filtered = germline_filter_per_sample[(
        germline_filter_per_sample["case_minus_normal"] > 0)]
    bulkpre_case = bulkpre[bulkpre["sample_name"] == case_name]
    bulkpre_case = bulkpre_case[bulkpre_case["str_id.1"].isin(
        germline_filter_per_sample_filtered.index)]
    bulkpre_case["Normal_GI"] = bulkpre_case["str_id.1"].map(
        germline_filter_per_sample_filtered["Normal_GI"])
    bulkpre_case["Case_GI"] = bulkpre_case["str_id.1"].map(
        germline_filter_per_sample_filtered["Case_GI"])
    bulkpre_case["Normal_Mosaic"] = bulkpre_case["str_id.1"].map(
        germline_filter_per_sample_filtered["Normal_Mosaic"])
    bulkpre_case["Case_Moasic"] = bulkpre_case["str_id.1"].map(
        germline_filter_per_sample_filtered["Case_Moasic"])
    bulkpre_case["case_minus_normal"] = bulkpre_case["str_id.1"].map(
        germline_filter_per_sample_filtered["case_minus_normal"])
    bulkpre_case["case_minus_normal_mosaic"] = bulkpre_case["str_id.1"].map(
        germline_filter_per_sample_filtered["case_minus_normal_mosaic"])
    bulkpre_case["normal_minus_case"] = bulkpre_case["str_id.1"].map(
        germline_filter_per_sample_filtered["normal_minus_case"])
    bulkpre_case["normal_minus_case_mosaic"] = bulkpre_case["str_id.1"].map(
        germline_filter_per_sample_filtered["normal_minus_case_mosaic"])
    if bulkpre_case.empty:
        pass
    else:
        bulkpre_case[['all_alleles', 'case_used_all_alleles_dp', 'case_allele_length_dp', 'normal_used_all_alleles_dp',
                      'normal_allele_length_dp']] = bulkpre_case.apply(lambda row: get_vcf_infors(row, vcf_in, case_name, normal_name), axis=1)
        bulkpre_case[['status_pair',
                      'filter1_pair',
                      'filter2_pair',
                      'filter3_pair',
                      'filter4_pair',
                      'normal_dp_pair',
                      'case_dp_pair',
                      'germ_allele_pair',
                      'source_allele_pair',
                      'mutation_allele_pair',
                      'case_muttype_pair',
                      'mut_source_seq_pair',
                      'mut_target_seq_pair',
                      'mut_source_seq_pos_tuple_0_based_leftrightclose_pair',
                      'mut_target_seq_pos_tuple_0_based_leftrightclose_pair',
                      'reference_start_index_0_based_include_pair',
                      'reference_end_index_0_based_include_pair',
                      'reference_start_coordinate_1_based_include_pair',
                      'reference_end_coordinate_1_based_include_pair',
                      'mutated_reference_seq_pair',
                      'relative_to_reference_germ_seq_pair',
                      'relative_to_reference_source_seq_pair',
                      'relative_to_reference_mut_seq_pair',
                      'alleles_mut_type_pair',
                      'mutation_site_type_pair',
                      'mutation_location_type_pair',
                      'germ_allele_seq_pair',
                      'source_allele_seq_pair',
                      'mutation_allele_seq_pair',
                      'obs_mut_vaf_new_case_pair',
                      'obs_mut_vaf_new_normal_pair',
                      'pvalue_pair'
                      ]] = bulkpre_case.apply(
            lambda row: case_pair_mosaic_calling(row, pysam_fasta), axis=1)
        bulkpre_case["popAF_mut_new"] = np.select(
        [
        bulkpre_case["mutation_allele_seq_pair"] == bulkpre_case["germ_seq"],
        bulkpre_case["mutation_allele_seq_pair"] == bulkpre_case["source_seq"],
            bulkpre_case["mutation_allele_seq_pair"] == bulkpre_case["mut_seq"],
        ],
        [
            bulkpre_case["external_germ_popAF"],
            bulkpre_case["external_source_popAF"],
            bulkpre_case["external_mosaic_popAF"],
        ], default=np.nan)
        bulkpre_case[["paired_mode_prediction"]] = bulkpre_case.apply(
            lambda row: mutation_decision(row), axis=1)
        bulkpre_case = bulkpre_case[bulkpre_case["paired_mode_prediction"]
                                    == "Mosaic mutations"]
        bulkpre_case["mutation_id"] = bulkpre_case["chrom"] + "_" + bulkpre_case["reference_start_coordinate_1_based_include_pair"].astype(
            str) + "_"+bulkpre_case["mut_source_seq_pair"].astype(str)+"_"+bulkpre_case["mut_target_seq_pair"].astype(str)
        bulkpre_case["mutation_base_pairs"] = bulkpre_case["mutation_allele_seq_pair"].apply(
            lambda x: len(x)) - bulkpre_case["source_allele_seq_pair"].apply(lambda x: len(x))
        bulkpre_case["mutant_dp_pair"] = bulkpre_case["case_dp_pair"].astype(
            int)*bulkpre_case["obs_mut_vaf_new_case_pair"].astype(float)
        bulkpre_case["mutant_dp_pair"] = bulkpre_case["mutant_dp_pair"].apply(
            lambda x: round(x))
        bulkpre_case["case alt/dp"] = bulkpre_case["mutant_dp_pair"].astype(str) + "/" + bulkpre_case["case_dp_pair"].astype(str)
        bulkpre_case["normal_mutant_dp_pair"] = bulkpre_case["normal_dp_pair"].astype(
            int)*bulkpre_case["obs_mut_vaf_new_normal_pair"].astype(float)
        bulkpre_case["normal_mutant_dp_pair"] = bulkpre_case["normal_mutant_dp_pair"].apply(
            lambda x: round(x))
        bulkpre_case["control alt/dp"] = bulkpre_case["normal_mutant_dp_pair"].astype(str) + "/" + bulkpre_case["normal_dp_pair"].astype(str)
        match_selected_col = {
            "mutation_id": "mutation_id",
            "chrom": "chrom",
            "reference_start_coordinate_1_based_include_pair": "reference_start_coordinate_1_based",
            "mut_source_seq_pair": "mutated_seq",
            "mut_target_seq_pair": "mutation_seq",
            "sample_name": "sample_name",
            "STRSTART": "STR_start",
            "STREND": "STR_end",
            "MOTIF_length": "motif length",
            "PERIOD": "period",
            "str_id.1": "STR id from HipSTR panel",
            "MOTIF": "motif",
            "alleles_mut_type_pair": "alleles_mut_type",
            "case_muttype_pair": "muttype_pair",
            "mutation_base_pairs": "mutation_base_pairs",
            "used_read_num_in_genotyping": "used_read_num_in_genotyping",
            "case alt/dp": "case alt/dp",
            "control alt/dp": "control alt/dp",
            "no_norm_len_revise_depth_ratio": "depth_ratio",
            "obs_mut_vaf_new_case_pair": "vaf_obs",
            "mutant_dp_pair": "mutant_dp",
            "popAF_mut_new": "popAF_mut",
            "inframe_single_step_prob": 'inframe_step_size_prob',
            'inframe_ins_prob': 'inframe_ins_prob',
            'inframe_del_prob': 'inframe_del_prob',
            'outframe_single_step_prob': "outframe_step_size_prob",
            'outframe_ins_prob': 'outframe_ins_prob',
            "outframe_del_prob": "outframe_del_prob",
            'mutated_reference_seq_pair': "mutated_reference_seq",
            'relative_to_reference_germ_seq_pair': "relative_to_reference_germ_seq",
            'relative_to_reference_source_seq_pair': "relative_to_reference_source_seq",
            'relative_to_reference_mut_seq_pair': "relative_to_reference_mut_seq",
            "germ_allele_seq_pair": "germ_seq",
            "source_allele_seq_pair": "source_seq",
            "mutation_allele_seq_pair": "mut_seq"
        }

        bulkpre_case_selected = bulkpre_case[match_selected_col.keys()]
        bulkpre_case_selected.columns = match_selected_col.values()
        bulkpre_case_selected.to_csv(output, header=True, index=True)
