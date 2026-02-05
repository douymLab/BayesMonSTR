ALIGNMENT_MODE = "STR-specific"  # Needleman-Wunsch
REF_ALIGNMENT_BASED_FLANKING_JUDGE = False
from Bio import Align
import numpy as np
import pdb

SHRINK_ALLELE = False
PADDING_BPS = 5  # 10 # revise_padding


def no_stutter_str_alignment(hap, reads_str_block):
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    alignment = hap == reads_str_block
    return alignment


def stutter_deletion(hap, reads_str_block):
    deletion_size = len(hap) - len(reads_str_block)
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    all_deletion_location = len(hap) - deletion_size + 1
    # sequencing_error_likelihood_dict={}
    deletion_site = 0
    max_score = 0
    output_alignment = (
        np.delete(
            hap,
            np.arange(0, 0 + deletion_size),
        )
        == reads_str_block
    )
    for deletion_location in np.arange(0, all_deletion_location):
        hap_deletion = np.delete(
            hap,
            np.arange(deletion_location, deletion_location + deletion_size),
        )
        alignment = hap_deletion == reads_str_block
        alignment_score = np.sum(alignment)
        if alignment_score > max_score:
            max_score = alignment_score
            deletion_site = deletion_location
            output_alignment = alignment

    # if deletion_site == 0:
    #     output_alignment = ["-"] * deletion_size + list(
    #         output_alignment[deletion_site:]
    #     )
    # else:
    #     output_alignment = (
    #         list(output_alignment[:deletion_site])
    #         + ["-"] * deletion_size
    #         + list(output_alignment[deletion_site:])
    #     )
    return deletion_site, output_alignment


def stutter_insertion(hap, reads_str_block):
    insertion_size = len(reads_str_block) - len(hap)
    hap = np.array(list(hap))
    reads_str_block = np.array(list(reads_str_block))
    all_insertion_location = len(hap) + 1
    insertion_site = 0
    max_score = 0
    output_alignment = (
        np.delete(
            reads_str_block,
            np.arange(0, 0 + insertion_size),
        )
        == hap
    )
    for insertion_location in np.arange(0, all_insertion_location):
        reads_str_block_remove_insertion = np.delete(
            reads_str_block,
            np.arange(insertion_location, insertion_location + insertion_size),
        )
        reads_str_block_remove_insertion_alignment = (
            reads_str_block_remove_insertion == hap
        )
        alignment_score = np.sum(reads_str_block_remove_insertion_alignment)

        if alignment_score > max_score:
            max_score = alignment_score
            insertion_site = insertion_location
            output_alignment = reads_str_block_remove_insertion_alignment
    # if insertion_site == 0:
    #     output_alignment = ["-"] * insertion_size + list(
    #         output_alignment[insertion_site:]
    #     )
    # else:
    #     output_alignment = (
    #         list(output_alignment[:insertion_site])
    #         + ["-"] * insertion_size
    #         + list(output_alignment[insertion_site:])
    #     )
    return insertion_site, output_alignment


def find_first_and_last_false(arr):
    first_index = None
    last_index = None
    # print(arr)
    # 遍历数组以找到第一个和最后一个 False 的索引
    for i, value in enumerate(arr):
        if value == False:
            if first_index is None:
                first_index = i
            last_index = i

    return first_index, last_index


def STR_specific_alignment_only(seq1, seq2):
    # This function is to align two sequences based on the STR specific alignment
    if len(seq1) == len(seq2):
        alignment = no_stutter_str_alignment(seq1, seq2)
        indel_site = None
        mutation_size = 0
    elif len(seq1) > len(seq2):
        indel_site, alignment = stutter_deletion(seq1, seq2)
        mutation_size = len(seq2) - len(seq1)
    elif len(seq1) < len(seq2):
        indel_site, alignment = stutter_insertion(seq1, seq2)
        mutation_size = len(seq2) - len(seq1)
    return alignment, mutation_size, indel_site


def STR_specific_alignment(seq1, seq2, first_pad_base):
    # HACK: TODO: when seq1 or seq2 is "", may produce errors: TypeError: 'bool' object is not iterable # TODO TO DEBUG
    # This function is to align two sequences based on the STR specific alignment
    if len(seq1) == len(seq2):
        alignment = no_stutter_str_alignment(seq1, seq2)
        indel_site = None
        mutation_size = 0
    elif len(seq1) > len(seq2):
        indel_site, alignment = stutter_deletion(seq1, seq2)
        mutation_size = len(seq1) - len(seq2)
    elif len(seq1) < len(seq2):
        indel_site, alignment = stutter_insertion(seq1, seq2)
        mutation_size = len(seq2) - len(seq1)
    # pdb.set_trace()
    first_index, last_index = find_first_and_last_false(alignment)
    if indel_site is None:
        if first_index == last_index:
            type = "SNV"
            mutation_site_seq1 = [
                first_index,
                last_index,
            ]  # left and right included
            mutation_site_seq2 = [
                first_index,
                last_index,
            ]  # left and right included
        else:
            if True in alignment[first_index : last_index + 1]:
                type = "noncontinuous_MNV"
            else:
                type = "continuous_MNV"
            mutation_site_seq1 = [
                first_index,
                last_index,
            ]  # left and right included
            mutation_site_seq2 = [
                first_index,
                last_index,
            ]  # left and right included
    else:
        if last_index == None:
            if len(seq1) > len(seq2):
                type = "Deletion"
                mutation_site_seq1 = [
                    indel_site - 1,
                    indel_site + mutation_size - 1,
                ]  # left+1 and right included is deleted,left is keeped
                mutation_site_seq2 = [indel_site - 1, indel_site - 1]
            else:
                type = "Insertion"
                mutation_site_seq1 = [indel_site - 1, indel_site - 1]
                mutation_site_seq2 = [
                    indel_site - 1,
                    indel_site + mutation_size - 1,
                ]  # left+1 and right included is deleted,left is keeped
        # extra process when indel_site is 0
        else:
            if first_index == last_index:
                if len(seq1) > len(seq2):
                    type = "Deletion_SNV"
                    if indel_site - 1 >= last_index:
                        mutation_site_seq1 = [
                            last_index,
                            indel_site + mutation_size - 1,
                        ]
                        mutation_site_seq2 = [last_index, indel_site - 1]
                    else:
                        mutation_site_seq1 = [
                            indel_site,
                            last_index + mutation_size,
                        ]
                        mutation_site_seq2 = [indel_site, last_index]
                else:
                    type = "Insertion_SNV"
                    if indel_site - 1 >= last_index:
                        mutation_site_seq1 = [last_index, indel_site - 1]
                        mutation_site_seq2 = [
                            last_index,
                            indel_site + mutation_size - 1,
                        ]
                    else:
                        mutation_site_seq1 = [indel_site, last_index]
                        mutation_site_seq2 = [
                            indel_site,
                            last_index + mutation_size,
                        ]
            else:
                if len(seq1) > len(seq2):
                    type = "Deletion_MNV"
                    if indel_site - 1 >= last_index:
                        mutation_site_seq1 = [
                            first_index,
                            indel_site + mutation_size - 1,
                        ]
                        mutation_site_seq2 = [first_index, indel_site - 1]
                    elif indel_site - 1 <= first_index:
                        mutation_site_seq1 = [
                            indel_site,
                            last_index + mutation_size,
                        ]
                        mutation_site_seq2 = [indel_site, last_index]
                    elif (
                        indel_site - 1 > first_index
                        and indel_site - 1 < last_index
                    ):
                        mutation_site_seq1 = [
                            first_index,
                            last_index + mutation_size,
                        ]
                        mutation_site_seq2 = [first_index, last_index]
                else:
                    type = "Insertion_MNV"
                    if indel_site - 1 >= last_index:
                        mutation_site_seq1 = [first_index, indel_site - 1]
                        mutation_site_seq2 = [
                            first_index,
                            indel_site + mutation_size - 1,
                        ]
                    elif indel_site - 1 <= first_index:
                        mutation_site_seq1 = [indel_site, last_index]
                        mutation_site_seq2 = [
                            indel_site,
                            last_index + mutation_size,
                        ]
                    elif (
                        indel_site - 1 > first_index
                        and indel_site - 1 < last_index
                    ):
                        mutation_site_seq1 = [first_index, last_index]
                        mutation_site_seq2 = [
                            first_index,
                            last_index + mutation_size,
                        ]
    if mutation_site_seq1[0] == -1:
        source_seq = first_pad_base + seq1[: mutation_site_seq1[1] + 1]
    else:
        source_seq = seq1[mutation_site_seq1[0] : mutation_site_seq1[1] + 1]
    if mutation_site_seq2[0] == -1:
        target_seq = first_pad_base + seq2[: mutation_site_seq2[1] + 1]
    else:
        target_seq = seq2[mutation_site_seq2[0] : mutation_site_seq2[1] + 1]
    (
        mutation_site_seq1,
        mutation_site_seq2,
        source_seq,
        target_seq,
    ) = normalize_vcf(
        type, mutation_site_seq1, mutation_site_seq2, source_seq, target_seq
    )
    return type, mutation_site_seq1, mutation_site_seq2, source_seq, target_seq


def normalize_vcf(
    type, mutation_site_seq1, mutation_site_seq2, source_seq, target_seq
):
    min_length = min(len(source_seq), len(target_seq))

    # 初始化计数器
    count = 0

    # 遍历字符串，比较每个字符
    for i in range(min_length):
        if source_seq[i] == target_seq[i]:
            count += 1
        else:
            break
    if count >= min_length:
        count = min_length - 1
    else:
        count = count
    mutation_site_seq1[0] = mutation_site_seq1[0] + count
    mutation_site_seq2[0] = mutation_site_seq2[0] + count
    source_seq, target_seq = source_seq[count:], target_seq[count:]
    min_length = min(len(source_seq), len(target_seq))
    # if type == "Insertion" or type == "Deletion":
    #     count = count - 1
    # if count == 0 or count == -1:
    #     pass
    # else:
    #     mutation_site_seq1[0] = mutation_site_seq1[0] + count
    #     mutation_site_seq2[0] = mutation_site_seq2[0] + count

    # 初始化计数器
    count_suffix = 0

    # 遍历字符串，从末尾开始比较每个字符

    for i in range(1, min_length + 1):
        if source_seq[-i] == target_seq[-i]:
            count_suffix += 1
        else:
            break
    if count_suffix >= min_length:
        count_suffix = min_length - 1
    else:
        count_suffix = count_suffix
    mutation_site_seq1[1] = mutation_site_seq1[1] - count_suffix
    mutation_site_seq2[1] = mutation_site_seq2[1] - count_suffix
    # HACK: Revise
    if count_suffix == 0:
        pass
    else:
        source_seq, target_seq = (
            source_seq[:-count_suffix],
            target_seq[:-count_suffix],
        )
    # source_seq, target_seq = (
    #     source_seq[:-count_suffix],
    #     target_seq[:-count_suffix],
    # )
    return mutation_site_seq1, mutation_site_seq2, source_seq, target_seq

    # if count_suffix == 0:
    #     if count == 0 or count == -1:
    #         pass
    #     else:
    #         source_seq, target_seq = source_seq[count:], target_seq[count:]
    #     return mutation_site_seq1, mutation_site_seq2, source_seq, target_seq
    # else:
    #     mutation_site_seq1[1] = mutation_site_seq1[1] - count_suffix
    #     mutation_site_seq2[1] = mutation_site_seq2[1] - count_suffix
    #     if count == 0 or count == -1:
    #         source_seq, target_seq = (
    #             source_seq[:-count_suffix],
    #             target_seq[:-count_suffix],
    #         )
    #     else:
    #         source_seq, target_seq = (
    #             source_seq[count:-count_suffix],
    #             target_seq[count:-count_suffix],
    #         )
    #     return mutation_site_seq1, mutation_site_seq2, source_seq, target_seq


## 目的: (1) 找出突变的位置在 STR 还是 Flanking（2）判读突变发生的类型是单位点还是多位点 （3）将 vcf 的格式改为标准化的 vcf 格式
def alignment_based_on_global_alignment(
    refseq,
    germline_allele,
    source_mutated_allele,
    mutant_allele,
    refseq_left_padding_1bp,
    flanking_length,
):
    if ALIGNMENT_MODE == "STR-specific":
        (
            type,
            mutation_site_source_mutated_allele,
            mutation_site_mutant_allele,
            source_mutated_allele_seq,
            mutant_allele_seq,
        ) = STR_specific_alignment(
            source_mutated_allele, mutant_allele, refseq_left_padding_1bp
        )
        if (
            (type == "SNV")
            or (type == "continuous_MNV")
            or (type == "Deletion")
            or (type == "Insertion")
        ):
            mutation_site_type = "Single"
        else:
            mutation_site_type = "Complex"
        if REF_ALIGNMENT_BASED_FLANKING_JUDGE:
            # TODO: 目前不用先不 DEBUG
            # DEBUG: 按照下面 else 的方法
            if refseq == mutant_allele:
                mutation_location_type = "Unknown"
            else:
                (
                    ref_mutant_type,
                    mutation_site_refseq,
                    mutation_site_mutant_allele_to_refseq,
                    refseq_seq,
                    mutant_allele_seq_to_refseq,
                ) = STR_specific_alignment(
                    refseq, mutant_allele, refseq_left_padding_1bp
                )
                if mutation_site_refseq[1] < flanking_length:
                    mutation_location_type = "flanking"
                elif mutation_site_refseq[0] >= (
                    len(refseq) - flanking_length
                ):
                    mutation_location_type = "flanking"
                elif (
                    mutation_site_refseq[0] >= flanking_length
                    and mutation_site_refseq[1] < len(refseq) - flanking_length
                ):
                    mutation_location_type = "STR"
                else:
                    mutation_location_type = "STR_flanking_spanning"
        else:
            (
                source_mutant_type,
                mutation_site_source_mutated_allele,
                mutation_site_mutant_allele_to_source,
                source_mutated_allele_seq_to_mutant,
                mutant_allele_seq_to_source,
            ) = STR_specific_alignment(
                source_mutated_allele, mutant_allele, refseq_left_padding_1bp
            )
            (
                alignment,
                mutation_size_no_abs,
                indel_site,
            ) = STR_specific_alignment_only(refseq, source_mutated_allele)
            mutation_size = abs(mutation_size_no_abs)
            # print("DEBUG:indel_site: ", indel_site)
            # print("DEBUG:mutation_size: ", mutation_size)
            if mutation_size_no_abs == 0:
                source_mutated_allele_STR_region = [
                    flanking_length,
                    len(source_mutated_allele) - flanking_length - 1,
                ]  # right-close
            elif mutation_size_no_abs > 0:  # right-close
                if indel_site < flanking_length:
                    source_mutated_allele_STR_region = [
                        flanking_length + mutation_size,
                        len(source_mutated_allele) - flanking_length - 1,
                    ]
                elif indel_site > len(refseq) - flanking_length:
                    source_mutated_allele_STR_region = [
                        flanking_length,
                        len(source_mutated_allele)
                        - flanking_length
                        - 1
                        - mutation_size,
                    ]
                else:
                    source_mutated_allele_STR_region = [
                        flanking_length,
                        len(source_mutated_allele) - flanking_length - 1,
                    ]
            else:  # right-close
                if indel_site < flanking_length:
                    if indel_site + mutation_size - 1 < flanking_length:
                        source_mutated_allele_STR_region = [
                            flanking_length - mutation_size,
                            len(source_mutated_allele) - flanking_length - 1,
                        ]
                    else:
                        source_mutated_allele_STR_region = [
                            indel_site,
                            len(source_mutated_allele) - flanking_length - 1,
                        ]  # TODO: 假如从左边的 Flanking 删除到右边的 Flanking 则计算错误,但是应该不会出现这种状况
                elif indel_site > len(refseq) - flanking_length:
                    source_mutated_allele_STR_region = [
                        flanking_length,
                        len(source_mutated_allele)
                        - flanking_length
                        + mutation_size
                        - 1,
                    ]
                else:
                    if (
                        indel_site + mutation_size - 1
                        < len(refseq) - flanking_length
                    ):
                        source_mutated_allele_STR_region = [
                            flanking_length,
                            len(source_mutated_allele) - flanking_length - 1,
                        ]
                    else:
                        offset = (
                            indel_site
                            + mutation_size
                            - 1
                            - (len(refseq) - flanking_length)
                            + 1
                        )
                        source_mutated_allele_STR_region = [
                            flanking_length,
                            len(source_mutated_allele)
                            - flanking_length
                            - 1
                            + offset,
                        ]
            # all double close #
            # if (
            #     mutation_site_source_mutated_allele[1]
            #     < source_mutated_allele_STR_region[0]
            # ):
            #     mutation_location_type = "flanking"
            # elif (
            #     mutation_site_source_mutated_allele[0]
            #     > source_mutated_allele_STR_region[1]
            # ):
            #     mutation_location_type = "flanking"
            # elif (
            #     mutation_site_source_mutated_allele[0]
            #     >= source_mutated_allele_STR_region[0]
            #     and mutation_site_source_mutated_allele[1]
            #     <= source_mutated_allele_STR_region[1]
            # ):
            #     mutation_location_type = "STR"
            # else:
            #     mutation_location_type = "STR_flanking_spanning"
            # Finish DEBUG
            if (
                mutation_site_source_mutated_allele[1]
                < source_mutated_allele_STR_region[0]
            ):
                # mutation_location_type = "flanking"
                ## SNV MNV non-con-MNV del snv_del mnv_del ##
                if type in [
                    "SNV",
                    "continuous_MNV",
                    "noncontinuous_MNV",
                    "Deletion",
                    "Deletion_MNV",
                    "Deletion_SNV",
                ]:
                    mutation_location_type = "flanking"
                ## ins
                elif type == "Insertion":
                    if mutation_site_source_mutated_allele[1] == (
                        source_mutated_allele_STR_region[0] - 1
                    ):
                        mutation_location_type = "STR"
                    else:
                        mutation_location_type = "flanking"
                ## ins-snv ins-mnv
                else:
                    if mutation_site_source_mutated_allele[1] == (
                        source_mutated_allele_STR_region[0] - 1
                    ):
                        mutation_location_type = "STR_flanking_spanning"
                    else:
                        mutation_location_type = "flanking"
            elif (
                mutation_site_source_mutated_allele[0]
                > source_mutated_allele_STR_region[1]
            ):
                if mutation_site_source_mutated_allele[0] == (
                    source_mutated_allele_STR_region[1] + 1
                ):
                    if type in ["Insertion_SNV", "Insertion_MNV"]:
                        mutation_location_type = "STR_flanking_spanning"
                    else:
                        mutation_location_type = "flanking"
                else:
                    mutation_location_type = "flanking"
            elif (
                mutation_site_source_mutated_allele[0]
                >= source_mutated_allele_STR_region[0]
                and mutation_site_source_mutated_allele[1]
                <= source_mutated_allele_STR_region[1]
            ):
                mutation_location_type = "STR"
            else:
                if mutation_site_source_mutated_allele[0] == (
                    source_mutated_allele_STR_region[0] - 1
                ):
                    if type in ["Deletion"]:  # "Deletion_MNV","Deletion_SNV"
                        mutation_location_type = "STR"
                    else:
                        mutation_location_type = "STR_flanking_spanning"
                elif mutation_site_source_mutated_allele[0] == (
                    source_mutated_allele_STR_region[1]
                ):
                    if type in ["Deletion"]:  # "Deletion_MNV","Deletion_SNV"
                        mutation_location_type = "flanking"
                    else:
                        mutation_location_type = "STR_flanking_spanning"
                else:
                    mutation_location_type = "STR_flanking_spanning"
        # print("DEBUG:")
        # print(mutation_site_source_mutated_allele)
        # print(source_mutated_allele_STR_region)
        if refseq == mutant_allele:
            ref_allele_is_ref = True
            mutation_site_refseq_mutant_allele = [np.inf, -np.inf]
        else:
            (
                ref_mutant_type,
                mutation_site_refseq_mutant_allele,
                mutation_site_mutant_allele_to_refseq,
                refseq_seq_mutant_allele,
                mutant_allele_seq_to_refseq,
            ) = STR_specific_alignment(
                refseq, mutant_allele, refseq_left_padding_1bp
            )
        if refseq == germline_allele:
            germline_allele_is_ref = True
            mutation_site_refseq_germ_allele = [np.inf, -np.inf]
        else:
            (
                ref_germ_type,
                mutation_site_refseq_germ_allele,
                mutation_site_germ_allele_to_refseq,
                refseq_seq_germ_allele,
                germ_allele_seq_to_refseq,
            ) = STR_specific_alignment(
                refseq, germline_allele, refseq_left_padding_1bp
            )
        if refseq == source_mutated_allele:
            source_mutated_allele_is_ref = True
            mutation_site_refseq_source_allele = [np.inf, -np.inf]
        else:
            (
                ref_source_type,
                mutation_site_refseq_source_allele,
                mutation_site_source_allele_to_refseq,
                refseq_seq_source_allele,
                source_allele_seq_to_refseq,
            ) = STR_specific_alignment(
                refseq, source_mutated_allele, refseq_left_padding_1bp
            )

        refseq_left_index = min(
            mutation_site_refseq_germ_allele[0],
            mutation_site_refseq_source_allele[0],
            mutation_site_refseq_mutant_allele[0],
        )
        refseq_right_index = max(
            mutation_site_refseq_germ_allele[1],
            mutation_site_refseq_source_allele[1],
            mutation_site_refseq_mutant_allele[1],
        )
        if refseq_left_index == -1:
            refseq_seq_reference = refseq[
                0 : refseq_right_index + 1
            ]
        else:
            refseq_seq_reference = refseq[
                refseq_left_index : refseq_right_index + 1
            ]
        if refseq == mutant_allele:
            mutant_allele_seq_to_refseq_alternative = "0"
        else:
            if refseq_left_index == -1:
                mutant_left_offset = mutation_site_refseq_mutant_allele[0]
            else:
                mutant_left_offset = (
                    mutation_site_refseq_mutant_allele[0] - refseq_left_index
                )
            mutant_right_offset = (
                refseq_right_index - mutation_site_refseq_mutant_allele[1]
            )
            mutant_allele_seq_to_refseq_alternative = mutant_allele[
                mutation_site_mutant_allele_to_refseq[0]
                - mutant_left_offset : mutation_site_mutant_allele_to_refseq[1]
                + mutant_right_offset
                + 1
            ]
        if refseq == germline_allele:
            germ_allele_seq_to_refseq_alternative = "0"
        else:
            if refseq_left_index == -1:
                germ_left_offset = mutation_site_refseq_germ_allele[0]
            else:
                germ_left_offset = (
                    mutation_site_refseq_germ_allele[0] - refseq_left_index
                )
            germ_right_offset = (
                refseq_right_index - mutation_site_refseq_germ_allele[1]
            )
            germ_allele_seq_to_refseq_alternative = germline_allele[
                mutation_site_germ_allele_to_refseq[0]
                - germ_left_offset : mutation_site_germ_allele_to_refseq[1]
                + germ_right_offset
                + 1
            ]
        if refseq == source_mutated_allele:
            source_allele_seq_to_refseq_alternative = "0"
        else:
            if refseq_left_index == -1:
                source_left_offset = mutation_site_refseq_source_allele[0]
            else:
                source_left_offset = (
                    mutation_site_refseq_source_allele[0] - refseq_left_index
                )
            source_right_offset = (
                refseq_right_index - mutation_site_refseq_source_allele[1]
            )
            source_allele_seq_to_refseq_alternative = source_mutated_allele[
                mutation_site_source_allele_to_refseq[0]
                - source_left_offset : mutation_site_source_allele_to_refseq[1]
                + source_right_offset
                + 1
            ]
        if refseq_left_index == -1:
            refseq_seq_reference = (
                refseq_left_padding_1bp + refseq_seq_reference
            )
            if refseq != mutant_allele:
                mutant_allele_seq_to_refseq_alternative = (
                    refseq_left_padding_1bp
                    + mutant_allele_seq_to_refseq_alternative
                )
            if refseq != germline_allele:
                germ_allele_seq_to_refseq_alternative = (
                    refseq_left_padding_1bp + germ_allele_seq_to_refseq_alternative
                )
            if refseq != source_mutated_allele:
                source_allele_seq_to_refseq_alternative = (
                    refseq_left_padding_1bp
                    + source_allele_seq_to_refseq_alternative
                )
        return (
            mutation_site_type,
            mutation_location_type,
            refseq_seq_reference,
            germ_allele_seq_to_refseq_alternative,
            source_allele_seq_to_refseq_alternative,
            mutant_allele_seq_to_refseq_alternative,
            refseq_left_index,
            refseq_right_index,
        )
    elif ALIGNMENT_MODE == "Needleman-Wunsch":
        pass  # TODO: support Needleman-Wunsch alignment


def adjust_pos_and_allele(vcf_pysam, var_record):
    ## exist the allele index, GT MGT PGT PMGT GI
    new_allele_index_list = []
    all_alleles = var_record.alleles
    new_allele_index_list.append(all_alleles[0])
    all_sample_new_GT_index_dict = {}
    sample_names = list(vcf_pysam.header.samples)
    all_samples_alleles = []
    str_start_zero_based_include = var_record.info["STRSTART"]
    str_end_zero_based_exclude = var_record.info["STREND"]
    for sample_name in sample_names:
        sample_GT_index = var_record.samples[sample_name]["GT"]
        sample_PGT_index = var_record.samples[sample_name]["PGT"]
        sample_MGT_index = var_record.samples[sample_name]["MGT"]
        sample_PMGT_index = var_record.samples[sample_name]["PMGT"]
        sample_GI_index = var_record.samples[sample_name]["GI"]
        if sample_GT_index[0] == None:
            continue
        else:
            all_samples_alleles.append(int(sample_GT_index[0]))
            all_samples_alleles.append(int(sample_GT_index[1]))
            # if sample_PGT_index[0] == ".":
            #     pass
            # else:
            #     all_samples_alleles.append(int(sample_PGT_index[0].split("|")[0]))
            #     all_samples_alleles.append(int(sample_PGT_index[0].split("|")[1]))
            if sample_MGT_index[0] == ".":
                pass
            else:
                all_samples_alleles.append(
                    int(sample_MGT_index[0].split("/")[0])
                )
                all_samples_alleles.append(
                    int(sample_MGT_index[0].split("/")[1])
                )
            # if sample_PMGT_index[0] == ".":
            #     pass
            # else:
            #     all_samples_alleles.append(int(sample_PMGT_index[0].split("|")[0]))
            #     all_samples_alleles.append(int(sample_PMGT_index[0].split("|")[1]))
            if sample_GI_index[0] == ".":
                pass
            else:
                all_samples_alleles.append(
                    int(sample_GI_index[0].split("/")[0])
                )
                all_samples_alleles.append(
                    int(sample_GI_index[0].split("/")[1])
                )
    all_samples_alleles = list(set(all_samples_alleles))
    if 0 in all_samples_alleles:
        all_samples_alleles.remove(0)
    all_allele_seq_array = np.array(all_alleles)[np.array(all_samples_alleles)]
    new_allele_index_list = new_allele_index_list + list(all_allele_seq_array)
    # update sample gt
    for sample_name in sample_names:
        sample_GT_index = var_record.samples[sample_name]["GT"]
        sample_PGT_index = var_record.samples[sample_name]["PGT"]
        sample_MGT_index = var_record.samples[sample_name]["MGT"]
        sample_PMGT_index = var_record.samples[sample_name]["PMGT"]
        sample_GI_index = var_record.samples[sample_name]["GI"]
        if sample_GT_index[0] == None:
            continue
        else:
            all_sample_new_GT_index_dict[sample_name] = {}
            GT1 = new_allele_index_list.index(
                all_alleles[int(sample_GT_index[0])]
            )
            GT2 = new_allele_index_list.index(
                all_alleles[int(sample_GT_index[1])]
            )
            all_sample_new_GT_index_dict[sample_name]["GT"] = (
                str(GT1) + "/" + str(GT2)
            )
            if sample_PGT_index[0] == ".":
                all_sample_new_GT_index_dict[sample_name]["PGT"] = "."
            else:
                PGT1 = new_allele_index_list.index(
                    all_alleles[int(sample_PGT_index[0].split("|")[0])]
                )
                PGT2 = new_allele_index_list.index(
                    all_alleles[int(sample_PGT_index[0].split("|")[1])]
                )
                all_sample_new_GT_index_dict[sample_name]["PGT"] = (
                    str(PGT1) + "|" + str(PGT2)
                )
            if sample_MGT_index[0] == ".":
                all_sample_new_GT_index_dict[sample_name]["MGT"] = "."
            else:
                MGT1 = new_allele_index_list.index(
                    all_alleles[int(sample_MGT_index[0].split("/")[0])]
                )
                MGT2 = new_allele_index_list.index(
                    all_alleles[int(sample_MGT_index[0].split("/")[1])]
                )
                all_sample_new_GT_index_dict[sample_name]["MGT"] = (
                    str(MGT1) + "|" + str(MGT2)
                )
            if sample_PMGT_index[0] == ".":
                all_sample_new_GT_index_dict[sample_name]["PMGT"] = "."
            else:
                PMGT1 = new_allele_index_list.index(
                    all_alleles[int(sample_PMGT_index[0].split("|")[0])]
                )
                PMGT2 = new_allele_index_list.index(
                    all_alleles[int(sample_PMGT_index[0].split("|")[1])]
                )
                all_sample_new_GT_index_dict[sample_name]["PMGT"] = (
                    str(PMGT1) + "|" + str(PMGT2)
                )
            if sample_GI_index[0] == ".":
                all_sample_new_GT_index_dict[sample_name]["GI"] = "."
            else:
                GI1 = new_allele_index_list.index(
                    all_alleles[int(sample_GI_index[0].split("/")[0])]
                )
                GI2 = new_allele_index_list.index(
                    all_alleles[int(sample_GI_index[0].split("/")[1])]
                )
                all_sample_new_GT_index_dict[sample_name]["GI"] = (
                    str(GI1) + "/" + str(GI2)
                )
    if SHRINK_ALLELE:
        # left shrink and right shrink
        # left shrink
        prefix_consistent_bps = check_prefix_length(
            new_allele_index_list, max_length=PADDING_BPS
        )
        new_allele_index_list_reverse = new_allele_index_list[::-1]
        # right shrink
        suffix_consistent_bps = check_prefix_length(
            new_allele_index_list_reverse, max_length=PADDING_BPS
        )
        shrink_new_allele_index_list = []
        for i in new_allele_index_list:
            new_allele = i[prefix_consistent_bps:-suffix_consistent_bps]
            shrink_new_allele_index_list.append(new_allele)
        if "" in shrink_new_allele_index_list:
            shrink_new_allele_index_list = new_allele_index_list
            pos_start_zero_based_include = (
                str_start_zero_based_include - PADDING_BPS
            )
            pos_end_zero_based_exclude = (
                str_end_zero_based_exclude + PADDING_BPS
            )
        else:
            pos_start_zero_based_include = (
                str_start_zero_based_include
                - PADDING_BPS
                + prefix_consistent_bps
            )
            pos_end_zero_based_exclude = (
                str_end_zero_based_exclude
                + PADDING_BPS
                - suffix_consistent_bps
            )
    else:
        shrink_new_allele_index_list = new_allele_index_list
        pos_start_zero_based_include = (
            str_start_zero_based_include - PADDING_BPS
        )
        pos_end_zero_based_exclude = str_end_zero_based_exclude + PADDING_BPS
    return (
        all_sample_new_GT_index_dict,
        shrink_new_allele_index_list,
        pos_start_zero_based_include,
        pos_end_zero_based_exclude,
    )


def check_prefix_length(lst, max_length=PADDING_BPS):
    if not lst:
        return 0  # 空列表返回0

    # 遍历前1到max_length个字符
    for length in range(1, max_length + 1):
        # 取第一个元素的前length个字符作为基准
        reference_prefix = lst[0][:length]

        # 检查所有元素的前length个字符是否与基准一致
        for item in lst:
            if item[:length] != reference_prefix:
                return length - 1  # 返回前一个长度

    return max_length  # 所有元素的前max_length个字符一致


# HACK: Revise and Add New
def check_zero(allele, reference_allele):
    if allele == "0":
        return reference_allele
    else:
        return allele


def normalize_global_vcf(
    refseq_seq_reference,
    germ_allele_seq_to_refseq_alternative,
    source_allele_seq_to_refseq_alternative,
    mutant_allele_seq_to_refseq_alternative,
    refseq_left_index,
    refseq_right_index,
):
    germ_allele_seq_to_refseq_alternative = check_zero(
        germ_allele_seq_to_refseq_alternative, refseq_seq_reference
    )
    source_allele_seq_to_refseq_alternative = check_zero(
        source_allele_seq_to_refseq_alternative, refseq_seq_reference
    )
    mutant_allele_seq_to_refseq_alternative = check_zero(
        mutant_allele_seq_to_refseq_alternative, refseq_seq_reference
    )
    min_length = min(
        len(refseq_seq_reference),
        len(germ_allele_seq_to_refseq_alternative),
        len(source_allele_seq_to_refseq_alternative),
        len(mutant_allele_seq_to_refseq_alternative),
    )

    # 初始化计数器
    count = 0

    # 遍历字符串，比较每个字符
    for i in range(min_length):
        if (
            (
                germ_allele_seq_to_refseq_alternative[i]
                == source_allele_seq_to_refseq_alternative[i]
            )
            and (
                germ_allele_seq_to_refseq_alternative[i]
                == mutant_allele_seq_to_refseq_alternative[i]
            )
            and (
                germ_allele_seq_to_refseq_alternative[i]
                == refseq_seq_reference[i]
            )
        ):
            count += 1
        else:
            break
    if count >= min_length:
        count = min_length - 1
    else:
        count = count
    refseq_left_index = refseq_left_index + count
    refseq_seq_reference = refseq_seq_reference[count:]
    germ_allele_seq_to_refseq_alternative = (
        germ_allele_seq_to_refseq_alternative[count:]
    )
    source_allele_seq_to_refseq_alternative = (
        source_allele_seq_to_refseq_alternative[count:]
    )
    mutant_allele_seq_to_refseq_alternative = (
        mutant_allele_seq_to_refseq_alternative[count:]
    )
    min_length = min(
        len(refseq_seq_reference),
        len(germ_allele_seq_to_refseq_alternative),
        len(source_allele_seq_to_refseq_alternative),
        len(mutant_allele_seq_to_refseq_alternative),
    )
    count_suffix = 0
    for i in range(1, min_length + 1):
        if (
            (
                germ_allele_seq_to_refseq_alternative[-i]
                == source_allele_seq_to_refseq_alternative[-i]
            )
            and (
                germ_allele_seq_to_refseq_alternative[-i]
                == mutant_allele_seq_to_refseq_alternative[-i]
            )
            and (
                germ_allele_seq_to_refseq_alternative[-i]
                == refseq_seq_reference[-i]
            )
        ):
            count_suffix += 1
        else:
            break
    if count_suffix >= min_length:
        count_suffix = min_length - 1
    else:
        count_suffix = count_suffix
    refseq_right_index = refseq_right_index - count_suffix
    if count_suffix == 0:
        pass
    else:
        refseq_seq_reference = refseq_seq_reference[:-count_suffix]
        germ_allele_seq_to_refseq_alternative = (
            germ_allele_seq_to_refseq_alternative[:-count_suffix]
        )
        source_allele_seq_to_refseq_alternative = (
            source_allele_seq_to_refseq_alternative[:-count_suffix]
        )
        mutant_allele_seq_to_refseq_alternative = (
            mutant_allele_seq_to_refseq_alternative[:-count_suffix]
        )
    return (
        refseq_seq_reference,
        germ_allele_seq_to_refseq_alternative,
        source_allele_seq_to_refseq_alternative,
        mutant_allele_seq_to_refseq_alternative,
        refseq_left_index,
        refseq_right_index,
    )
