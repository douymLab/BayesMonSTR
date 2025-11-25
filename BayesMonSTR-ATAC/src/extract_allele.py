import numpy as np
import re

# import sys

# sys.path.append("/Users/lid/Github/MosaicSTR/configs")
# from ..configs import config_params
import config_params

ALLELE_EXTRACT = config_params.ALLELE_EXTRACT
MAPQ_CUTOFF = ALLELE_EXTRACT["MAPQ_CUTOFF"]  # 20
BASEQ_CUTOFF = ALLELE_EXTRACT["BASEQ_CUTOFF"]  # 20
MISMATCHES_CUTOFF = ALLELE_EXTRACT["MISMATCHES_CUTOFF"]  # 15
USE_MIS_FRAC = ALLELE_EXTRACT["USE_MIS_FRAC"]  # True # Default value and No use for if-else
MIS_FRAC_CUTOFF = ALLELE_EXTRACT["MIS_FRAC_CUTOFF"]  # 0.1
READS_FLANKING_LENGTH_CUTOFF = ALLELE_EXTRACT[
    "READS_FLANKING_LENGTH_CUTOFF"
]  # 5 # 10 # revise_padding
READS_SOFT_CLIPPING_CUTOFF = ALLELE_EXTRACT[
    "READS_SOFT_CLIPPING_CUTOFF"
]  # 100000000 # 1000 # 不过滤 clip reads 因为 long reads 过滤了也不太好
READS_HARD_CLIPPING_CUTOFF = ALLELE_EXTRACT[
    "READS_HARD_CLIPPING_CUTOFF"
]  # 100000000 # 1000 # 不过滤 clip reads 因为 long reads 过滤了也不太好
USE_SEQUENCING_LENGTH = ALLELE_EXTRACT[
    "USE_SEQUENCING_LENGTH"
]  # False, for read-level features
FLANKING_INDEL_CHECK_RANGE = ALLELE_EXTRACT[
    "FLANKING_INDEL_CHECK_RANGE"
]  # 15, for read-level features
PADDING_BPS = ALLELE_EXTRACT[
    "PADDING_BPS"
]  # 5 # 10 # revise_padding
AVOID_DIE_CYCLE = 15000000  # For long reads read-level features
FLANKING_INDEL_CHECK_RANGE = 25
FIXED_FLANKING_INDEL_CHECK_RANGE = True
PCRLibrary = ALLELE_EXTRACT["PCRLibrary"]  # "PCR" # "PCR-free"
# HipSTR mentioned that:
# In practice, SNP calls in and around STR regions are likely to be error prone.
# We therefore exclude SNPs that are within 15 bp of the STR region when computing the phasing likelihoods.
# If no phased SNP information is available, HipSTR assigns equal phasing likelihoods to both of the strands.
# XXX: Current extract_allele can use in illumina and pacbio and element and ONT datasets


def judge_indel_exist_given_region(
    pysam_read, zero_base_start_include, zero_based_end_exclude
):
    align_block = pysam_read.get_blocks()
    # print(pysam_read) # 比对起始位置是 1-based
    # print(pysam_read.reference_start) # 比对起始位置是 1-based
    # print(pysam_read.get_blocks()) # 比对起始和结束位置是 0-based，左闭右开
    # print(pysam_read.get_blocks()) # soft/hard clip blocks 不会出现在这里，所以不会影响 indel 的判断和计数
    if len(align_block) == 1:
        return 0
    else:
        prev_block_end = block[0][1]
        for block in align_block[1:]:
            block_start = block[0]
            if block_start == prev_block_end:
                if (
                    zero_base_start_include
                    < block_start
                    < zero_based_end_exclude
                ):
                    return 1
                else:
                    pass
            else:
                if (
                    prev_block_end >= zero_base_start_include
                    and prev_block_end < zero_based_end_exclude
                ):
                    return 1
                else:
                    pass
            prev_block_end = block[1]
        return 0


def count_mismatches_given_region(
    mis_pos_zero_based_list, zero_base_start_include, zero_based_end_exclude
):
    # for STR
    mis_number_given_region = 0
    for mis_pos in mis_pos_zero_based_list:
        if zero_base_start_include <= mis_pos < zero_based_end_exclude:
            mis_number_given_region += 1
    return mis_number_given_region


def judge_left_spanning_del_given_region(
    del_pos_zero_based_list, zero_based_left_fl_end_include
):
    for del_pos, del_length in del_pos_zero_based_list:
        if (del_pos <= zero_based_left_fl_end_include) and (
            del_pos + del_length - 1 > zero_based_left_fl_end_include
        ):
            return True, del_pos
    return False, False


def judge_right_spanning_del_given_region(
    del_pos_zero_based_list, zero_based_right_fl_start_include
):
    for del_pos, del_length in del_pos_zero_based_list:
        if (del_pos < zero_based_right_fl_start_include) and (
            del_pos + del_length - 1 >= zero_based_right_fl_start_include
        ):
            return True, del_pos
    return False, False


## 准则: Padding 后 STR 与 Flanking 间隔的 insertion 不属于 STR，但属于 Flanking
## 准则: Padding 前 STR 与 Flanking 间隔的 insertion 属于 STR，但不属于 Flanking


def judge_left_fl_indel_exist_given_region(
    ins_before_zero_based_pos_list,
    del_pos_zero_based_list,
    zero_based_start_include,
    zero_based_end_include,
):
    # For left flanking 25bp
    # PADDING BPS INS: double end ins is flanking ins
    # NO PADDING BPS INS: left end ins is flanking ins and right end ins is not flanking ins
    # DEL: overlap is flanking del
    for ins_before_pos, _ in ins_before_zero_based_pos_list:
        if PADDING_BPS > 0:
            if (
                zero_based_start_include
                <= ins_before_pos
                <= zero_based_end_include + 1
            ):
                return True
        else:
            if (
                zero_based_start_include
                <= ins_before_pos
                < zero_based_end_include + 1
            ):
                return True
    for del_pos, del_length in del_pos_zero_based_list:
        if zero_based_start_include <= del_pos <= zero_based_end_include:
            return True
        if (
            zero_based_start_include
            <= (del_pos + del_length - 1)
            <= zero_based_end_include
        ):
            return True
        if (
            (del_pos < zero_based_start_include)
            and ((del_pos + del_length - 1) >= zero_based_start_include)
        ):
            return True  # HACK: missing status debug Done
    return False


def judge_right_fl_indel_exist_given_region(
    ins_before_zero_based_pos_list,
    del_pos_zero_based_list,
    zero_based_start_include,
    zero_based_end_include,
):
    # For right flanking 25bp
    # only right end included
    for ins_before_pos, _ in ins_before_zero_based_pos_list:
        if PADDING_BPS > 0:
            if (
                zero_based_start_include
                <= ins_before_pos
                <= zero_based_end_include + 1
            ):
                return True
        else:
            if (
                zero_based_start_include
                < ins_before_pos
                <= zero_based_end_include + 1
            ):
                return True
    for del_pos, del_length in del_pos_zero_based_list:
        if zero_based_start_include <= del_pos <= zero_based_end_include:
            return True
        if (
            zero_based_start_include
            <= del_pos + del_length - 1
            <= zero_based_end_include
        ):
            return True
        if (
            (del_pos < zero_based_start_include)
            and ((del_pos + del_length - 1) >= zero_based_start_include)
        ):
            return True  # HACK: missing status debug Done
    return False


# def judge_left_fl_indel_exist_given_region(
#     ins_before_zero_based_pos_list,
#     del_pos_zero_based_list,
#     zero_base_start_include,
#     zero_based_end_exclude,
# ):
#     # For left flanking 25bp
#     # only left end included
#     for ins_before_pos, _ in ins_before_zero_based_pos_list:
#         if zero_base_start_include <= ins_before_pos < zero_based_end_exclude:
#             return True
#     for del_pos, del_length in del_pos_zero_based_list:
#         if zero_base_start_include <= del_pos < zero_based_end_exclude:
#             return True
#         if (
#             zero_base_start_include
#             < del_pos + del_length
#             <= zero_based_end_exclude
#         ):
#             return True
#     return False


# def judge_right_fl_indel_exist_given_region(
#     ins_before_zero_based_pos_list,
#     del_pos_zero_based_list,
#     zero_base_start_include,
#     zero_based_end_exclude,
# ):
#     # For right flanking 25bp
#     # only right end included
#     for ins_before_pos, _ in ins_before_zero_based_pos_list:
#         if zero_base_start_include < ins_before_pos <= zero_based_end_exclude:
#             return True
#     for del_pos, del_length in del_pos_zero_based_list:
#         if zero_base_start_include <= del_pos < zero_based_end_exclude:
#             return True
#         if (
#             zero_base_start_include
#             < del_pos + del_length
#             <= zero_based_end_exclude
#         ):
#             return True
#     return False


def count_indels_number_given_region(
    ins_before_zero_based_pos_list,
    del_pos_zero_based_list,
    zero_base_start_include,
    zero_based_end_exclude,
):
    # for STR
    # both end excluded ins for padding STR regions
    indel_num = 0
    for ins_before_pos, _ in ins_before_zero_based_pos_list:
        if (
            zero_base_start_include < ins_before_pos < zero_based_end_exclude
        ):  ## HACK: 因为定义的 STR 区域是 padding 的区域，靠近边界的 ins 不认为 STR 的 ins，有 del 时往 STR 区域内缩小
            indel_num += 1
    for del_pos, del_length in del_pos_zero_based_list:
        if (zero_base_start_include <= del_pos < zero_based_end_exclude) or (
            zero_base_start_include
            < del_pos + del_length
            <= zero_based_end_exclude
        ):
            indel_num += 1
    return indel_num
    ## TODO: DEBUG spanning del across STR and Flanking may be counted twice in flanking and STR


def count_indels_number_exclude_region(
    pysam_read, zero_base_start_include, zero_based_end_exclude
):
    indels_num = 0
    zero_based_pos = pysam_read.reference_start
    for cigar, length in pysam_read.cigartuples:
        if cigar in [1, 2]:
            if not (
                zero_base_start_include
                <= zero_based_pos
                < zero_based_end_exclude
            ):
                indels_num += 1
        if cigar in [0, 2, 3, 7, 8]:
            zero_based_pos += length

    return indels_num


def parse_cigar_and_md(pysam_read):
    """
    Parse the CIGAR string and MD tag to find mismatches, insertions, and deletions.
    cigar_tuples: A list of tuples from read.cigartuples.
    md_tag: The MD tag string from the read.
    Returns mismatches, insertions, and deletions with their positions.
    """

    ref_pos = pysam_read.reference_start  # Position in the reference
    if ref_pos is None:
        return [], [], []
    cigar_tuples = pysam_read.cigartuples
    md_tag = pysam_read.get_tag("MD") if pysam_read.has_tag("MD") else ""
    mismatches = []
    insertions = []
    deletions = []

    # Process CIGAR
    for op, length in cigar_tuples:
        if op == 0:  # Match or mismatch
            ref_pos += length
        elif op == 1:  # Insertion to the reference
            insertions.append((ref_pos, length))
        elif op == 2:  # Deletion from the reference
            deletions.append((ref_pos, length))
            ref_pos += length
        elif op in [3, 7, 8]:
            ref_pos += length
    # Process MD tag for mismatches (since we've handled insertions and deletions)
    md_pos = (
        pysam_read.reference_start
    )  # Position in the reference according to MD tag
    # <unknown>:2: SyntaxWarning: invalid escape sequence '\d'
    pattern = re.compile(r"(\d+)|([A-Z])|(\^[A-Z]+)")
    for match in pattern.finditer(md_tag):
        if match.group(1):  # Number: match
            md_pos += int(match.group(1))
        elif match.group(2):  # Letter: mismatch
            mismatches.append(md_pos)
            md_pos += 1
        elif match.group(
            3
        ):  # Caret followed by letters: deletion (already handled by CIGAR)
            md_pos += (
                len(match.group(3)) - 1
            )  # Skip caret and move past deletion

    return mismatches, insertions, deletions

# /data/wangweixiang/1122_013/chr21_20000000_30000000/code/extract_allele.py:311: SyntaxWarning: invalid escape sequence '\d'
#   pattern = re.compile("(\d+)|([A-Z])|(\^[A-Z]+)")

class PerReadFeature:
    def __init__(
        self, pysam_read, str_zero_based_included, str_zero_based_end_excluded
    ):
        self.pysam_read = pysam_read
        self.reference_start = (
            pysam_read.reference_start
        )  # 0-based leftmost coordinate
        self.reference_end = (
            pysam_read.reference_end
        )  # reference_end points to one past the last aligned residue.
        self.STR_start_from_0_based_bed_included = str_zero_based_included
        self.STR_stop_from_0_based_bed_excluded = str_zero_based_end_excluded

    def max_lik_read_assignment(self):
        pass  # TODO

    def read_is_usable_for_STAR(self):
        pass  # TODO

    def read_is_usable_for_bwa(self):
        self.barcode = self.pysam_read.get_tag("CB") if self.pysam_read.has_tag("CB") else ""
        if self.pysam_read.reference_end is None:
            # XXX: reference_end points to one past the last aligned residue. Returns None if not available (read is unmapped or no cigar alignment present).
            self.is_usable = False
            self.unusable_reason = "unmap_issue"
            return
        if self.pysam_read.query_alignment_qualities is None:  # XXX: Pacbio or Nanopore data may be unavailable for query_alignment_qualities in some reads
            self.is_usable = False
            self.unusable_reason = "library_issue"
            return
        if self.pysam_read.has_tag("MD"):  # XXX: Pacbio or Nanopore data may be unavailable for MD tag in some reads
            ## XXX: BWA mem and minimap2 will produce MD tags but pbmm won't produce MD tags so need add MD through samtools calmd
            try:
                aligned_pairs = self.pysam_read.get_aligned_pairs(
                    with_seq=True, matches_only=True
                )  # zero-based coordinates
                # BUG FROM BAM: HG002_GRCh38_ONT-UL_GIAB_20200122_phased
                # BUG FROM BAM: AssertionError: Invalid MD tag: MD length 3813089307 mismatch with CIGAR length 36322 and 568 insertions
            except AssertionError:
                self.is_usable = False
                self.unusable_reason = "unmap_issue"
                return
        else:
            self.is_usable = False
            self.unusable_reason = "unmap_issue"
            return
        if PCRLibrary == "PCR":
            self.read_is_duplicate = int(self.pysam_read.is_duplicate)
        else:
            self.read_is_duplicate = int(False)
        self.read_is_supplementary = int(self.pysam_read.is_supplementary)
        self.read_is_secondary = int(self.pysam_read.is_secondary)
        self.read_is_unmapped = int(self.pysam_read.is_unmapped)
        self.read_is_qcfail = int(self.pysam_read.is_qcfail)
        self.read_is_proper_pair = int(
            (not self.pysam_read.is_paired) or self.pysam_read.is_proper_pair
        )
        self.read_mq = int(self.pysam_read.mapping_quality >= MAPQ_CUTOFF)
        self.read_bq = int(
            np.mean(self.pysam_read.query_alignment_qualities) >= BASEQ_CUTOFF
        )
        self.read_cigar = self.pysam_read.get_cigar_stats()
        mismatch_number = (
                self.read_cigar[0][-1]
                - self.read_cigar[0][1]
                - self.read_cigar[0][2]
            )
        read_alignment_length = self.pysam_read.query_alignment_length
        self.read_mismatch = int((mismatch_number/read_alignment_length) <= MIS_FRAC_CUTOFF)   # HACK: 150 for debug mosaicgt become more
        # self.read_mismatch = int(
        #     (
        #         self.read_cigar[0][-1]
        #         - self.read_cigar[0][1]
        #         - self.read_cigar[0][2]
        #     )
        #     <= MISMATCHES_CUTOFF
        # )
        self.read_edit_distance_all_mismatches_indels = self.read_cigar[0][
            -1
        ]  # XXX DON'T CONSIDER INDEL avoid big INDELs in STR regions is variants
        # self.read_cigar[0][-1] or read.get_tag("NM") NM means edit distance
        # The definition of the NM tag is Edit distance to the reference, including ambiguous bases but excluding clipping.
        self.STR_left_flanking_start = (
            self.STR_start_from_0_based_bed_included
            - READS_FLANKING_LENGTH_CUTOFF
        )
        self.STR_right_flanking_end = (
            self.STR_stop_from_0_based_bed_excluded
            + READS_FLANKING_LENGTH_CUTOFF
        )
        self.read_spanning = int(
            (self.pysam_read.reference_end >= self.STR_right_flanking_end)
            and (
                self.pysam_read.reference_start <= self.STR_left_flanking_start
            )
        )
        self.read_soft_clipping = int(
            self.read_cigar[0][4] <= READS_SOFT_CLIPPING_CUTOFF
        )
        self.read_soft_clipping_num = self.read_cigar[0][4]
        self.read_hard_clipping = int(
            self.read_cigar[0][5] <= READS_HARD_CLIPPING_CUTOFF
        )
        self.read_hard_clipping_num = self.read_cigar[0][5]
        self.map_issue = (
            self.read_is_supplementary
            + self.read_is_secondary
            + self.read_is_unmapped
            + (1 - self.read_soft_clipping)
            + (1 - self.read_hard_clipping)
            + (1 - self.read_is_proper_pair)
            + (1 - self.read_mq)
        )
        self.library_issue = (
            self.read_is_duplicate
            + self.read_is_qcfail
            + (1 - self.read_bq)
            + (1 - self.read_mismatch)
        )
        self.spanning_issue = 1 - self.read_spanning
        if (
            self.read_is_proper_pair  # This depends on what you want to do. If you are using paired-end reads, the 0x2 flag means that both ends of the read were mapped and they were mapped within a reasonable distance given the expected distance
            + self.read_mq
            + self.read_bq
            + self.read_mismatch
            + self.read_spanning
            + self.read_soft_clipping
            + self.read_hard_clipping
        ) == 7 and (
            self.read_is_duplicate
            + self.read_is_supplementary
            + self.read_is_secondary
            + self.read_is_unmapped
            + self.read_is_qcfail
        ) == 0:
            self.is_usable = True
            self.unusable_reason = "None"
        elif self.library_issue > 0:
            self.is_usable = False
            self.unusable_reason = "library_issue"
        elif self.map_issue > 0:
            self.is_usable = False
            self.unusable_reason = "map_issue"
        elif self.spanning_issue > 0:
            self.is_usable = False
            self.unusable_reason = "spanning_issue"
        # else:
        #     self.is_usable = False
        #     self.unusable_reason = f"map_issue{self.map_issue};library_issue{self.library_issue};spanning_issue{self.spanning_issue}"

    def read_level_features(self):
        # self.barcode = self.pysam_read.get_tag("CB") if self.pysam_read.has_tag("CB") else ""
        # other per-locus features such as mappability
        if USE_SEQUENCING_LENGTH:
            read_length = (
                self.pysam_read.infer_read_length()
            )  # does not include hard-clipped bases
        else:
            read_length = (
                self.pysam_read.infer_query_length()
            )  # including hard-clipped bases
        self.read_length = read_length
        readCIGAR = self.pysam_read.get_cigar_stats()
        (
            mismatches_loci_list,
            insertions_loci_list,
            deletions_loci_list,
        ) = parse_cigar_and_md(self.pysam_read)
        all_sbs_num = len(mismatches_loci_list)
        all_ins_num = len(insertions_loci_list)
        all_del_num = len(deletions_loci_list)
        all_indel_num = all_ins_num + all_del_num
        # left_fl_zero_based_start_included = (
        #     self.STR_start_from_0_based_bed_included
        #     - FLANKING_INDEL_CHECK_RANGE
        # )
        # right_fl_zero_based_end_excluded = (
        #     self.STR_stop_from_0_based_bed_excluded
        #     + FLANKING_INDEL_CHECK_RANGE
        # )
        # is_flanking_indel = judge_left_fl_indel_exist_given_region(
        #     insertions_loci_list,
        #     deletions_loci_list,
        #     left_fl_zero_based_start_included,
        #     self.STR_start_from_0_based_bed_included,
        # ) or judge_right_fl_indel_exist_given_region(
        #     insertions_loci_list,
        #     deletions_loci_list,
        #     self.STR_stop_from_0_based_bed_excluded,
        #     right_fl_zero_based_end_excluded,
        # )
        if PADDING_BPS > 0:
            STR_indel_num = count_indels_number_given_region(
                insertions_loci_list,
                deletions_loci_list,
                self.STR_start_from_0_based_bed_included - PADDING_BPS,
                self.STR_stop_from_0_based_bed_excluded + PADDING_BPS,
            )
            STR_sbs_num = count_mismatches_given_region(
                mismatches_loci_list,
                self.STR_start_from_0_based_bed_included - PADDING_BPS,
                self.STR_stop_from_0_based_bed_excluded + PADDING_BPS,
            )
            left_flanking_sbs_num = count_mismatches_given_region(
                mismatches_loci_list,
                self.reference_start,
                self.STR_start_from_0_based_bed_included - PADDING_BPS,
            )
            right_flanking_sbs_num = count_mismatches_given_region(
                mismatches_loci_list,
                self.STR_stop_from_0_based_bed_excluded + PADDING_BPS,
                self.reference_end,
            )
        else:
            STR_indel_num = count_indels_number_given_region(
                insertions_loci_list,
                deletions_loci_list,
                self.STR_start_from_0_based_bed_included,
                self.STR_stop_from_0_based_bed_excluded,
            )
            STR_sbs_num = count_mismatches_given_region(
                mismatches_loci_list,
                self.STR_start_from_0_based_bed_included,
                self.STR_stop_from_0_based_bed_excluded,
            )
            left_flanking_sbs_num = count_mismatches_given_region(
                mismatches_loci_list,
                self.reference_start,
                self.STR_start_from_0_based_bed_included,
            )
            right_flanking_sbs_num = count_mismatches_given_region(
                mismatches_loci_list,
                self.STR_stop_from_0_based_bed_excluded,
                self.reference_end,
            )
        flanking_sbs_num = all_sbs_num - STR_sbs_num
        flanking_indel_num = all_indel_num - STR_indel_num
        aligned_pairs = self.pysam_read.get_aligned_pairs(
            with_seq=True, matches_only=True
        )  # zero-based coordinates
        refpos_queryindex = {i[1]: [i[0], i[2]] for i in aligned_pairs}
        str_start_zero_based = self.STR_start_from_0_based_bed_included
        k = 0
        while True:
            if refpos_queryindex.get(str_start_zero_based) is None:
                str_start_zero_based += 1
                k += 1
            else:
                str_start_query_index_include = refpos_queryindex.get(
                    str_start_zero_based
                )[0]
                break
            if k >= AVOID_DIE_CYCLE:
                str_start_query_index_include = None
        str_end_zero_based_include = (
            self.STR_stop_from_0_based_bed_excluded - 1
        )
        k = 0
        while True:
            if refpos_queryindex.get(str_end_zero_based_include) is None:
                str_end_zero_based_include -= 1
                k += 1
            else:
                str_end_query_index_include = refpos_queryindex.get(
                    str_end_zero_based_include
                )[0]
                break
            if k >= AVOID_DIE_CYCLE:
                str_end_query_index_include = None
        k = 0
        str_start_zero_based_include_padding_bps = (
            self.STR_start_from_0_based_bed_included - PADDING_BPS
        )
        while True:
            if (
                refpos_queryindex.get(str_start_zero_based_include_padding_bps)
                is None
            ):
                str_start_zero_based_include_padding_bps += 1
                k += 1
            else:
                str_padding_bps_start_query_index_include = (
                    refpos_queryindex.get(
                        str_start_zero_based_include_padding_bps
                    )[0]
                )
                break
            if k >= AVOID_DIE_CYCLE:
                str_padding_bps_start_query_index_include = None
        k = 0
        str_end_zero_based_include_padding_bps = (
            self.STR_stop_from_0_based_bed_excluded + PADDING_BPS - 1
        )
        while True:
            if (
                refpos_queryindex.get(str_end_zero_based_include_padding_bps)
                is None
            ):
                str_end_zero_based_include_padding_bps -= 1
                k += 1
            else:
                str_padding_bps_end_query_index_include = (
                    refpos_queryindex.get(
                        str_end_zero_based_include_padding_bps
                    )[0]
                )
                break
            if k >= AVOID_DIE_CYCLE:
                str_padding_bps_end_query_index_include = None
        reference_start_query_index_include = refpos_queryindex.get(
            self.reference_start
        )[0]
        reference_end_query_index_include = refpos_queryindex.get(
            self.reference_end - 1
        )[0]
        if PADDING_BPS > 0:
            left_flanking_bps_num = (
                str_padding_bps_start_query_index_include
                - 1
                - reference_start_query_index_include
                + 1
            )
            right_flanking_bps_num = (
                reference_end_query_index_include
                - (str_padding_bps_end_query_index_include + 1)
                + 1
            )
        else:
            left_flanking_bps_num = (
                str_start_query_index_include
                - 1
                - reference_start_query_index_include
                + 1
            )
            right_flanking_bps_num = (
                reference_end_query_index_include
                - (str_end_query_index_include + 1)
                + 1
            )
        if FIXED_FLANKING_INDEL_CHECK_RANGE:
            if PADDING_BPS > 0:
                read_left_flanking_start_include = (
                    self.STR_start_from_0_based_bed_included
                    - PADDING_BPS
                    - 1
                    - FLANKING_INDEL_CHECK_RANGE
                    + 1
                )
                read_left_flanking_end_include = (
                    self.STR_start_from_0_based_bed_included - PADDING_BPS - 1
                )
                read_right_flanking_start_include = (
                    self.STR_stop_from_0_based_bed_excluded + PADDING_BPS
                )
                read_right_flanking_end_include = (
                    self.STR_stop_from_0_based_bed_excluded
                    + PADDING_BPS
                    + FLANKING_INDEL_CHECK_RANGE
                    - 1
                )
            else:
                read_left_flanking_start_include = (
                    self.STR_start_from_0_based_bed_included
                    - 1
                    - FLANKING_INDEL_CHECK_RANGE
                    + 1
                )
                read_left_flanking_end_include = (
                    self.STR_start_from_0_based_bed_included - 1
                )
                read_right_flanking_start_include = (
                    self.STR_stop_from_0_based_bed_excluded
                )
                read_right_flanking_end_include = (
                    self.STR_stop_from_0_based_bed_excluded
                    + FLANKING_INDEL_CHECK_RANGE
                    - 1
                )
        else:
            if PADDING_BPS > 0:
                read_left_flanking_start_include = self.reference_start
                read_left_flanking_end_include = (
                    self.STR_start_from_0_based_bed_included - PADDING_BPS - 1
                )
                read_right_flanking_start_include = (
                    self.STR_stop_from_0_based_bed_excluded + PADDING_BPS
                )
                read_right_flanking_end_include = self.reference_end - 1
            else:
                read_left_flanking_start_include = self.reference_start
                read_left_flanking_end_include = (
                    self.STR_start_from_0_based_bed_included - 1
                )
                read_right_flanking_start_include = (
                    self.STR_stop_from_0_based_bed_excluded
                )
                read_right_flanking_end_include = self.reference_end - 1
        is_flanking_indel = judge_left_fl_indel_exist_given_region(
            insertions_loci_list,
            deletions_loci_list,
            read_left_flanking_start_include,
            read_left_flanking_end_include,
        ) or judge_right_fl_indel_exist_given_region(
            insertions_loci_list,
            deletions_loci_list,
            read_right_flanking_start_include,
            read_right_flanking_end_include,
        )
        (
            left_spanning_str_flanking_indel,
            left_span_del_pos,
        ) = judge_left_spanning_del_given_region(
            deletions_loci_list,
            read_left_flanking_end_include,
        )
        (
            right_spanning_str_flanking_indel,
            right_span_del_pos,
        ) = judge_right_spanning_del_given_region(
            deletions_loci_list,
            read_right_flanking_start_include,
        )
        is_spanning_str_flanking_indel = (
            left_spanning_str_flanking_indel
            or right_spanning_str_flanking_indel
        )
        if (
            left_spanning_str_flanking_indel
            and right_spanning_str_flanking_indel
        ):
            if left_span_del_pos == right_span_del_pos:
                the_same_del_for_left_and_right_spanning_str_flanking = True
            else:
                the_same_del_for_left_and_right_spanning_str_flanking = False
        else:
            the_same_del_for_left_and_right_spanning_str_flanking = False
        flanking_indel_num = (
            flanking_indel_num
            + left_spanning_str_flanking_indel
            + right_spanning_str_flanking_indel
            - the_same_del_for_left_and_right_spanning_str_flanking
        )
        if self.pysam_read.is_forward: # zero based to one based
            str_pcr_start = str_start_query_index_include + 1
            str_pcr_end = str_end_query_index_include + 1
            str_padding_pcr_start = str_padding_bps_start_query_index_include + 1
            str_padding_pcr_end = str_padding_bps_end_query_index_include + 1
        else: # zero based to one based
            str_pcr_start = read_length - str_end_query_index_include
            str_pcr_end = read_length - str_start_query_index_include
            str_padding_pcr_start = (
                read_length - str_padding_bps_end_query_index_include
            )
            str_padding_pcr_end = (
                read_length - str_padding_bps_start_query_index_include
            )
        # 末端有几个非 str 区域碱基
        str_pcr_start_terminal1 = str_pcr_start - 1
        str_pcr_end_terminal2 = read_length - str_pcr_end
        str_padding_pcr_start_terminal1 = str_padding_pcr_start - 1
        str_padding_pcr_end_terminal2 = read_length - str_padding_pcr_end
        self.read_features = {
            "zero_based_leftmost_coordinate": self.reference_start,
            "strand": self.pysam_read.is_forward,
            "mapping_quality": self.pysam_read.mapping_quality,
            "read1": self.pysam_read.is_read1,
            "read_softclipped": readCIGAR[0][4],
            "read_hardclipped": readCIGAR[0][5],
            "read_all_sbs_num_based_NM_edit_distance": (
                readCIGAR[0][-1] - readCIGAR[0][1] - readCIGAR[0][2]
            ),
            "read_flanking_indel_num": flanking_indel_num,
            "is_flanking_indel": is_flanking_indel,
            "is_spanning_str_flanking_indel": is_spanning_str_flanking_indel,
            "read_all_sbs_num_based_on_MD": all_sbs_num,
            "read_flanking_sbs_num": flanking_sbs_num,
            # "str_pcr_start": str_start_query_index_include,
            # "str_pcr_end": str_end_query_index_include,
            # "str_padding_pcr_start": str_padding_bps_start_query_index_include,
            # "str_padding_pcr_end": str_padding_bps_end_query_index_include,
            "str_pcr_start": str_pcr_start,
            "str_pcr_end": str_pcr_end,
            "str_padding_pcr_start": str_padding_pcr_start,
            "str_padding_pcr_end": str_padding_pcr_end,
            "read_length": read_length,
            "left_flanking_sbs_num": left_flanking_sbs_num,
            "right_flanking_sbs_num": right_flanking_sbs_num,
            "left_flanking_bps_num": left_flanking_bps_num,
            "right_flanking_bps_num": right_flanking_bps_num,
            "str_pcr_start_terminal1": str_pcr_start_terminal1,
            "str_pcr_end_terminal2": str_pcr_end_terminal2,
            "str_padding_pcr_start_terminal1": str_padding_pcr_start_terminal1,
            "str_padding_pcr_end_terminal2": str_padding_pcr_end_terminal2,
        }


# print(pysam_read) # 1-based alignment location for 4th cols
# print(mismatches_loci_list) # 0-based
# print(insertions_loci_list) # 0-based next base location of insertion location
# print(deletions_loci_list) # 0-based first deletion base
# print(pysam_read.get_blocks()) # 0-based
# print(pysam_read.reference_start) # 0-based
# print(pysam_read.get_aligned_pairs()) # 0-based
