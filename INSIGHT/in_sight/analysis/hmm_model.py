# Author: Weixiang Wang
# Date: 2023-04-20
# Description: Hidden Markov Model for Read Segmentation

import sys
import numpy as np
import pysam

_DEBUG = False

# ----------------------------- Hidden Markov Model Parameters Begin ----------------------------- #
ALL_MOTIF__PARAMS = {}

_PARAMS_1 = {}
_PARAMS_1["mismatch_rate"] = 0.005
_PARAMS_1["error_rate"] = 0.001
_PARAMS_1["mis_2_mis_rate"] = 0.005
ALL_MOTIF__PARAMS[1] = _PARAMS_1

_PARAMS_2 = {}
_PARAMS_2["mismatch_rate"] = 0.01
_PARAMS_2["insertion_rate"] = 0.009
_PARAMS_2["deletion_rate"] = 0.009
_PARAMS_2["error_rate"] = 0.001
_PARAMS_2["ins_2_ins_rate"] = 0.009
_PARAMS_2["ins_2_mis_rate"] = 0.01
_PARAMS_2["mis_2_ins_rate"] = 0.009
_PARAMS_2["mis_2_mis_rate"] = 0.01
ALL_MOTIF__PARAMS[2] = _PARAMS_2

_PARAMS_3 = {}
_PARAMS_3["mismatch_rate"] = 0.0090
_PARAMS_3["insertion_rate"] = 0.0080
_PARAMS_3["deletion_rate"] = 0.0080
_PARAMS_3["out_frame_rate"] = 0.000001
_PARAMS_3["continous_deletion_rate"] = 0.000001
_PARAMS_3["error_rate"] = 0.001
_PARAMS_3["mis_2_out_frame_rate"] = 0.000001
_PARAMS_3["ins_2_out_frame_rate"] = 0.000001
_PARAMS_3["ins_2_ins_rate"] = 0.0080
_PARAMS_3["ins_2_mis_rate"] = 0.0090
_PARAMS_3["mis_2_ins_rate"] = 0.0080
_PARAMS_3["mis_2_mis_rate"] = 0.0090
ALL_MOTIF__PARAMS[3] = _PARAMS_3

_PARAMS_4 = {}
_PARAMS_4["mismatch_rate"] = 0.006
_PARAMS_4["insertion_rate"] = 0.007
_PARAMS_4["deletion_rate"] = 0.007
_PARAMS_4["out_frame_rate"] = 0.000001
_PARAMS_4["continous_deletion_rate"] = 0.000001
_PARAMS_4["error_rate"] = 0.001
_PARAMS_4["mis_2_out_frame_rate"] = 0.000001
_PARAMS_4["ins_2_out_frame_rate"] = 0.000001
_PARAMS_4["ins_2_ins_rate"] = 0.007
_PARAMS_4["ins_2_mis_rate"] = 0.006
_PARAMS_4["mis_2_ins_rate"] = 0.007
_PARAMS_4["mis_2_mis_rate"] = 0.006
ALL_MOTIF__PARAMS[4] = _PARAMS_4

_PARAMS_5 = {}
_PARAMS_5["mismatch_rate"] = 0.0088
_PARAMS_5["insertion_rate"] = 0.009
_PARAMS_5["deletion_rate"] = 0.006
_PARAMS_5["out_frame_rate"] = 0.000001
_PARAMS_5["continous_deletion_rate"] = 0.000001
_PARAMS_5["error_rate"] = 0.001
_PARAMS_5["mis_2_out_frame_rate"] = 0.000001
_PARAMS_5["ins_2_out_frame_rate"] = 0.000001
_PARAMS_5["ins_2_ins_rate"] = 0.009
_PARAMS_5["ins_2_mis_rate"] = 0.0088
_PARAMS_5["mis_2_ins_rate"] = 0.009
_PARAMS_5["mis_2_mis_rate"] = 0.0088
ALL_MOTIF__PARAMS[5] = _PARAMS_5

_PARAMS_6 = {}
_PARAMS_6["mismatch_rate"] = 0.0048
_PARAMS_6["insertion_rate"] = 0.007
_PARAMS_6["deletion_rate"] = 0.006
_PARAMS_6["out_frame_rate"] = 0.000001
_PARAMS_6["continous_deletion_rate"] = 1.000000e-07
_PARAMS_6["error_rate"] = 0.001
_PARAMS_6["mis_2_out_frame_rate"] = 0.000001
_PARAMS_6["ins_2_out_frame_rate"] = 0.000001
_PARAMS_6["ins_2_ins_rate"] = 0.007
_PARAMS_6["ins_2_mis_rate"] = 0.0048
_PARAMS_6["mis_2_ins_rate"] = 0.007
_PARAMS_6["mis_2_mis_rate"] = 0.0048
ALL_MOTIF__PARAMS[6] = _PARAMS_6

L_STEPS_DICT = {1: 5, 2: 6, 3: 6, 4: 8, 5: 10, 6: 12}
BASE_INDEX_DICT = {"A": 0, "T": 1, "C": 2, "G": 3}
DEFAULT_PROBABILITY = 1e-8
# ----------------------------- Hidden Markov Model Parameters End ----------------------------- #


# ----------------------------- Privacy Function Start ----------------------------- #
def _baseq_2_error_rate(baseq_array) -> np.array:
    """Convert base quality to error rate.

    Args:
    ----
        baseq_array ([np.array]): [phred-scaled base quality]

    Returns:
    -------
        [np.array]: [error rate]
    """
    return np.power(10, np.array(baseq_array) / (-10))


def _read_is_usable(
    read, STR_start_from_0_based_bed, STR_stop_from_0_based_bed
) -> bool:
    """Flter reads that are not usable for the analysis.

    The following criteria are used:
    1. Read is not a duplicate
    2. Read is not supplementary
    3. Read is not secondary
    4. Read is not unmapped
    5. Read is not qcfail
    6. Read is proper pair
    7. Read mapping quality is >= 20
    8. Read base quality is >= 20
    9. Read mismatch is <= 10
    10. Read is spanning(coordinates-level) if there are more than 14 bases on both sides of STR region
    11. Read soft clipping is <= 1000
    12. Read hard clipping is <= 1000

    Args:
    ----
        read ([pysam.AlignedSegment]): [A pysam.AlignedSegment read which overlaps with STR region]
        STR_start_from_0_based_bed ([int]): [STR start site from 0-based reference bed file]
        STR_stop_from_0_based_bed ([int]): [STR end site from 0-based reference bed file]

    Returns:
    -------
        [bool]: True if the read is usable, False if the read is not usable
    """
    read_is_duplicate = read.is_duplicate
    read_is_supplementary = read.is_supplementary
    read_is_secondary = read.is_secondary
    read_is_unmapped = read.is_unmapped
    read_is_qcfail = read.is_qcfail
    read_is_proper_pair = (not read.is_paired) or read.is_proper_pair
    read_mq = read.mapping_quality >= 20
    read_bq = np.mean(read.query_alignment_qualities) >= 20
    read_cigar = read.get_cigar_stats()
    read_mismatch = (read_cigar[0][-1] - read_cigar[0][1] - read_cigar[0][2]) <= 10
    STR_left_flanking_start = STR_start_from_0_based_bed - 15
    STR_right_flanking_end = STR_stop_from_0_based_bed + 15
    if read.reference_end is None:
        # reference_end points to one past the last aligned residue. Returns None if not available (read is unmapped or no cigar alignment present).
        return False
    read_spanning = (read.reference_end >= STR_right_flanking_end) and (
        read.reference_start <= STR_left_flanking_start
    )
    read_soft_clipping = read_cigar[0][4] <= 1000
    read_hard_clipping = read_cigar[0][5] <= 1000
    if (
        int(read_is_proper_pair)
        + int(read_mq)
        + int(read_bq)
        + int(read_mismatch)
        + int(read_spanning)
        + int(read_soft_clipping)
        + int(read_hard_clipping)
    ) == 7 and (
        int(read_is_duplicate)
        + int(read_is_supplementary)
        + int(read_is_secondary)
        + int(read_is_unmapped)
        + int(read_is_qcfail)
    ) == 0:
        return True
    else:
        return False
# ----------------------------- Privacy Function End ----------------------------- #


# ----------------------------- HMM Class Start ----------------------------- #
class HMMSeggerInit:
    """Initialize HMM parameters for HMM segmentation."""

    def __init__(
        self, all_motif_params, L_steps_dict, base_index_dict, default_probability
    ):
        self.all_motif_params = all_motif_params
        self.L_steps_dict = L_steps_dict
        self.base_index_dict = base_index_dict
        self.default_probability = default_probability
        self.hidden_states_dict = self.__make_hidden_states_dict()
        self.trans_matrix_dict = self.__make_trans_matrix_dict(
            self.hidden_states_dict, default_probability, all_motif_params
        )
        self.start_matrix_dict = self.__make_start_matrix_dict(
            self.hidden_states_dict, default_probability, all_motif_params
        )
        self.init_state = True

    def __str__(self) -> str:
        return "HMM initialization object"

    def __make_hidden_states(self, motif_length) -> list:
        hidden_states = []
        if motif_length == 1:
            state_order = ["S", "M"]
        elif motif_length == 2:
            state_order = ["S", "I", "M"]
        else:
            state_order = ["S", "I", "M", "O"]
        for i in state_order:
            for base in range(1, motif_length + 1):
                hidden_states.append(i + "_" + str(base))
        hidden_states.append("F1")
        hidden_states.append("F2")
        hidden_states.append("F3")
        hidden_states.append("F")
        return hidden_states

    def __make_hidden_states_dict(self) -> dict:
        hidden_states_dict = {}
        for i in range(1, 7):
            hidden_states_dict[i] = self.__make_hidden_states(i)
        return hidden_states_dict

    def __make_start_probability_matrix(
        self, hidden_states, motif_length, default_probability, **params
    ) -> np.array:
        states_num = len(hidden_states)
        start_matrix = np.full((states_num, 1), default_probability)
        mismatch_rate = params["mismatch_rate"]
        if motif_length == 1:
            start_matrix[1, :] = mismatch_rate
            start_matrix[0, :] = 1 - mismatch_rate - start_matrix[2] * 4
        else:
            for i in range(
                motif_length * 2, motif_length * 3
            ):  # For mismatch start rate
                start_matrix[i, :] = mismatch_rate
            if states_num == motif_length * 3 + 4:
                for i in range(motif_length):  # For STR start rate
                    start_matrix[i, :] = (
                        1
                        - motif_length * mismatch_rate
                        - default_probability * (motif_length + 4)
                    ) / motif_length
            elif states_num == motif_length * 4 + 4:
                for i in range(motif_length):  # For STR start rate
                    start_matrix[i, :] = (
                        1
                        - motif_length * mismatch_rate
                        - default_probability * (motif_length * 2 + 4)
                    ) / motif_length
        return np.squeeze(start_matrix)

    def __make_start_matrix_dict(
        self, hidden_states_dict, default_probability, params
    ) -> dict:
        start_matrix_dict = {}
        for i in range(1, 7):
            start_matrix_dict[i] = self.__make_start_probability_matrix(
                hidden_states_dict[i], i, default_probability, **(params[i])
            )
        return start_matrix_dict

    def __make_mononucleotide_trans_matrix(
        self,
        mononucleotide_hidden_states,
        to_flanking_L_step,
        default_probability,
        **params,
    ) -> np.ndarray:
        trans_matrix = np.full(
            (len(mononucleotide_hidden_states), len(mononucleotide_hidden_states)),
            default_probability,
        )
        trans_matrix[0, 2] = 1 / to_flanking_L_step  # Flanking F1
        trans_matrix[0, 5] = 1 / (to_flanking_L_step + 3)  # Flanking F
        trans_matrix[0, 1] = params["mismatch_rate"]  # Mismatch
        trans_matrix[0, 0] = 1 - sum(trans_matrix[0, :]) + default_probability

        trans_matrix[1, 1] = params["mis_2_mis_rate"]
        trans_matrix[
            1, 5
        ] = default_probability  # Here only STR can transfer to flanking hidden state
        trans_matrix[1, 0] = 1 - sum(trans_matrix[1, :]) + default_probability

        trans_matrix[2, 5] = 1 / 3  # Flanking F
        trans_matrix[2, 3] = (
            1 - sum(trans_matrix[2, :]) + default_probability
        )  # Flanking F2

        trans_matrix[3, 5] = 1 / 3  # Flanking F
        trans_matrix[3, 4] = (
            1 - sum(trans_matrix[3, :]) + default_probability
        )  # Flanking F3

        trans_matrix[4, 5] = (
            1 - sum(trans_matrix[4, :]) + default_probability
        )  # Flanking F

        trans_matrix[5, 5] = (
            1 - sum(trans_matrix[5, :]) + default_probability
        )  # Flanking F

        return trans_matrix.T

    def __make_dinucleotide_trans_matrix(
        self,
        dinucleotide_hidden_states,
        motif_length,
        to_flanking_L_step,
        default_probability,
        **params,
    ) -> np.ndarray:
        trans_matrix = np.full(
            (len(dinucleotide_hidden_states), len(dinucleotide_hidden_states)),
            default_probability,
        )

        for i in range(motif_length):  # For STR
            trans_matrix[i, 3 * motif_length] = 1 / to_flanking_L_step  # Flanking F1
            trans_matrix[i, 3 * motif_length + 3] = 1 / (
                to_flanking_L_step + 3
            )  # Flanking F
            trans_matrix[i, i] = params["deletion_rate"]  # Del
            trans_matrix[i, i + motif_length] = params["insertion_rate"]  # Ins
            if i != (motif_length - 1):
                trans_matrix[i, i + 2 * motif_length + 1] = params[
                    "mismatch_rate"
                ]  # Mis
                trans_matrix[i, i + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
            else:
                trans_matrix[i, 2 * motif_length] = params["mismatch_rate"]  # Mis
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR

        for i in range(motif_length, motif_length * 2):  # For Ins
            # trans_matrix[i,3*motif_length]=1/to_flanking_L_step # Here don't permit ins to flanking
            trans_matrix[i, i] = params["ins_2_ins_rate"]  # Ins
            if i != (2 * motif_length - 1):
                trans_matrix[i, i + motif_length + 1] = params["ins_2_mis_rate"]  # Mis
                trans_matrix[i, i - motif_length + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
            else:
                trans_matrix[i, 2 * motif_length] = params["ins_2_mis_rate"]  # Mis
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR

        for i in range(motif_length * 2, motif_length * 3):  # for Mis
            # trans_matrix[i,3*motif_length]=1/to_flanking_L_step # Don't permit mis to flanking
            trans_matrix[i, i - motif_length] = params["mis_2_ins_rate"]  # Ins
            if i != (3 * motif_length - 1):
                trans_matrix[i, i + 1] = params["mis_2_mis_rate"]  # Mis
                trans_matrix[i, i - motif_length * 2 + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
            else:
                trans_matrix[i, 2 * motif_length] = params["mis_2_mis_rate"]  # Mis
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
        trans_matrix[3 * motif_length, 3 * motif_length + 3] = 1 / 3
        trans_matrix[3 * motif_length, 3 * motif_length + 1] = (
            1 - sum(trans_matrix[3 * motif_length, :]) + default_probability
        )
        trans_matrix[3 * motif_length + 1, 3 * motif_length + 3] = 1 / 3
        trans_matrix[3 * motif_length + 1, 3 * motif_length + 2] = (
            1 - sum(trans_matrix[3 * motif_length + 1, :]) + default_probability
        )
        trans_matrix[3 * motif_length + 2, 3 * motif_length + 3] = (
            1 - sum(trans_matrix[3 * motif_length + 2, :]) + default_probability
        )
        trans_matrix[3 * motif_length + 3, 3 * motif_length + 3] = (
            1 - sum(trans_matrix[3 * motif_length + 3, :]) + default_probability
        )
        return trans_matrix.T

    def __make_multipolymer_trans_matrix(
        self,
        multipolymer_hidden_states,
        motif_length,
        to_flanking_L_step,
        default_probability,
        **params,
    ) -> np.ndarray:
        trans_matrix = np.full(
            (len(multipolymer_hidden_states), len(multipolymer_hidden_states)),
            default_probability,
        )

        for i in range(motif_length):  # For STR
            trans_matrix[i, 4 * motif_length] = 1 / to_flanking_L_step  # Flanking F1
            trans_matrix[i, 4 * motif_length + 3] = 1 / (
                to_flanking_L_step + 3
            )  # Flanking F
            trans_matrix[i, i + motif_length] = params["insertion_rate"]  # Ins
            trans_matrix[i, i + 3 * motif_length] = params[
                "out_frame_rate"
            ]  # Out-frame interruption
            if i != (motif_length - 1) and i != (motif_length - 2):
                trans_matrix[i, i + 2 * motif_length + 1] = params[
                    "mismatch_rate"
                ]  # Mis
                for j in range(motif_length):  # STR and deletion
                    if j == i + 1:
                        pass
                    elif j == i + 2:
                        trans_matrix[i, j] = params["deletion_rate"]  # Del
                    else:
                        trans_matrix[i, j] = params[
                            "continous_deletion_rate"
                        ]  # Continous del
                trans_matrix[i, i + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
            elif i == (motif_length - 2):
                trans_matrix[i, i + 2 * motif_length + 1] = params[
                    "mismatch_rate"
                ]  # Mis
                for j in range(motif_length):  # STR and deletion
                    if j == i + 1:
                        pass
                    elif j == 0:
                        trans_matrix[i, j] = params["deletion_rate"]  # Del
                    else:
                        trans_matrix[i, j] = params[
                            "continous_deletion_rate"
                        ]  # Continous del
                trans_matrix[i, i + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR

            else:
                trans_matrix[i, 2 * motif_length] = params["mismatch_rate"]  # Mis
                for j in range(motif_length):  # for STR and deletion
                    if j == 0:
                        pass
                    elif j == 1:
                        trans_matrix[i, j] = params["deletion_rate"]  # del
                    else:
                        trans_matrix[i, j] = params[
                            "continous_deletion_rate"
                        ]  # Continous del
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expectation STR

        for i in range(motif_length, motif_length * 2):  # For ins
            # trans_matrix[i,4*motif_length]=1/to_flanking_L_step # Don't permit mis to flanking
            trans_matrix[i, i] = params["ins_2_ins_rate"]  # ins
            trans_matrix[i, i + motif_length * 2] = params[
                "ins_2_out_frame_rate"
            ]  # Out-frame interruption
            if i != (2 * motif_length - 1):
                trans_matrix[i, i + motif_length + 1] = params["ins_2_mis_rate"]  # Mis
                trans_matrix[i, i - motif_length + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
            else:
                trans_matrix[i, 2 * motif_length] = params["ins_2_mis_rate"]  # Mis
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR

        for i in range(motif_length * 2, motif_length * 3):  # for mis
            # trans_matrix[i,4*motif_length]=1/to_flanking_L_step # Don't permit mis to flanking
            trans_matrix[i, i - motif_length] = params["mis_2_ins_rate"]  # Ins
            trans_matrix[i, i + motif_length] = params[
                "mis_2_out_frame_rate"
            ]  # Out-frame interruption
            if i != (3 * motif_length - 1):
                trans_matrix[i, i + 1] = params["mis_2_mis_rate"]  # Mis
                trans_matrix[i, i - motif_length * 2 + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
            else:
                trans_matrix[i, 2 * motif_length] = params["mis_2_mis_rate"]  # Mis
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR

        for i in range(
            motif_length * 3, motif_length * 4
        ):  # For out-frame interruption
            # trans_matrix[i,4*motif_length]=1/to_flanking_L_step # Don't permit mis to flanking
            row_sum = (
                1 - sum(trans_matrix[i, :]) + default_probability * (motif_length - 2)
            )
            for j in range(0, motif_length):
                if i - motif_length * 3 + 2 < motif_length:
                    if i - motif_length * 3 + 2 == j or i - motif_length * 3 + 1 == j:
                        pass
                    else:
                        trans_matrix[i, j] = row_sum / (
                            motif_length - 2
                        )  # Expected STR
                elif i - motif_length * 3 + 1 < motif_length:
                    if i - motif_length * 4 + 2 == j or i - motif_length * 3 + 1 == j:
                        pass
                    else:
                        trans_matrix[i, j] = row_sum / (
                            motif_length - 2
                        )  # Expected STR
                else:
                    if i - motif_length * 4 + 2 == j or i - motif_length * 4 + 1 == j:
                        pass
                    else:
                        trans_matrix[i, j] = row_sum / (
                            motif_length - 2
                        )  # Expected STR
        trans_matrix[4 * motif_length, 4 * motif_length + 3] = 1 / 3
        trans_matrix[4 * motif_length, 4 * motif_length + 1] = (
            1 - sum(trans_matrix[4 * motif_length, :]) + default_probability
        )

        trans_matrix[4 * motif_length + 1, 4 * motif_length + 3] = 1 / 3
        trans_matrix[4 * motif_length + 1, 4 * motif_length + 2] = (
            1 - sum(trans_matrix[4 * motif_length + 1, :]) + default_probability
        )

        trans_matrix[4 * motif_length + 2, 4 * motif_length + 3] = (
            1 - sum(trans_matrix[4 * motif_length + 2, :]) + default_probability
        )
        trans_matrix[4 * motif_length + 3, 4 * motif_length + 3] = (
            1 - sum(trans_matrix[4 * motif_length + 3, :]) + default_probability
        )

        return trans_matrix.T

    def __make_trans_matrix_dict(
        self, hidden_states_dict, default_probability, params
    ) -> dict:
        trans_matrix_dict = {}
        for i in range(1, 7):
            to_flanking_L_step = self.L_steps_dict[i]
            if i == 1:
                trans_matrix_dict[1] = self.__make_mononucleotide_trans_matrix(
                    hidden_states_dict[i],
                    to_flanking_L_step,
                    default_probability,
                    **(params[1]),
                )
            elif i == 2:
                trans_matrix_dict[2] = self.__make_dinucleotide_trans_matrix(
                    hidden_states_dict[i],
                    i,
                    to_flanking_L_step,
                    default_probability,
                    **(params[2]),
                )
            else:
                trans_matrix_dict[i] = self.__make_multipolymer_trans_matrix(
                    hidden_states_dict[i],
                    i,
                    to_flanking_L_step,
                    default_probability,
                    **(params[i]),
                )
        return trans_matrix_dict

    def make_mononucleotide_emis_matrix(
        self, hidden_states, motif, base_index_dict, error_rate, flanking_sequence
    ) -> np.ndarray:
        emis_matrix = np.full((len(hidden_states), 4), error_rate / 3)
        emis_matrix[0, base_index_dict[motif[0]]] = 1 - error_rate
        emis_matrix[1, :] = (3 - error_rate) / 9
        emis_matrix[1, base_index_dict[motif[0]]] = error_rate / 3

        emis_matrix[2, base_index_dict[flanking_sequence[0]]] = 1 - error_rate
        emis_matrix[3, base_index_dict[flanking_sequence[1]]] = 1 - error_rate
        emis_matrix[4, base_index_dict[flanking_sequence[2]]] = 1 - error_rate

        emis_matrix[5, :] = 1 / 4
        return emis_matrix

    def make_dinucleotide_emis_matrix(
        self,
        hidden_states,
        motif,
        motif_length,
        base_index_dict,
        error_rate,
        flanking_sequence,
    ) -> np.ndarray:
        emis_matrix = np.full((len(hidden_states), 4), error_rate / 3)
        for i in range(motif_length):  # For STR
            emis_matrix[i, base_index_dict[motif[i]]] = 1 - error_rate
        for i in range(motif_length, motif_length * 2):  # For ins
            emis_matrix[i, :] = (3 - error_rate) / 9
            if i != (2 * motif_length - 1):
                emis_matrix[i, base_index_dict[motif[i - motif_length + 1]]] = (
                    error_rate / 3
                )
            else:
                emis_matrix[i, base_index_dict[motif[0]]] = error_rate / 3
        for i in range(motif_length * 2, motif_length * 3):  # For mis
            emis_matrix[i, :] = (3 - error_rate) / 9
            emis_matrix[i, base_index_dict[motif[i - motif_length * 2]]] = (
                error_rate / 3
            )

        emis_matrix[3 * motif_length, base_index_dict[flanking_sequence[0]]] = (
            1 - error_rate
        )
        emis_matrix[3 * motif_length + 1, base_index_dict[flanking_sequence[1]]] = (
            1 - error_rate
        )
        emis_matrix[3 * motif_length + 2, base_index_dict[flanking_sequence[2]]] = (
            1 - error_rate
        )

        emis_matrix[3 * motif_length + 3, :] = 1 / 4
        return emis_matrix

    def make_multipolymer_emis_matrix(
        self,
        hidden_states,
        motif,
        motif_length,
        base_index_dict,
        error_rate,
        flanking_sequence,
    ) -> np.ndarray:
        emis_matrix = np.full((len(hidden_states), 4), error_rate / 3)
        for i in range(motif_length):  # For STR
            emis_matrix[i, base_index_dict[motif[i]]] = 1 - error_rate
        for i in range(motif_length, motif_length * 2):  # For ins
            emis_matrix[i, :] = (3 - error_rate) / 9
            if i != (2 * motif_length - 1):
                emis_matrix[i, base_index_dict[motif[i - motif_length + 1]]] = (
                    error_rate / 3
                )
            else:
                emis_matrix[i, base_index_dict[motif[0]]] = error_rate / 3
        for i in range(motif_length * 2, motif_length * 3):  # For mis
            emis_matrix[i, :] = (3 - error_rate) / 9
            emis_matrix[i, base_index_dict[motif[i - motif_length * 2]]] = (
                error_rate / 3
            )

        for i in range(
            motif_length * 3, motif_length * 4
        ):  # For out-frame interruption
            emis_matrix[i, :] = (3 - error_rate) / 9
            if i != (4 * motif_length - 1):
                emis_matrix[i, base_index_dict[motif[i - motif_length * 3 + 1]]] = (
                    error_rate / 3
                )
            else:
                emis_matrix[i, base_index_dict[motif[0]]] = error_rate / 3
        emis_matrix[4 * motif_length, base_index_dict[flanking_sequence[0]]] = (
            1 - error_rate
        )
        emis_matrix[4 * motif_length + 1, base_index_dict[flanking_sequence[1]]] = (
            1 - error_rate
        )
        emis_matrix[4 * motif_length + 2, base_index_dict[flanking_sequence[2]]] = (
            1 - error_rate
        )

        emis_matrix[4 * motif_length + 3, :] = 1 / 4
        return emis_matrix

    def read_segmentation_DP(
        self,
        seq,
        seq_quality,
        motif,
        hidden_states,
        start_matrix,
        trans_matrix,
        base_index_dict,
        flanking_sequence,
        state_map,
    ) -> list:
        """HMM dynamic programming to decode hidden states."""
        if len(motif) == 1:
            emis_matrix = self.make_mononucleotide_emis_matrix(
                hidden_states, motif, base_index_dict, seq_quality[0], flanking_sequence
            )
        elif len(motif) == 2:
            emis_matrix = self.make_dinucleotide_emis_matrix(
                hidden_states,
                motif,
                len(motif),
                base_index_dict,
                seq_quality[0],
                flanking_sequence,
            )
        else:
            emis_matrix = self.make_multipolymer_emis_matrix(
                hidden_states,
                motif,
                len(motif),
                base_index_dict,
                seq_quality[0],
                flanking_sequence,
            )
        path_prob = np.log(start_matrix) + np.log(
            emis_matrix[:, base_index_dict[seq[0]]]
        )
        path = []
        for o, q in zip(seq[1:], seq_quality[1:]):
            if len(motif) == 1:
                emis_matrix = self.make_mononucleotide_emis_matrix(
                    hidden_states, motif, base_index_dict, q, flanking_sequence
                )
            elif len(motif) == 2:
                emis_matrix = self.make_dinucleotide_emis_matrix(
                    hidden_states,
                    motif,
                    len(motif),
                    base_index_dict,
                    q,
                    flanking_sequence,
                )
            else:
                emis_matrix = self.make_multipolymer_emis_matrix(
                    hidden_states,
                    motif,
                    len(motif),
                    base_index_dict,
                    q,
                    flanking_sequence,
                )
            path_prob = path_prob + np.log(trans_matrix)
            best_state = np.argmax(path_prob, axis=1)
            best_prob = path_prob[np.arange(len(best_state)), best_state]
            path_prob = best_prob + np.log(emis_matrix[:, base_index_dict[o]])
            path.append(best_state)
        best_state = np.argmax(path_prob)
        # final_best_prob = path_prob[best_state] # Likelihood of best path
        best_state_sequence = [state_map[best_state]]
        for prev_state in path[::-1]:
            best_state = prev_state[best_state]
            best_state_sequence.append(state_map[best_state])
            final_best_path = best_state_sequence[::-1]
        return final_best_path


class HMMSeggerLocus:
    """Load STR locus information from bed file and initialize HMM Segger."""

    def __init__(self, hmm_segger_init, chrom, start, end, str_id, str_unit, left_flanking_three_bp, right_flanking_three_bp):
        self.hmm_segger_init = hmm_segger_init
        self.load_STR_locus_info(chrom, start, end, str_id, str_unit, left_flanking_three_bp, right_flanking_three_bp)
        self.load_str_locus = True

    def __str__(self):
        return f"STR locus {self.STR_id} at {self.chrom}:{self.start}-{self.end}"

    def load_STR_locus_info(self, chrom, start, end, str_id, str_unit, left_flanking_three_bp, right_flanking_three_bp):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.STR_id = str_id
        self.motif = str_unit.upper()
        self.motif_length = len(self.motif)
        self.STR_length = end - start
        self.left_flanking_three_bp = left_flanking_three_bp
        self.right_flanking_three_bp = right_flanking_three_bp
        self.reverse_motif = self.motif[::-1]
        self.reverse_left_flanking_three_bp = self.left_flanking_three_bp[::-1]
        self.trans_mat = self.hmm_segger_init.trans_matrix_dict[self.motif_length]
        self.init_mat = self.hmm_segger_init.start_matrix_dict[self.motif_length]
        self.hidden_states = self.hmm_segger_init.hidden_states_dict[self.motif_length]
        self.L_step = self.hmm_segger_init.L_steps_dict[self.motif_length]
        state_map = {}
        for i, j in zip(
            [k for k in range(len(self.hidden_states))], self.hidden_states
        ):
            state_map[i] = j
        self.state_map = state_map


class SegRead:
    """Segment single read into Flanking segment and STR segment."""

    def __init__(self, hmmsegger_init, hmmsegger_locus, read):
        self.hmmsegger_init = hmmsegger_init
        self.hmmsegger_locus = hmmsegger_locus
        self.pysam_read = read
        self.reference_start = read.reference_start
        self.reference_end = read.reference_end
        (
            self.left_seg_seq,
            self.right_seg_seq,
            self.left_seg_baseq_error_rate,
            self.right_seg_baseq_error_rate,
            self.left_seq_seg_start_index_include_this_base,
            self.right_seq_seg_start_index_include_this_base,
        ) = self.extract_segmentation_seq(read)
        if (
            (self.left_seg_seq in ["InDels in anchor bigger than 18","InDels in anchor bigger than 15"])
            or (self.left_seg_seq == "No enough length for segmentation")
            or ("N" in self.left_seg_seq)
            or ("N" in self.right_seg_seq)
        ):
            print("Warning: InDels in anchor bigger than 18 or 15 or No enough length for segmentation or N in left or right seg seq in read")
        #     if _DEBUG is True:
        #         sys.stderr.write(
        #             "Warning: InDels in anchor bigger than 12 or No enough length for segmentation or N in left or right seg seq in read: %s \n"
        #             % read.query_name
        #         )
        else:
            self.segmentation_by_hmm_per_read()
            if self.STR_length == "STR length out of range":
                print("Warning: STR length out of range in read")
                # if _DEBUG is True:
                #     sys.stderr.write(
                #         "Warning: STR length out of range in read: %s" % read.query_name
                #     )
            else:
                pass
                if _DEBUG is True:
                    print(f"STR length of read {read.query_name} is: {self.STR_length}")

    def __str__(self):
        return f"Segmented read {self.pysam_read.query_name} at {self.reference_start}-{self.reference_end}"

    def extract_segmentation_seq(self, read):
        try:
            aligned_pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
        except ValueError as e:
            if "MD tag not present" in str(e):
                # Handle the case where MD tag is missing
                # You might want to skip this read or use an alternative method
                print(f"MD tag not present for read {read.query_name}. Skipping.")
                return None, None, None, None, None, None, None, None
            else:
                raise  # Re-raise if it's a different ValueError
        self.query_qualities = read.query_qualities
        aligned_pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
        self.refpos_queryindex = {i[1]: [i[0], i[2]] for i in aligned_pairs}
        self.read_seq = read.query_alignment_sequence
        read_baseq = read.query_alignment_qualities
        read_baseq_error_rate = _baseq_2_error_rate(read_baseq)
        if read_baseq_error_rate.min() == 0:
            read_baseq_error_rate[read_baseq_error_rate == 0] = read_baseq_error_rate[
                read_baseq_error_rate.nonzero()
            ].min()
        if read_baseq_error_rate.max() == 1:
            read_baseq_error_rate[read_baseq_error_rate == 1] = read_baseq_error_rate[
                read_baseq_error_rate != 1
            ].max()
        self.read_baseq_accuray = 1 - read_baseq_error_rate
        self.alignment_start_index = read.query_alignment_start
        self.alignment_end_index = read.query_alignment_end
        left_flanking_last_base_pos = self.hmmsegger_locus.start - 1
        k = 0
        while True:
            if self.refpos_queryindex.get(left_flanking_last_base_pos) is None:
                left_flanking_last_base_pos -= 1
                k += 1
            else:
                flanking_left_segmentation_site_index = self.refpos_queryindex.get(
                    left_flanking_last_base_pos
                )[0]
                break
            if k >= 15:
                return "InDels in anchor bigger than 15", None, None, None, None, None
        right_flanking_first_base_pos = self.hmmsegger_locus.end
        k = 0
        while True:
            if self.refpos_queryindex.get(right_flanking_first_base_pos) is None:
                right_flanking_first_base_pos += 1
                k += 1
            else:
                flanking_right_segmentation_site_index = self.refpos_queryindex.get(
                    right_flanking_first_base_pos
                )[0]
                break
            if k >= 15:
                return "InDels in anchor bigger than 15", None, None, None, None, None

        left_STR_first_base_pos = left_flanking_last_base_pos + 1
        k = 0
        while True:
            if self.refpos_queryindex.get(left_STR_first_base_pos) is None:
                left_STR_first_base_pos += 1
                k += 1
            else:
                STR_left_segmentation_site_index = self.refpos_queryindex.get(
                    left_STR_first_base_pos
                )[0]
                break
            if k >= 18:
                return "InDels in anchor bigger than 18", None, None, None, None, None
            
        right_STR_last_base_pos = right_flanking_first_base_pos - 1
        k = 0
        while True:
            if self.refpos_queryindex.get(right_STR_last_base_pos) is None:
                right_STR_last_base_pos -= 1
                k += 1
            else:
                STR_right_segmentation_site_index = self.refpos_queryindex.get(
                    right_STR_last_base_pos
                )[0]
                break
            if k >= 18:
                return "InDels in anchor bigger than 18", None, None, None, None, None

        self.read_spanning = (
            flanking_left_segmentation_site_index - self.alignment_start_index + 1 >= 10
        ) and (
            self.alignment_end_index - flanking_right_segmentation_site_index + 1
        ) >= 10
        self.enoughSTR = (
            STR_right_segmentation_site_index - STR_left_segmentation_site_index + 1
        ) >= self.hmmsegger_locus.L_step

        if self.read_spanning and self.enoughSTR:
            # base pairs number level based read spanning and enough STR
            left_seg_seq = self.read_seq[
                flanking_left_segmentation_site_index
                - self.alignment_start_index
                - 10
                + 1 : STR_left_segmentation_site_index
                - self.alignment_start_index
                + self.hmmsegger_locus.L_step
            ]
            right_seg_seq = self.read_seq[
                STR_right_segmentation_site_index
                - self.alignment_start_index
                - self.hmmsegger_locus.L_step
                + 1 : flanking_right_segmentation_site_index
                - self.alignment_start_index
                + 10
            ]
            left_seg_baseq_error_rate = read_baseq_error_rate[
                flanking_left_segmentation_site_index
                - self.alignment_start_index
                - 10
                + 1 : STR_left_segmentation_site_index
                - self.alignment_start_index
                + self.hmmsegger_locus.L_step
            ]
            right_seg_baseq_error_rate = read_baseq_error_rate[
                STR_right_segmentation_site_index
                - self.alignment_start_index
                - self.hmmsegger_locus.L_step
                + 1 : flanking_right_segmentation_site_index
                - self.alignment_start_index
                + 10
            ]
            left_seq_seg_start_index_include_this_base = (
                STR_left_segmentation_site_index
                - self.alignment_start_index
                + self.hmmsegger_locus.L_step
                - 1
            )
            right_seq_seg_start_index_include_this_base = (
                STR_right_segmentation_site_index
                - self.alignment_start_index
                - self.hmmsegger_locus.L_step
                + 1
            )
            return (
                left_seg_seq,
                right_seg_seq,
                left_seg_baseq_error_rate,
                right_seg_baseq_error_rate,
                left_seq_seg_start_index_include_this_base,
                right_seq_seg_start_index_include_this_base,
            )
        else:
            return "No enough length for segmentation", None, None, None, None, None

    def segmentation_seq(self, seq, path):
        if "F1" in path:
            anchor_FLK_first_base_0_based = path.index("F1")
        elif "F2" in path:
            anchor_FLK_first_base_0_based = path.index("F2")
        elif "F3" in path:
            anchor_FLK_first_base_0_based = path.index("F3")
        elif "F" in path:
            anchor_FLK_first_base_0_based = path.index("F")
        else:
            anchor_FLK_first_base_0_based = len(seq)
        return anchor_FLK_first_base_0_based

    def segmentation_by_hmm_per_read(self):
        reverse_left_seg_seq = self.left_seg_seq[::-1].upper()
        reverse_left_baseq_error_rate = self.left_seg_baseq_error_rate[::-1]
        right_seg_seq = self.right_seg_seq.upper()
        self.left_final_best_path = self.hmmsegger_init.read_segmentation_DP(
            reverse_left_seg_seq,
            reverse_left_baseq_error_rate,
            self.hmmsegger_locus.reverse_motif,
            self.hmmsegger_locus.hidden_states,
            self.hmmsegger_locus.init_mat,
            self.hmmsegger_locus.trans_mat,
            self.hmmsegger_init.base_index_dict,
            self.hmmsegger_locus.reverse_left_flanking_three_bp,
            self.hmmsegger_locus.state_map,
        )

        left_anchor_FLK_first_base_0_based = self.segmentation_seq(
            reverse_left_seg_seq, self.left_final_best_path
        )
        self.right_final_best_path = self.hmmsegger_init.read_segmentation_DP(
            right_seg_seq,
            self.right_seg_baseq_error_rate,
            self.hmmsegger_locus.motif,
            self.hmmsegger_locus.hidden_states,
            self.hmmsegger_locus.init_mat,
            self.hmmsegger_locus.trans_mat,
            self.hmmsegger_init.base_index_dict,
            self.hmmsegger_locus.right_flanking_three_bp,
            self.hmmsegger_locus.state_map,
        )
        right_anchor_FLK_first_base_0_based = self.segmentation_seq(
            right_seg_seq, self.right_final_best_path
        )

        if (left_anchor_FLK_first_base_0_based == len(reverse_left_seg_seq)) or (
            right_anchor_FLK_first_base_0_based == len(right_seg_seq)
        ):
            # May exist repeat expansion
            self.STR_length = "STR length out of range"
        else:
            left_STR_first_base_location = (
                self.left_seq_seg_start_index_include_this_base
                - left_anchor_FLK_first_base_0_based
                + 1
            )
            right_STR_last_base_location = (
                self.right_seq_seg_start_index_include_this_base
                + right_anchor_FLK_first_base_0_based
                - 1
            )
            if right_STR_last_base_location + 1 <= left_STR_first_base_location:
                # There may be very short STRs and odd interruptions,
                # resulting in segmentation where the right site is on the left of the left site
                self.STR_length = "STR length out of range"
            else:
                self.STR_first_base_location = left_STR_first_base_location
                self.STR_last_base_location = right_STR_last_base_location
                self.STR_position = (self.STR_first_base_location + self.reference_start, self.STR_last_base_location + self.reference_start)
                self.STR_seq = self.read_seq[
                    left_STR_first_base_location : right_STR_last_base_location + 1
                ]
                self.STR_baseq_accuray = self.read_baseq_accuray[
                    left_STR_first_base_location : right_STR_last_base_location + 1
                ]
                self.STR_seq_qualities = self.query_qualities[left_STR_first_base_location : right_STR_last_base_location + 1]
                self.STR_length = len(self.STR_seq)
                self.left_flanking_ten_bp = self.read_seq[left_STR_first_base_location-10:left_STR_first_base_location]
                self.right_flanking_ten_bp = self.read_seq[right_STR_last_base_location+1:right_STR_last_base_location+11]
                self.left_flanking_ten_bp_qualities = self.query_qualities[left_STR_first_base_location-10:left_STR_first_base_location]
                self.right_flanking_ten_bp_qualities = self.query_qualities[right_STR_last_base_location+1:right_STR_last_base_location+11]
                self.left_flanking_ten_bp_baseq_accuray = self.read_baseq_accuray[left_STR_first_base_location-10:left_STR_first_base_location]
                self.right_flanking_ten_bp_baseq_accuray = self.read_baseq_accuray[right_STR_last_base_location+1:right_STR_last_base_location+11]
# ----------------------------- HMM Class End ----------------------------- #

class StrReads:
    def __init__(self):
        self.str_data = {}

    def compute_str_data(self, bamfile, chrom, start_bed, end_bed, str_id, str_unit, left_flanking_three_bp, right_flanking_three_bp):

        try:
            hmm_seg_init = HMMSeggerInit(
                ALL_MOTIF__PARAMS, L_STEPS_DICT, BASE_INDEX_DICT, DEFAULT_PROBABILITY
            )
            hmm_seg_locus = HMMSeggerLocus(
                hmm_seg_init, chrom, start=start_bed, end=end_bed, str_id=str_id, str_unit=str_unit, 
                left_flanking_three_bp=left_flanking_three_bp, right_flanking_three_bp=right_flanking_three_bp
            )
        except Exception as e:
            print(f"Error initializing HMMSegger: {str(e)}")
            return

        hmm_seg_locus = HMMSeggerLocus(
            hmm_seg_init, chrom, start=start_bed, end=end_bed, str_id=str_id, str_unit=str_unit, left_flanking_three_bp=left_flanking_three_bp, right_flanking_three_bp=right_flanking_three_bp
        )

        fetched_reads = bamfile.fetch(chrom, start_bed, end_bed)
        
        if fetched_reads is None:
            print(f"No reads found in the specified region: {chrom}:{start_bed}-{end_bed}")
            return

        for read in fetched_reads:
            unique_id = f"{read.query_name}_{read.reference_start}"
            try:
                readsegger = SegRead(hmm_seg_init, hmm_seg_locus, read)
            except (ValueError, AttributeError) as e:
                print(f"Error processing read {unique_id}: {str(e)}")
                continue
            if (
                (readsegger.left_seg_seq in ["InDels in anchor bigger than 18", "InDels in anchor bigger than 15"])
                or (readsegger.left_seg_seq == "No enough length for segmentation")
                or ("N" in readsegger.left_seg_seq)
                or ("N" in readsegger.right_seg_seq)
                or readsegger.STR_length == "STR length out of range"
            ):
                continue
            else:
                result = {
                    'STR_seq': readsegger.STR_seq,
                    'Left_flanking_seq': readsegger.left_flanking_ten_bp,
                    'Right_flanking_seq': readsegger.right_flanking_ten_bp,
                    'STR_seq_qualities': readsegger.STR_seq_qualities,
                    'Left_flanking_seq_qualities':  readsegger.left_flanking_ten_bp_qualities,
                    'Right_flanking_seq_qualities': readsegger.right_flanking_ten_bp_qualities,
                    'STR_position': [readsegger.STR_first_base_location, readsegger.STR_last_base_location],
                    'Left_three_flanking_seq': left_flanking_three_bp,
                    'Right_three_flanking_seq': right_flanking_three_bp,
                    'sequence': read.query_sequence
                }
                self.str_data[unique_id] = result

    def items(self):
        return self.str_data.items()

# ----------------------------- HMM Debug Start ----------------------------- #
if _DEBUG is True:
    if __name__ == "__main__":
        chrom = "10"
        start_bed = 13237583-1
        end_bed = 90360149
        hmm_seg_init = HMMSeggerInit(
            ALL_MOTIF__PARAMS, L_STEPS_DICT, BASE_INDEX_DICT, DEFAULT_PROBABILITY
        )
        hmm_seg_locus = HMMSeggerLocus(
            hmm_seg_init, chrom, start = start_bed, end = end_bed, str_id = 'Human_STR_137314', str_unit = "ATTT", left_flanking_three_bp = "ACG", right_flanking_three_bp = "TTT"
        )
        bam = "/storage/douyanmeiLab/fanwenxuan/data/phs001485.v3.p1/SRR13989893.bam"
        pysamAlignmentFile = pysam.AlignmentFile(bam, "rc")
        for read in pysamAlignmentFile.fetch(chrom, start_bed, end_bed):
            readsegger = SegRead(hmm_seg_init, hmm_seg_locus, read)
            if (
                (readsegger.left_seg_seq in
                    ["InDels in anchor bigger than 18", "InDels in anchor bigger than 15"])
                or (
                    readsegger.left_seg_seq
                    == "No enough length for segmentation"
                )
                or ("N" in readsegger.left_seg_seq)
                or ("N" in readsegger.right_seg_seq)
            ):
                continue
            else:
                if readsegger.STR_length == "STR length out of range":
                    continue
                else:
                    print(read)
                    print(readsegger.STR_seq)
                    print(readsegger.left_flanking_ten_bp)
                    print(readsegger.right_flanking_ten_bp)
# ----------------------------- HMM Debug End ----------------------------- #
