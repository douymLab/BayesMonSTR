# Author: Weixiang Wang
# Date: 2024-03-11
# Description: Hidden Markov Model for Read Segmentation

import sys
import numpy as np
import pysam
import config_params

COORDINATE_SEG = config_params.COORDINATE_SEG
GLOBAL_FOR_STR_AND_FLANKING = COORDINATE_SEG[
    "GLOBAL_FOR_STR_AND_FLANKING"
]  # True
PADDING_BPS = 5  # 10 # revise_padding
PADDING_BPS = COORDINATE_SEG["PADDING_BPS"]  # 5 # 10 # revise_padding
DYNAMIC_PADDING = COORDINATE_SEG["DYNAMIC_PADDING"]  # False
DYNAMIC_PADDING_DICT = COORDINATE_SEG[
    "DYNAMIC_PADDING_DICT"
]  # {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

_DEBUG = False

# ----------------------------- Hidden Markov Model Parameters Begin ----------------------------- #
ALL_MOTIF_PARAMS = {}

_PARAMS_1 = {}
_PARAMS_1["mismatch_rate"] = 0.005
_PARAMS_1["error_rate"] = 0.001
_PARAMS_1["mis_2_mis_rate"] = 0.005
ALL_MOTIF_PARAMS[1] = _PARAMS_1

_PARAMS_2 = {}
_PARAMS_2["mismatch_rate"] = 0.01
_PARAMS_2["insertion_rate"] = 0.009
_PARAMS_2["deletion_rate"] = 0.009
_PARAMS_2["error_rate"] = 0.001
_PARAMS_2["ins_2_ins_rate"] = 0.009
_PARAMS_2["ins_2_mis_rate"] = 0.01
_PARAMS_2["mis_2_ins_rate"] = 0.009
_PARAMS_2["mis_2_mis_rate"] = 0.01
ALL_MOTIF_PARAMS[2] = _PARAMS_2

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
ALL_MOTIF_PARAMS[3] = _PARAMS_3

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
ALL_MOTIF_PARAMS[4] = _PARAMS_4

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
ALL_MOTIF_PARAMS[5] = _PARAMS_5

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
ALL_MOTIF_PARAMS[6] = _PARAMS_6

DEFAULT_PROBABILITY = 1e-8
BASE_INDEX_DICT = {"A": 0, "T": 1, "C": 2, "G": 3}

MAX_ANCHOR_INDEL_BP = 1000000  # 未用, 用对于 TGS True
MIN_FLANKING_SPANING_BP = 5  # 未用 # revise_padding
FLANKING_ALLELE_BASE_PAIRS = 5  # 未用
HMM_FLANKING_LENGTH = 5  # 未用
MAX_STR_LENGTH = 150  # 150
L_STEPS_DICT = {1: 5, 2: 6, 3: 6, 4: 8, 5: 10, 6: 12}  # 未用
ALLOW_MIS_OR_INS_2_F = True  # 未用
DONT_ALLOW_N = True  # Don't allow reads with a N base in Genotyping regions or the whole read sequence
TGS = True


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


# ----------------------------- HMM Class Start ----------------------------- #
class HMMSeggerInit:
    """Initialize HMM parameters for HMM segmentation."""

    def __init__(
        self,
        all_motif_params,
        L_steps_dict,
        base_index_dict,
        default_probability,
        flanking_length,
    ):
        self.flanking_length = flanking_length
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
        for i in range(1, self.flanking_length + 1):
            hidden_states.append(f"F{i}")
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
            start_matrix[0, :] = (
                1
                - mismatch_rate
                - start_matrix[2] * (self.flanking_length + 1)
            )
        else:
            for i in range(
                motif_length * 2, motif_length * 3
            ):  # For mismatch start rate
                start_matrix[i, :] = mismatch_rate
            if states_num == motif_length * 3 + self.flanking_length + 1:
                for i in range(motif_length):  # For STR start rate
                    start_matrix[i, :] = (
                        1
                        - motif_length * mismatch_rate
                        - default_probability
                        * (motif_length + self.flanking_length + 1)
                    ) / motif_length
            elif states_num == motif_length * 4 + self.flanking_length + 1:
                for i in range(motif_length):  # For STR start rate
                    start_matrix[i, :] = (
                        1
                        - motif_length * mismatch_rate
                        - default_probability
                        * (motif_length * 2 + self.flanking_length + 1)
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
            (
                len(mononucleotide_hidden_states),
                len(mononucleotide_hidden_states),
            ),
            default_probability,
        )
        trans_matrix[0, 2] = 1 / to_flanking_L_step  # Flanking F1
        trans_matrix[0, 2 + self.flanking_length] = 1 / (
            to_flanking_L_step + self.flanking_length
        )  # Flanking F
        trans_matrix[0, 1] = params["mismatch_rate"]  # Mismatch
        trans_matrix[0, 0] = 1 - sum(trans_matrix[0, :]) + default_probability

        trans_matrix[1, 1] = params["mis_2_mis_rate"]
        # TODO: ALLOW MIS 2 FLANKING
        if ALLOW_MIS_OR_INS_2_F:
            trans_matrix[1, 2 + self.flanking_length] = 1 / (
                to_flanking_L_step + self.flanking_length
            )  # Flanking F
            trans_matrix[1, 2] = 1 / (to_flanking_L_step)  # Flanking F1
        trans_matrix[
            1, 2 + self.flanking_length
        ] = default_probability  # Here only STR can transfer to flanking hidden state
        trans_matrix[1, 0] = 1 - sum(trans_matrix[1, :]) + default_probability

        for i in range(2, self.flanking_length + 2 - 1):
            trans_matrix[i, 2 + self.flanking_length] = (
                1 / self.flanking_length
            )  # Flanking F
            trans_matrix[i, i + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # Flanking F2
        trans_matrix[
            self.flanking_length + 2 - 1, self.flanking_length + 2
        ] = (
            1
            - sum(trans_matrix[self.flanking_length + 2 - 1, :])
            + default_probability
        )  # Flanking F

        trans_matrix[self.flanking_length + 2, self.flanking_length + 2] = (
            1
            - sum(trans_matrix[self.flanking_length + 2, :])
            + default_probability
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
            trans_matrix[i, 3 * motif_length] = (
                1 / to_flanking_L_step
            )  # Flanking F1
            trans_matrix[i, 3 * motif_length + self.flanking_length] = 1 / (
                to_flanking_L_step + self.flanking_length
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
                trans_matrix[i, 2 * motif_length] = params[
                    "mismatch_rate"
                ]  # Mis
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR

        for i in range(motif_length, motif_length * 2):  # For Ins
            # trans_matrix[i,3*motif_length]=1/to_flanking_L_step # Here don't permit ins to flanking
            # TODO: ALLOW INS TO FLANKING
            if ALLOW_MIS_OR_INS_2_F:
                trans_matrix[i, 3 * motif_length] = (
                    1 / to_flanking_L_step
                )  # Flanking F1
                trans_matrix[
                    i, 3 * motif_length + self.flanking_length
                ] = 1 / (
                    to_flanking_L_step + self.flanking_length
                )  # Flanking F
            trans_matrix[i, i] = params["ins_2_ins_rate"]  # Ins
            if i != (2 * motif_length - 1):
                trans_matrix[i, i + motif_length + 1] = params[
                    "ins_2_mis_rate"
                ]  # Mis
                trans_matrix[i, i - motif_length + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
            else:
                trans_matrix[i, 2 * motif_length] = params[
                    "ins_2_mis_rate"
                ]  # Mis
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
                trans_matrix[i, 2 * motif_length] = params[
                    "mis_2_mis_rate"
                ]  # Mis
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR

        for i in range(
            3 * motif_length, 3 * motif_length + self.flanking_length - 1
        ):
            trans_matrix[i, 3 * motif_length + self.flanking_length] = (
                1 / self.flanking_length
            )  # Flanking F
            trans_matrix[i, i + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # Flanking F2
        trans_matrix[
            3 * motif_length + self.flanking_length - 1,
            3 * motif_length + self.flanking_length,
        ] = (
            1
            - sum(trans_matrix[3 * motif_length + self.flanking_length - 1, :])
            + default_probability
        )
        trans_matrix[
            3 * motif_length + self.flanking_length,
            3 * motif_length + self.flanking_length,
        ] = (
            1
            - sum(trans_matrix[3 * motif_length + self.flanking_length, :])
            + default_probability
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
            trans_matrix[i, 4 * motif_length] = (
                1 / to_flanking_L_step
            )  # Flanking F1
            trans_matrix[i, 4 * motif_length + self.flanking_length] = 1 / (
                to_flanking_L_step + self.flanking_length
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
                trans_matrix[i, 2 * motif_length] = params[
                    "mismatch_rate"
                ]  # Mis
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
            # TODO: ALLOW INS TO FLANKING
            if ALLOW_MIS_OR_INS_2_F:
                trans_matrix[i, 4 * motif_length] = (
                    1 / to_flanking_L_step
                )  # Flanking F1
                trans_matrix[
                    i, 4 * motif_length + self.flanking_length
                ] = 1 / (
                    to_flanking_L_step + self.flanking_length
                )  # Flanking F
            trans_matrix[i, i] = params["ins_2_ins_rate"]  # ins
            trans_matrix[i, i + motif_length * 2] = params[
                "ins_2_out_frame_rate"
            ]  # Out-frame interruption
            if i != (2 * motif_length - 1):
                trans_matrix[i, i + motif_length + 1] = params[
                    "ins_2_mis_rate"
                ]  # Mis
                trans_matrix[i, i - motif_length + 1] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR
            else:
                trans_matrix[i, 2 * motif_length] = params[
                    "ins_2_mis_rate"
                ]  # Mis
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
                trans_matrix[i, 2 * motif_length] = params[
                    "mis_2_mis_rate"
                ]  # Mis
                trans_matrix[i, 0] = (
                    1 - sum(trans_matrix[i, :]) + default_probability
                )  # Expected STR

        for i in range(
            motif_length * 3, motif_length * 4
        ):  # For out-frame interruption
            # trans_matrix[i,4*motif_length]=1/to_flanking_L_step # Don't permit mis to flanking
            row_sum = (
                1
                - sum(trans_matrix[i, :])
                + default_probability * (motif_length - 2)
            )
            for j in range(0, motif_length):
                if i - motif_length * 3 + 2 < motif_length:
                    if (
                        i - motif_length * 3 + 2 == j
                        or i - motif_length * 3 + 1 == j
                    ):
                        pass
                    else:
                        trans_matrix[i, j] = row_sum / (
                            motif_length - 2
                        )  # Expected STR
                elif i - motif_length * 3 + 1 < motif_length:
                    if (
                        i - motif_length * 4 + 2 == j
                        or i - motif_length * 3 + 1 == j
                    ):
                        pass
                    else:
                        trans_matrix[i, j] = row_sum / (
                            motif_length - 2
                        )  # Expected STR
                else:
                    if (
                        i - motif_length * 4 + 2 == j
                        or i - motif_length * 4 + 1 == j
                    ):
                        pass
                    else:
                        trans_matrix[i, j] = row_sum / (
                            motif_length - 2
                        )  # Expected STR

        for i in range(
            4 * motif_length, 4 * motif_length + self.flanking_length - 1
        ):
            trans_matrix[i, 4 * motif_length + self.flanking_length] = (
                1 / self.flanking_length
            )  # Flanking F
            trans_matrix[i, i + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # Flanking F2
        trans_matrix[
            4 * motif_length + self.flanking_length - 1,
            4 * motif_length + self.flanking_length,
        ] = (
            1
            - sum(trans_matrix[4 * motif_length + self.flanking_length - 1, :])
            + default_probability
        )
        trans_matrix[
            4 * motif_length + self.flanking_length,
            4 * motif_length + self.flanking_length,
        ] = (
            1
            - sum(trans_matrix[4 * motif_length + self.flanking_length, :])
            + default_probability
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
        self,
        hidden_states,
        motif,
        base_index_dict,
        error_rate,
        flanking_sequence,
    ) -> np.ndarray:
        emis_matrix = np.full((len(hidden_states), 4), error_rate / 3)
        emis_matrix[0, base_index_dict[motif[0]]] = 1 - error_rate
        emis_matrix[1, :] = (3 - error_rate) / 9
        emis_matrix[1, base_index_dict[motif[0]]] = error_rate / 3

        for i in range(2, self.flanking_length + 2):
            emis_matrix[i, base_index_dict[flanking_sequence[i - 2]]] = (
                1 - error_rate
            )
        emis_matrix[self.flanking_length + 2, :] = 1 / 4
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
                emis_matrix[
                    i, base_index_dict[motif[i - motif_length + 1]]
                ] = (error_rate / 3)
            else:
                emis_matrix[i, base_index_dict[motif[0]]] = error_rate / 3
        for i in range(motif_length * 2, motif_length * 3):  # For mis
            emis_matrix[i, :] = (3 - error_rate) / 9
            emis_matrix[i, base_index_dict[motif[i - motif_length * 2]]] = (
                error_rate / 3
            )

        for i in range(
            3 * motif_length, 3 * motif_length + self.flanking_length
        ):
            emis_matrix[
                i, base_index_dict[flanking_sequence[i - 3 * motif_length]]
            ] = (1 - error_rate)
        emis_matrix[3 * motif_length + self.flanking_length, :] = 1 / 4
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
                emis_matrix[
                    i, base_index_dict[motif[i - motif_length + 1]]
                ] = (error_rate / 3)
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
                emis_matrix[
                    i, base_index_dict[motif[i - motif_length * 3 + 1]]
                ] = (error_rate / 3)
            else:
                emis_matrix[i, base_index_dict[motif[0]]] = error_rate / 3
        for i in range(
            4 * motif_length, 4 * motif_length + self.flanking_length
        ):
            emis_matrix[
                i, base_index_dict[flanking_sequence[i - 4 * motif_length]]
            ] = (1 - error_rate)
        emis_matrix[4 * motif_length + self.flanking_length, :] = 1 / 4
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
                hidden_states,
                motif,
                base_index_dict,
                seq_quality[0],
                flanking_sequence,
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
        final_best_path = best_state_sequence[
            ::-1
        ]  # Not should be in for loop
        return final_best_path


class CoordiSeggerLocus:
    """Load STR locus information from bed file and initialize Coordinate Segger."""

    def __init__(self, pysam_bed_row, pysam_fasta=None):
        self.pysam_bed_row = pysam_bed_row
        self.load_STR_locus_info(pysam_bed_row)
        self.load_str_locus = True
        if DYNAMIC_PADDING:
            self.PADDING_BPS = DYNAMIC_PADDING_DICT[self.motif_length]
        else:
            self.PADDING_BPS = PADDING_BPS
        self.ref_STR_sequence = pysam_fasta.fetch(
            self.chrom, self.start, self.end
        ).upper()
        # A region can either be specified by reference, start and end. start and end denote 0-based, half-open intervals.
        self.ref_STR_flanking_sequence = pysam_fasta.fetch(
            self.chrom,
            self.start - self.PADDING_BPS,
            self.end + self.PADDING_BPS,
        ).upper()
        self.left_flanking_padding_bp = pysam_fasta.fetch(
            self.chrom, self.start - self.PADDING_BPS, self.start
        ).upper()
        self.right_flanking_padding_bp = pysam_fasta.fetch(
            self.chrom, self.end, self.end + self.PADDING_BPS
        ).upper()
        if GLOBAL_FOR_STR_AND_FLANKING:
            self.ref_allele_length = self.STR_length + 2 * self.PADDING_BPS
            self.ref_allele_seq = self.ref_STR_flanking_sequence
        else:
            self.ref_allele_length = self.STR_length
            self.ref_allele_seq = self.ref_STR_sequence
        if self.is_usable:
            if "N" in self.ref_STR_flanking_sequence:
                self.is_usable = False
                self.unusable_reason = "NinReferenceAllele"
                if _DEBUG is True:
                    sys.stderr.write(
                        "Warning: N in STR flanking sequence in STR locus:"
                        f" {self.STR_id} at"
                        f" {self.chrom}:{self.start}-{self.end} \n"
                    )

    def __str__(self):
        return (
            f"STR locus {self.STR_id} at {self.chrom}:{self.start}-{self.end}"
        )

    def load_STR_locus_info(self, pysam_bed_row):
        self.bed_row = pysam_bed_row
        self.chrom = self.bed_row.split("\t")[0]
        self.start = int(self.bed_row.split("\t")[1])
        self.end = int(self.bed_row.split("\t")[2])
        self.STR_id = self.bed_row.split("\t")[5]
        motif = self.bed_row.split("\t")[6]
        self.motif = motif.upper()
        self.motif_length = int(self.bed_row.split("\t")[3])
        self.period = float(self.bed_row.split("\t")[4])
        self.STR_length = self.end - self.start
        # self.left_flanking_anchor_bp = str(self.bed_row.split("\t")[7]).upper()
        # self.right_flanking_anchor_bp = str(
        #     self.bed_row.split("\t")[8]
        # ).upper()
        # # TODO: if no N bps are in the flanking sequence, then how to do ?
        # # abandom this locus because Telomeres and centriole in reference preparation scripts
        # self.ref_STR_seq = str(self.bed_row.split("\t")[9]).upper()  # TODO: ADD N inference
        # if (
        #     "N" in self.ref_STR_seq
        #     or "N" in self.left_flanking_anchor_bp
        #     or "N" in self.right_flanking_anchor_bp
        # ):
        #     self.is_usable = False
        #     self.unusable_reason = "NinReferenceAllele"
        #     if _DEBUG is True:
        #         sys.stderr.write(
        #             "Warning: N in STR sequence or left or right flanking"
        #             f" anchor bp in STR locus: {self.STR_id} at"
        #             f" {self.chrom}:{self.start}-{self.end} \n"
        #         )
        if "N" in self.motif:
            self.is_usable = False
            self.unusable_reason = "NinMotif"
            if _DEBUG is True:
                sys.stderr.write(
                    f"Warning: N in motif in STR locus: {self.STR_id} at"
                    f" {self.chrom}:{self.start}-{self.end} \n"
                )
        elif "/" in self.motif:
            self.is_usable = False
            self.unusable_reason = "ContinuousSTR"
            if _DEBUG is True:
                sys.stderr.write(
                    f"Warning: / in motif in STR locus: {self.STR_id} at"
                    f" {self.chrom}:{self.start}-{self.end} \n"
                )
        elif self.STR_length > MAX_STR_LENGTH:
            self.is_usable = False
            self.unusable_reason = "STRLengthOutOfRange"
            if _DEBUG is True:
                sys.stderr.write(
                    f"Warning: STR length longer than {MAX_STR_LENGTH}:"
                    f" {self.STR_id} at"
                    f" {self.chrom}:{self.start}-{self.end} \n"
                )
        else:
            self.is_usable = True
            self.unusable_reason = None


class CoordinateSegRead:
    """Segment single read into flanking segment and STR segment."""

    # Two methods: (1) STR global (2) STR and Flanking
    def __init__(self, coordinatesegger_locus, read):
        self.coordinatesegger_locus = coordinatesegger_locus
        self.pysam_read = read
        self.reference_start = read.reference_start
        self.reference_end = read.reference_end
        (
            self.STR_flanking_seq,
            self.STR_flanking_length,
            self.STR_flanking_error_rate,
            self.STR_seq,
            self.STR_length,
            self.STR_error_rate,
            self.left_flanking_seq,
            self.left_flanking_length,
            self.left_flanking_error_rate,
            self.right_flanking_seq,
            self.right_flanking_length,
            self.right_flanking_error_rate,
        ) = self.extract_segmentation_seq(read)
        # if (
        #     self.STR_flanking_seq == f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}"
        # ):  # 不可能发生
        #     self.is_usable = False
        #     self.unusable_reason = "SegmentConditionFail"
        #     if _DEBUG is True:
        #         sys.stderr.write(
        #             "Warning: InDels in anchor bigger than"
        #             f" {MAX_ANCHOR_INDEL_BP} in read: {read.query_name} \n"
        #         )
        if GLOBAL_FOR_STR_AND_FLANKING:
            if self.STR_flanking_length is None or self.STR_flanking_length <= 0:  # TODO: DEBUG 可能等于零，因为存在大片段的 deletions, 目前如果存在大片段的 deletion 导致 STR allele 检测不到，这样的 reads 会直接被我丢弃掉，Don't debug now
                self.is_usable = False
                self.unusable_reason = "SegmentResultFail"
                if _DEBUG is True:
                    sys.stderr.write(
                        "Warning: No enough hap sequence in read: %s"
                        % read.query_name
                    )
            else:
                if DONT_ALLOW_N:
                    if (
                        "N" in self.STR_seq
                        or "N" in self.left_flanking_seq
                        or "N" in self.right_flanking_seq
                    ):  # TODO TO DEBUG 其实使用包含 N 的 reads 也可以，可以使用在以后的版本，但是目前这是一个小细节，不是那么重要哈
                        self.is_usable = False
                        self.unusable_reason = "ReadN"
                        if _DEBUG is True:
                            sys.stderr.write(
                                "Warning: N in STR sequence or left or right flanking"
                                f" sequence in read: {read.query_name} \n"
                            )  # XXX: Filter out reads with a N base in Genotyping regions or the whole read sequence
                    else:
                        self.is_usable = True
                        self.unusable_reason = None
                        if _DEBUG is True:
                            print(
                                "STR sequence of read %s is: %s"
                                % (read.query_name, self.STR_seq)
                            )
                            print(self.STR_error_rate)
                            print(self.STR_length)
                            print(self.left_flanking_seq)
                            print(self.left_flanking_error_rate)
                            print(self.right_flanking_seq)
                            print(self.right_flanking_error_rate)
                        pass
                else:
                    self.is_usable = True
                    self.unusable_reason = None
                    if _DEBUG is True:
                        print(
                            "STR sequence of read %s is: %s"
                            % (read.query_name, self.STR_seq)
                        )
                        print(self.STR_error_rate)
                        print(self.STR_length)
                        print(self.left_flanking_seq)
                        print(self.left_flanking_error_rate)
                        print(self.right_flanking_seq)
                        print(self.right_flanking_error_rate)
                    pass
        else:
            if self.STR_length is None or self.STR_length <= 0:
                self.is_usable = False
                self.unusable_reason = "SegmentResultFail"
                if _DEBUG is True:
                    sys.stderr.write(
                        "Warning: No enough hap sequence in read: %s"
                        % read.query_name
                    )
            else:
                if DONT_ALLOW_N:
                    if (
                        "N" in self.STR_seq
                        or "N" in self.left_flanking_seq
                        or "N" in self.right_flanking_seq
                    ):
                        self.is_usable = False
                        self.unusable_reason = "ReadN"
                        if _DEBUG is True:
                            sys.stderr.write(
                                "Warning: N in STR sequence or left or right flanking"
                                f" sequence in read: {read.query_name} \n"
                            )  # XXX: Filter out reads with a N base in Genotyping regions or the whole read sequence
                    else:
                        self.is_usable = True
                        self.unusable_reason = None
                        if _DEBUG is True:
                            print(
                                "STR sequence of read %s is: %s"
                                % (read.query_name, self.STR_seq)
                            )
                            print(self.STR_error_rate)
                            print(self.STR_length)
                            print(self.left_flanking_seq)
                            print(self.left_flanking_error_rate)
                            print(self.right_flanking_seq)
                            print(self.right_flanking_error_rate)
                        pass
                else:
                    self.is_usable = True
                    self.unusable_reason = None
                    if _DEBUG is True:
                        print(
                            "STR sequence of read %s is: %s"
                            % (read.query_name, self.STR_seq)
                        )
                        print(self.STR_error_rate)
                        print(self.STR_length)
                        print(self.left_flanking_seq)
                        print(self.left_flanking_error_rate)
                        print(self.right_flanking_seq)
                        print(self.right_flanking_error_rate)
                    pass
        if self.is_usable:
            if GLOBAL_FOR_STR_AND_FLANKING:
                self.read_allele_length = self.STR_flanking_length
                self.read_allele_seq = self.STR_flanking_seq
                self.read_allele_accuracy = 1 - self.STR_flanking_error_rate
            else:
                self.read_allele_length = self.STR_length
                self.read_allele_seq = self.STR_seq
                self.read_allele_accuracy = 1 - self.STR_error_rate
        # if (
        #     self.left_seg_seq
        #     == f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}"
        # ) or (self.left_seg_seq == "No enough length for segmentation"):
        #     self.is_usable = False
        #     self.unusable_reason = "SegmentConditionFail"
        #     if _DEBUG is True:
        #         sys.stderr.write(
        #             "Warning: InDels in anchor bigger than"
        #             f" {MAX_ANCHOR_INDEL_BP} or No enough length for"
        #             " segmentation or N in left or right seg seq in read:"
        #             " %s \n"
        #             % read.query_name
        #         )
        # elif ("N" in self.left_seg_seq) or ("N" in self.right_seg_seq):
        #     self.is_usable = False
        #     self.unusable_reason = "ReadN"
        # else:
        # self.segmentation_by_hmm_per_read()
        # if self.STR_length == "STR length out of range":
        #     self.is_usable = False
        #     self.unusable_reason = "SegmentResultFail"
        #     if _DEBUG is True:
        #         sys.stderr.write(
        #             "Warning: STR length out of range in read: %s"
        #             % read.query_name
        #         )
        # elif self.STR_length == "No enough flanking sequence":  # XXX: temp fix for the missing flanking sequence
        #     self.is_usable = False
        #     self.unusable_reason = "SegmentResultFail"
        #     if _DEBUG is True:
        #         sys.stderr.write(
        #             "Warning: No enough flanking sequence in read: %s"
        #             % read.query_name
        #         )
        # else:
        #     self.is_usable = True
        #     self.unusable_reason = None
        #     if _DEBUG is True:
        #         print(
        #             "STR sequence of read %s is: %s"
        #             % (read.query_name, self.STR_seq)
        #         )
        #     pass

    def __str__(self):
        return (
            f"Segmented read {self.pysam_read.query_name} at"
            f" {self.reference_start}-{self.reference_end}"
        )

    def extract_segmentation_seq(self, read):
        # How to process hard-clipping and soft-clipping sequencing reads
        aligned_pairs = read.get_aligned_pairs(
            with_seq=True, matches_only=True
        )
        self.refpos_queryindex = {i[1]: [i[0], i[2]] for i in aligned_pairs}
        self.read_seq = read.query_alignment_sequence.upper()
        read_baseq = read.query_alignment_qualities
        read_baseq_error_rate = _baseq_2_error_rate(read_baseq)
        if read_baseq_error_rate.min() == 0:
            read_baseq_error_rate[
                read_baseq_error_rate == 0
            ] = read_baseq_error_rate[read_baseq_error_rate.nonzero()].min()
        if read_baseq_error_rate.max() == 1:
            read_baseq_error_rate[
                read_baseq_error_rate == 1
            ] = read_baseq_error_rate[read_baseq_error_rate != 1].max()
        self.read_baseq_accuray = 1 - read_baseq_error_rate
        self.alignment_start_index = read.query_alignment_start
        self.alignment_end_index = read.query_alignment_end
        left_flanking_last_base_pos = self.coordinatesegger_locus.start - 1
        k = 0
        while True:
            if self.refpos_queryindex.get(left_flanking_last_base_pos) is None:
                left_flanking_last_base_pos -= 1
                k += 1
            else:
                flanking_left_segmentation_site_index = (
                    self.refpos_queryindex.get(left_flanking_last_base_pos)[0]
                )
                break
            if ((k >= MAX_ANCHOR_INDEL_BP) and TGS):
                return (
                    f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                )
        right_flanking_first_base_pos = self.coordinatesegger_locus.end
        k = 0
        while True:
            if (
                self.refpos_queryindex.get(right_flanking_first_base_pos)
                is None
            ):
                right_flanking_first_base_pos += 1
                k += 1
            else:
                flanking_right_segmentation_site_index = (
                    self.refpos_queryindex.get(right_flanking_first_base_pos)[
                        0
                    ]
                )
                break
            if ((k >= MAX_ANCHOR_INDEL_BP) and TGS):
                return (
                    f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                )

        left_padding_first_base_pos = (
            self.coordinatesegger_locus.start
            - 1
            - self.coordinatesegger_locus.PADDING_BPS
            + 1
        )
        k = 0
        while True:
            if self.refpos_queryindex.get(left_padding_first_base_pos) is None:
                left_padding_first_base_pos += 1
                k += 1
            else:
                left_padding_first_base_pos_index = self.refpos_queryindex.get(
                    left_padding_first_base_pos
                )[0]
                break
            if ((k >= MAX_ANCHOR_INDEL_BP) and TGS):
                return (
                    f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                )
        right_padding_last_base_pos = (
            self.coordinatesegger_locus.end
            + self.coordinatesegger_locus.PADDING_BPS
            - 1
        )
        k = 0
        while True:
            if self.refpos_queryindex.get(right_padding_last_base_pos) is None:
                right_padding_last_base_pos -= 1
                k += 1
            else:
                right_padding_last_base_pos_index = self.refpos_queryindex.get(
                    right_padding_last_base_pos
                )[0]
                break
            if ((k >= MAX_ANCHOR_INDEL_BP) and TGS):
                return (
                    f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                )

        # self.read_spanning = (
        #     flanking_left_segmentation_site_index
        #     - self.alignment_start_index
        #     + 1
        #     >= MIN_FLANKING_SPANING_BP
        # ) and (
        #     self.alignment_end_index
        #     - flanking_right_segmentation_site_index
        #     + 1
        # ) >= MIN_FLANKING_SPANING_BP
        # self.enoughSTR = (
        #     STR_right_segmentation_site_index
        #     - STR_left_segmentation_site_index
        #     + 1
        # ) >= self.hmmsegger_locus.L_step
        # if self.read_spanning and self.enoughSTR:
        # base pairs number level based read spanning and enough STR
        # segmented sequence include flanking MIN_FLANKING_SPANING_BP base pairs(base pairs number level and not coordinates level)
        # L_Step STR base pairs (base pairs number level and not coordinates level)
        # And inserted base pairs between STR and Flanking
        STR_flanking_seq = self.read_seq[
            left_padding_first_base_pos_index
            - self.alignment_start_index : right_padding_last_base_pos_index
            - self.alignment_start_index
            + 1
        ]
        STR_flanking_length = len(STR_flanking_seq)
        STR_flanking_error_rate = read_baseq_error_rate[
            left_padding_first_base_pos_index
            - self.alignment_start_index : right_padding_last_base_pos_index
            - self.alignment_start_index
            + 1
        ]
        STR_seq = self.read_seq[
            flanking_left_segmentation_site_index
            - self.alignment_start_index
            + 1 : flanking_right_segmentation_site_index
            - self.alignment_start_index
        ]
        STR_length = len(STR_seq)
        STR_error_rate = read_baseq_error_rate[
            flanking_left_segmentation_site_index
            - self.alignment_start_index
            + 1 : flanking_right_segmentation_site_index
            - self.alignment_start_index
        ]
        left_flanking_seq = self.read_seq[
            left_padding_first_base_pos_index
            - self.alignment_start_index : flanking_left_segmentation_site_index
            - self.alignment_start_index
            + 1
        ]
        left_flanking_length = len(left_flanking_seq)
        left_flanking_error_rate = read_baseq_error_rate[
            left_padding_first_base_pos_index
            - self.alignment_start_index : flanking_left_segmentation_site_index
            - self.alignment_start_index
            + 1
        ]
        right_flanking_seq = self.read_seq[
            flanking_right_segmentation_site_index
            - self.alignment_start_index : right_padding_last_base_pos_index
            - self.alignment_start_index
            + 1
        ]
        right_flanking_length = len(right_flanking_seq)
        right_flanking_error_rate = read_baseq_error_rate[
            flanking_right_segmentation_site_index
            - self.alignment_start_index : right_padding_last_base_pos_index
            - self.alignment_start_index
            + 1
        ]
        self.all_left_flanking_baseq = read_baseq[
             0 : left_padding_first_base_pos_index
            - self.alignment_start_index
        ]
        self.all_right_flanking_baseq = read_baseq[
            right_padding_last_base_pos_index
            - self.alignment_start_index
            + 1 : 
        ]
        return (
            STR_flanking_seq,
            STR_flanking_length,
            STR_flanking_error_rate,
            STR_seq,
            STR_length,
            STR_error_rate,
            left_flanking_seq,
            left_flanking_length,
            left_flanking_error_rate,
            right_flanking_seq,
            right_flanking_length,
            right_flanking_error_rate
        )

        # left_seg_seq = self.read_seq[
        #     flanking_left_segmentation_site_index
        #     - self.alignment_start_index
        #     - MIN_FLANKING_SPANING_BP
        #     + 1 : STR_left_segmentation_site_index
        #     - self.alignment_start_index
        #     + self.hmmsegger_locus.L_step
        # ]
        # right_seg_seq = self.read_seq[
        #     STR_right_segmentation_site_index
        #     - self.alignment_start_index
        #     - self.hmmsegger_locus.L_step
        #     + 1 : flanking_right_segmentation_site_index
        #     - self.alignment_start_index
        #     + MIN_FLANKING_SPANING_BP
        # ]
        # left_seg_baseq_error_rate = read_baseq_error_rate[
        #     flanking_left_segmentation_site_index
        #     - self.alignment_start_index
        #     - MIN_FLANKING_SPANING_BP
        #     + 1 : STR_left_segmentation_site_index
        #     - self.alignment_start_index
        #     + self.hmmsegger_locus.L_step
        # ]
        # right_seg_baseq_error_rate = read_baseq_error_rate[
        #     STR_right_segmentation_site_index
        #     - self.alignment_start_index
        #     - self.hmmsegger_locus.L_step
        #     + 1 : flanking_right_segmentation_site_index
        #     - self.alignment_start_index
        #     + MIN_FLANKING_SPANING_BP
        # ]
        # left_seq_seg_start_index_include_this_base = (
        #     STR_left_segmentation_site_index
        #     - self.alignment_start_index
        #     + self.hmmsegger_locus.L_step
        #     - 1
        # )
        # right_seq_seg_start_index_include_this_base = (
        #     STR_right_segmentation_site_index
        #     - self.alignment_start_index
        #     - self.hmmsegger_locus.L_step
        #     + 1
        # )
        # return (
        #     left_seg_seq,
        #     right_seg_seq,
        #     left_seg_baseq_error_rate,
        #     right_seg_baseq_error_rate,
        #     left_seq_seg_start_index_include_this_base,
        #     right_seq_seg_start_index_include_this_base,
        # )
        # else:
        #     return (
        #         "No enough length for segmentation",
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #     )


class HMMSeggerLocus:
    """Load STR locus information from bed file and initialize HMM Segger."""

    def __init__(self, hmm_segger_init, pysam_bed_row, pysam_fasta=None):
        self.hmm_segger_init = hmm_segger_init
        self.pysam_bed_row = pysam_bed_row
        self.load_STR_locus_info(pysam_bed_row)
        self.load_str_locus = True
        if DYNAMIC_PADDING:
            self.PADDING_BPS = DYNAMIC_PADDING_DICT[self.motif_length]
        else:
            self.PADDING_BPS = PADDING_BPS
        self.ref_STR_sequence = pysam_fasta.fetch(
            self.chrom, self.start, self.end
        ).upper()
        # A region can either be specified by reference, start and end. start and end denote 0-based, half-open intervals.
        self.ref_STR_flanking_sequence = pysam_fasta.fetch(
            self.chrom,
            self.start - self.PADDING_BPS,
            self.end + self.PADDING_BPS,
        ).upper()
        self.left_flanking_padding_bp = pysam_fasta.fetch(
            self.chrom, self.start - self.PADDING_BPS, self.start
        ).upper()
        self.right_flanking_padding_bp = pysam_fasta.fetch(
            self.chrom, self.end, self.end + self.PADDING_BPS
        ).upper()
        if GLOBAL_FOR_STR_AND_FLANKING:
            self.ref_allele_length = self.STR_length + 2 * self.PADDING_BPS
            self.ref_allele_seq = self.ref_STR_flanking_sequence
        else:
            self.ref_allele_length = self.STR_length
            self.ref_allele_seq = self.ref_STR_sequence
        if self.is_usable:
            if "N" in self.ref_STR_flanking_sequence:
                self.is_usable = False
                self.unusable_reason = "NinReferenceAllele"
                if _DEBUG is True:
                    sys.stderr.write(
                        "Warning: N in STR flanking sequence in STR locus:"
                        f" {self.STR_id} at"
                        f" {self.chrom}:{self.start}-{self.end} \n"
                    )

    def __str__(self):
        return (
            f"STR locus {self.STR_id} at {self.chrom}:{self.start}-{self.end}"
        )

    def load_STR_locus_info(self, pysam_bed_row):
        self.bed_row = pysam_bed_row
        self.chrom = self.bed_row.split("\t")[0]
        self.start = int(self.bed_row.split("\t")[1])
        self.end = int(self.bed_row.split("\t")[2])
        self.STR_id = self.bed_row.split("\t")[5]
        motif = self.bed_row.split("\t")[6]
        self.motif = motif.upper()
        self.motif_length = int(self.bed_row.split("\t")[3])
        self.period = float(self.bed_row.split("\t")[4])
        self.STR_length = self.end - self.start
        self.left_flanking_anchor_bp = str(self.bed_row.split("\t")[7]).upper()
        self.right_flanking_anchor_bp = str(
            self.bed_row.split("\t")[8]
        ).upper()
        # TODO: if no N bps are in the flanking sequence, then how to do ?
        # abandom this locus because Telomeres and centriole in reference preparation scripts
        self.ref_STR_seq = str(self.bed_row.split("\t")[9]).upper()
        if (
            "N" in self.ref_STR_seq
            or "N" in self.left_flanking_anchor_bp
            or "N" in self.right_flanking_anchor_bp
        ):
            self.is_usable = False
            self.unusable_reason = "NinReferenceAllele"
            if _DEBUG is True:
                sys.stderr.write(
                    "Warning: N in STR sequence or left or right flanking"
                    f" anchor bp in STR locus: {self.STR_id} at"
                    f" {self.chrom}:{self.start}-{self.end} \n"
                )
        elif "N" in self.motif:
            self.is_usable = False
            self.unusable_reason = "NinMotif"
            if _DEBUG is True:
                sys.stderr.write(
                    f"Warning: N in motif in STR locus: {self.STR_id} at"
                    f" {self.chrom}:{self.start}-{self.end} \n"
                )
        elif "/" in self.motif:
            self.is_usable = False
            self.unusable_reason = "ContinuousSTR"
            if _DEBUG is True:
                sys.stderr.write(
                    f"Warning: / in motif in STR locus: {self.STR_id} at"
                    f" {self.chrom}:{self.start}-{self.end} \n"
                )
        elif self.STR_length > MAX_STR_LENGTH:
            self.is_usable = False
            self.unusable_reason = "STRLengthOutOfRange"
            if _DEBUG is True:
                sys.stderr.write(
                    f"Warning: STR length longer than {MAX_STR_LENGTH}:"
                    f" {self.STR_id} at"
                    f" {self.chrom}:{self.start}-{self.end} \n"
                )
        else:
            self.is_usable = True
            self.unusable_reason = None
        self.reverse_motif = self.motif[::-1]
        self.reverse_left_flanking_anchor_bp = self.left_flanking_anchor_bp[
            ::-1
        ]
        self.trans_mat = self.hmm_segger_init.trans_matrix_dict[
            self.motif_length
        ]
        self.init_mat = self.hmm_segger_init.start_matrix_dict[
            self.motif_length
        ]
        self.hidden_states = self.hmm_segger_init.hidden_states_dict[
            self.motif_length
        ]
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
            self.left_seg_seq
            == f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}"
        ) or (self.left_seg_seq == "No enough length for segmentation"):
            self.is_usable = False
            self.unusable_reason = "SegmentConditionFail"
            if _DEBUG is True:
                sys.stderr.write(
                    "Warning: InDels in anchor bigger than"
                    f" {MAX_ANCHOR_INDEL_BP} or No enough length for"
                    " segmentation or N in left or right seg seq in read:"
                    " %s \n"
                    % read.query_name
                )
        elif ("N" in self.left_seg_seq) or ("N" in self.right_seg_seq):
            self.is_usable = False
            self.unusable_reason = "ReadN"
        else:
            self.segmentation_by_hmm_per_read()
            if self.STR_length == "STR length out of range":
                self.is_usable = False
                self.unusable_reason = "SegmentResultFail"
                if _DEBUG is True:
                    sys.stderr.write(
                        "Warning: STR length out of range in read: %s"
                        % read.query_name
                    )
            elif (
                self.STR_length == "No enough flanking sequence"
            ):  # XXX: temp fix for the missing flanking sequence
                self.is_usable = False
                self.unusable_reason = "SegmentResultFail"
                if _DEBUG is True:
                    sys.stderr.write(
                        "Warning: No enough flanking sequence in read: %s"
                        % read.query_name
                    )
            else:
                self.is_usable = True
                self.unusable_reason = None
                if _DEBUG is True:
                    print(
                        "STR sequence of read %s is: %s"
                        % (read.query_name, self.STR_seq)
                    )
                pass

    def __str__(self):
        return (
            f"Segmented read {self.pysam_read.query_name} at"
            f" {self.reference_start}-{self.reference_end}"
        )

    def extract_segmentation_seq(self, read):
        # How to process hard-clipping and soft-clipping sequencing reads
        aligned_pairs = read.get_aligned_pairs(
            with_seq=True, matches_only=True
        )
        self.refpos_queryindex = {i[1]: [i[0], i[2]] for i in aligned_pairs}
        self.read_seq = read.query_alignment_sequence.upper()
        read_baseq = read.query_alignment_qualities
        read_baseq_error_rate = _baseq_2_error_rate(read_baseq)
        if read_baseq_error_rate.min() == 0:
            read_baseq_error_rate[
                read_baseq_error_rate == 0
            ] = read_baseq_error_rate[read_baseq_error_rate.nonzero()].min()
        if read_baseq_error_rate.max() == 1:
            read_baseq_error_rate[
                read_baseq_error_rate == 1
            ] = read_baseq_error_rate[read_baseq_error_rate != 1].max()
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
                flanking_left_segmentation_site_index = (
                    self.refpos_queryindex.get(left_flanking_last_base_pos)[0]
                )
                break
            if k >= MAX_ANCHOR_INDEL_BP:
                return (
                    f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
        right_flanking_first_base_pos = self.hmmsegger_locus.end
        k = 0
        while True:
            if (
                self.refpos_queryindex.get(right_flanking_first_base_pos)
                is None
            ):
                right_flanking_first_base_pos += 1
                k += 1
            else:
                flanking_right_segmentation_site_index = (
                    self.refpos_queryindex.get(right_flanking_first_base_pos)[
                        0
                    ]
                )
                break
            if k >= MAX_ANCHOR_INDEL_BP:
                return (
                    f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}",
                    None,
                    None,
                    None,
                    None,
                    None,
                )

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
            if k >= MAX_ANCHOR_INDEL_BP:
                return (
                    f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
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
            if k >= MAX_ANCHOR_INDEL_BP:
                return (
                    f"InDels in anchor bigger than {MAX_ANCHOR_INDEL_BP}",
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        self.read_spanning = (
            flanking_left_segmentation_site_index
            - self.alignment_start_index
            + 1
            >= MIN_FLANKING_SPANING_BP
        ) and (
            self.alignment_end_index
            - flanking_right_segmentation_site_index
            + 1
        ) >= MIN_FLANKING_SPANING_BP
        self.enoughSTR = (
            STR_right_segmentation_site_index
            - STR_left_segmentation_site_index
            + 1
        ) >= self.hmmsegger_locus.L_step

        if self.read_spanning and self.enoughSTR:
            # base pairs number level based read spanning and enough STR
            # segmented sequence include flanking MIN_FLANKING_SPANING_BP base pairs(base pairs number level and not coordinates level)
            # L_Step STR base pairs (base pairs number level and not coordinates level)
            # And inserted base pairs between STR and Flanking
            left_seg_seq = self.read_seq[
                flanking_left_segmentation_site_index
                - self.alignment_start_index
                - MIN_FLANKING_SPANING_BP
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
                + MIN_FLANKING_SPANING_BP
            ]
            left_seg_baseq_error_rate = read_baseq_error_rate[
                flanking_left_segmentation_site_index
                - self.alignment_start_index
                - MIN_FLANKING_SPANING_BP
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
                + MIN_FLANKING_SPANING_BP
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
            return (
                "No enough length for segmentation",
                None,
                None,
                None,
                None,
                None,
            )

    def segmentation_seq(self, seq, path):
        anchor_FLK_first_base_0_based = "NoF"
        for i in range(1, self.hmmsegger_init.flanking_length + 1):
            f_state = "F" + str(i)
            if f_state in path:
                anchor_FLK_first_base_0_based = path.index(f_state)
                break
        if anchor_FLK_first_base_0_based == "NoF":
            if "F" in path:
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
            self.hmmsegger_locus.reverse_left_flanking_anchor_bp,
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
            self.hmmsegger_locus.right_flanking_anchor_bp,
            self.hmmsegger_locus.state_map,
        )
        right_anchor_FLK_first_base_0_based = self.segmentation_seq(
            right_seg_seq, self.right_final_best_path
        )

        if (
            left_anchor_FLK_first_base_0_based == len(reverse_left_seg_seq)
        ) or (right_anchor_FLK_first_base_0_based == len(right_seg_seq)):
            # May exist repeat expansion (insertions located in flanking is more than MIN_FLANKING_SPANING_BP)
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
            if (
                right_STR_last_base_location + 1
                <= left_STR_first_base_location
            ):
                # There may be very short STRs and odd interruptions,
                # resulting in segmentation where the right site is on the left of the left site
                self.STR_length = "STR length out of range"
            else:
                if (
                    left_STR_first_base_location - FLANKING_ALLELE_BASE_PAIRS
                ) < 0 or (
                    (
                        right_STR_last_base_location
                        + 1
                        + FLANKING_ALLELE_BASE_PAIRS
                    )
                    > len(self.read_seq)
                ):  # XXX: temp fix for the missing flanking sequence
                    self.STR_length = "No enough flanking sequence"
                else:
                    self.STR_first_base_location = left_STR_first_base_location
                    self.STR_last_base_location = right_STR_last_base_location
                    self.STR_seq = self.read_seq[
                        left_STR_first_base_location : right_STR_last_base_location
                        + 1
                    ]
                    self.STR_baseq_accuray = self.read_baseq_accuray[
                        left_STR_first_base_location : right_STR_last_base_location
                        + 1
                    ]
                    self.STR_length = len(self.STR_seq)
                    self.left_flanking_allele_bp = self.read_seq[
                        left_STR_first_base_location
                        - FLANKING_ALLELE_BASE_PAIRS : left_STR_first_base_location
                    ]
                    self.right_flanking_allele_bp = self.read_seq[
                        right_STR_last_base_location
                        + 1 : right_STR_last_base_location
                        + 1
                        + FLANKING_ALLELE_BASE_PAIRS
                    ]
                    self.left_flanking_allele_bp_baseq_accuray = self.read_baseq_accuray[
                        left_STR_first_base_location
                        - FLANKING_ALLELE_BASE_PAIRS : left_STR_first_base_location
                    ]
                    self.right_flanking_allele_bp_baseq_accuray = (
                        self.read_baseq_accuray[
                            right_STR_last_base_location
                            + 1 : right_STR_last_base_location
                            + 1
                            + FLANKING_ALLELE_BASE_PAIRS
                        ]
                    )
                if _DEBUG:  # revise_padding
                    print(self.STR_baseq_accuray)
                    print(
                        self.read_seq[
                            left_STR_first_base_location
                            - PADDING_BPS : left_STR_first_base_location
                        ]
                    )
                    print(
                        self.read_seq[
                            right_STR_last_base_location
                            + 1 : right_STR_last_base_location
                            + 1
                            + PADDING_BPS
                        ]
                    )
                    print(
                        self.read_baseq_accuray[
                            left_STR_first_base_location
                            - PADDING_BPS : left_STR_first_base_location
                        ]
                    )
                    print(
                        self.read_baseq_accuray[
                            right_STR_last_base_location
                            + 1 : right_STR_last_base_location
                            + 1
                            + PADDING_BPS
                        ]
                    )
                    if (
                        self.read_seq[
                            left_STR_first_base_location
                            - PADDING_BPS : left_STR_first_base_location
                        ]
                        == ""
                    ):
                        print(self.read_seq)


# ----------------------------- HMM Class End ----------------------------- #

# hmm_seg_init = HMMSeggerInit(
#     ALL_MOTIF_PARAMS,
#     L_STEPS_DICT,
#     BASE_INDEX_DICT,
#     DEFAULT_PROBABILITY,
#     HMM_FLANKING_LENGTH,
# )

## TODO
## segmentation methods 2, based on coordinate and flanking indel ##
## hap alignment methods from DINDEL ##
