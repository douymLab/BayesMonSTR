import numpy as np

from ..utils import FLK_BP, FLK_BP_SEG, tools

All_motif_params = {}

params_1 = {}
params_1["mismatch_rate"] = 0.005
params_1["error_rate"] = 0.001
params_1["mis_2_mis_rate"] = 0.005
All_motif_params[1] = params_1

params_2 = {}
params_2["mismatch_rate"] = 0.01
params_2["insertion_rate"] = 0.009
params_2["deletion_rate"] = 0.009
params_2["error_rate"] = 0.001
params_2["ins_2_ins_rate"] = 0.009
params_2["ins_2_mis_rate"] = 0.01
params_2["mis_2_ins_rate"] = 0.009
params_2["mis_2_mis_rate"] = 0.01
All_motif_params[2] = params_2

params_3 = {}
params_3["mismatch_rate"] = 0.0090
params_3["insertion_rate"] = 0.0080
params_3["deletion_rate"] = 0.0080
params_3["out_frame_rate"] = 0.000001
params_3["continous_deletion_rate"] = 0.000001
params_3["error_rate"] = 0.001
params_3["mis_2_out_frame_rate"] = 0.000001
params_3["ins_2_out_frame_rate"] = 0.000001
params_3["ins_2_ins_rate"] = 0.0080
params_3["ins_2_mis_rate"] = 0.0090
params_3["mis_2_ins_rate"] = 0.0080
params_3["mis_2_mis_rate"] = 0.0090
All_motif_params[3] = params_3

params_4 = {}
params_4["mismatch_rate"] = 0.006
params_4["insertion_rate"] = 0.007
params_4["deletion_rate"] = 0.007
params_4["out_frame_rate"] = 0.000001
params_4["continous_deletion_rate"] = 0.000001
params_4["error_rate"] = 0.001
params_4["mis_2_out_frame_rate"] = 0.000001
params_4["ins_2_out_frame_rate"] = 0.000001
params_4["ins_2_ins_rate"] = 0.007
params_4["ins_2_mis_rate"] = 0.006
params_4["mis_2_ins_rate"] = 0.007
params_4["mis_2_mis_rate"] = 0.006
All_motif_params[4] = params_4

params_5 = {}
params_5["mismatch_rate"] = 0.0088
params_5["insertion_rate"] = 0.009
params_5["deletion_rate"] = 0.006
params_5["out_frame_rate"] = 0.000001
params_5["continous_deletion_rate"] = 0.000001
params_5["error_rate"] = 0.001
params_5["mis_2_out_frame_rate"] = 0.000001
params_5["ins_2_out_frame_rate"] = 0.000001
params_5["ins_2_ins_rate"] = 0.009
params_5["ins_2_mis_rate"] = 0.0088
params_5["mis_2_ins_rate"] = 0.009
params_5["mis_2_mis_rate"] = 0.0088
All_motif_params[5] = params_5

params_6 = {}
params_6["mismatch_rate"] = 0.0048
params_6["insertion_rate"] = 0.007
params_6["deletion_rate"] = 0.006
params_6["out_frame_rate"] = 0.000001
params_6["continous_deletion_rate"] = 1.000000e-07
params_6["error_rate"] = 0.001
params_6["mis_2_out_frame_rate"] = 0.000001
params_6["ins_2_out_frame_rate"] = 0.000001
params_6["ins_2_ins_rate"] = 0.007
params_6["ins_2_mis_rate"] = 0.0048
params_6["mis_2_ins_rate"] = 0.007
params_6["mis_2_mis_rate"] = 0.0048
All_motif_params[6] = params_6

L_steps_dict = {1: 5, 2: 6, 3: 6, 4: 8, 5: 10, 6: 12}
base_index_dict = {"A": 0, "T": 1, "C": 2, "G": 3}
default_probability = 1e-8


def make_hidden_states(motif_length):
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


def make_hidden_states_dict():
    hidden_states_dict = {}
    for i in range(1, 7):
        hidden_states_dict[i] = make_hidden_states(i)
    return hidden_states_dict


def make_start_probability_matrix(
    hidden_states, motif_length, default_probability, **params
):
    states_num = len(hidden_states)
    start_matrix = np.full((states_num, 1), default_probability)
    mismatch_rate = params["mismatch_rate"]
    if motif_length == 1:
        start_matrix[1, :] = mismatch_rate
        start_matrix[0, :] = 1 - mismatch_rate - start_matrix[2] * 4
    else:
        for i in range(motif_length * 2, motif_length * 3):  # for mis
            start_matrix[i, :] = mismatch_rate
        if states_num == motif_length * 3 + 4:
            for i in range(motif_length):  # for STR
                start_matrix[i, :] = (
                    1
                    - motif_length * mismatch_rate
                    - default_probability * (motif_length + 4)
                ) / motif_length
        elif states_num == motif_length * 4 + 4:
            for i in range(motif_length):  # for STR
                start_matrix[i, :] = (
                    1
                    - motif_length * mismatch_rate
                    - default_probability * (motif_length * 2 + 4)
                ) / motif_length
    return np.squeeze(start_matrix)


def make_start_matrix_dict(hidden_states_dict, default_probability, params):
    start_matrix_dict = {}
    for i in range(1, 7):
        start_matrix_dict[i] = make_start_probability_matrix(
            hidden_states_dict[i], i, default_probability, **(params[i])
        )
    return start_matrix_dict


def make_mononucleotide_trans_matrix(
    mononucleotide_hidden_states, to_flanking_L_step, default_probability, **params
):
    trans_matrix = np.full(
        (len(mononucleotide_hidden_states), len(mononucleotide_hidden_states)),
        default_probability,
    )
    trans_matrix[0, 2] = 1 / to_flanking_L_step  # flanking F1
    trans_matrix[0, 5] = 1 / (to_flanking_L_step + 3)  # flanking F
    trans_matrix[0, 1] = params["mismatch_rate"]  # mismatch
    trans_matrix[0, 0] = 1 - sum(trans_matrix[0, :]) + default_probability

    trans_matrix[1, 1] = params["mis_2_mis_rate"]
    trans_matrix[1, 5] = (
        default_probability  # Here set only STR can transfer to flanking hidden state
    )
    trans_matrix[1, 0] = 1 - sum(trans_matrix[1, :]) + default_probability

    trans_matrix[2, 5] = 1 / 3  # flanking F
    trans_matrix[2, 3] = (
        1 - sum(trans_matrix[2, :]) + default_probability
    )  # flanking F2

    trans_matrix[3, 5] = 1 / 3  # flanking F
    trans_matrix[3, 4] = (
        1 - sum(trans_matrix[3, :]) + default_probability
    )  # flanking F3

    trans_matrix[4, 5] = 1 - sum(trans_matrix[4, :]) + default_probability  # flanking F

    trans_matrix[5, 5] = 1 - sum(trans_matrix[5, :]) + default_probability  # flanking F

    return trans_matrix.T


def make_dinucleotide_trans_matrix(
    dinucleotide_hidden_states,
    motif_length,
    to_flanking_L_step,
    default_probability,
    **params,
):
    trans_matrix = np.full(
        (len(dinucleotide_hidden_states), len(dinucleotide_hidden_states)),
        default_probability,
    )

    for i in range(motif_length):  # for STR
        trans_matrix[i, 3 * motif_length] = 1 / to_flanking_L_step  # flanking F1
        trans_matrix[i, 3 * motif_length + 3] = 1 / (
            to_flanking_L_step + 3
        )  # flanking F
        trans_matrix[i, i] = params["deletion_rate"]  # del
        trans_matrix[i, i + motif_length] = params["insertion_rate"]  # ins
        if i != (motif_length - 1):
            trans_matrix[i, i + 2 * motif_length + 1] = params["mismatch_rate"]  # mis
            trans_matrix[i, i + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR
        else:
            trans_matrix[i, 2 * motif_length] = params["mismatch_rate"]  # mis
            trans_matrix[i, 0] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR

    for i in range(motif_length, motif_length * 2):  # for ins
        # trans_matrix[i,3*motif_length]=1/to_flanking_L_step # flanking, Don't permit ins to flanking
        trans_matrix[i, i] = params["ins_2_ins_rate"]  # ins
        if i != (2 * motif_length - 1):
            trans_matrix[i, i + motif_length + 1] = params["ins_2_mis_rate"]  # mis
            trans_matrix[i, i - motif_length + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR
        else:
            trans_matrix[i, 2 * motif_length] = params["ins_2_mis_rate"]  # mis
            trans_matrix[i, 0] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR

    for i in range(motif_length * 2, motif_length * 3):  # for mis
        # trans_matrix[i,3*motif_length]=1/to_flanking_L_step # flanking, Don't permit mis to flanking
        trans_matrix[i, i - motif_length] = params["mis_2_ins_rate"]  # ins
        if i != (3 * motif_length - 1):
            trans_matrix[i, i + 1] = params["mis_2_mis_rate"]  # mis
            trans_matrix[i, i - motif_length * 2 + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR
        else:
            trans_matrix[i, 2 * motif_length] = params["mis_2_mis_rate"]  # mis
            trans_matrix[i, 0] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR
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


def make_multipolymer_trans_matrix(
    multipolymer_hidden_states,
    motif_length,
    to_flanking_L_step,
    default_probability,
    **params,
):
    trans_matrix = np.full(
        (len(multipolymer_hidden_states), len(multipolymer_hidden_states)),
        default_probability,
    )

    for i in range(motif_length):  # for STR
        trans_matrix[i, 4 * motif_length] = 1 / to_flanking_L_step  # flanking F1
        trans_matrix[i, 4 * motif_length + 3] = 1 / (
            to_flanking_L_step + 3
        )  # flanking F
        trans_matrix[i, i + motif_length] = params["insertion_rate"]  # ins
        trans_matrix[i, i + 3 * motif_length] = params[
            "out_frame_rate"
        ]  # out-frame interruption
        if i != (motif_length - 1) and i != (motif_length - 2):
            trans_matrix[i, i + 2 * motif_length + 1] = params["mismatch_rate"]  # mis
            for j in range(motif_length):  # for STR and deletion
                if j == i + 1:
                    pass
                elif j == i + 2:
                    trans_matrix[i, j] = params["deletion_rate"]  # del
                else:
                    trans_matrix[i, j] = params[
                        "continous_deletion_rate"
                    ]  # continous del
            trans_matrix[i, i + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR
        elif i == (motif_length - 2):
            trans_matrix[i, i + 2 * motif_length + 1] = params["mismatch_rate"]  # mis
            for j in range(motif_length):  # for STR and deletion
                if j == i + 1:
                    pass
                elif j == 0:
                    trans_matrix[i, j] = params["deletion_rate"]  # del
                else:
                    trans_matrix[i, j] = params[
                        "continous_deletion_rate"
                    ]  # continous del
            trans_matrix[i, i + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR

        else:
            trans_matrix[i, 2 * motif_length] = params["mismatch_rate"]  # mis
            for j in range(motif_length):  # for STR and deletion
                if j == 0:
                    pass
                elif j == 1:
                    trans_matrix[i, j] = params["deletion_rate"]  # del
                else:
                    trans_matrix[i, j] = params[
                        "continous_deletion_rate"
                    ]  # continous del
            trans_matrix[i, 0] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expectation STR

    for i in range(motif_length, motif_length * 2):  # for ins
        # trans_matrix[i,4*motif_length]=1/to_flanking_L_step # flanking, Don't permit mis to flanking
        trans_matrix[i, i] = params["ins_2_ins_rate"]  # ins
        trans_matrix[i, i + motif_length * 2] = params[
            "ins_2_out_frame_rate"
        ]  # out-frame interruption
        if i != (2 * motif_length - 1):
            trans_matrix[i, i + motif_length + 1] = params["ins_2_mis_rate"]  # mis
            trans_matrix[i, i - motif_length + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR
        else:
            trans_matrix[i, 2 * motif_length] = params["ins_2_mis_rate"]  # mis
            trans_matrix[i, 0] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR

    for i in range(motif_length * 2, motif_length * 3):  # for mis
        # trans_matrix[i,4*motif_length]=1/to_flanking_L_step # flanking, Don't permit mis to flanking
        trans_matrix[i, i - motif_length] = params["mis_2_ins_rate"]  # ins
        trans_matrix[i, i + motif_length] = params[
            "mis_2_out_frame_rate"
        ]  # out-frame interruption
        if i != (3 * motif_length - 1):
            trans_matrix[i, i + 1] = params["mis_2_mis_rate"]  # mis
            trans_matrix[i, i - motif_length * 2 + 1] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR
        else:
            trans_matrix[i, 2 * motif_length] = params["mis_2_mis_rate"]  # mis
            trans_matrix[i, 0] = (
                1 - sum(trans_matrix[i, :]) + default_probability
            )  # expected STR

    for i in range(motif_length * 3, motif_length * 4):  # for out-frame interruption
        # trans_matrix[i,4*motif_length]=1/to_flanking_L_step # flanking, Don't permit mis to flanking
        row_sum = 1 - sum(trans_matrix[i, :]) + default_probability * (motif_length - 2)
        for j in range(0, motif_length):
            if i - motif_length * 3 + 2 < motif_length:
                if i - motif_length * 3 + 2 == j or i - motif_length * 3 + 1 == j:
                    pass
                else:
                    trans_matrix[i, j] = row_sum / (motif_length - 2)  # expected STR
            elif i - motif_length * 3 + 1 < motif_length:
                if i - motif_length * 4 + 2 == j or i - motif_length * 3 + 1 == j:
                    pass
                else:
                    trans_matrix[i, j] = row_sum / (motif_length - 2)  # expected STR
            else:
                if i - motif_length * 4 + 2 == j or i - motif_length * 4 + 1 == j:
                    pass
                else:
                    trans_matrix[i, j] = row_sum / (motif_length - 2)  # expected STR
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


def make_trans_matrix_dict(hidden_states_dict, default_probability, params):
    trans_matrix_dict = {}
    for i in range(1, 7):
        to_flanking_L_step = L_steps_dict[i]
        if i == 1:
            trans_matrix_dict[1] = make_mononucleotide_trans_matrix(
                hidden_states_dict[i],
                to_flanking_L_step,
                default_probability,
                **(params[1]),
            )
        elif i == 2:
            trans_matrix_dict[2] = make_dinucleotide_trans_matrix(
                hidden_states_dict[i],
                i,
                to_flanking_L_step,
                default_probability,
                **(params[2]),
            )
        else:
            trans_matrix_dict[i] = make_multipolymer_trans_matrix(
                hidden_states_dict[i],
                i,
                to_flanking_L_step,
                default_probability,
                **(params[i]),
            )
    return trans_matrix_dict


def make_mononucleotide_emis_matrix(
    hidden_states, motif, base_index_dict, error_rate, flanking_sequence
):
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
    hidden_states, motif, motif_length, base_index_dict, error_rate, flanking_sequence
):
    emis_matrix = np.full((len(hidden_states), 4), error_rate / 3)
    for i in range(motif_length):  # for STR
        emis_matrix[i, base_index_dict[motif[i]]] = 1 - error_rate
    for i in range(motif_length, motif_length * 2):  # for ins
        emis_matrix[i, :] = (3 - error_rate) / 9
        if i != (2 * motif_length - 1):
            emis_matrix[i, base_index_dict[motif[i - motif_length + 1]]] = (
                error_rate / 3
            )
        else:
            emis_matrix[i, base_index_dict[motif[0]]] = error_rate / 3
    for i in range(motif_length * 2, motif_length * 3):  # for mis
        emis_matrix[i, :] = (3 - error_rate) / 9
        emis_matrix[i, base_index_dict[motif[i - motif_length * 2]]] = error_rate / 3

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
    hidden_states, motif, motif_length, base_index_dict, error_rate, flanking_sequence
):
    emis_matrix = np.full((len(hidden_states), 4), error_rate / 3)
    for i in range(motif_length):  # for STR
        emis_matrix[i, base_index_dict[motif[i]]] = 1 - error_rate
    for i in range(motif_length, motif_length * 2):  # for ins
        emis_matrix[i, :] = (3 - error_rate) / 9
        if i != (2 * motif_length - 1):
            emis_matrix[i, base_index_dict[motif[i - motif_length + 1]]] = (
                error_rate / 3
            )
        else:
            emis_matrix[i, base_index_dict[motif[0]]] = error_rate / 3
    for i in range(motif_length * 2, motif_length * 3):  # for mis
        emis_matrix[i, :] = (3 - error_rate) / 9
        emis_matrix[i, base_index_dict[motif[i - motif_length * 2]]] = error_rate / 3

    for i in range(motif_length * 3, motif_length * 4):  # for out-frame interruption
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


def baseQ_2_error_rate(baseQ_array):
    return np.power(10, np.array(baseQ_array) / (-10))


hidden_states_dict = make_hidden_states_dict()
trans_matrix_dict = make_trans_matrix_dict(
    hidden_states_dict, default_probability, All_motif_params
)
start_matrix_dict = make_start_matrix_dict(
    hidden_states_dict, default_probability, All_motif_params
)


def extract_segmentation_seq(
    read, STR_start_from_0_based_bed, STR_stop_from_0_based_bed, L_step
):
    ## How to process hard-clipping and soft-clipping sequencing reads
    aligned_pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    refpos_queryindex = {i[1]: [i[0], i[2]] for i in aligned_pairs}
    read_seq = read.query_alignment_sequence
    read_baseq = read.query_alignment_qualities
    read_baseq_error_rate = baseQ_2_error_rate(read_baseq)
    if read_baseq_error_rate.min() == 0:
        read_baseq_error_rate[read_baseq_error_rate == 0] = read_baseq_error_rate[
            read_baseq_error_rate.nonzero()
        ].min()
    if read_baseq_error_rate.max() == 1:
        read_baseq_error_rate[read_baseq_error_rate == 1] = read_baseq_error_rate[
            read_baseq_error_rate != 1
        ].max()
    read_baseq_accuray = 1 - read_baseq_error_rate
    alignment_start_index = read.query_alignment_start
    alignment_end_index = read.query_alignment_end
    left_flanking_last_base_pos = STR_start_from_0_based_bed - 1
    k = 0
    while True:
        if refpos_queryindex.get(left_flanking_last_base_pos) is None:
            left_flanking_last_base_pos -= 1
            k += 1
        else:
            flanking_left_segmentation_site_index = refpos_queryindex.get(
                left_flanking_last_base_pos
            )[0]
            break
        # if k >= 12:
        #     return (
        #         "InDels in anchor bigger than 12",
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #     )
    right_flanking_first_base_pos = STR_stop_from_0_based_bed
    k = 0
    while True:
        if refpos_queryindex.get(right_flanking_first_base_pos) is None:
            right_flanking_first_base_pos += 1
            k += 1
        else:
            flanking_right_segmentation_site_index = refpos_queryindex.get(
                right_flanking_first_base_pos
            )[0]
            break
        # if k >= 12:
        #     return (
        #         "InDels in anchor bigger than 12",
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #     )

    left_STR_first_base_pos = left_flanking_last_base_pos + 1
    k = 0
    while True:
        if refpos_queryindex.get(left_STR_first_base_pos) is None:
            left_STR_first_base_pos += 1
            k += 1
        else:
            STR_left_segmentation_site_index = refpos_queryindex.get(
                left_STR_first_base_pos
            )[0]
            break
        # if k >= 12:
        #     return (
        #         "InDels in anchor bigger than 12",
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #     )
    right_STR_last_base_pos = right_flanking_first_base_pos - 1
    k = 0
    while True:
        if refpos_queryindex.get(right_STR_last_base_pos) is None:
            right_STR_last_base_pos -= 1
            k += 1
        else:
            STR_right_segmentation_site_index = refpos_queryindex.get(
                right_STR_last_base_pos
            )[0]
            break
        # if k >= 12:
        #     return (
        #         "InDels in anchor bigger than 12",
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #     )

    readSpanning = (
        flanking_left_segmentation_site_index - alignment_start_index + 1 >= FLK_BP_SEG
    ) and (
        alignment_end_index - flanking_right_segmentation_site_index + 1
    ) >= FLK_BP_SEG
    enoughSTR = (
        STR_right_segmentation_site_index - STR_left_segmentation_site_index + 1
    ) >= L_step

    if readSpanning and enoughSTR:
        left_seg_seq = read_seq[
            flanking_left_segmentation_site_index
            - alignment_start_index
            - FLK_BP_SEG
            + 1 : STR_left_segmentation_site_index - alignment_start_index + L_step
        ]
        right_seg_seq = read_seq[
            STR_right_segmentation_site_index
            - alignment_start_index
            - L_step
            + 1 : flanking_right_segmentation_site_index
            - alignment_start_index
            + FLK_BP_SEG
        ]
        left_seg_baseq_error_rate = read_baseq_error_rate[
            flanking_left_segmentation_site_index
            - alignment_start_index
            - FLK_BP_SEG
            + 1 : STR_left_segmentation_site_index - alignment_start_index + L_step
        ]
        right_seg_baseq_error_rate = read_baseq_error_rate[
            STR_right_segmentation_site_index
            - alignment_start_index
            - L_step
            + 1 : flanking_right_segmentation_site_index
            - alignment_start_index
            + FLK_BP_SEG
        ]
        left_seq_seg_start_index_include_this_base = (
            STR_left_segmentation_site_index - alignment_start_index + L_step - 1
        )
        right_seq_seg_start_index_include_this_base = (
            STR_right_segmentation_site_index - alignment_start_index - L_step + 1
        )
        return (
            left_seg_seq,
            right_seg_seq,
            left_seg_baseq_error_rate,
            right_seg_baseq_error_rate,
            read_seq,
            read_baseq_accuray,
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
            alignment_start_index,
            STR_left_segmentation_site_index,
            STR_right_segmentation_site_index,
        )


def segmentation_seq(seq, path):
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


def read_segmentation_DP(
    seq,
    seq_quality,
    motif,
    hidden_states,
    start_matrix,
    trans_matrix,
    base_index_dict,
    flanking_sequence,
    state_map,
):
    """
    HMM dynamic programming to decode hidden states
    """
    if len(motif) == 1:
        emis_matrix = make_mononucleotide_emis_matrix(
            hidden_states, motif, base_index_dict, seq_quality[0], flanking_sequence
        )
    elif len(motif) == 2:
        emis_matrix = make_dinucleotide_emis_matrix(
            hidden_states,
            motif,
            len(motif),
            base_index_dict,
            seq_quality[0],
            flanking_sequence,
        )
    else:
        emis_matrix = make_multipolymer_emis_matrix(
            hidden_states,
            motif,
            len(motif),
            base_index_dict,
            seq_quality[0],
            flanking_sequence,
        )
    path_prob = np.log(start_matrix) + np.log(emis_matrix[:, base_index_dict[seq[0]]])
    path = []
    for o, q in zip(seq[1:], seq_quality[1:]):
        if len(motif) == 1:
            emis_matrix = make_mononucleotide_emis_matrix(
                hidden_states, motif, base_index_dict, q, flanking_sequence
            )
        elif len(motif) == 2:
            emis_matrix = make_dinucleotide_emis_matrix(
                hidden_states, motif, len(motif), base_index_dict, q, flanking_sequence
            )
        else:
            emis_matrix = make_multipolymer_emis_matrix(
                hidden_states, motif, len(motif), base_index_dict, q, flanking_sequence
            )
        path_prob = path_prob + np.log(trans_matrix)
        best_state = np.argmax(path_prob, axis=1)
        best_prob = path_prob[np.arange(len(best_state)), best_state]
        path_prob = best_prob + np.log(emis_matrix[:, base_index_dict[o]])
        path.append(best_state)
    best_state = np.argmax(path_prob)
    final_best_prob = path_prob[best_state]
    best_state_sequence = [state_map[best_state]]
    for prev_state in path[::-1]:
        best_state = prev_state[best_state]
        best_state_sequence.append(state_map[best_state])
        final_best_path = best_state_sequence[::-1]
    return final_best_path, final_best_prob


def segmentation_by_hmm_per_locus_info(
    motif, motif_length, left_flanking_three_bp, L_steps_dict
):
    reverse_motif = motif[::-1]
    reverse_left_flanking_three_bp = left_flanking_three_bp[::-1]
    trans_mat = trans_matrix_dict[motif_length]
    init_mat = start_matrix_dict[motif_length]
    hidden_states = hidden_states_dict[motif_length]
    L_step = L_steps_dict[motif_length]
    state_map = {}
    for i, j in zip([k for k in range(len(hidden_states))], hidden_states):
        state_map[i] = j
    return (
        L_step,
        reverse_motif,
        reverse_left_flanking_three_bp,
        trans_mat,
        init_mat,
        hidden_states,
        state_map,
    )


def segmentation_by_hmm_per_read(
    read_seq,
    read_baseq_accuray,
    motif,
    reverse_motif,
    reverse_left_flanking_three_bp,
    trans_mat,
    init_mat,
    hidden_states,
    right_flanking_three_bp,
    left_seg_seq,
    right_seg_seq,
    left_seg_baseq_error_rate,
    right_seg_baseq_error_rate,
    left_seq_seg_start_index_include_this_base,
    right_seq_seg_start_index_include_this_base,
    state_map,
):
    reverse_left_seg_seq = left_seg_seq[::-1].upper()
    reverse_left_baseq_error_rate = left_seg_baseq_error_rate[::-1]
    right_seg_seq = right_seg_seq.upper()
    left_final_best_path, left_final_best_prob = read_segmentation_DP(
        reverse_left_seg_seq,
        reverse_left_baseq_error_rate,
        reverse_motif,
        hidden_states,
        init_mat,
        trans_mat,
        base_index_dict,
        reverse_left_flanking_three_bp,
        state_map,
    )

    left_anchor_FLK_first_base_0_based = segmentation_seq(
        reverse_left_seg_seq, left_final_best_path
    )
    right_final_best_path, right_final_best_prob = read_segmentation_DP(
        right_seg_seq,
        right_seg_baseq_error_rate,
        motif,
        hidden_states,
        init_mat,
        trans_mat,
        base_index_dict,
        right_flanking_three_bp,
        state_map,
    )
    right_anchor_FLK_first_base_0_based = segmentation_seq(
        right_seg_seq, right_final_best_path
    )
    if (left_anchor_FLK_first_base_0_based == len(reverse_left_seg_seq)) or (
        right_anchor_FLK_first_base_0_based == len(right_seg_seq)
    ):
        return ("STR length out of range", "STR length out of range")
    else:
        left_STR_first_base_location = (
            left_seq_seg_start_index_include_this_base
            - left_anchor_FLK_first_base_0_based
            + 1
        )
        right_STR_last_base_location = (
            right_seq_seg_start_index_include_this_base
            + right_anchor_FLK_first_base_0_based
            - 1
        )
        if right_STR_last_base_location + 1 <= left_STR_first_base_location:
            return ("STR length out of range", "STR length out of range")
        else:
            return left_STR_first_base_location - 1, right_STR_last_base_location + 1
            # STR_seq = read_seq[
            #     left_STR_first_base_location : right_STR_last_base_location + 1
            # ]
            # STR_baseq_accuray = read_baseq_accuray[
            #     left_STR_first_base_location : right_STR_last_base_location + 1
            # ]
            # STR_length = len(STR_seq)
            # return STR_length, STR_seq, STR_baseq_accuray


def STR_locus_info(bed_row):
    chrs = bed_row.split("\t")[0]
    start = int(bed_row.split("\t")[1])
    end = int(bed_row.split("\t")[2])
    STR_id = bed_row.split("\t")[5]
    motif = bed_row.split("\t")[6]
    motif = motif.upper()
    motif_length = int(bed_row.split("\t")[3])
    period = float(bed_row.split("\t")[4])
    STR_length = end - start
    left_three_flanking = str(bed_row.split("\t")[7])
    right_three_flanking = str(bed_row.split("\t")[8])
    ref_STR_seq = str(bed_row.split("\t")[9])
    return (
        chrs,
        start,
        end,
        STR_id,
        motif,
        motif_length,
        period,
        STR_length,
        left_three_flanking,
        right_three_flanking,
        ref_STR_seq,
    )


################# hmm_segmentation_single_cell #################
# A filtered STR reference panel should be used in /storage/douyanmeiLab/data/STR_reference_panel_for_HMM
# import seg
# import pysam


def _segmentation(STR_locus, read):
    # HipSTR_GRCH37_RefPanel = pysam.TabixFile(HipSTR_GRCH37_RefPanel_file)
    # locus seg
    (
        contig,
        STR_start_from_0_based_bed,
        STR_stop_from_0_based_bed,
        STR_id,
        motif,
        motif_length,
        period,
        STR_length,
        left_three_flanking,
        right_three_flanking,
        ref_STR_seq,
    ) = STR_locus_info(STR_locus)
    (
        L_step,
        reverse_motif,
        reverse_left_flanking_three_bp,
        trans_mat,
        init_mat,
        hidden_states,
        state_map,
    ) = segmentation_by_hmm_per_locus_info(
        motif, motif_length, left_three_flanking, L_steps_dict
    )
    # read seg
    (
        left_seg_seq,
        right_seg_seq,
        left_seg_baseq_error_rate,
        right_seg_baseq_error_rate,
        read_seq,
        read_baseq_accuray,
        left_seq_seg_start_index_include_this_base,
        right_seq_seg_start_index_include_this_base,
    ) = extract_segmentation_seq(
        read, STR_start_from_0_based_bed, STR_stop_from_0_based_bed, L_step
    )
    if (
        (left_seg_seq == "InDels in anchor bigger than 12")
        or (left_seg_seq == "No enough length for segmentation")
        or ("N" in left_seg_seq)
        or ("N" in right_seg_seq)
    ):
        raise ValueError("seq error")  # HACK: return None?
    else:
        STR_length, STR_seq, STR_baseq_accuray = segmentation_by_hmm_per_read(
            read_seq,
            read_baseq_accuray,
            motif,
            reverse_motif,
            reverse_left_flanking_three_bp,
            trans_mat,
            init_mat,
            hidden_states,
            right_three_flanking,  # right_flanking_three_bp -> right_three_flanking
            left_seg_seq,
            right_seg_seq,
            left_seg_baseq_error_rate,
            right_seg_baseq_error_rate,
            left_seq_seg_start_index_include_this_base,
            right_seq_seg_start_index_include_this_base,
            state_map,
        )
        if STR_length == "STR length out of range":
            raise ValueError("STR length out of range")
        else:
            return STR_seq, STR_baseq_accuray
            pass  # Your code to process STR_length and STR_seq #


class SegModel:
    def __init__(self, STR_locus, region_dict) -> None:
        super().__init__()
        self.region_dict = region_dict
        # locus seg
        (
            self.contig,
            self.STR_start_from_0_based_bed,
            self.STR_stop_from_0_based_bed,
            self.STR_id,
            self.motif,
            self.motif_length,
            self.period,
            self.STR_length,
            self.left_three_flanking,
            self.right_three_flanking,
            self.ref_STR_seq,
        ) = STR_locus_info(STR_locus)
        (
            self.L_step,
            self.reverse_motif,
            self.reverse_left_flanking_three_bp,
            self.trans_mat,
            self.init_mat,
            self.hidden_states,
            self.state_map,
        ) = segmentation_by_hmm_per_locus_info(
            self.motif, self.motif_length, self.left_three_flanking, L_steps_dict
        )

    def seg(self, read):
        # read seg
        (
            left_seg_seq,
            right_seg_seq,
            left_seg_baseq_error_rate,
            right_seg_baseq_error_rate,
            read_seq,
            read_baseq_accuray,
            left_seq_seg_start_index_include_this_base,
            right_seq_seg_start_index_include_this_base,
        ) = extract_segmentation_seq(
            read,
            self.STR_start_from_0_based_bed,
            self.STR_stop_from_0_based_bed,
            self.L_step,
        )
        if ("InDels in anchor" in left_seg_seq) or (
            left_seg_seq == "No enough length for segmentation"
        ):
            # alignment_start_index = read_baseq_accuray
            # STR_left_segmentation_site_index = (
            #     left_seq_seg_start_index_include_this_base
            # )
            # STR_right_segmentation_site_index = (
            #     right_seq_seg_start_index_include_this_base
            # )
            # anchor_l = STR_left_segmentation_site_index - alignment_start_index
            # anchor_r = STR_right_segmentation_site_index - alignment_start_index
            anchors = tools.extract_hap(read, self.region_dict)
            if anchors:
                anchor_l, anchor_r = anchors
            else:
                return None
            if anchor_l > anchor_r:
                return 0, 0, 0
            else:
                return anchor_l, anchor_r, 0
        elif ("N" in left_seg_seq) or ("N" in right_seg_seq):
            return None
        else:
            # STR_length, STR_seq, STR_baseq_accuray = segmentation_by_hmm_per_read(
            anchor_l, anchor_r = segmentation_by_hmm_per_read(
                read_seq,
                read_baseq_accuray,
                self.motif,
                self.reverse_motif,
                self.reverse_left_flanking_three_bp,
                self.trans_mat,
                self.init_mat,
                self.hidden_states,
                self.right_three_flanking,  # right_flanking_three_bp -> right_three_flanking
                left_seg_seq,
                right_seg_seq,
                left_seg_baseq_error_rate,
                right_seg_baseq_error_rate,
                left_seq_seg_start_index_include_this_base,
                right_seq_seg_start_index_include_this_base,
                self.state_map,
            )
            # if STR_length == "STR length out of range":
            if (
                anchor_l == "STR length out of range"
                or "N" in read_seq[anchor_l + 1 : anchor_r]
            ):
                return None
                raise ValueError("STR length out of range")
            else:
                # return STR_seq, STR_baseq_accuray
                if (
                    anchor_r <= read.query_alignment_length - FLK_BP
                    and anchor_l >= FLK_BP
                ):
                    return anchor_l, anchor_r, 1
                else:
                    return None
