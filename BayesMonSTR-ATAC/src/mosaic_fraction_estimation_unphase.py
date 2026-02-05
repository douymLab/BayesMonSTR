from scipy import stats
import numpy as np
from scipy.stats import geom
from scipy.special import logsumexp
from scipy.optimize import minimize
import logging

# import os
import hap_alignment
import myutils
import config_params
import logger_config
import mutation_filtering
import pdb
## 相比于 weixiang2 和 phase 先改了 unordered mosaicGT 和 select initial guess 对于 unique mosaic fraction mode

DEBUG = False
PLOT = False

# if DEBUG:
#     import sys
#     import matplotlib.pyplot as plt

#     sys.path.append(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     )
#     from myutils import myutils
#     from configs import config_params, logger_config
# else:
#     from ..myutils import myutils
#     from ..configs import config_params, logger_config

LOG_ONE_HALF = config_params.CONSTANT["LOG_ONE_HALF"]  # np.log(0.5)
LOG_TWO = config_params.CONSTANT["LOG_TWO"]  # np.log(2)

EM_PARAMS = config_params.EM_PARAMS
MIN_AVERAGE_COVERAGE = EM_PARAMS["loci_filter"]["MIN_AVERAGE_COVERAGE"]  # 5
CHECK_CONVERGE_PERIOD = EM_PARAMS["iteration_params"][
    "CHECK_CONVERGE_PERIOD"
]  # 2
DIFFERENCE_CONVERGE = EM_PARAMS["iteration_params"][
    "DIFFERENCE_CONVERGE"
]  # 0.0001
DIFFERENCE_CONVERGE_MosaicFrac = 0.001
MAX_TOLERANCE = EM_PARAMS["iteration_params"][
    "MAX_TOLERANCE"
]  # np.log(1 / 10)
MIN_STUTTER_INS = EM_PARAMS["init_params"]["MIN_STUTTER_INS"]  # 0.000001
MIN_STUTTER_DEL = EM_PARAMS["init_params"]["MIN_STUTTER_DEL"]  # 0.000001
MAX_SINGLE_STEP_PROB = EM_PARAMS["init_params"][
    "MAX_SINGLE_STEP_PROB"
]  # 0.999

MOSAIC_FRACTION_ESTIMATION_PARAMS = (
    config_params.MOSAIC_FRACTION_ESTIMATION_PARAMS
)
MAX_ITERATION = MOSAIC_FRACTION_ESTIMATION_PARAMS["MAX_ITERATION"]  # 500
MIN_ITERATION = MOSAIC_FRACTION_ESTIMATION_PARAMS["MIN_ITERATION"]  # 10
NON_MUTATION_RATE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "non_mutation_rate"
]  # 1 - 10**-4,1 - 10**-4
MS_MUTATION_RATE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "ms_mutation_rate"
]  # 10**-4
INITIAL_MOSAIC_FRACTION = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "initial_mosaic_fraction"
]  # 0.1
USE_MINOR_CELLS_POP_AS_MUT_CELLS = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "use_minor_cells_pop_as_mut_cells"
]  # False
ALLELE_FILTER = MOSAIC_FRACTION_ESTIMATION_PARAMS["allele_filter"]  # True
MAX_ALLOWABLE_ALLELE_NUM = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "max_allowable_allele_num"
]  # 8
ALLOWABLE_HET2HOM = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "allowable_het2hom"
]  # True
ITERATION_MODE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "iteration_mode"
]  # "cell-based"
INIT_TOTAL_MOSAIC_LOG_PRIOR_LIK = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "init_total_mosaic_log_prior_lik"
]  # -100000000.0
LIKELIHOOD_MODE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "likelihood_mode"
]  # "seq-based"
# STR_SPECIFIC_ALIGNMENT_PROBS = MOSAIC_FRACTION_ESTIMATION_PARAMS["STR_specific_alignment_probs"]
ADD_FLANKING_PROB = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "ADD_FLANKING_PROB"
]  # False
# FLANKING_ALLELE_LENGTH = MOSAIC_FRACTION_ESTIMATION_PARAMS["FLANKING_ALLELE_LENGTH"]
HET2HOM_SET_ZERO_MUT_READS = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "het2hom_set_zero_mut_reads"
]  # False
MF_MORE_THAN_HALT_SET_ZERO = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "mf_more_than_halt_set_zero"
]  # False
MAX_LIK_ASSIGNMENT = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "max_lik_assignment"
]  # False
GERM_2_GERM_MUTATION_RATE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "germ_2_germ_mutation_rate"
]  # 10**-4
ALLOWABLE_TWO_MUT_ALLELES = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "allowable_two_mut_alleles"
]  # False

HOM2HET_AND_HET2HET_TWO_MOSAIC_FRACTION_MODE = (
    MOSAIC_FRACTION_ESTIMATION_PARAMS[
        "hom2het_and_het2het_two_mosaic_fraction_mode"
    ]
)  # False
HET2HET_TWO_ALLELES_MOSAIC_FRACTION = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "het2het_two_alleles_mosaic_fraction"
]  # 0
ONE_ALLELE_MOSAIC_FRACTION = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "one_allele_mosaic_fraction"
]  # 0
ALL_GT_TRANSFORMATION_MOSAIC_FRACTION = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "all_gt_transformation_mosaic_fraction"
]  # True
MLE_OPTIMIZE_METHOD = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "mle_optimize_method"
]  # "SLSQP"
MOSAIC_FRACTION_BOUNDS = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "mosaic_fraction_bounds"
]  # (0, 1)
POSTERIOR_METHOD = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "posterior_method"
]  # Gi-free "Gi-free" # Likelihood
POSTERIOR_GERM_INCLUDED = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "posterior_germ_included"
]  # False(But true in fact through CONSIDER_GERM2GERM_IN_POSTERIOR)
POSTERIOR_MUT_TYPE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "posterior_mut_type"
]  # 1mut
POSTERIOR_ALLOW_MOSAIC_FRACTION = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "posterior_allow_mosaic_fraction"
]  # all
ESTIMATION_MODE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "estimation_mode"
]  # unique
NUMERIC_CALCULATION_TOLERANCE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "numeric_calculation_tolerance"
]  # 1e-3
PHASE_MODE = MOSAIC_FRACTION_ESTIMATION_PARAMS["phase_mode"]  # True
LOG_LIK_AND_LOG_POSTERIOR_MEANTIME = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "log_lik_and_log_posterior_meantime"
]  # True
ONLY_USE_MLE_MOSAIC_FRACTION = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "only_use_mle_mosaic_fraction"
]  # False
BINOMINAL_FILTER = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "binominal_filter"
]  # False
ALLELE_SORT = MOSAIC_FRACTION_ESTIMATION_PARAMS["allele_sort"]  # True
USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "use_gi_prior_update_mosaic_fraction"
]  # "OnlyLik"
CONSIDER_GERM2GERM_IN_PARAMS_UPDATE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "consider_germ2germ_in_params_update"
]  # False
UNORDER_MOSAIC_GT = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "unorder_mosaic_gt"
]  # True
USE_INITIAL_GUESS = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "use_initial_guess"
]  # True
INITIAL_GUESS_NUMBER = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "initial_guess_number"
]  # 50 # 20
FILTER_BIG_STUTTER_SIZE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "FILTER_BIG_STUTTER_SIZE"
]  # True
BIG_STUTTER_SIZE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "BIG_STUTTER_SIZE"
]  # {1: 20, 2: 10, 3: 10, 4: 9, 5: 8, 6: 7}
BIG_STUTTER_DEPTH_CUTOFF = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "BIG_STUTTER_DEPTH_CUTOFF"
]  # 1
LIKELIHOOD_MODE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "likelihood_mode"
]  # "seq-based"
CONSIDER_GERM2GERM_IN_POSTERIOR = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "consider_germ2germ_in_posterior"
]  # True
DOUBLE_HET_COMBA_2x_PROB = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "double_het_comba_2x_prob"
]  # False # TODO: I need calculation heteryzygous with a 2x lik than hom 1x lik ???
HAP_IDENTIFY_POA = True
ALLELE_DEPTH_CUTOFF = 1
MAX_ALLELE_CLUSTER_NUM = 5
MAX_ALLOWABLE_ALLELE_NUM_BEFORE_FEATURE = 20
MAX_LOCI_DEPTH = 10000
MIS_NUM_TOLERENT = 100  # HACK: Change from 5 to 100 to tolerent more random recurrent mismatches # HACK: Change from 2 to 5 for filtering the hap with more random recurrent mismatches, observed from complex mutation types with multiple sequencing errors are called out #
STRAND_BIAS_DEPTH_CUTOFF = 5  # 0.03125 小概率事件,(1/2)^5
OBSERVE_SF_CUTOFF = 0.99
MLE_SF_CUTOFF = 0.9
ALWAYS_TRUE_DEPTHEST_ALLELE = True
OVERALL_SF_CUTOFF = 0.5
# BASEQ_ACCURACY_CUTOFF = 0.95
# BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR = 0.99
BASEQ_ACCURACY_CUTOFF = 13
BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR = 20
# import pyabpoa
USE_POA = False
# [wangweixiang@login01 mosaic_fraction_estimation_results]$ pip show pyabpoa
# Name: pyabpoa
# Version: 1.5.1
# POA = pyabpoa.msa_aligner(aln_mode="g")
SOURCE_CHECK = True
# DEFAULT_MAX_STEP = 100
# HACK: For default steps and max step limitation for KeyErrors
# HACK: 预计 MIS_NUM_TOLERENT 和 BASEQ_ACCURACY_CUTOFF/BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR 以及 only_baseq_filter_allele = True 后过滤 allele 会非常凶，所以导致程序跑得快得可怕
# HACK:
# 另外增加 initial test 的数量以及放宽收敛的次数和收敛的条件，也会导致程序跑得快得可怕
# 跑得快得可怕，需要进一步进行check
# 我害怕快得可怕是因为 EM 没收敛或者迭代不彻底，导致 MF 估计出现错误，但是实际并没有发生，因为 pearson r = 0.9
# 目前的 收敛条件
# MAX_ITERATION = MOSAIC_FRACTION_ESTIMATION_PARAMS["MAX_ITERATION"]  # 500 不改
# MIN_ITERATION = MOSAIC_FRACTION_ESTIMATION_PARAMS["MIN_ITERATION"]  # 10 不改
# USE_INITIAL_GUESS = MOSAIC_FRACTION_ESTIMATION_PARAMS[
#     "use_initial_guess"
# ]  # True 不改
# INITIAL_GUESS_NUMBER = MOSAIC_FRACTION_ESTIMATION_PARAMS[
#     "initial_guess_number"
# ]  # 50 # 20 不改目前，如果要进一步增加迭代速度可与 MIN_ITERATION 同时改，INITIAL_GUESS_NUMBER 变大，MIN_ITERATION 变小，但目前不改
# DIFFERENCE_CONVERGE = EM_PARAMS["iteration_params"][
#     "DIFFERENCE_CONVERGE"
# ]  # 0.0001  # 不改，如果要进一步增加迭代速度可以改为 0.001
# HACK: 另外迭代方面还可以改的东西包括: 不管 likelihood 是否收敛，只关注参数的变化是否收敛即可，但是目前两者都关注，但是满足其一即可收敛，不知道 lik 设置的收敛条件的阀值是否是合理的，但是目前不改了，因为参数估计的结果还比较符合预期了哈
# NUMERIC_CALCULATION_TOLERANCE # 1e-3 不改目前除非测序深度为 1w
# 我害怕 allele 过滤得太凶影响 sensitivity，但是，2-3 条 reads 时还是可以保留至少 0.8 的 sensitivity
# 目前的 baseq cutoff
# BASEQ_ACCURACY_CUTOFF = 0.95 应该改
# BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR = 0.99 应该改
# only_baseq_filter_allele = True # 不改
# ACCURACY_FRACTION_CUTOFF = 0.5 # 不改
# SOURCE_CHECK_TARGET_DEPTH_CUTOFF = 3 # 不改
# 所以目前要改的内容为 BASEQ_ACCURACY_CUTOFF 和 BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR 相关内容
# check 最新一期 1485 跑出来的突变量是多少，观察是否需要改 cutoff
# 完毕结束了哈


import math
def accuracy_to_phred(accuracy: float) -> float:
    """
    Transforms base accuracy (probability) to baseQ Phred score.

    Parameters:
    accuracy (float): The base accuracy as a probability (0 to 1).

    Returns:
    float: The corresponding baseQ Phred score.
    """
    if accuracy == "NA":
        return "NA"
    # Ensure the accuracy is within the valid range
    if not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be between 0 and 1.")

    # Convert accuracy probability to probability of error
    probability_of_error = 1 - accuracy

    # Calculate the Phred score
    if probability_of_error == 0:
        # Return a very high Phred score if there is no error (implies very high accuracy)
        return 40

    phred_score = int(round(-10 * math.log10(probability_of_error), 0))

    return phred_score

class MosaicFractionPerSampleEstimator:
    def __init__(
        self,
        reads_list,
        reads_accuracy_list,
        pysam_reads_list,
        alleles_STR_length_list,
        alleles_pysam_read_dict,
        alleles_reads_accuracy_dict,
        alleles_depth_dict,
        all_alleles_list_for_gt_output,  # alleles_sequence_depth_dict,
        stutter_model_params,
        in_frame_stutter_model,
        out_frame_stutter_model,
        locus_infos,
        other_params,
    ):
        [
            logging.root.removeHandler(handler)
            for handler in logging.root.handlers[:]
        ]
        logger_config.configure_logger(other_params["log_file_path"])
        if ALLELE_SORT:
            alleles_depth_dict = dict(
                sorted(alleles_depth_dict.items(), key=lambda x: x[1])
            )
        self.reads_list = reads_list
        self.reads_accuracy_list = reads_accuracy_list
        self.reads_number = len(self.reads_list)
        # print(self.reads_number)
        self.pysam_reads_list = pysam_reads_list
        self.alleles_STR_length_list = alleles_STR_length_list
        self.all_alleles_list_for_gt_output = all_alleles_list_for_gt_output
        self.alleles_pysam_read_dict = alleles_pysam_read_dict
        self.alleles_reads_accuracy_dict = alleles_reads_accuracy_dict
        self.alleles_depth_dict = alleles_depth_dict
        # self.alleles_sequence_depth_dict = alleles_sequence_depth_dict
        self.alleles_order = list(
            self.alleles_depth_dict.keys()
        )  # XXX: Depth Need Be From low to high sort, alleles_depth_dict need be sorted before input
        # self.alleles_seq_order = self.alleles_sequence_depth_dict.keys()
        self.sample_alleles_number = len(self.alleles_order)
        self.alleles_depth = list(
            self.alleles_depth_dict.values()
        )  # XXX: corresponding to alleles_order
        if len(self.alleles_STR_length_list) > 1:
            self.max_stutter_steps = max(self.alleles_STR_length_list) - min(
                self.alleles_STR_length_list
            )
        else:
            self.max_stutter_steps = 1
        self.stutter_model_params = stutter_model_params
        # TODO: performace refine for input stutter model because all samples share a stutter model per locus
        self.inframe_stutter_model_dict = in_frame_stutter_model
        # self.stutter_model(
        #     stutter_model_params["inframe_single_step_prob"],
        #     stutter_model_params["inframe_ins_prob"],
        #     stutter_model_params["inframe_del_prob"],
        #     self.max_stutter_steps,
        # )
        self.outframe_stutter_model_dict = out_frame_stutter_model
        # self.stutter_model(
        #     stutter_model_params["outframe_single_step_prob"],
        #     stutter_model_params["outframe_ins_prob"],
        #     stutter_model_params["outframe_del_prob"],
        #     self.max_stutter_steps,
        # )
        self.log_no_stutter_lik = np.log(
            np.exp(self.inframe_stutter_model_dict[0])
            + np.exp(self.outframe_stutter_model_dict[0])
            - 1
        )
        self.locus_infos = locus_infos
        self.other_params = other_params
        if ALLELE_FILTER:
            self.allele_filter()
        else:
            self.alleles_used_for_gt = self.alleles_order
            # self.alleles_seq_used_for_gt = self.alleles_seq_order
        self.alleles_number_used_for_gt = len(self.alleles_used_for_gt)
        self.check_loci_callable()
        if self.callable:
            if ESTIMATION_MODE == "double":
                self.hom2het_trained = False
                self.hom2het_total_niter = 0
                self.hom2het_n_train = 0
                self.hom2het_converged = False
                self.hom2het_converged_list = [False, False, False]
                self.hom2het_locus_state = "WaitTrain"
                self.het2het_trained = False
                self.het2het_total_niter = 0
                self.het2het_n_train = 0
                self.het2het_converged = False
                self.het2het_converged_list = [False, False, False]
                self.het2het_locus_state = "WaitTrain"
                self.estimate_mode = "double"
            elif ESTIMATION_MODE == "unique":
                self.trained = False
                self.total_niter = 0
                self.n_train = 0
                self.converged = False
                self.converged_list = [False, False, False]
                self.locus_state = "WaitTrain"
                self.estimate_mode = "unique"
            elif ESTIMATION_MODE == "allmle":
                self.estimate_mode = "allmle"
            else:
                raise ValueError("Unknown estimation mode")
            self.genotypes = (
                myutils.get_all_gt_unordered_combinations_with_replacement(
                    list(range(self.alleles_number_used_for_gt)), 2
                )
            )
            # XXX: unordered and if gt1 (0,1) gt2 (1,2),we think mosaic is single allele mutation with priority
            # because single allele mutation is more likely than two alleles mutation (as our prior/experience/assumption/infinite site mutations hypothesis)
            # because unordered, we only find mosaic allele through gt comparison, which mean 0->2
            # XXX: unphase is unordered
            self.gt_number = len(self.genotypes)
            if UNORDER_MOSAIC_GT:
                self.mosaic_genotypes = (
                    myutils.get_all_gt_unordered_combinations_with_replacement(
                        list(range(self.gt_number)), 2
                    )
                )
            else:
                self.mosaic_genotypes = (
                    myutils.get_all_gt_ordered_permutations_with_replacement(
                        list(range(self.gt_number)), 2
                    )
                )
            if HAP_IDENTIFY_POA and (LIKELIHOOD_MODE != "length-based"):
                if self.all_singleton_flag:
                    self.all_reads_all_alleles_log_lik()
                else:
                    pass
            else:
                self.all_reads_all_alleles_log_lik()
            self.all_reads_all_gts_log_lik()
            self.cal_mutation_rate_prior()
            self.get_different_levels_mosaic_list()
        else:
            logging.info(
                f"This locus is not callable because {self.loci_calling}."
            )
            pass  # TODO: Need format information for output

        if DEBUG:
            if ESTIMATION_MODE == "double":
                self.hom2het_total_mosaic_log_prior_lik_list = []
                self.het2het_total_mosaic_log_prior_lik_list = []
            elif ESTIMATION_MODE == "unique":
                self.total_mosaic_log_prior_lik_list = []
            else:
                pass

    def still_output_germline_gt_and_post_when_uncallable(self):
        # Only use likelihood to judge genotypes
        # Don't consider ordered and unordered GT, only use unordered GT
        # But don't use 2*Lik for Het and 1*Lik for Hom
        self.genotypes = (
            myutils.get_all_gt_unordered_combinations_with_replacement(
                list(range(self.alleles_number_used_for_gt)), 2
            )
        )
        if (self.reads_number == 0) or (self.alleles_number_used_for_gt == 0):
            GI = "."
            GIP = "."
            GIQ = "."
            GIQP = "."
        elif self.loci_calling == "CoverageFail":
            if self.alleles_number_used_for_gt == 1:
                GI = self.genotypes[0]
                GIP = 1
                GIQ = "."
                GIQP = "."
            else:
                self.gt_number = len(self.genotypes)
                if HAP_IDENTIFY_POA and (LIKELIHOOD_MODE != "length-based"):
                    if self.all_singleton_flag:
                        self.all_reads_all_alleles_log_lik()
                    else:
                        pass
                else:
                    self.all_reads_all_alleles_log_lik()
                self.all_reads_all_gts_log_lik()
                (
                    MaxGermlineGT,
                    MaxGermlineLogPosterior,
                    MaxGermlineLogLik,
                    SecondGermlineGT,
                    SecondGermlineLogPosterior,
                    SecondGermlineLogLik,
                ) = self.cal_germline_log_posterior()
                germline_freedom = 2 - len(
                    set(MaxGermlineGT) & set(SecondGermlineGT)
                )
                (
                    best_germ_vs_second_germ_or,
                    best_germ_vs_second_germ_pvalue,
                ) = mutation_filtering.ht_log_likelihood_ratio_test(
                    MaxGermlineLogLik,
                    SecondGermlineLogLik,
                    germline_freedom,
                )
                GI = MaxGermlineGT
                GIP = np.exp(MaxGermlineLogPosterior)
                GIQ = best_germ_vs_second_germ_or
                GIQP = best_germ_vs_second_germ_pvalue
        elif self.loci_calling == "Germline":
            GI = self.genotypes[0]
            GIP = 1
            GIQ = "."
            GIQP = "."
        else:
            if self.alleles_number_used_for_gt == 1:
                GI = self.genotypes[0]
                GIP = 1
                GIQ = "."
                GIQP = "."
            else:
                self.gt_number = len(self.genotypes)
                if HAP_IDENTIFY_POA and (LIKELIHOOD_MODE != "length-based"):
                    if self.all_singleton_flag:
                        self.all_reads_all_alleles_log_lik()
                    else:
                        pass
                else:
                    self.all_reads_all_alleles_log_lik()
                self.all_reads_all_gts_log_lik()
                (
                    MaxGermlineGT,
                    MaxGermlineLogPosterior,
                    MaxGermlineLogLik,
                    SecondGermlineGT,
                    SecondGermlineLogPosterior,
                    SecondGermlineLogLik,
                ) = self.cal_germline_log_posterior()
                germline_freedom = 2 - len(
                    set(MaxGermlineGT) & set(SecondGermlineGT)
                )
                (
                    best_germ_vs_second_germ_or,
                    best_germ_vs_second_germ_pvalue,
                ) = mutation_filtering.ht_log_likelihood_ratio_test(
                    MaxGermlineLogLik,
                    SecondGermlineLogLik,
                    germline_freedom,
                )
                GI = MaxGermlineGT
                GIP = np.exp(MaxGermlineLogPosterior)
                GIQ = best_germ_vs_second_germ_or
                GIQP = best_germ_vs_second_germ_pvalue
        return GI, GIP, GIQ, GIQP

    def stutter_model(
        self,
        stutter_single_step_prob,
        stutter_ins_prob,
        stutter_del_prob,
        max_stutter_steps,
    ):
        # XXX Avoid python imprecision float calculation to produce the value > 1 or < 0
        # HACK: A more optimal solution for stutter_single_step_prob = 1 is necessary
        stutter_single_step_prob = min(
            [stutter_single_step_prob, MAX_SINGLE_STEP_PROB]
        )
        stutter_ins_prob = max([stutter_ins_prob, MIN_STUTTER_INS])
        stutter_del_prob = max([stutter_del_prob, MIN_STUTTER_DEL])
        max_stutter_steps = max(
            [max_stutter_steps, 1]
        )  # Prevent errors when invoking the homopolymer outframe stutter model function.
        # XXX Avoid key error because stutter model don‘t consider other samples’ alleles when initialized
        steps = np.arange(1, max_stutter_steps + 1, 1)
        step_probs = geom.pmf(steps, stutter_single_step_prob)
        if step_probs.min() == 0.0:
            step_probs[step_probs == 0.0] = step_probs[
                step_probs.nonzero()
            ].min()
        stutter_sizes = (
            list(map(lambda x: -x, steps[::-1])) + [0] + list(steps)
        )
        stutter_probs = (
            list(np.log(stutter_del_prob) + np.log(step_probs[::-1]))
            + [np.log(1.0 - stutter_del_prob - stutter_ins_prob)]
            + list(np.log(stutter_ins_prob) + np.log(step_probs))
        )
        log_stutter_model_dic = dict(zip(stutter_sizes, stutter_probs))
        return log_stutter_model_dic

    def allele_filter(self):   # HACK: allele filter 的效果没有非常好的原因: 初始 allele 很多，大部分的 allele 仅仅 assign 到它本身，只有 depth 为 1 的 reads 被 assign，所以大部分 allele mle depth 很低，可能都没满足 DEPTH_CUTOFF，因此导致过滤效果不理想
        # binominal test to filter noise
        self.all_singleton_flag = False
        motif_and_str_length_judge = myutils.post_homopolymer_motif_check(self.locus_infos["motif"], self.locus_infos["ref_allele_length"])
        alleles_used_for_gt = self.alleles_order
        error_rate = (
            self.stutter_model_params["inframe_del_prob"]
            + self.stutter_model_params["inframe_ins_prob"]
            + self.stutter_model_params["outframe_del_prob"]
            + self.stutter_model_params["outframe_ins_prob"]
        )
        if (
            FILTER_BIG_STUTTER_SIZE
        ):  # HACK: stutter 过滤了 big stutter size reads，那么 calling mosaic 时也要过滤这条 read，否则容易造成 hom calling 成 het
            # HACK: 因为 calling mosaic 的时候，只考虑到单个样本的 reads 信息，所以无法过滤掉 all samples low depth big stutter size reads，因此当前倾向于 FILTER_BIG_STUTTER_SIZE = False
            self.big_stutter_size_reads_number = 0
            self.big_stutter_size_alleles_number = 0
            max_stutter_size = (
                BIG_STUTTER_SIZE[self.locus_infos["motif_length"]]
                * self.locus_infos["motif_length"]
            )
            for index, row in self.alleles_depth_dict.items():
                if LIKELIHOOD_MODE == "length-based":
                    bp_diff = abs(
                        index - self.locus_infos["ref_allele_length"]
                    )
                else:
                    bp_diff = abs(
                        len(index[1]) - self.locus_infos["ref_allele_length"]
                    )
                if (
                    bp_diff > max_stutter_size
                    and row <= BIG_STUTTER_DEPTH_CUTOFF
                ):
                    alleles_used_for_gt.remove(index)
                    self.big_stutter_size_reads_number += row
                    self.big_stutter_size_alleles_number += 1
        if HAP_IDENTIFY_POA and (LIKELIHOOD_MODE != "length-based"):
            # HACK: A Temp Solution For Without Representative Haplotype (cluster need refine before POA)
            self.singleton_alleles_number = 0
            self.allele_with_depth_more_than_n_allele_number = 0
            self.filtered_out_allele_num_more_than_max = 0
            allele_with_depth_more_than_n_list = []
            singleton_allele_list = []
            final_alleles_used_for_gt = []
            final_alleles_lik_array_index = []
            for allele in alleles_used_for_gt:
                if (
                    self.alleles_depth_dict.get(allele, 0)
                    >= ALLELE_DEPTH_CUTOFF
                ):
                    allele_with_depth_more_than_n_list.append(allele)
                    self.allele_with_depth_more_than_n_allele_number += 1
                else:
                    singleton_allele_list.append(allele[1])
                    self.singleton_alleles_number += 1
            if USE_POA:
                cluster_num = int(round(self.singleton_alleles_number / 2))
                cluster_num = min(MAX_ALLELE_CLUSTER_NUM, cluster_num)
                consensus_pass_allele = []
                # print(self.locus_infos["STR_id"])
                # print("singleton:")
                # print(singleton_allele_list)
                if (cluster_num > 0) and (self.singleton_alleles_number > 1):
                    poa_msa = POA.msa(
                        singleton_allele_list,
                        True,
                        False,
                        min_freq=2 / self.singleton_alleles_number,
                        max_n_cons=cluster_num,
                    )
                    for con_seq, cluster_size in zip(
                        poa_msa.cons_seq, poa_msa.clu_n_seq
                    ):
                        if cluster_size >= ALLELE_DEPTH_CUTOFF:
                            if (
                                tuple(["", con_seq, ""])
                                in allele_with_depth_more_than_n_list
                            ):
                                pass
                            else:
                                consensus_pass_allele.append(
                                    tuple(["", con_seq, ""])
                                )
                        else:
                            if len(allele_with_depth_more_than_n_list) > 0:
                                pass
                            else:
                                consensus_pass_allele.append(
                                    tuple(["", con_seq, ""])
                                )
                else:
                    pass
            else:
                consensus_pass_allele = []
            # print("consensus")
            # print(consensus_pass_allele)
            new_alleles_used_for_gt = (
                consensus_pass_allele + allele_with_depth_more_than_n_list
            )
            if (
                len(new_alleles_used_for_gt)
                > MAX_ALLOWABLE_ALLELE_NUM_BEFORE_FEATURE
            ):
                self.filtered_out_allele_num_more_than_max = (
                    len(new_alleles_used_for_gt)
                    - MAX_ALLOWABLE_ALLELE_NUM_BEFORE_FEATURE
                )
                new_alleles_used_for_gt_no_more_than_max = (
                    new_alleles_used_for_gt[
                        self.filtered_out_allele_num_more_than_max :
                    ]
                )
            else:
                # self.filtered_out_allele_num_more_than_max = 0
                new_alleles_used_for_gt_no_more_than_max = (
                    new_alleles_used_for_gt
                )
            self.alleles_used_for_gt = new_alleles_used_for_gt_no_more_than_max
            self.all_reads_all_alleles_log_lik()  # TODO: This function is called replicately three times, and extra one calling is unnecessary, Done
            if len(self.alleles_used_for_gt) > 1:
                (
                    haplotype_depth_array,
                    hap_depth_list,
                    unknown_assign_num,
                ) = myutils.cal_mle_haplotype_depth_dict_unphase(
                    self.all_reads_all_alleles_log_lik_array
                )
                observed_hap_depth_list = [self.alleles_depth_dict.get(
                        i, 0
                    ) for i in self.alleles_used_for_gt]
                sort_mle_hap_depth_alleles = np.lexsort((observed_hap_depth_list, list(hap_depth_list)))
                # 这边是从小到大排序 right !
                # sort_mle_hap_depth_alleles = np.argsort(hap_depth_list)
                # HACK: 如果 MLE depth 一样的时候如何排序呢？按照观察深度更高的优先
                # DEBUG DONE
                overall_strand_list = [
                    i.is_reverse for i in self.pysam_reads_list
                ]
                overall_strand_fraction = sum(overall_strand_list) / len(
                    overall_strand_list
                )
                overall_strand_fraction = myutils.cal_undirectional_value(
                    overall_strand_fraction
                )
                removed_allele_list = []
                allele_source_dict = {}
                allele_mle_depth_list = []
                allele_obs_depth_list = []
                allele_source_edit_dis_dict = {}
                for allele_order, allele_idx in enumerate(
                    sort_mle_hap_depth_alleles
                ):
                    allele_seq = self.alleles_used_for_gt[allele_idx][1]
                    allele_depth_mle = hap_depth_list[allele_idx]
                    allele_tuple = tuple(["", allele_seq, ""])
                    allele_depth_observe = self.alleles_depth_dict.get(
                        allele_tuple, 0
                    )
                    allele_accuracy_list = (
                        self.alleles_reads_accuracy_dict.get(allele_tuple, [])
                    )
                    sort_mle_hap_depth_alleles_reverse = (
                        sort_mle_hap_depth_alleles[allele_order + 1:][::-1]
                    )  # 这边是从大到小排序，right !
                    edit_distance_list = []
                    mis_location_list = []
                    prior_allele_id_list = []
                    best_source_allele = ""
                    best_source_id = ""
                    flanking_first_bp = False
                    edit_distance_one = False
                    flanking_mismatch = False
                    if ((allele_depth_mle >= ALLELE_DEPTH_CUTOFF) or (
                        allele_depth_observe >= ALLELE_DEPTH_CUTOFF
                    )):
                        for (
                            allele_idx_reverse
                        ) in sort_mle_hap_depth_alleles_reverse:
                            prior_allele_seq = self.alleles_used_for_gt[
                                allele_idx_reverse
                            ][1]
                            if len(allele_seq) != len(prior_allele_seq):
                                edit_distance = np.inf
                                mis_location = "NA"
                            else:
                                (
                                    edit_distance,
                                    mis_location,
                                ) = myutils.cal_edit_distance_based_on_mismatch(
                                    allele_seq, prior_allele_seq
                                )
                            if edit_distance == 1:
                                best_source_allele = prior_allele_seq
                                best_source_id = allele_idx_reverse
                                best_mis_location = mis_location
                                allele_source_dict[allele_tuple] = tuple(["", best_source_allele, ""])
                                allele_source_edit_dis_dict[allele_tuple] = edit_distance
                                for i in best_mis_location:
                                    if myutils.approximation_flanking_first_bp_check(i, len(allele_seq)):
                                        flanking_first_bp = True
                                        break
                                    if myutils.approximation_flanking_mismatch_check(i, len(allele_seq)):
                                        flanking_mismatch = True
                                        break
                                edit_distance_one = True
                                break
                            else:
                                prior_allele_id_list.append(allele_idx_reverse)
                                edit_distance_list.append(edit_distance)
                                mis_location_list.append(mis_location)
                        if best_source_allele != "":
                            pass
                        else:
                            if edit_distance_list == []:
                                pass
                            else:
                                if min(edit_distance_list) <= MIS_NUM_TOLERENT:
                                    min_edit_distance_index = np.argmin(
                                        edit_distance_list
                                    )
                                    best_source_allele = (
                                        self.alleles_used_for_gt[
                                            prior_allele_id_list[
                                                min_edit_distance_index
                                            ]
                                        ][1]
                                    )
                                    best_source_id = prior_allele_id_list[
                                        min_edit_distance_index
                                    ]
                                    best_mis_location = mis_location_list[
                                        min_edit_distance_index
                                    ]
                                    allele_source_dict[allele_tuple] = tuple(["", best_source_allele, ""])
                                    allele_source_edit_dis_dict[allele_tuple] = edit_distance_list[
                                        min_edit_distance_index
                                    ]
                                    for i in best_mis_location:
                                        if myutils.approximation_flanking_first_bp_check(i, len(allele_seq)):
                                            flanking_first_bp = True
                                            break
                                        if myutils.approximation_flanking_mismatch_check(i, len(allele_seq)):
                                            flanking_mismatch = True
                                            break
                                else:
                                    pass
                        allele_reads_list = self.alleles_pysam_read_dict.get(
                            allele_tuple, []
                        )
                        allele_strand_list = [
                            i.is_reverse for i in allele_reads_list
                        ]
                        if allele_reads_list == []:
                            observed_strand_fraction = "NA"
                            observed_average_accuracy = "NA"
                        else:
                            observed_strand_fraction = sum(
                                allele_strand_list
                            ) / len(allele_strand_list)
                            observed_strand_fraction = (
                                myutils.cal_undirectional_value(
                                    observed_strand_fraction
                                )
                            )
                            if best_source_allele == "":
                                observed_average_accuracy = "NA"
                            else:
                                observed_mis_accuracy_list = []
                                for mis_loc in best_mis_location:
                                    mis_loc_base_accuracy_list = [
                                        accuracy_to_phred(allele_accuracy[1][mis_loc])
                                        for allele_accuracy in allele_accuracy_list
                                    ]
                                    observed_mis_accuracy_list.extend(
                                        mis_loc_base_accuracy_list
                                    )
                                if observed_mis_accuracy_list == []:
                                    observed_average_accuracy = "NA"
                                else:
                                    observed_average_accuracy = np.mean(
                                        observed_mis_accuracy_list
                                    )
                        mle_read_order_list = haplotype_depth_array[
                            :, allele_idx
                        ]
                        mle_read_order_selected_list = np.where(
                            mle_read_order_list == 1
                        )[0].tolist()
                        mle_strand_list = [
                            self.pysam_reads_list[i].is_reverse
                            for i in mle_read_order_selected_list
                        ]
                        mle_seq_list = [
                            self.reads_list[i][1]
                            for i in mle_read_order_selected_list
                        ]
                        mle_accuracy_list = [
                            self.reads_accuracy_list[i][1]
                            for i in mle_read_order_selected_list
                        ]
                        if mle_read_order_selected_list == []:
                            mle_strand_fraction = "NA"
                            mle_average_accuracy = "NA"
                        else:
                            mle_strand_fraction = sum(mle_strand_list) / len(
                                mle_strand_list
                            )
                            mle_strand_fraction = (
                                myutils.cal_undirectional_value(
                                    mle_strand_fraction
                                )
                            )
                            if best_source_allele == "":
                                mle_average_accuracy = "NA"
                            else:
                                mle_mis_accuracy_list = []
                                for read_seq, read_accuracy in zip(
                                    mle_seq_list, mle_accuracy_list
                                ):
                                    (
                                        log_str_align_prob,
                                        mis_score,
                                        max_alignment_length,
                                        base_accuracy_list,
                                    ) = myutils.cal_per_read_log_likelihood_given_allele_seq_based(
                                        allele_seq,
                                        read_seq,
                                        read_accuracy,
                                        self.log_no_stutter_lik,
                                        self.locus_infos["motif_length"],
                                        self.inframe_stutter_model_dict,
                                        self.outframe_stutter_model_dict,
                                        best_mis_location,
                                    )
                                    mle_mis_accuracy_list.extend(
                                        [accuracy_to_phred(bq_acc) for bq_acc in base_accuracy_list]
                                    )
                                if mle_mis_accuracy_list != []:
                                    mle_average_accuracy = np.mean(
                                        mle_mis_accuracy_list
                                    )
                                else:
                                    mle_average_accuracy = "NA"
                        # BaseQ and mis location, baseQ can use but mis location hard to locate specially for no representative hap reads, so only use baseQ
                        # What about final allele ?
                        # 如何扫描和过滤？
                        # 先看全局的 strand bias，假如全局 strand bias 很大，则 strand bias 难以过滤，只用 baseQ 过滤
                        # 最大 bundle 的 allele 无论如何都要保留
                        # 其余可以过滤根据 strand fraction 和 mis loc accuracy baseQ
                        # reads 大于等于 5 条可以 strand fraction 因为可能性 (1/2)^5 = 1/32 = 0.03125 小于 0.05
                        # reads 大于等于 5 的非最后 allele 假如 obs strand fraction = 1 和 mle strand fraction > 0.9 则过滤
                        # reads 小于 5 的非最后 allele 假如 obs strand fraction = 1 和 mle strand fraction > 0.9 则过滤
                        # accuracy 小于 Q13/Q20
                        # 需不需要使用全局 baseQ 指标
                        # 所以现在的规则: 没有 baseQ 只有 strand，需要 strand 在 深度在 5 以上，且有 strand bias
                        # 有 baseQ 和 strand，需要 strand 在深度在 5 以上，且 strand bias 或者 baseQ 有问题
                        # 什么规则对于 "NA" 和 阀值进行进一步思考和代码书写
                        # 全局有 strand bias 的话

                        if overall_strand_fraction >= OVERALL_SF_CUTOFF:  # HACK 在观察中发现只要聚焦 mutant allele 的 strand 就可以了，全局的 strand 不必管得太多，所以改为 0.5
                            use_obs_strand_filter = False
                            use_mle_strand_filter = False
                        else:
                            if (
                                allele_depth_observe
                                >= STRAND_BIAS_DEPTH_CUTOFF
                            ):
                                use_obs_strand_filter = True
                            else:
                                use_obs_strand_filter = False
                            if allele_depth_mle >= STRAND_BIAS_DEPTH_CUTOFF:
                                use_mle_strand_filter = True
                            else:
                                use_mle_strand_filter = False
                        if (flanking_first_bp or motif_and_str_length_judge):  # TODO: 这边条件设置还是挺松散的，但是即使这么松散，还是留了下大量假阳性的 mismatch artifacts，所以暂时没必要把条件放宽哈
                            mean_baseq_cutoff = BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR
                            fraction_baseq_cutoff = BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR
                        elif flanking_mismatch:  # HACK: Because flanking mismatches artifacts are enrichment still in mosaic output, so use a higher baseQ cutoff and more strict hard cut-off, Done!
                            mean_baseq_cutoff = BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR
                            fraction_baseq_cutoff = BASEQ_ACCURACY_CUTOFF_FOR_HIGH_ERROR
                        else:
                            mean_baseq_cutoff = BASEQ_ACCURACY_CUTOFF
                            fraction_baseq_cutoff = BASEQ_ACCURACY_CUTOFF
                        # HACK: direct low baseQ filter 还是需要低于 13, 目前不设置低于 20 的过滤, 仍然使用一个保守的过滤条件  # TODO List
                        if observed_average_accuracy != "NA":
                            low_accuracy_fraction_check = myutils.check_low_accuracy_fraction_check(observed_mis_accuracy_list, fraction_baseq_cutoff)
                            low_mean_accuracy_check = (observed_average_accuracy <= mean_baseq_cutoff)
                            if (
                                low_mean_accuracy_check or low_accuracy_fraction_check
                            ):
                                baseq_filter_allele = True
                            else:
                                baseq_filter_allele = False
                            very_low_mean_accuracy_check = (observed_average_accuracy <= BASEQ_ACCURACY_CUTOFF)
                            very_low_accuracy_fraction_check = myutils.check_low_accuracy_fraction_check(observed_mis_accuracy_list, BASEQ_ACCURACY_CUTOFF)
                            if (
                                very_low_mean_accuracy_check
                                or very_low_accuracy_fraction_check
                            ):
                                only_baseq_filter_allele = True
                            else:
                                only_baseq_filter_allele = False
                        elif mle_average_accuracy != "NA":
                            low_accuracy_fraction_check = myutils.check_low_accuracy_fraction_check(mle_mis_accuracy_list, fraction_baseq_cutoff)
                            low_mean_accuracy_check = (mle_average_accuracy <= mean_baseq_cutoff)
                            if (
                                low_mean_accuracy_check or low_accuracy_fraction_check
                            ):
                                baseq_filter_allele = True
                            else:
                                baseq_filter_allele = False
                            very_low_mean_accuracy_check = (mle_average_accuracy <= BASEQ_ACCURACY_CUTOFF)
                            very_low_accuracy_fraction_check = myutils.check_low_accuracy_fraction_check(mle_mis_accuracy_list, BASEQ_ACCURACY_CUTOFF)
                            if (
                                very_low_mean_accuracy_check
                                or very_low_accuracy_fraction_check
                            ):
                                only_baseq_filter_allele = True
                            else:
                                only_baseq_filter_allele = False
                        else:
                            baseq_filter_allele = "NA"
                            only_baseq_filter_allele = False
                        if baseq_filter_allele == "NA":
                            if (best_source_allele == ""):  # XXX 没有来源的 allele 的时候，就保留这个 allele just DEBUG For missing germline allele
                                strand_filter_allele = False
                            else:
                                if use_obs_strand_filter:
                                    if observed_strand_fraction != "NA":
                                        if (
                                            observed_strand_fraction
                                            >= OBSERVE_SF_CUTOFF
                                        ):
                                            strand_filter_allele = True
                                        else:
                                            strand_filter_allele = False
                                    else:
                                        if use_mle_strand_filter:
                                            if mle_strand_fraction != "NA":
                                                if (
                                                    mle_strand_fraction
                                                    >= MLE_SF_CUTOFF
                                                ):
                                                    strand_filter_allele = True
                                                else:
                                                    strand_filter_allele = False
                                            else:
                                                strand_filter_allele = False
                                        else:
                                            strand_filter_allele = False
                                elif use_mle_strand_filter:
                                    if mle_strand_fraction != "NA":
                                        if mle_strand_fraction >= MLE_SF_CUTOFF:
                                            if observed_strand_fraction != "NA":
                                                if (
                                                    observed_strand_fraction
                                                    >= OBSERVE_SF_CUTOFF
                                                ):
                                                    strand_filter_allele = True
                                                else:
                                                    strand_filter_allele = False
                                            else:
                                                strand_filter_allele = True
                                        else:
                                            strand_filter_allele = False
                                    else:
                                        strand_filter_allele = False
                                else:
                                    strand_filter_allele = False
                        elif baseq_filter_allele:
                            if only_baseq_filter_allele:
                                strand_filter_allele = True
                            else:
                                if observed_strand_fraction != "NA":
                                    if (
                                        observed_strand_fraction
                                        >= OBSERVE_SF_CUTOFF
                                    ):
                                        strand_filter_allele = True
                                    else:
                                        strand_filter_allele = False
                                elif mle_strand_fraction != "NA":
                                    if mle_strand_fraction >= MLE_SF_CUTOFF:
                                        strand_filter_allele = True
                                    else:
                                        strand_filter_allele = False
                                else:
                                    strand_filter_allele = True  # HACK: 质量低，没有 strand 证据来支持也可以进行过滤
                        else:
                            # HACK: 质量很高的时候满足特定的条件需不需要进行过滤呢? 这边还是高门槛条件过滤 #
                            # strand_filter_allele = False
                            if (best_source_allele == ""):  # XXX 没有来源的 allele 的时候，就保留这个 allele just DEBUG For missing germline allele
                                strand_filter_allele = False
                            else:  # HACK: 有来源 alleles 质量很高时的过滤条件，只过滤 post-homopolymer issues 情况, 目前仍然使用一个保守的过滤条件 # TODO List
                                if (edit_distance_one and flanking_first_bp and (self.locus_infos["motif_length"] == 1)):
                                    if use_obs_strand_filter:
                                        if observed_strand_fraction != "NA":
                                            if (
                                                observed_strand_fraction
                                                >= OBSERVE_SF_CUTOFF
                                            ):
                                                strand_filter_allele = True
                                            else:
                                                strand_filter_allele = False
                                        else:
                                            if use_mle_strand_filter:
                                                if mle_strand_fraction != "NA":
                                                    if (
                                                        mle_strand_fraction
                                                        >= MLE_SF_CUTOFF
                                                    ):
                                                        strand_filter_allele = True
                                                    else:
                                                        strand_filter_allele = False
                                                else:
                                                    strand_filter_allele = False
                                            else:
                                                strand_filter_allele = False
                                    elif use_mle_strand_filter:
                                        if mle_strand_fraction != "NA":
                                            if mle_strand_fraction >= MLE_SF_CUTOFF:
                                                if observed_strand_fraction != "NA":
                                                    if (
                                                        observed_strand_fraction
                                                        >= OBSERVE_SF_CUTOFF
                                                    ):
                                                        strand_filter_allele = True
                                                    else:
                                                        strand_filter_allele = False
                                                else:
                                                    strand_filter_allele = True
                                            else:
                                                strand_filter_allele = False
                                        else:
                                            strand_filter_allele = False
                                    else:
                                        strand_filter_allele = False
                                else:
                                    strand_filter_allele = False
                        if strand_filter_allele:
                            removed_allele_list.append(allele_tuple)
                        else:
                            final_alleles_used_for_gt.append(allele_tuple)
                            final_alleles_lik_array_index.append(allele_idx)
                            allele_mle_depth_list.append(allele_depth_mle)
                            allele_obs_depth_list.append(allele_depth_observe)
                    else:
                        continue
                if ALWAYS_TRUE_DEPTHEST_ALLELE:
                    depthest_allele_index = sort_mle_hap_depth_alleles[-1]
                    depthest_allele = self.alleles_used_for_gt[
                        depthest_allele_index
                    ]
                    if depthest_allele in final_alleles_used_for_gt:
                        pass
                    else:
                        final_alleles_used_for_gt.append(depthest_allele)
                        final_alleles_lik_array_index.append(
                            depthest_allele_index
                        )
                        allele_mle_depth_list.append(
                            hap_depth_list[depthest_allele_index]
                        )
                        allele_obs_depth_list.append(
                            self.alleles_depth_dict.get(depthest_allele, 0)
                        )
                else:
                    pass
                self.alleles_used_for_gt = final_alleles_used_for_gt
                self.all_reads_all_alleles_log_lik_array = (
                    self.all_reads_all_alleles_log_lik_array[
                        :, final_alleles_lik_array_index
                    ]
                )
                if SOURCE_CHECK:
                    final_used_allele_list, used_index_list = myutils.source_remove_checked(self.alleles_used_for_gt, removed_allele_list, allele_source_dict, allele_mle_depth_list, allele_obs_depth_list, allele_source_edit_dis_dict)
                    self.alleles_used_for_gt = final_used_allele_list
                    self.all_reads_all_alleles_log_lik_array = (
                        self.all_reads_all_alleles_log_lik_array[
                            :, used_index_list
                        ]
                    )
            else:
                if len(self.alleles_used_for_gt) == 0:
                    self.alleles_used_for_gt = alleles_used_for_gt
                    self.all_singleton_flag = True
                    # final_alleles_used_for_gt = self.alleles_used_for_gt
                else:
                    pass
                    # final_alleles_used_for_gt = self.alleles_used_for_gt
            # self.extra_consensus_allele_from_poa = list(
            #     set(final_alleles_used_for_gt) - set(self.alleles_order)
            # )
            # self.all_alleles_list_for_gt_output.extend(
            #     self.extra_consensus_allele_from_poa
            # )
        else:
            # pass
            # self.extra_consensus_allele_from_poa = []
            self.alleles_used_for_gt = []
            self.singleton_alleles_number = 0
            self.allele_with_depth_more_than_n_allele_number = 0
            allele_with_depth_more_than_n_list = []
            singleton_allele_list = []
            for allele in alleles_used_for_gt:
                if (
                    self.alleles_depth_dict.get(allele, 0)
                    >= ALLELE_DEPTH_CUTOFF
                ):
                    self.alleles_used_for_gt.append(allele)
                    allele_with_depth_more_than_n_list.append(allele)
                    self.allele_with_depth_more_than_n_allele_number += 1
                else:
                    singleton_allele_list.append(allele[1])
                    self.singleton_alleles_number += 1
        # allele list for gt index ok
        # allele sort and new list ok
        # mle haplotype depth array and # TODO, Done!
        # final gt mle allele depth for unphased genotypes # TODO, Done!
        # replication calculation for likelihood # TODO, Done!
        # self.alleles_used_for_gt ok
        # final_alleles_used_for_gt ok
        # MAX_ALLOWABLE_ALLELE_NUM 8 ok
        # if len(self.alleles_used_for_gt) <= MAX_ALLOWABLE_ALLELE_NUM:
        #     # self.alleles_used_for_gt = alleles_used_for_gt
        #     return
        # if BINOMINAL_FILTER:
        #     for allele, allele_depth in self.alleles_depth_dict.items():
        #         pvalue = stats.binomtest(
        #             allele_depth,
        #             n=self.reads_number,
        #             p=error_rate,
        #             alternative="less",
        #         ).pvalue
        #         if pvalue > 0.05:
        #             continue
        #             # break
        #         else:
        #             self.alleles_used_for_gt.remove(allele)
        # filter alleles to max allowable allele number
        # But keep all allele reads
        # alleles_used_for_gt = self.alleles_order[binominal_filter_number:]
        # remove_allele = self.alleles_order[:binominal_filter_number]
        # alleles_seq_used_for_gt = self.alleles_seq_used_for_gt[binominal_filter_number:]
        if len(self.alleles_used_for_gt) > MAX_ALLOWABLE_ALLELE_NUM:
            remove_allele_num = len(self.alleles_used_for_gt) - MAX_ALLOWABLE_ALLELE_NUM
            self.alleles_used_for_gt = self.alleles_used_for_gt[
                remove_allele_num:
            ]
            if HAP_IDENTIFY_POA and (LIKELIHOOD_MODE != "length-based"):
                if self.all_singleton_flag:
                    pass
                else:
                    self.all_reads_all_alleles_log_lik_array = (
                        self.all_reads_all_alleles_log_lik_array[
                            :, remove_allele_num:
                        ]
                    )
            # self.alleles_seq_used_for_gt = alleles_seq_used_for_gt[
            #    len(alleles_seq_used_for_gt) - MAX_ALLOWABLE_ALLELE_NUM :
            # ]
        else:
            pass
            # self.alleles_used_for_gt = alleles_used_for_gt
            # self.alleles_seq_used_for_gt = alleles_seq_used_for_gt
        self.extra_consensus_allele_from_poa = list(
                set(self.alleles_used_for_gt) - set(self.alleles_order)
            )
        self.all_alleles_list_for_gt_output.extend(
            self.extra_consensus_allele_from_poa
        )

    def check_loci_callable(self):
        # TODO: Add Max Coverage and Min Coverage Filter, Done #
        coverage_state = (self.reads_number < MIN_AVERAGE_COVERAGE) or (
            self.reads_number > MAX_LOCI_DEPTH
        )
        alleles_number_state = self.alleles_number_used_for_gt <= 1
        if coverage_state:
            self.loci_calling = "CoverageFail"
            self.callable = False
        elif alleles_number_state:
            self.loci_calling = "Germline"
            self.callable = False
        # elif self.alleles_number_used_for_gt > MAX_ALLELE_NUM:
        #     self.loci_calling = "TooManyAllele"
        #     self.callable = False
        else:
            self.loci_calling = "CandidateMosaic"
            self.callable = True
        if self.all_singleton_flag:
            self.singleton_judge = "all_singleton_obs_allele"
        else:
            self.singleton_judge = "not_all_singleton_obs_allele"

    def all_reads_all_alleles_log_lik(self):
        self.all_reads_all_alleles_log_lik_array = np.zeros(
            (len(self.reads_list), len(self.alleles_used_for_gt)),
            dtype=np.float64,
        )
        if LIKELIHOOD_MODE == "length-based":
            for read_index, read in enumerate(self.reads_list):
                for allele_index, allele in enumerate(
                    self.alleles_used_for_gt
                ):
                    self.all_reads_all_alleles_log_lik_array[
                        read_index, allele_index
                    ] = self.cal_per_read_log_likelihood_given_allele_len_based(
                        allele, read
                    )
        else:
            for read_index, read in enumerate(self.reads_list):
                for allele_index, allele in enumerate(
                    self.alleles_used_for_gt
                ):
                    self.all_reads_all_alleles_log_lik_array[
                        read_index, allele_index
                    ] = self.cal_per_read_log_likelihood_given_allele_seq_based(
                        allele, read, self.reads_accuracy_list[read_index]
                    )
            pass  # TODO

    def cal_per_read_log_likelihood_given_allele_len_based(
        self, allele_str_length, read_str_length
    ):
        bp_diff = read_str_length - allele_str_length
        if bp_diff == 0.0:
            per_read_log_ll = self.log_no_stutter_lik
        else:
            quotient, remainder = divmod(
                abs(bp_diff), self.locus_infos["motif_length"]
            )
            if remainder == 0.0:
                if bp_diff > 0.0:
                    per_read_log_ll = self.inframe_stutter_model_dict.get(
                        int((bp_diff) / self.locus_infos["motif_length"]), self.inframe_stutter_model_dict[max(self.inframe_stutter_model_dict.keys())]
                    )
                else:
                    per_read_log_ll = self.inframe_stutter_model_dict.get(
                        int((bp_diff) / self.locus_infos["motif_length"]), self.inframe_stutter_model_dict[min(self.inframe_stutter_model_dict.keys())]
                    )
            else:
                if bp_diff > 0.0:
                    bp_diff_2_continuous_value = bp_diff - quotient
                    per_read_log_ll = self.outframe_stutter_model_dict.get(
                        bp_diff_2_continuous_value, self.outframe_stutter_model_dict[max(self.outframe_stutter_model_dict.keys())]
                    )
                else:
                    bp_diff_2_continuous_value = bp_diff + quotient
                    per_read_log_ll = self.outframe_stutter_model_dict.get(
                        bp_diff_2_continuous_value, self.outframe_stutter_model_dict[min(self.outframe_stutter_model_dict.keys())]
                    )
        return per_read_log_ll

    def cal_per_read_log_likelihood_given_allele_seq_based(
        self, allele_seq, read_seq, read_accuracy
    ):
        # XXX:(-4)%3 = 2
        bp_diff = len(read_seq[1]) - len(allele_seq[1])
        if bp_diff == 0.0:
            log_str_align_prob = hap_alignment.no_stutter_str_alignment(
                allele_seq[1],
                read_seq[1],
                read_accuracy[1],
                self.log_no_stutter_lik,
            )
        else:
            quotient, remainder = divmod(
                abs(bp_diff), self.locus_infos["motif_length"]
            )
            if remainder == 0.0:
                steps = int((bp_diff) / self.locus_infos["motif_length"])
                if bp_diff > 0:
                    log_str_align_prob = hap_alignment.stutter_insertion(
                        allele_seq[1],
                        read_seq[1],
                        read_accuracy[1],
                        steps,
                        self.inframe_stutter_model_dict
                        # ins_rate,
                        # p_geom,
                        # self.locus_infos["motif_length"]
                    )
                else:
                    log_str_align_prob = hap_alignment.stutter_deletion(
                        # in_or_out_frame,
                        allele_seq[1],
                        read_seq[1],
                        read_accuracy[1],
                        steps,
                        self.inframe_stutter_model_dict
                        # del_rate,
                        # p_geom,
                        # self.locus_infos["motif_length"]
                    )
            else:
                if bp_diff > 0:
                    steps = bp_diff - quotient
                    log_str_align_prob = hap_alignment.stutter_insertion(
                        allele_seq[1],
                        read_seq[1],
                        read_accuracy[1],
                        steps,
                        self.outframe_stutter_model_dict
                        # ins_rate,
                        # p_geom,
                        # self.locus_infos["motif_length"]
                    )
                else:
                    steps = bp_diff + quotient
                    log_str_align_prob = hap_alignment.stutter_deletion(
                        # in_or_out_frame,
                        allele_seq[1],
                        read_seq[1],
                        read_accuracy[1],
                        steps,
                        self.outframe_stutter_model_dict
                        # del_rate,
                        # p_geom,
                        # self.locus_infos["motif_length"]
                    )
        # if abs(len(allele_seq[1]) - len(read_seq[1]))%self.locus_infos["motif_length"]==0:
        #     log_stutter_model_dic = self.inframe_stutter_model_dict
        # else:
        #     log_stutter_model_dic = self.outframe_stutter_model_dict
        # if len(allele_seq[1]) > len(read_seq[1]):

        #     log_str_align_prob = hap_alignment.stutter_deletion(
        #         # in_or_out_frame,
        #         allele_seq[1],
        #         read_seq[1],
        #         read_accuracy[1],

        #         # del_rate,
        #         # p_geom,
        #         # self.locus_infos["motif_length"]
        #     )
        # elif len(allele_seq[1]) == len(read_seq[1]):

        # elif len(allele_seq[1]) < len(read_seq[1]):
        #     log_str_align_prob = hap_alignment.stutter_insertion(
        #         in_or_out_frame,
        #         allele_seq[1],
        #         read_seq[1],
        #         read_accuracy[1],
        #         ins_rate,
        #         p_geom,
        #         self.locus_infos["motif_length"]
        #     )
        if ADD_FLANKING_PROB:
            log_left_flk_align_prob = hap_alignment.align_flk(
                allele_seq[0], read_seq[0], tuple(read_accuracy[0])
            )
            log_right_flk_align_prob = hap_alignment.align_flk(
                allele_seq[2], read_seq[2], tuple(read_accuracy[2])
            )
        else:
            log_left_flk_align_prob = 0
            log_right_flk_align_prob = 0
        per_read_log_ll = (
            log_str_align_prob
            + log_left_flk_align_prob
            + log_right_flk_align_prob
        )
        return per_read_log_ll

    def all_reads_all_gts_log_lik(self):
        self.all_reads_all_gts_log_lik_array = np.zeros(
            (len(self.reads_list), len(self.genotypes)), dtype=np.float64
        )
        for read_index, read in enumerate(self.reads_list):
            for gt_index, gt in enumerate(self.genotypes):
                self.all_reads_all_gts_log_lik_array[
                    read_index, gt_index
                ] = self.cal_per_read_given_gt_log_lik(read_index, gt)
        # XXX: per-read likelihood calculation and gt likelihood calculation
        self.all_gts_log_lik_array = np.sum(
            self.all_reads_all_gts_log_lik_array, axis=0
        )

    def cal_per_read_given_gt_log_lik(self, read_index, gt):
        read_allele1_log_lik = self.all_reads_all_alleles_log_lik_array[
            read_index, gt[0]
        ]
        read_allele2_log_lik = self.all_reads_all_alleles_log_lik_array[
            read_index, gt[1]
        ]
        read_gt_log_lik = logsumexp(
            [read_allele1_log_lik, read_allele2_log_lik], b=[0.5, 0.5]
        )
        return read_gt_log_lik

    def cal_mutation_rate_prior(self):
        self.log_mutation_rate_prior = []
        self.log_mutation_rate_prior_with_germ2germ_rate_array = []
        self.mosaic_gts_types = []
        for germ_cells, mut_cells in self.mosaic_genotypes:
            if germ_cells == mut_cells:
                self.log_mutation_rate_prior.append(
                    GERM_2_GERM_MUTATION_RATE
                )  # HACK:Non-mutation use zero prior -np.inf rather than NON_MUTATION_RATE
                self.log_mutation_rate_prior_with_germ2germ_rate_array.append(
                    NON_MUTATION_RATE
                )
                self.mosaic_gts_types.append("germ2germ")
            elif (
                len(
                    set(self.genotypes[germ_cells])
                    & set(self.genotypes[mut_cells])
                )
                == 1
            ):
                if len(set(self.genotypes[mut_cells])) == 1:
                    self.log_mutation_rate_prior.append(MS_MUTATION_RATE)
                    self.log_mutation_rate_prior_with_germ2germ_rate_array.append(
                        MS_MUTATION_RATE
                    )
                    self.mosaic_gts_types.append("het2hom")
                else:
                    self.log_mutation_rate_prior.append(MS_MUTATION_RATE)
                    self.log_mutation_rate_prior_with_germ2germ_rate_array.append(
                        MS_MUTATION_RATE
                    )
                    self.mosaic_gts_types.append("homhet2het")
            else:
                self.log_mutation_rate_prior.append(MS_MUTATION_RATE * 2)
                self.log_mutation_rate_prior_with_germ2germ_rate_array.append(
                    MS_MUTATION_RATE * 2
                )
                self.mosaic_gts_types.append("2mut")
        self.log_mutation_rate_prior_with_germ2germ_rate_array = np.array(
            self.log_mutation_rate_prior_with_germ2germ_rate_array
        )

    def get_different_levels_mosaic_list(self):
        self.homhet_list = []
        self.hethet_list = []
        self.germ2germ_list = []
        self.hom2hom_two_mut = []
        for mosaic_index, cells in enumerate(self.mosaic_genotypes):
            germ_cells_index = cells[0]
            mut_cells_index = cells[1]
            germ_cells_gt = self.genotypes[germ_cells_index]
            mut_cells_gt = self.genotypes[mut_cells_index]
            if germ_cells_gt == mut_cells_gt:
                self.germ2germ_list.append(mosaic_index)
            elif (len(set(germ_cells_gt)) + len(set(mut_cells_gt))) == 2:
                self.hom2hom_two_mut.append(mosaic_index)
            elif (len(set(germ_cells_gt)) + len(set(mut_cells_gt))) == 3:
                self.homhet_list.append(mosaic_index)
            else:
                self.hethet_list.append(mosaic_index)

    def mosaic_fraction_estimation(self):
        if ESTIMATION_MODE == "unique":
            self.train_single_mosaic_fraction_mode()
            if self.locus_state == "CoverageFail":
                self.loci_calling = "CoverageFail"
                self.posterior_cal_logic = False
                self.estimated_mosaic_fraction = 0
            elif "Alleles" in self.locus_state:
                self.loci_calling = "Germline"
                self.posterior_cal_logic = False
                self.estimated_mosaic_fraction = 0
            elif self.locus_state == "HasConverged":
                if (
                    abs(self.mosaic_fraction - 1)
                    < NUMERIC_CALCULATION_TOLERANCE
                ) or (
                    abs(self.mosaic_fraction - 0)
                    < NUMERIC_CALCULATION_TOLERANCE
                ):
                    # self.loci_calling = "Germline"
                    # self.posterior_cal_logic = False
                    # self.estimated_mosaic_fraction = 0
                    self.loci_calling = "CandidateGermline"
                    self.posterior_cal_logic = True
                    self.estimated_mosaic_fraction = [
                        self.mosaic_fraction
                    ] * len(self.mosaic_genotypes)
                    self.mosaic_fraction_type = ["mosaic"] * len(
                        self.mosaic_genotypes
                    )
                else:
                    self.loci_calling = "CandidateMosaic"
                    self.posterior_cal_logic = True
                    self.estimated_mosaic_fraction = [
                        self.mosaic_fraction
                    ] * len(self.mosaic_genotypes)
                    self.mosaic_fraction_type = ["mosaic"] * len(
                        self.mosaic_genotypes
                    )
            else:
                self.loci_calling = self.locus_state
                self.posterior_cal_logic = False
                self.estimated_mosaic_fraction = 0
        elif ESTIMATION_MODE == "double":
            self.train_hom2het()
            self.train_het2het()
            print(f"hom2het_mosaic_fraction:{self.hom2het_mosaic_fraction}")
            print(f"het2het_mosaic_fraction:{self.het2het_mosaic_fraction}")
            if self.hom2het_locus_state == "CoverageFail":
                self.loci_calling = "CoverageFail"
                self.posterior_cal_logic = False
                self.estimated_mosaic_fraction = [0, 0]
            elif "Alleles" in self.hom2het_locus_state:
                self.loci_calling = "Germline"
                self.posterior_cal_logic = False
                self.estimated_mosaic_fraction = [0, 0]
            elif (self.hom2het_locus_state == "HasConverged") and (
                self.het2het_locus_state == "HasConverged"
            ):
                estimated_mosaic_fraction = [
                    self.hom2het_mosaic_fraction,
                    self.het2het_mosaic_fraction,
                ]
                (
                    mosaic_state,
                    self.estimated_mosaic_fraction,
                    self.mosaic_fraction_type,
                ) = self.check_double_mosaic_fraction(
                    estimated_mosaic_fraction
                )
                if mosaic_state == "germline":
                    self.loci_calling = "Germline"
                    self.posterior_cal_logic = False
                else:
                    self.loci_calling = "CandidateMosaic"
                    self.posterior_cal_logic = True
            elif (self.hom2het_locus_state == "HasConverged") or (
                self.het2het_locus_state == "HasConverged"
            ):  # XXX: Use converged mutation type mosaic_fraction
                if self.hom2het_locus_state == "HasConverged":
                    estimated_mosaic_fraction = [
                        self.hom2het_mosaic_fraction,
                        self.hom2het_mosaic_fraction,
                    ]
                elif self.het2het_locus_state == "HasConverged":
                    estimated_mosaic_fraction = [
                        self.het2het_mosaic_fraction,
                        self.het2het_mosaic_fraction,
                    ]
                (
                    mosaic_state,
                    self.estimated_mosaic_fraction,
                    self.mosaic_fraction_type,
                ) = self.check_double_mosaic_fraction(
                    estimated_mosaic_fraction
                )
                if mosaic_state == "germline":
                    self.posterior_cal_logic = False
                    self.loci_calling = f"hom2het:{self.hom2het_locus_state};het2het:{self.het2het_locus_state};Germline"
                else:
                    self.posterior_cal_logic = True
                    self.loci_calling = f"hom2het:{self.hom2het_locus_state};het2het:{self.het2het_locus_state};CandidateMosaic"
            else:
                self.loci_calling = f"hom2het:{self.hom2het_locus_state};het2het:{self.het2het_locus_state}"
                self.posterior_cal_logic = False
                self.estimated_mosaic_fraction = [
                    self.hom2het_mosaic_fraction,
                    self.het2het_mosaic_fraction,
                ]

        elif ESTIMATION_MODE == "allmle":
            (
                self.estimated_mosaic_fraction,
                self.mosaic_fraction_type,
            ) = self.cal_all_gts_trans_mosaic_fraction_based_on_mle()
            if "mosaic" in self.mosaic_fraction_type:
                self.posterior_cal_logic = True
                self.loci_calling = "CandidateMosaic"
            else:
                self.posterior_cal_logic = False
                self.loci_calling = "Germline"
        else:
            raise ValueError("Unknown estimation mode")

    def find_EM_initial_guess(self):
        initial_value = 1 / INITIAL_GUESS_NUMBER
        all_mosaic_log_prior_lik_total_list = []
        for i in np.arange(initial_value, 1, initial_value):
            all_mosaic_log_prior_lik_list = []
            for mosaic_index, cells in enumerate(self.mosaic_genotypes):
                germ_cells_index = cells[0]
                mut_cells_index = cells[1]
                germ_cells_gt = self.genotypes[germ_cells_index]
                mut_cells_gt = self.genotypes[mut_cells_index]
                if CONSIDER_GERM2GERM_IN_PARAMS_UPDATE:
                    pass
                else:
                    if germ_cells_gt == mut_cells_gt:
                        continue
                if ALLOWABLE_HET2HOM:
                    pass
                else:
                    if (len(set(mut_cells_gt))) == 1 and (
                        len(set(germ_cells_gt))
                    ) == 2:  # XXX
                        continue
                if ALLOWABLE_TWO_MUT_ALLELES:
                    pass
                else:
                    if len(set(germ_cells_gt) & set(mut_cells_gt)) == 0:
                        continue
                mosaic_fraction = i
                if USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GIandMutRate":
                    log_mosaic_prior_lik = (
                        self.all_gts_log_lik_array[germ_cells_index]
                        + self.log_mutation_rate_prior[mosaic_index]
                        + self.cal_mosaic_gt_log_lik(cells, mosaic_fraction)
                    )
                elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GI":
                    log_mosaic_prior_lik = self.all_gts_log_lik_array[
                        germ_cells_index
                    ] + self.cal_mosaic_gt_log_lik(cells, mosaic_fraction)
                elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "MutRate":
                    log_mosaic_prior_lik = self.log_mutation_rate_prior[
                        mosaic_index
                    ] + self.cal_mosaic_gt_log_lik(cells, mosaic_fraction)
                elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "OnlyLik":
                    log_mosaic_prior_lik = self.cal_mosaic_gt_log_lik(
                        cells, mosaic_fraction
                    )
                else:
                    logging.error("GI prior update mosaic fraction error.")
                    raise ValueError("GI prior update mosaic fraction error.")
                all_mosaic_log_prior_lik_list.append(log_mosaic_prior_lik)
            # 最大期然是乘起来还是加起来呢
            # 0.1 * 0.01 * 0.001，0.1 + 0.01 + 0.001
            # 0.2 * 0.01 * 0.0001，0.2 + 0.01 + 0.0001
            # 乘起来 0.2 的变更小，加起来的 0.2 的变更大
            # XXX: 看起来加起来更合理，MUTEA 是所有 allele 加起来，所有样本乘起来
            all_mosaic_log_prior_lik = logsumexp(all_mosaic_log_prior_lik_list)
            all_mosaic_log_prior_lik_total_list.append(
                all_mosaic_log_prior_lik
            )
        max_mosaic_initial_guess = np.argmax(
            all_mosaic_log_prior_lik_total_list
        )
        return initial_value * (max_mosaic_initial_guess + 1)

    def check_double_mosaic_fraction(self, mosaic_fraction):
        hom2het_mosaic_fraction = mosaic_fraction[0]
        het2het_mosaic_fraction = mosaic_fraction[1]
        collate_mosaic_fraction = []
        all_gts_trans_mosaic_fraction_type = []
        mosaic_fraction_type = {}
        if (
            (abs(hom2het_mosaic_fraction - 1) < NUMERIC_CALCULATION_TOLERANCE)
            or (
                abs(hom2het_mosaic_fraction - 0)
                < NUMERIC_CALCULATION_TOLERANCE
            )
        ) and (
            (abs(het2het_mosaic_fraction - 1) < NUMERIC_CALCULATION_TOLERANCE)
            or (
                abs(het2het_mosaic_fraction - 0)
                < NUMERIC_CALCULATION_TOLERANCE
            )
        ):
            mosaic_state = "germline"
        elif (
            (abs(hom2het_mosaic_fraction - 1) > NUMERIC_CALCULATION_TOLERANCE)
            or (
                abs(hom2het_mosaic_fraction - 0)
                > NUMERIC_CALCULATION_TOLERANCE
            )
        ) and (
            (abs(het2het_mosaic_fraction - 1) < NUMERIC_CALCULATION_TOLERANCE)
            or (
                abs(het2het_mosaic_fraction - 0)
                < NUMERIC_CALCULATION_TOLERANCE
            )
        ):
            mosaic_state = "mosaic"
            mosaic_fraction_type["hom2het_mosaic_fraction"] = "mosaic"
            mosaic_fraction_type[
                "het2het_mosaic_fraction"
            ] = "zero_one_mosaic_fraction"
        elif (
            (abs(hom2het_mosaic_fraction - 1) < NUMERIC_CALCULATION_TOLERANCE)
            or (
                abs(hom2het_mosaic_fraction - 0)
                < NUMERIC_CALCULATION_TOLERANCE
            )
        ) and (
            (abs(het2het_mosaic_fraction - 1) > NUMERIC_CALCULATION_TOLERANCE)
            or (
                abs(het2het_mosaic_fraction - 0)
                > NUMERIC_CALCULATION_TOLERANCE
            )
        ):
            mosaic_state = "mosaic"
            mosaic_fraction_type[
                "hom2het_mosaic_fraction"
            ] = "zero_one_mosaic_fraction"
            mosaic_fraction_type["het2het_mosaic_fraction"] = "mosaic"
        else:
            mosaic_state = "mosaic"
            mosaic_fraction_type["hom2het_mosaic_fraction"] = "mosaic"
            mosaic_fraction_type["het2het_mosaic_fraction"] = "mosaic"
        for mosaic_index, cells in enumerate(self.mosaic_genotypes):
            germ_cells_index = cells[0]
            mut_cells_index = cells[1]
            germ_cells_gt = self.genotypes[germ_cells_index]
            mut_cells_gt = self.genotypes[mut_cells_index]
            if germ_cells_gt == mut_cells_gt:
                collate_mosaic_fraction.append(0)
                all_gts_trans_mosaic_fraction_type.append("germline")
            elif (len(set(germ_cells_gt)) == 1) and (
                len(set(mut_cells_gt)) == 2
            ):
                collate_mosaic_fraction.append(hom2het_mosaic_fraction)
                all_gts_trans_mosaic_fraction_type.append(
                    mosaic_fraction_type["hom2het_mosaic_fraction"]
                )
            elif (len(set(germ_cells_gt)) == 2) and (
                len(set(mut_cells_gt)) == 1
            ):
                if ALLOWABLE_HET2HOM:
                    collate_mosaic_fraction.append(het2het_mosaic_fraction)
                    all_gts_trans_mosaic_fraction_type.append(
                        mosaic_fraction_type["het2het_mosaic_fraction"]
                    )
                else:
                    collate_mosaic_fraction.append(0)
                    all_gts_trans_mosaic_fraction_type.append("het2hom")
            elif (len(set(germ_cells_gt)) == 2) and (
                len(set(mut_cells_gt)) == 2
            ):
                collate_mosaic_fraction.append(het2het_mosaic_fraction)
                all_gts_trans_mosaic_fraction_type.append(
                    mosaic_fraction_type["het2het_mosaic_fraction"]
                )
            else:
                collate_mosaic_fraction.append(0)
                all_gts_trans_mosaic_fraction_type.append("hom2hom_two_mut")
        return (
            mosaic_state,
            collate_mosaic_fraction,
            all_gts_trans_mosaic_fraction_type,
        )

    def train_single_mosaic_fraction_mode(self):
        if self.alleles_number_used_for_gt <= 1:
            self.locus_state = f"{self.alleles_number_used_for_gt}Alleles"
            self.trained = True
            self.converged = True
            self.niter = 0
            self.mosaic_fraction = 0
            return
        if self.locus_state == "CoverageFail":
            logging.info(
                "This locus will not be trained due to insufficient coverage."
            )
            return
        elif self.trained and self.converged:
            logging.info(
                "We invoked the train function previously,"
                " and this locus has now converged."
            )
            return
        elif self.trained and (1 - self.converged):
            self.niter = MIN_ITERATION
            logging.info(
                "We invoked the train function previously,"
                " but this locus did not converge."
                " We will continue the training"
                " using the previous set of parameters."
            )
        else:
            logging.info(
                "The train function has not been invoked before,"
                " and we will train this locus from the beginning."
            )
            self.niter = 0
            self.total_mosaic_log_prior_lik = INIT_TOTAL_MOSAIC_LOG_PRIOR_LIK

            coverage_state = self.reads_number < MIN_AVERAGE_COVERAGE
        if coverage_state:
            logging.info(
                f"{self.locus_infos['STR_id']} has a sample depth less than"
                f" {MIN_AVERAGE_COVERAGE}"
            )
            self.locus_state = "CoverageFail"
            return
        if USE_INITIAL_GUESS:
            self.mosaic_fraction = self.find_EM_initial_guess()
        else:
            self.mosaic_fraction = INITIAL_MOSAIC_FRACTION
        while (self.niter < MIN_ITERATION) or (
            (not self.converged) and (self.niter < MAX_ITERATION)
        ):
            prev_total_mosaic_log_prior_lik = self.total_mosaic_log_prior_lik
            if ITERATION_MODE == "allele-based":  # XXX
                (
                    mosaic_fraction,
                    self.total_mosaic_log_prior_lik,
                    self.all_mosaic_log_prior_lik_list,
                    self.all_mosaic_gt_list,
                ) = self.update_mosaic_fraction_based_mut_allele()
            elif ITERATION_MODE == "cell-based":
                (
                    mosaic_fraction,
                    self.total_mosaic_log_prior_lik,
                    self.all_mosaic_log_prior_lik_list,
                    self.all_mosaic_gt_list,
                ) = self.update_mosaic_fraction_based_mut_cells()
            else:
                logging.error("Iteration mode error.")
                raise ValueError("Iteration mode error.")
            total_mosaic_log_prior_lik_diff = (
                self.total_mosaic_log_prior_lik
                - prev_total_mosaic_log_prior_lik
            )
            converged0 = total_mosaic_log_prior_lik_diff < MAX_TOLERANCE
            if converged0:  # XXX
                logging.warning(
                    "The total prior log-likelihood of"
                    f" {self.locus_infos['STR_id']} unexpectedly decreased to"
                    " less than 1/10."
                )
                self.converged = "LL_Decreased"
                self.converged_list = [converged0, "unknown", "unknown"]
                break
            prev_mosaic_fraction = self.mosaic_fraction
            if DEBUG:
                self.total_mosaic_log_prior_lik_list.append(
                    self.total_mosaic_log_prior_lik
                )
                print(f"iter:{self.niter}")
                print(mosaic_fraction)
            self.mosaic_fraction = mosaic_fraction
            if USE_MINOR_CELLS_POP_AS_MUT_CELLS:  # XXX
                if self.mosaic_fraction < 0 or self.mosaic_fraction > 0.5:
                    logging.warning(
                        "The mosaic fraction of"
                        f" {self.locus_infos['STR_id']} is less than 0 or"
                        " greater than 0.5."
                    )
                    self.converged = "MosaicFraction_OutOfRange"
                    self.converged_list = [converged0, "unknown", "unknown"]
                    break
            else:
                if self.mosaic_fraction < 0 or self.mosaic_fraction > 1:
                    logging.warning(
                        "The mosaic fraction of"
                        f" {self.locus_infos['STR_id']} is less than 0 or"
                        " greater than 1."
                    )
                    self.converged = "MosaicFraction_OutOfRange"
                    self.converged_list = [converged0, "unknown", "unknown"]
                    break
            if self.niter % CHECK_CONVERGE_PERIOD == 0:
                mosaic_fraction_diff = abs(
                    self.mosaic_fraction - prev_mosaic_fraction
                )
                converged1 = (
                    (
                        (-total_mosaic_log_prior_lik_diff)
                        / prev_total_mosaic_log_prior_lik
                    )
                    < DIFFERENCE_CONVERGE
                ) and (total_mosaic_log_prior_lik_diff < DIFFERENCE_CONVERGE)
                converged2 = mosaic_fraction_diff < DIFFERENCE_CONVERGE_MosaicFrac
                # BUG: Check converge condition and converged0 should be in the before of stutter params updating and break in time
                # HAS SOLVED
                self.converged = converged1 or converged2
                self.converged_list = [converged0, converged1, converged2]
            self.niter += 1
            self.total_niter += 1
        self.n_train += 1
        self.trained = True
        if self.converged == "LL_Decreased":
            logging.info(
                f"{self.locus_infos['STR_id']}: The total mosaic prior"
                " log-likelihood unexpectedly decreased to less than 1/10."
            )
            self.locus_state = "LLDecreased"
        elif self.converged == "MosaicFraction_OutOfRange":
            if USE_MINOR_CELLS_POP_AS_MUT_CELLS:
                logging.info(
                    f"{self.locus_infos['STR_id']}: The mosaic fraction is"
                    " less than 0 or greater than 0.5."
                )
            else:
                logging.info(
                    f"{self.locus_infos['STR_id']}: The mosaic fraction is"
                    " less than 0 or greater than 1."
                )
            self.locus_state = "MosaicFractionOutOfRange"
        elif (
            self.converged == True
        ):  # BUG numpy.bool_ don't support 'is' operator: HAS SOLVED
            logging.info(f"{self.locus_infos['STR_id']} has converged.")
            self.locus_state = "HasConverged"
        else:
            logging.info(
                f"{self.locus_infos['STR_id']}: Maximum iteration reached."
            )
            self.locus_state = "MaxIterationReached"
        if DEBUG:
            print(f"Current alleles depth dict: {self.alleles_depth_dict}")
            max_mosaic_index = np.argmax(self.all_mosaic_log_prior_lik_list)
            max_mosaic_gt = self.all_mosaic_gt_list[max_mosaic_index]
            print(
                "Max mosaic"
                f" transformation:{self.mosaic_genotypes[max_mosaic_gt]}"
            )
            print(f"GT:{self.genotypes}")
            print(f"Alleles:{self.alleles_used_for_gt}")
            print(
                f"All_mosaic_log_prior_lik_list:{self.all_mosaic_log_prior_lik_list}"
            )

    def train_hom2het(self):
        if self.alleles_number_used_for_gt <= 1:
            self.hom2het_locus_state = (
                f"{self.alleles_number_used_for_gt}Alleles"
            )
            self.hom2het_trained = True
            self.hom2het_converged = True
            self.hom2het_niter = 0
            self.hom2het_mosaic_fraction = 0
            return
        if self.hom2het_locus_state == "CoverageFail":
            logging.info(
                "This locus will not be trained due to insufficient coverage."
            )
            return
        elif self.hom2het_trained and self.hom2het_converged:
            logging.info(
                "We invoked the train function previously,"
                " and this locus has now converged for hom2het."
            )
            return
        elif self.hom2het_trained and (1 - self.hom2het_converged):
            self.hom2het_niter = MIN_ITERATION
            logging.info(
                "We invoked the train function previously,"
                " but this locus did not converge."
                " We will continue the training"
                " using the previous set of parameters."
            )
        else:
            logging.info(
                "The train function has not been invoked before,"
                " and we will train this locus from the beginning."
            )
            self.hom2het_niter = 0
            self.hom2het_total_mosaic_log_prior_lik = (
                INIT_TOTAL_MOSAIC_LOG_PRIOR_LIK
            )

            coverage_state = self.reads_number < MIN_AVERAGE_COVERAGE
        self.hom2het_mosaic_fraction = INITIAL_MOSAIC_FRACTION
        if coverage_state:
            logging.info(
                f"{self.locus_infos['STR_id']} has a sample depth less than"
                f" {MIN_AVERAGE_COVERAGE}"
            )
            self.hom2het_locus_state = "CoverageFail"
            return
        while (self.hom2het_niter < MIN_ITERATION) or (
            (not self.hom2het_converged)
            and (self.hom2het_niter < MAX_ITERATION)
        ):
            prev_total_mosaic_log_prior_lik = (
                self.hom2het_total_mosaic_log_prior_lik
            )
            if ITERATION_MODE == "allele-based":  # XXX
                (
                    hom2het_mosaic_fraction,
                    self.hom2het_total_mosaic_log_prior_lik,
                    self.hom2het_all_mosaic_log_prior_lik_list,
                    self.hom2het_all_mosaic_gt_list,
                ) = self.update_mosaic_fraction_based_mut_allele("hom2het")
            elif ITERATION_MODE == "cell-based":
                (
                    hom2het_mosaic_fraction,
                    self.hom2het_total_mosaic_log_prior_lik,
                    self.hom2het_all_mosaic_log_prior_lik_list,
                    self.hom2het_all_mosaic_gt_list,
                ) = self.update_mosaic_fraction_based_mut_cells("hom2het")
            else:
                logging.error("Iteration mode error.")
                raise ValueError("Iteration mode error.")
            total_mosaic_log_prior_lik_diff = (
                self.hom2het_total_mosaic_log_prior_lik
                - prev_total_mosaic_log_prior_lik
            )
            converged0 = total_mosaic_log_prior_lik_diff < MAX_TOLERANCE
            if converged0:  # XXX
                logging.warning(
                    "The total prior log-likelihood of"
                    f" {self.locus_infos['STR_id']} unexpectedly decreased to"
                    " less than 1/10."
                )
                self.hom2het_converged = "LL_Decreased"
                self.hom2het_converged_list = [
                    converged0,
                    "unknown",
                    "unknown",
                ]
                break
            prev_mosaic_fraction = self.hom2het_mosaic_fraction
            if DEBUG:
                self.hom2het_total_mosaic_log_prior_lik_list.append(
                    self.hom2het_total_mosaic_log_prior_lik
                )
                print(f"iter:{self.hom2het_niter}")
                print(hom2het_mosaic_fraction)
            self.hom2het_mosaic_fraction = hom2het_mosaic_fraction
            if USE_MINOR_CELLS_POP_AS_MUT_CELLS:  # XXX
                if (
                    self.hom2het_mosaic_fraction < 0
                    or self.hom2het_mosaic_fraction > 0.5
                ):
                    logging.warning(
                        "The mosaic fraction of"
                        f" {self.locus_infos['STR_id']} is less than 0 or"
                        " greater than 0.5."
                    )
                    self.hom2het_converged = "MosaicFraction_OutOfRange"
                    self.hom2het_converged_list = [
                        converged0,
                        "unknown",
                        "unknown",
                    ]
                    break
            else:
                if (
                    self.hom2het_mosaic_fraction < 0
                    or self.hom2het_mosaic_fraction > 1
                ):
                    logging.warning(
                        "The mosaic fraction of"
                        f" {self.locus_infos['STR_id']} is less than 0 or"
                        " greater than 1."
                    )
                    self.hom2het_converged = "MosaicFraction_OutOfRange"
                    self.hom2het_converged_list = [
                        converged0,
                        "unknown",
                        "unknown",
                    ]
                    break
            if self.hom2het_niter % CHECK_CONVERGE_PERIOD == 0:
                mosaic_fraction_diff = abs(
                    self.hom2het_mosaic_fraction - prev_mosaic_fraction
                )
                converged1 = (
                    (
                        (-total_mosaic_log_prior_lik_diff)
                        / prev_total_mosaic_log_prior_lik
                    )
                    < DIFFERENCE_CONVERGE
                ) and (total_mosaic_log_prior_lik_diff < DIFFERENCE_CONVERGE)
                converged2 = mosaic_fraction_diff < DIFFERENCE_CONVERGE_MosaicFrac
                # BUG: Check converge condition and converged0 should be in the before of stutter params updating and break in time
                # HAS SOLVED
                self.hom2het_converged = converged1 or converged2
                self.hom2het_converged_list = [
                    converged0,
                    converged1,
                    converged2,
                ]
            self.hom2het_niter += 1
            self.hom2het_total_niter += 1
        self.hom2het_n_train += 1
        self.hom2het_trained = True
        if self.hom2het_converged == "LL_Decreased":
            logging.info(
                f"{self.locus_infos['STR_id']}: The total mosaic prior"
                " log-likelihood unexpectedly decreased to less than 1/10."
            )
            self.hom2het_locus_state = "LLDecreased"
        elif self.hom2het_converged == "MosaicFraction_OutOfRange":
            if USE_MINOR_CELLS_POP_AS_MUT_CELLS:
                logging.info(
                    f"{self.locus_infos['STR_id']}: The mosaic fraction is"
                    " less than 0 or greater than 0.5."
                )
            else:
                logging.info(
                    f"{self.locus_infos['STR_id']}: The mosaic fraction is"
                    " less than 0 or greater than 1."
                )
            self.hom2het_locus_state = "MosaicFractionOutOfRange"
        elif (
            self.hom2het_converged == True
        ):  # BUG numpy.bool_ don't support 'is' operator: HAS SOLVED
            logging.info(f"{self.locus_infos['STR_id']} has converged.")
            self.hom2het_locus_state = "HasConverged"
        else:
            logging.info(
                f"{self.locus_infos['STR_id']}: Maximum iteration reached."
            )
            self.hom2het_locus_state = "MaxIterationReached"
        if DEBUG:
            print(f"Current alleles depth dict: {self.alleles_depth_dict}")
            max_mosaic_index = np.argmax(
                self.hom2het_all_mosaic_log_prior_lik_list
            )
            max_mosaic_gt = self.hom2het_all_mosaic_gt_list[max_mosaic_index]
            print(
                "Max mosaic"
                f" transformation:{self.mosaic_genotypes[max_mosaic_gt]}"
            )
            print(f"GT:{self.genotypes}")
            print(f"Alleles:{self.alleles_used_for_gt}")
            print(
                f"All_mosaic_log_prior_lik_list:{self.hom2het_all_mosaic_log_prior_lik_list}"
            )

    def train_het2het(self):
        if self.alleles_number_used_for_gt <= 2:
            self.het2het_locus_state = (
                f"{self.alleles_number_used_for_gt}Alleles"
            )
            self.het2het_trained = True
            self.het2het_converged = True
            self.het2het_niter = 0
            self.het2het_mosaic_fraction = 0
            return
        if self.het2het_locus_state == "CoverageFail":
            logging.info(
                "This locus will not be trained due to insufficient coverage."
            )
            return
        elif self.het2het_trained and self.het2het_converged:
            logging.info(
                "We invoked the train function previously,"
                " and this locus has now converged for het2het."
            )
            return
        elif self.het2het_trained and (1 - self.het2het_converged):
            self.het2het_niter = MIN_ITERATION
            logging.info(
                "We invoked the train function previously,"
                " but this locus did not converge."
                " We will continue the training"
                " using the previous set of parameters."
            )
        else:
            logging.info(
                "The train function has not been invoked before,"
                " and we will train this locus from the beginning."
            )
            self.het2het_niter = 0
            self.het2het_total_mosaic_log_prior_lik = (
                INIT_TOTAL_MOSAIC_LOG_PRIOR_LIK
            )

            coverage_state = self.reads_number < MIN_AVERAGE_COVERAGE
        self.het2het_mosaic_fraction = INITIAL_MOSAIC_FRACTION
        if coverage_state:
            logging.info(
                f"{self.locus_infos['STR_id']} has a sample depth less than"
                f" {MIN_AVERAGE_COVERAGE}"
            )
            self.het2het_locus_state = "CoverageFail"
            return
        while (self.het2het_niter < MIN_ITERATION) or (
            (not self.het2het_converged)
            and (self.het2het_niter < MAX_ITERATION)
        ):
            prev_total_mosaic_log_prior_lik = (
                self.het2het_total_mosaic_log_prior_lik
            )
            if ITERATION_MODE == "allele-based":  # XXX
                (
                    het2het_mosaic_fraction,
                    self.het2het_total_mosaic_log_prior_lik,
                    self.het2het_all_mosaic_log_prior_lik_list,
                    self.het2het_all_mosaic_gt_list,
                ) = self.update_mosaic_fraction_based_mut_allele("het2het")
            elif ITERATION_MODE == "cell-based":
                (
                    het2het_mosaic_fraction,
                    self.het2het_total_mosaic_log_prior_lik,
                    self.het2het_all_mosaic_log_prior_lik_list,
                    self.het2het_all_mosaic_gt_list,
                ) = self.update_mosaic_fraction_based_mut_cells("het2het")
            else:
                logging.error("Iteration mode error.")
                raise ValueError("Iteration mode error.")
            total_mosaic_log_prior_lik_diff = (
                self.het2het_total_mosaic_log_prior_lik
                - prev_total_mosaic_log_prior_lik
            )
            converged0 = total_mosaic_log_prior_lik_diff < MAX_TOLERANCE
            if converged0:  # XXX
                logging.warning(
                    "The total prior log-likelihood of"
                    f" {self.locus_infos['STR_id']} unexpectedly decreased to"
                    " less than 1/10."
                )
                self.het2het_converged = "LL_Decreased"
                self.het2het_converged_list = [
                    converged0,
                    "unknown",
                    "unknown",
                ]
                break
            prev_mosaic_fraction = self.het2het_mosaic_fraction
            if DEBUG:
                self.het2het_total_mosaic_log_prior_lik_list.append(
                    self.het2het_total_mosaic_log_prior_lik
                )
                print(f"iter:{self.het2het_niter}")
                print(het2het_mosaic_fraction)
            self.het2het_mosaic_fraction = het2het_mosaic_fraction
            if USE_MINOR_CELLS_POP_AS_MUT_CELLS:  # XXX
                if (
                    self.het2het_mosaic_fraction < 0
                    or self.het2het_mosaic_fraction > 0.5
                ):
                    logging.warning(
                        "The mosaic fraction of"
                        f" {self.locus_infos['STR_id']} is less than 0 or"
                        " greater than 0.5."
                    )
                    self.het2het_converged = "MosaicFraction_OutOfRange"
                    self.het2het_converged_list = [
                        converged0,
                        "unknown",
                        "unknown",
                    ]
                    break
            else:
                if (
                    self.het2het_mosaic_fraction < 0
                    or self.het2het_mosaic_fraction > 1
                ):
                    logging.warning(
                        "The mosaic fraction of"
                        f" {self.locus_infos['STR_id']} is less than 0 or"
                        " greater than 1."
                    )
                    self.het2het_converged = "MosaicFraction_OutOfRange"
                    self.het2het_converged_list = [
                        converged0,
                        "unknown",
                        "unknown",
                    ]
                    break
            if self.het2het_niter % CHECK_CONVERGE_PERIOD == 0:
                mosaic_fraction_diff = abs(
                    self.het2het_mosaic_fraction - prev_mosaic_fraction
                )
                converged1 = (
                    (
                        (-total_mosaic_log_prior_lik_diff)
                        / prev_total_mosaic_log_prior_lik
                    )
                    < DIFFERENCE_CONVERGE
                ) and (total_mosaic_log_prior_lik_diff < DIFFERENCE_CONVERGE)
                converged2 = mosaic_fraction_diff < DIFFERENCE_CONVERGE_MosaicFrac
                # BUG: Check converge condition and converged0 should be in the before of stutter params updating and break in time
                # HAS SOLVED
                self.het2het_converged = converged1 or converged2
                self.het2het_converged_list = [
                    converged0,
                    converged1,
                    converged2,
                ]
            self.het2het_niter += 1
            self.het2het_total_niter += 1
        self.het2het_n_train += 1
        self.het2het_trained = True
        if self.het2het_converged == "LL_Decreased":
            logging.info(
                f"{self.locus_infos['STR_id']}: The total mosaic prior"
                " log-likelihood unexpectedly decreased to less than 1/10."
            )
            self.het2het_locus_state = "LLDecreased"
        elif self.het2het_converged == "MosaicFraction_OutOfRange":
            if USE_MINOR_CELLS_POP_AS_MUT_CELLS:
                logging.info(
                    f"{self.locus_infos['STR_id']}: The mosaic fraction is"
                    " less than 0 or greater than 0.5."
                )
            else:
                logging.info(
                    f"{self.locus_infos['STR_id']}: The mosaic fraction is"
                    " less than 0 or greater than 1."
                )
            self.het2het_locus_state = "MosaicFractionOutOfRange"
        elif (
            self.het2het_converged == True
        ):  # BUG numpy.bool_ don't support 'is' operator: HAS SOLVED
            logging.info(f"{self.locus_infos['STR_id']} has converged.")
            self.het2het_locus_state = "HasConverged"
        else:
            logging.info(
                f"{self.locus_infos['STR_id']}: Maximum iteration reached."
            )
            self.het2het_locus_state = "MaxIterationReached"
        if DEBUG:
            print(f"Current alleles depth dict: {self.alleles_depth_dict}")
            max_mosaic_index = np.argmax(
                self.het2het_all_mosaic_log_prior_lik_list
            )
            max_mosaic_gt = self.het2het_all_mosaic_gt_list[max_mosaic_index]
            print(
                "Max mosaic"
                f" transformation:{self.mosaic_genotypes[max_mosaic_gt]}"
            )
            print(f"GT:{self.genotypes}")
            print(f"Alleles:{self.alleles_used_for_gt}")
            print(
                f"All_mosaic_log_prior_lik_list:{self.het2het_all_mosaic_log_prior_lik_list}"
            )

    def update_mosaic_fraction_based_mut_allele(self, mut_type="all"):
        self.log_total_reads_number = []
        self.log_mut_alleles_number = []
        all_mosaic_log_prior_lik_list = []
        all_mosaic_gt_list = []
        for mosaic_index, cells in enumerate(self.mosaic_genotypes):
            germ_cells_index = cells[0]
            mut_cells_index = cells[1]
            germ_cells_gt = self.genotypes[germ_cells_index]
            mut_cells_gt = self.genotypes[mut_cells_index]
            if CONSIDER_GERM2GERM_IN_PARAMS_UPDATE:
                pass
            else:
                if germ_cells_gt == mut_cells_gt:
                    continue
            if ESTIMATION_MODE == "double":
                if mut_type == "hom2het":
                    if ((len(set(mut_cells_gt))) == 2) and (
                        (len(set(germ_cells_gt))) == 1
                    ):
                        mosaic_fraction = self.hom2het_mosaic_fraction
                        niter = self.hom2het_niter
                        pass
                    else:
                        continue
                elif mut_type == "het2het":
                    if ALLOWABLE_HET2HOM:
                        if (len(set(germ_cells_gt))) == 2:
                            mosaic_fraction = self.het2het_mosaic_fraction
                            niter = self.het2het_niter
                            pass
                        else:
                            continue
                    else:
                        if ((len(set(mut_cells_gt))) == 2) and (
                            (len(set(germ_cells_gt))) == 2
                        ):
                            mosaic_fraction = self.het2het_mosaic_fraction
                            niter = self.het2het_niter
                            pass
                        else:
                            continue
                else:
                    mosaic_fraction = self.mosaic_fraction
                    niter = self.niter
            elif ESTIMATION_MODE == "unique":
                mosaic_fraction = self.mosaic_fraction
                niter = self.niter
            if ALLOWABLE_HET2HOM:  # XXX Allow het2hom
                if (mut_cells_gt[0] == germ_cells_gt[0]) and (
                    mut_cells_gt[1] != germ_cells_gt[1]
                ):  # XXX
                    mut_allele = mut_cells_gt[1]
                    mut_cell_ano_allele = mut_cells_gt[0]
                elif (mut_cells_gt[0] != germ_cells_gt[0]) and (
                    mut_cells_gt[1] == germ_cells_gt[1]
                ):  # XXX
                    mut_allele = mut_cells_gt[0]
                    mut_cell_ano_allele = mut_cells_gt[1]
                elif (mut_cells_gt[0] != germ_cells_gt[1]) and (
                    mut_cells_gt[1] == germ_cells_gt[0]
                ):
                    mut_allele = mut_cells_gt[0]
                    mut_cell_ano_allele = mut_cells_gt[1]
                elif (mut_cells_gt[0] == germ_cells_gt[1]) and (
                    mut_cells_gt[1] != germ_cells_gt[0]
                ):
                    mut_allele = mut_cells_gt[1]
                    mut_cell_ano_allele = mut_cells_gt[0]
                if (
                    (
                        (mut_cells_gt[0] == germ_cells_gt[0])
                        and (mut_cells_gt[1] != germ_cells_gt[1])
                    )
                    or (
                        (mut_cells_gt[0] != germ_cells_gt[0])
                        and (mut_cells_gt[1] == germ_cells_gt[1])
                    )
                    or (
                        (mut_cells_gt[0] != germ_cells_gt[1])
                        and (mut_cells_gt[1] == germ_cells_gt[0])
                    )
                    or (
                        (mut_cells_gt[0] == germ_cells_gt[1])
                        and (mut_cells_gt[1] != germ_cells_gt[0])
                    )
                ):  # XXX only allow one allele to mutate between gts
                    current_mut_allele_log_posterior = (
                        self.cal_per_read_mut_log_posterior(
                            germ_cells_gt[0],
                            germ_cells_gt[1],
                            mut_cell_ano_allele,
                            mut_allele,
                            mut_type,
                        )
                    )
                    pseudo_log_mut_alleles_number = logsumexp(
                        current_mut_allele_log_posterior
                    )
                    if USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GIandMutRate":
                        log_mosaic_prior_lik = (
                            self.all_gts_log_lik_array[germ_cells_index]
                            + self.log_mutation_rate_prior[mosaic_index]
                            + self.cal_mosaic_gt_log_lik(
                                cells, mosaic_fraction
                            )
                        )
                    elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GI":
                        log_mosaic_prior_lik = self.all_gts_log_lik_array[
                            germ_cells_index
                        ] + self.cal_mosaic_gt_log_lik(cells, mosaic_fraction)
                    elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "MutRate":
                        log_mosaic_prior_lik = self.log_mutation_rate_prior[
                            mosaic_index
                        ] + self.cal_mosaic_gt_log_lik(cells, mosaic_fraction)
                    elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "OnlyLik":
                        log_mosaic_prior_lik = self.cal_mosaic_gt_log_lik(
                            cells, mosaic_fraction
                        )
                    else:
                        logging.error("GI prior update mosaic fraction error.")
                        raise ValueError(
                            "GI prior update mosaic fraction error."
                        )
                    all_mosaic_log_prior_lik_list.append(log_mosaic_prior_lik)
                    self.log_total_reads_number.append(
                        log_mosaic_prior_lik + np.log(self.reads_number)
                    )
                    if HET2HOM_SET_ZERO_MUT_READS and (
                        len(set(mut_cells_gt)) == 1
                    ):  # XXX: Allow het2hom but don't count mut allele counts, total reads counts are counted
                        self.log_mut_alleles_number.append(
                            log_mosaic_prior_lik + np.log(0)
                        )
                    else:
                        if USE_MINOR_CELLS_POP_AS_MUT_CELLS:
                            if (
                                pseudo_log_mut_alleles_number
                                - np.log(self.reads_number)
                            ) + LOG_TWO > LOG_ONE_HALF:  # XXX: Don't allow mosaic allele with AF bigger than 0.25
                                if MF_MORE_THAN_HALT_SET_ZERO:  # XXX: set zero
                                    self.log_mut_alleles_number.append(
                                        log_mosaic_prior_lik + np.log(0)
                                    )
                                else:  # XXX:change mutation directions, may produce negative reads number
                                    pass  # HACK may produce negative reads number
                                    # self.log_mut_alleles_number.append(
                                    #     log_mosaic_prior_lik
                                    #     + np.log(
                                    #         self.reads_number
                                    #         - np.exp(pseudo_log_mut_alleles_number)
                                    #     )
                                    # )
                            else:
                                self.log_mut_alleles_number.append(
                                    log_mosaic_prior_lik
                                    + pseudo_log_mut_alleles_number
                                )
                        else:
                            self.log_mut_alleles_number.append(
                                log_mosaic_prior_lik
                                + pseudo_log_mut_alleles_number
                            )
                    all_mosaic_gt_list.append(mosaic_index)
                else:
                    continue
            else:
                if (len(set(mut_cells_gt))) == 1 and (
                    len(set(germ_cells_gt))
                ) == 2:  # XXX
                    continue
                else:
                    if (mut_cells_gt[0] == germ_cells_gt[0]) and (
                        mut_cells_gt[1] != germ_cells_gt[1]
                    ):  # XXX
                        mut_allele = mut_cells_gt[1]
                        mut_cell_ano_allele = mut_cells_gt[0]
                    elif (mut_cells_gt[0] != germ_cells_gt[0]) and (
                        mut_cells_gt[1] == germ_cells_gt[1]
                    ):  # XXX
                        mut_allele = mut_cells_gt[0]
                        mut_cell_ano_allele = mut_cells_gt[1]
                    elif (mut_cells_gt[0] != germ_cells_gt[1]) and (
                        mut_cells_gt[1] == germ_cells_gt[0]
                    ):
                        mut_allele = mut_cells_gt[0]
                        mut_cell_ano_allele = mut_cells_gt[1]
                    elif (mut_cells_gt[0] == germ_cells_gt[1]) and (
                        mut_cells_gt[1] != germ_cells_gt[0]
                    ):
                        mut_allele = mut_cells_gt[1]
                        mut_cell_ano_allele = mut_cells_gt[0]
                    if (
                        (
                            (mut_cells_gt[0] == germ_cells_gt[0])
                            and (mut_cells_gt[1] != germ_cells_gt[1])
                        )
                        or (
                            (mut_cells_gt[0] != germ_cells_gt[0])
                            and (mut_cells_gt[1] == germ_cells_gt[1])
                        )
                        or (
                            (mut_cells_gt[0] != germ_cells_gt[1])
                            and (mut_cells_gt[1] == germ_cells_gt[0])
                        )
                        or (
                            (mut_cells_gt[0] == germ_cells_gt[1])
                            and (mut_cells_gt[1] != germ_cells_gt[0])
                        )
                    ):  # XXX only allow one allele to mutate between gts
                        current_mut_allele_log_posterior = (
                            self.cal_per_read_mut_log_posterior(
                                germ_cells_gt[0],
                                germ_cells_gt[1],
                                mut_cell_ano_allele,
                                mut_allele,
                                mut_type,
                            )
                        )
                        pseudo_log_mut_alleles_number = logsumexp(
                            current_mut_allele_log_posterior
                        )
                        if (
                            USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION
                            == "GIandMutRate"
                        ):
                            log_mosaic_prior_lik = (
                                self.all_gts_log_lik_array[germ_cells_index]
                                + self.log_mutation_rate_prior[mosaic_index]
                                + self.cal_mosaic_gt_log_lik(
                                    cells, mosaic_fraction
                                )
                            )
                        elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GI":
                            log_mosaic_prior_lik = self.all_gts_log_lik_array[
                                germ_cells_index
                            ] + self.cal_mosaic_gt_log_lik(
                                cells, mosaic_fraction
                            )
                        elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "MutRate":
                            log_mosaic_prior_lik = (
                                self.log_mutation_rate_prior[mosaic_index]
                                + self.cal_mosaic_gt_log_lik(
                                    cells, mosaic_fraction
                                )
                            )
                        elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "OnlyLik":
                            log_mosaic_prior_lik = self.cal_mosaic_gt_log_lik(
                                cells, mosaic_fraction
                            )
                        else:
                            logging.error(
                                "GI prior update mosaic fraction error."
                            )
                            raise ValueError(
                                "GI prior update mosaic fraction error."
                            )
                        all_mosaic_log_prior_lik_list.append(
                            log_mosaic_prior_lik
                        )
                        self.log_total_reads_number.append(
                            log_mosaic_prior_lik + np.log(self.reads_number)
                        )
                        if USE_MINOR_CELLS_POP_AS_MUT_CELLS:  # XXX
                            if (
                                pseudo_log_mut_alleles_number
                                - np.log(self.reads_number)
                            ) + LOG_TWO > LOG_ONE_HALF:
                                if MF_MORE_THAN_HALT_SET_ZERO:  # XXX: set zero
                                    self.log_mut_alleles_number.append(
                                        log_mosaic_prior_lik + np.log(0)
                                    )
                                else:  # XXX:change mutation directions, may produce negative reads number
                                    pass  # HACK may produce negative reads number
                                    # self.log_mut_alleles_number.append(
                                    #     log_mosaic_prior_lik
                                    #     + np.log(
                                    #         self.reads_number
                                    #         - np.exp(pseudo_log_mut_alleles_number)
                                    #     )
                                    # )
                            else:
                                self.log_mut_alleles_number.append(
                                    log_mosaic_prior_lik
                                    + pseudo_log_mut_alleles_number
                                )
                        else:
                            self.log_mut_alleles_number.append(
                                log_mosaic_prior_lik
                                + pseudo_log_mut_alleles_number
                            )
                        all_mosaic_gt_list.append(mosaic_index)
        updated_mosaic_fraction = 2 * np.exp(
            logsumexp(self.log_mut_alleles_number)
            - logsumexp(self.log_total_reads_number)
        )
        total_mosaic_log_prior_lik = logsumexp(all_mosaic_log_prior_lik_list)
        if DEBUG:
            print(f"iter:{niter}")
            print(
                f"max_mosaic_log_prior_lik_list:{max(all_mosaic_log_prior_lik_list)}"
            )
        return (
            updated_mosaic_fraction,
            total_mosaic_log_prior_lik,
            all_mosaic_log_prior_lik_list,
            all_mosaic_gt_list,
        )

    def cal_per_read_mut_log_posterior(
        self, germ_allele1, germ_allele2, germ_allele3, mut_allele, mut_type
    ):
        current_mut_allele_log_posterior = []
        if ESTIMATION_MODE == "double":
            if mut_type == "hom2het":
                mosaic_fraction = self.hom2het_mosaic_fraction
            elif mut_type == "het2het":
                mosaic_fraction = self.het2het_mosaic_fraction
            else:
                mosaic_fraction = self.mosaic_fraction
        else:
            mosaic_fraction = self.mosaic_fraction
        for read_index, _ in enumerate(self.reads_list):
            per_read_germ1_log_lik = self.all_reads_all_alleles_log_lik_array[
                read_index, germ_allele1
            ]
            per_read_germ2_log_lik = self.all_reads_all_alleles_log_lik_array[
                read_index, germ_allele2
            ]
            per_read_mut_cell_ano_allele_log_lik = (
                self.all_reads_all_alleles_log_lik_array[
                    read_index, germ_allele3
                ]
            )
            per_read_mut_log_lik = self.all_reads_all_alleles_log_lik_array[
                read_index, mut_allele
            ]
            if MAX_LIK_ASSIGNMENT:
                if (
                    max(
                        [
                            per_read_germ1_log_lik,
                            per_read_germ2_log_lik,
                            per_read_mut_log_lik,
                            per_read_mut_cell_ano_allele_log_lik,
                        ]
                    )
                    == per_read_mut_log_lik
                ):
                    max_lik_number = [
                        per_read_germ1_log_lik,
                        per_read_germ2_log_lik,
                        per_read_mut_log_lik,
                        per_read_mut_cell_ano_allele_log_lik,
                    ].count(
                        per_read_mut_log_lik
                    )  # XXX
                    per_read_mut_log_posterior = np.log(1 / max_lik_number)
                else:
                    per_read_mut_log_posterior = -np.inf
            else:
                per_read_mut_log_posterior = (
                    np.log(mosaic_fraction / 2) + per_read_mut_log_lik
                ) - logsumexp(
                    [
                        per_read_germ1_log_lik,
                        per_read_germ2_log_lik,
                        per_read_mut_log_lik,
                        per_read_mut_cell_ano_allele_log_lik,
                    ],
                    b=[
                        (1 - mosaic_fraction) / 2,  # 1，
                        (1 - mosaic_fraction) / 2,  # 1，
                        mosaic_fraction / 2,  # 1，
                        mosaic_fraction / 2,  # 1，
                    ],
                )
            current_mut_allele_log_posterior.append(per_read_mut_log_posterior)
        return current_mut_allele_log_posterior

    def update_mosaic_fraction_based_mut_cells(self, mut_type="all"):
        self.log_total_cells_number = []
        self.log_mut_cells_number = []
        all_mosaic_log_prior_lik_list = []
        all_mosaic_gt_list = []
        for mosaic_index, cells in enumerate(self.mosaic_genotypes):
            germ_cells_index = cells[0]
            mut_cells_index = cells[1]
            germ_cells_gt = self.genotypes[germ_cells_index]
            mut_cells_gt = self.genotypes[mut_cells_index]
            if CONSIDER_GERM2GERM_IN_PARAMS_UPDATE:
                pass
            else:
                if germ_cells_gt == mut_cells_gt:
                    continue
            if ALLOWABLE_HET2HOM:
                pass
            else:
                if (len(set(mut_cells_gt))) == 1 and (
                    len(set(germ_cells_gt))
                ) == 2:  # XXX
                    continue
            if ALLOWABLE_TWO_MUT_ALLELES:
                pass
            else:
                if len(set(germ_cells_gt) & set(mut_cells_gt)) == 0:
                    continue
            if ESTIMATION_MODE == "double":
                if mut_type == "hom2het":
                    if ((len(set(mut_cells_gt))) == 2) and (
                        (len(set(germ_cells_gt))) == 1
                    ):
                        mosaic_fraction = self.hom2het_mosaic_fraction
                        niter = self.hom2het_niter
                        pass
                    else:
                        continue
                elif mut_type == "het2het":
                    if ALLOWABLE_HET2HOM:
                        if (len(set(germ_cells_gt))) == 2:
                            mosaic_fraction = self.het2het_mosaic_fraction
                            niter = self.het2het_niter
                            pass
                        else:
                            continue
                    else:
                        if ((len(set(mut_cells_gt))) == 2) and (
                            (len(set(germ_cells_gt))) == 2
                        ):
                            mosaic_fraction = self.het2het_mosaic_fraction
                            niter = self.het2het_niter
                            pass
                        else:
                            continue
                else:
                    mosaic_fraction = self.mosaic_fraction
                    niter = self.niter
            else:
                mosaic_fraction = self.mosaic_fraction
                niter = self.niter
            current_gt_log_posterior = []
            for read_index, _ in enumerate(self.reads_list):
                per_read_germ_log_lik = self.all_reads_all_gts_log_lik_array[
                    read_index, germ_cells_index
                ]
                per_read_mut_log_lik = self.all_reads_all_gts_log_lik_array[
                    read_index, mut_cells_index
                ]
                if MAX_LIK_ASSIGNMENT:
                    if (
                        max([per_read_germ_log_lik, per_read_mut_log_lik])
                        == per_read_mut_log_lik
                    ):
                        if per_read_germ_log_lik == per_read_mut_log_lik:
                            per_read_mut_log_posterior = np.log(1 / 2)  # XXX
                        else:
                            per_read_mut_log_posterior = np.log(1)
                    else:
                        per_read_mut_log_posterior = -np.inf
                else:
                    per_read_mut_log_posterior = (
                        np.log(mosaic_fraction) + per_read_mut_log_lik
                    ) - logsumexp(
                        [per_read_germ_log_lik, per_read_mut_log_lik],
                        b=[
                            1
                            - mosaic_fraction,  # 1 - self.mosaic_fraction,  # 1,
                            mosaic_fraction,  # self.mosaic_fraction  # 1,
                        ],
                    )  # XXX
                current_gt_log_posterior.append(per_read_mut_log_posterior)
            pseudo_log_mut_cells_number = logsumexp(current_gt_log_posterior)
            if USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GIandMutRate":
                log_mosaic_prior_lik = (
                    self.all_gts_log_lik_array[germ_cells_index]
                    + self.log_mutation_rate_prior[mosaic_index]
                    + self.cal_mosaic_gt_log_lik(cells, mosaic_fraction)
                )
            elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GI":
                log_mosaic_prior_lik = self.all_gts_log_lik_array[
                    germ_cells_index
                ] + self.cal_mosaic_gt_log_lik(cells, mosaic_fraction)
            elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "MutRate":
                log_mosaic_prior_lik = self.log_mutation_rate_prior[
                    mosaic_index
                ] + self.cal_mosaic_gt_log_lik(cells, mosaic_fraction)
            elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "OnlyLik":
                log_mosaic_prior_lik = self.cal_mosaic_gt_log_lik(
                    cells, mosaic_fraction
                )
            else:
                logging.error("GI prior update mosaic fraction error.")
                raise ValueError("GI prior update mosaic fraction error.")
            all_mosaic_log_prior_lik_list.append(log_mosaic_prior_lik)
            self.log_total_cells_number.append(
                log_mosaic_prior_lik + np.log(self.reads_number)
            )
            if USE_MINOR_CELLS_POP_AS_MUT_CELLS:  # XXX
                if (
                    pseudo_log_mut_cells_number - np.log(self.reads_number)
                ) > LOG_ONE_HALF:  # XXX
                    self.log_mut_cells_number.append(
                        log_mosaic_prior_lik
                        + np.log(
                            self.reads_number
                            - np.exp(pseudo_log_mut_cells_number)
                        )
                    )
                else:
                    self.log_mut_cells_number.append(
                        log_mosaic_prior_lik + pseudo_log_mut_cells_number
                    )
            else:
                self.log_mut_cells_number.append(
                    log_mosaic_prior_lik + pseudo_log_mut_cells_number
                )
            all_mosaic_gt_list.append(mosaic_index)
        updated_mosaic_fraction = np.exp(
            logsumexp(self.log_mut_cells_number)
            - logsumexp(self.log_total_cells_number)
        )
        total_mosaic_log_prior_lik = logsumexp(all_mosaic_log_prior_lik_list)
        if DEBUG:
            print(f"iter:{niter}")
            print(
                f"max_mosaic_log_prior_lik_list:{max(all_mosaic_log_prior_lik_list)}"
            )
        return (
            updated_mosaic_fraction,
            total_mosaic_log_prior_lik,
            all_mosaic_log_prior_lik_list,
            all_mosaic_gt_list,
        )

    def cal_mosaic_gt_log_lik(self, mosaic_gt, mosaic_fraction):
        per_read_mosaic_gt_log_lik_list = []
        for read_index, _ in enumerate(self.reads_list):
            read_mosaic_gt_log_lik = self.cal_per_read_given_mosaic_gt_log_lik(
                read_index, mosaic_gt, mosaic_fraction
            )
            per_read_mosaic_gt_log_lik_list.append(read_mosaic_gt_log_lik)
        mosaic_gt_log_lik = np.sum(per_read_mosaic_gt_log_lik_list)
        return mosaic_gt_log_lik

    def cal_per_read_given_mosaic_gt_log_lik(
        self, read_index, mosaic_gt, mosaic_fraction
    ):
        germ_cells_index = mosaic_gt[0]
        mut_cells_index = mosaic_gt[1]
        read_mosaic_gt_log_lik = logsumexp(
            [
                self.all_reads_all_gts_log_lik_array[
                    read_index, germ_cells_index
                ],
                self.all_reads_all_gts_log_lik_array[
                    read_index, mut_cells_index
                ],
            ],
            b=[1 - mosaic_fraction, mosaic_fraction],
        )
        return read_mosaic_gt_log_lik

    def mosaic_genotype_negloglik(self, cells, mosaic_fraction):
        mosaic_log_lik = 0
        germ_cells_index = cells[0]
        mut_cells_index = cells[1]
        for read_index, _ in enumerate(self.reads_list):
            per_read_germ_log_lik = self.all_reads_all_gts_log_lik_array[
                read_index, germ_cells_index
            ]
            per_read_mut_log_lik = self.all_reads_all_gts_log_lik_array[
                read_index, mut_cells_index
            ]
            per_read_mosaic_log_lik = logsumexp(
                [per_read_germ_log_lik, per_read_mut_log_lik],
                b=[1 - mosaic_fraction, mosaic_fraction],
            )
            mosaic_log_lik += per_read_mosaic_log_lik
        return -mosaic_log_lik

    def cal_all_gts_trans_mosaic_fraction_based_on_mle(self):
        all_gts_trans_mosaic_fraction_based_on_mle = []
        all_gts_trans_mosaic_fraction_type_based_on_mle = []
        for _, cells in enumerate(self.mosaic_genotypes):
            if cells[0] == cells[1]:
                all_gts_trans_mosaic_fraction_type_based_on_mle.append(
                    "germline"
                )
                all_gts_trans_mosaic_fraction_based_on_mle.append(0)
            else:
                result_with_bounds = minimize(
                    lambda x: self.mosaic_genotype_negloglik(
                        cells, x[0]
                    ),  # BUG SOLVED: ADD [0]
                    x0=[INITIAL_MOSAIC_FRACTION],
                    method=MLE_OPTIMIZE_METHOD,
                    options={},
                    bounds=[MOSAIC_FRACTION_BOUNDS],
                )
                mle_mosaic_fraction = result_with_bounds.x[0]
                all_gts_trans_mosaic_fraction_based_on_mle.append(
                    mle_mosaic_fraction
                )
                if (
                    abs(mle_mosaic_fraction - 1)
                    < NUMERIC_CALCULATION_TOLERANCE
                ):
                    all_gts_trans_mosaic_fraction_type_based_on_mle.append(
                        "one_mosaic_fraction"
                    )
                elif (
                    abs(mle_mosaic_fraction - 0)
                    < NUMERIC_CALCULATION_TOLERANCE
                ):
                    all_gts_trans_mosaic_fraction_type_based_on_mle.append(
                        "zero_mosaic_fraction"
                    )
                else:
                    all_gts_trans_mosaic_fraction_type_based_on_mle.append(
                        "mosaic"
                    )
        return (
            all_gts_trans_mosaic_fraction_based_on_mle,
            all_gts_trans_mosaic_fraction_type_based_on_mle,
        )

    def cal_germline_log_posterior(self):
        self.all_gts_log_posterior_array = (
            self.all_gts_log_lik_array - logsumexp(self.all_gts_log_lik_array)
        )
        MaxPostIndex = np.argmax(self.all_gts_log_lik_array)
        MaxGermlineGT = self.genotypes[MaxPostIndex]
        MaxGermlineLogPosterior = self.all_gts_log_posterior_array[
            MaxPostIndex
        ]
        MaxGermlineLogLik = self.all_gts_log_lik_array[MaxPostIndex]
        SecondPostIndex = np.argsort(self.all_gts_log_lik_array)[-2]
        SecondGermlineGT = self.genotypes[SecondPostIndex]
        SecondMosaicLogPosterior = self.all_gts_log_posterior_array[
            SecondPostIndex
        ]
        SecondMosaicLogLik = self.all_gts_log_lik_array[SecondPostIndex]
        return (
            MaxGermlineGT,
            MaxGermlineLogPosterior,
            MaxGermlineLogLik,
            SecondGermlineGT,
            SecondMosaicLogPosterior,
            SecondMosaicLogLik,
        )  # XXX: 没有 np.exp

    def cal_somatic_log_posterior(self):
        if self.posterior_cal_logic:
            if ONLY_USE_MLE_MOSAIC_FRACTION:
                mle_mosaic_fraction_mosaic_index = np.argmax(
                    self.all_mosaic_gts_log_lik_array
                )
                mle_mosaic_fraction = self.estimated_mosaic_fraction[
                    mle_mosaic_fraction_mosaic_index
                ]
                mle_mosaic_fraction_list = [mle_mosaic_fraction] * len(
                    self.estimated_mosaic_fraction
                )
                self.all_reads_all_mosaic_gts_log_lik(mle_mosaic_fraction_list)
            else:
                self.all_reads_all_mosaic_gts_log_lik(
                    self.estimated_mosaic_fraction
                )
            if POSTERIOR_METHOD == "Gi-Given":
                self.Gi_log_lik_with_mosaic_gt_order()
                self.cal_all_gts_trans_mosaic_log_posterior()
                (
                    AllMosaicGTLogLikLogPosterior_dict,
                    AllMosaicGTDict_dict,
                ) = self.cal_different_levels_log_mosaic_posterior("orderedGT")
                AllMosaicGTDict = AllMosaicGTDict_dict["posterior"]
                AllMosaicGTLogPosterior = AllMosaicGTLogLikLogPosterior_dict[
                    "posterior"
                ]
                AllMosaicGTLogLik = AllMosaicGTLogLikLogPosterior_dict["lik"]
                AllMosaicGTOrder = AllMosaicGTDict["AllMosaicGT"]
                MaxPostIndex = np.argmax(AllMosaicGTLogPosterior)
                MaxMosaicGT = AllMosaicGTOrder[MaxPostIndex]
                MaxMosaicGT_for_output = self.get_mosaic_gt(MaxMosaicGT)
                MaxMosaicLogPosterior = AllMosaicGTLogPosterior[MaxPostIndex]
                MaxMosaicLogLik = AllMosaicGTLogLik[MaxPostIndex]
                SecondPostIndex = np.argsort(AllMosaicGTLogPosterior)[-2]
                SecondMosaicGT = AllMosaicGTOrder[SecondPostIndex]
                SecondMosaicGT_for_output = self.get_mosaic_gt(SecondMosaicGT)
                SecondMosaicLogPosterior = AllMosaicGTLogPosterior[
                    SecondPostIndex
                ]
                SecondMosaicLogLik = AllMosaicGTLogLik[SecondPostIndex]
                MaxMosaicGT_GTindex = self.mosaic_genotypes[MaxMosaicGT]
                Germline_MosaicIndex = self.mosaic_genotypes.index(
                    (MaxMosaicGT_GTindex[0], MaxMosaicGT_GTindex[0])
                )
                GermlinePostMosaicIndex = AllMosaicGTOrder.index(
                    Germline_MosaicIndex
                )
                GermlineGT_for_output = self.genotypes[MaxMosaicGT_GTindex[0]]
                GermlineLogPosterior = AllMosaicGTLogPosterior[
                    GermlinePostMosaicIndex
                ]
                GermlineLogLik = self.all_mosaic_gts_log_lik_array[
                    Germline_MosaicIndex
                ]
                ConsistentMaxMosaicLogPosterior = MaxMosaicLogPosterior
            else:
                # XXX: unique and double mosaic fraction estimation method don't support for Gi-free currently
                self.cal_all_gts_trans_mosaic_log_posterior()
                if UNORDER_MOSAIC_GT:
                    # 因为 UNORDER_MOSAIC_GT 本来 mosaicGT 和 GT 就是无序的，所以不用再做其他的改变了，获得原始的 posterior 就行了
                    (
                        AllMosaicGTLogPosterior_dict,
                        AllMosaicGTDict_dict,
                    ) = self.cal_different_levels_log_mosaic_posterior(
                        "orderedGT"
                    )
                    AllMosaicGTDict = AllMosaicGTDict_dict["posterior"]
                    AllMosaicGTOrder = AllMosaicGTDict["AllMosaicGT"]
                else:
                    (
                        AllMosaicGTLogPosterior_dict,
                        AllMosaicGTDict_dict,
                    ) = self.cal_different_levels_log_mosaic_posterior(
                        "unorderedGTunorderedMutDirection"
                    )
                    AllMosaicGTDict = AllMosaicGTDict_dict["posterior"]
                    AllMosaicGTOrder = AllMosaicGTDict
                if POSTERIOR_METHOD == "Gi-free":
                    AllMosaicGTLogPosterior = AllMosaicGTLogPosterior_dict[
                        "posterior"
                    ]  # XXX: 是 LogPosterior
                    AllMosaicGTLogLik = AllMosaicGTLogPosterior_dict["lik"]  # XXX: 是 LogLikFraction
                elif POSTERIOR_METHOD == "Likelihood":
                    AllMosaicGTLogPosterior = AllMosaicGTLogPosterior_dict[
                        "posterior"
                    ]
                    AllMosaicGTLogLik = AllMosaicGTLogPosterior
                if UNORDER_MOSAIC_GT:  # True
                    # AllMosaicGTOrder = AllMosaicGTDict["AllMosaicGT"]
                    # GI-free，以下 posterior 是 normalize 包括 mutation rate 先验
                    # GI-free，以下 LogLik 是 normalize 不包括 mutation rate 先验
                    # 以下 LogLik 是 LogLikFrac (AllMosaicGTLogLik)
                    MaxPostIndex = np.argmax(AllMosaicGTLogPosterior)
                    MaxMosaicGT = AllMosaicGTOrder[MaxPostIndex]
                    MaxMosaicGT_for_output = self.get_mosaic_gt(MaxMosaicGT)
                    MaxMosaicLogPosterior = AllMosaicGTLogPosterior[
                        MaxPostIndex
                    ]
                    MaxMosaicLogLik = AllMosaicGTLogLik[MaxPostIndex]
                    SecondPostIndex = np.argsort(AllMosaicGTLogPosterior)[-2]
                    SecondMosaicGT = AllMosaicGTOrder[SecondPostIndex]
                    SecondMosaicGT_for_output = self.get_mosaic_gt(
                        SecondMosaicGT
                    )
                    SecondMosaicLogPosterior = AllMosaicGTLogPosterior[
                        SecondPostIndex
                    ]
                    SecondMosaicLogLik = AllMosaicGTLogLik[SecondPostIndex]
                    MaxMosaicGT_GTindex = self.mosaic_genotypes[MaxMosaicGT]
                    Germline1_MosaicIndex = self.mosaic_genotypes.index(
                        (MaxMosaicGT_GTindex[0], MaxMosaicGT_GTindex[0])
                    )
                    Germline1PostMosaicIndex = AllMosaicGTOrder.index(
                        Germline1_MosaicIndex
                    )
                    Germline1LogPosterior = AllMosaicGTLogPosterior[
                        Germline1PostMosaicIndex
                    ]
                    Germline1LogLik = AllMosaicGTLogLik[Germline1PostMosaicIndex]
                    # GermlineLogLik1 = self.all_mosaic_gts_log_lik_array[
                    #     Germline1_MosaicIndex
                    # ]  # DEBUG：这个值是真的 loglik 而不是 loglikfrac
                    Germline2_MosaicIndex = self.mosaic_genotypes.index(
                        (MaxMosaicGT_GTindex[1], MaxMosaicGT_GTindex[1])
                    )
                    Germline2PostMosaicIndex = AllMosaicGTOrder.index(
                        Germline2_MosaicIndex
                    )
                    Germline2LogPosterior = AllMosaicGTLogPosterior[
                        Germline2PostMosaicIndex
                    ]
                    Germline2LogLik = AllMosaicGTLogLik[Germline2PostMosaicIndex]
                    # GermlineLogLik2 = self.all_mosaic_gts_log_lik_array[
                    #     Germline2_MosaicIndex
                    # ] # DEBUG：这个值是真的 loglik 而不是 loglikfrac
                    if Germline1LogPosterior >= Germline2LogPosterior:
                        GermlineGT_for_output = self.genotypes[
                            MaxMosaicGT_GTindex[0]
                        ]  # GermlineGT_for_output 不是 pop output 但是后续会被处理
                        GermlineLogPosterior = Germline1LogPosterior
                        GermlineLogLik = Germline1LogLik
                    else:
                        GermlineGT_for_output = self.genotypes[
                            MaxMosaicGT_GTindex[1]
                        ]  # GermlineGT_for_output 不是 pop output 但是后续会被处理
                        GermlineLogPosterior = Germline2LogPosterior
                        GermlineLogLik = Germline2LogLik
                else:
                    AllMosaicGTLogPosterior_sorted = sorted(
                        AllMosaicGTLogPosterior.items(),
                        key=lambda item: item[1],
                    )
                    MaxMosaicGT_for_output = AllMosaicGTLogPosterior_sorted[
                        -1
                    ][0]
                    MaxMosaicLogPosterior = AllMosaicGTLogPosterior_sorted[-1][
                        1
                    ]
                    MaxMosaicLogLik = AllMosaicGTLogLik[MaxMosaicGT_for_output]
                    SecondMosaicGT_for_output = AllMosaicGTLogPosterior_sorted[
                        -2
                    ][0]
                    SecondMosaicLogPosterior = AllMosaicGTLogPosterior_sorted[
                        -2
                    ][1]
                    SecondMosaicLogLik = AllMosaicGTLogLik[
                        SecondMosaicGT_for_output
                    ]
                    GermlineGT1_for_output = tuple(
                        [MaxMosaicGT_for_output[0], MaxMosaicGT_for_output[0]]
                    )
                    GermlineGT2_for_output = tuple(
                        [MaxMosaicGT_for_output[1], MaxMosaicGT_for_output[1]]
                    )
                    Germline1LogPosterior = AllMosaicGTLogPosterior[
                        GermlineGT1_for_output
                    ]
                    Germline2LogPosterior = AllMosaicGTLogPosterior[
                        GermlineGT2_for_output
                    ]
                    GermlineLogLik1 = AllMosaicGTLogLik[GermlineGT1_for_output]
                    GermlineLogLik2 = AllMosaicGTLogLik[GermlineGT2_for_output]
                    if Germline1LogPosterior >= Germline2LogPosterior:
                        GermlineGT_for_output = GermlineGT1_for_output
                        GermlineLogPosterior = Germline1LogPosterior
                        GermlineLogLik = GermlineLogLik1
                    else:
                        GermlineGT_for_output = GermlineGT2_for_output
                        GermlineLogPosterior = Germline2LogPosterior
                        GermlineLogLik = GermlineLogLik2
                ConsistentMaxMosaicLogPosterior = logsumexp(
                    [MaxMosaicLogPosterior, MaxMosaicLogPosterior]
                ) - logsumexp(
                    [
                        MaxMosaicLogPosterior,
                        MaxMosaicLogPosterior,
                        Germline1LogPosterior,
                        Germline2LogPosterior,
                    ]
                )
            (
                Non_Germ2germLogLikLogPosterior_dict,
                _,
            ) = self.cal_different_levels_log_mosaic_posterior("allmosaic")
            if POSTERIOR_METHOD == "Likelihood":
                Non_Germ2germLogPosterior = (
                    Non_Germ2germLogLikLogPosterior_dict["posterior"]
                )
                Non_Germ2germLogLikFraction = Non_Germ2germLogPosterior
            else:
                Non_Germ2germLogPosterior = (
                    Non_Germ2germLogLikLogPosterior_dict["posterior"]
                )
                Non_Germ2germLogLikFraction = (
                    Non_Germ2germLogLikLogPosterior_dict["lik"]
                )
            return (
                MaxMosaicGT_for_output,
                MaxMosaicLogPosterior,
                MaxMosaicLogLik,
                SecondMosaicGT_for_output,
                SecondMosaicLogPosterior,
                SecondMosaicLogLik,
                GermlineGT_for_output,
                GermlineLogPosterior,
                GermlineLogLik,
                Germline1LogLik,
                Germline2LogLik,
                Non_Germ2germLogPosterior,
                Non_Germ2germLogLikFraction,
                ConsistentMaxMosaicLogPosterior,
            )  # XXX: 没有 np.exp
        else:
            return tuple([False] * 14)
        # pass  # TODO: Need format information for output

    def all_reads_all_mosaic_gts_log_lik(self, mosaic_fraction):
        self.all_reads_all_mosaic_gts_log_lik_array = np.zeros(
            (len(self.reads_list), len(self.mosaic_genotypes)),
            dtype=np.float64,
        )
        for read_index, read in enumerate(self.reads_list):
            for mosaic_index, cells in enumerate(self.mosaic_genotypes):
                germ_cells_index = cells[0]
                mut_cells_index = cells[1]
                per_read_germ_log_lik = self.all_reads_all_gts_log_lik_array[
                    read_index, germ_cells_index
                ]
                per_read_mut_log_lik = self.all_reads_all_gts_log_lik_array[
                    read_index, mut_cells_index
                ]
                per_read_mosaic_log_lik = logsumexp(
                    [per_read_germ_log_lik, per_read_mut_log_lik],
                    b=[
                        1 - mosaic_fraction[mosaic_index],
                        mosaic_fraction[mosaic_index],
                    ],
                )
                self.all_reads_all_mosaic_gts_log_lik_array[
                    read_index, mosaic_index
                ] = per_read_mosaic_log_lik
        self.all_mosaic_gts_log_lik_array = np.sum(
            self.all_reads_all_mosaic_gts_log_lik_array, axis=0
        )

    def Gi_log_lik_with_mosaic_gt_order(self):
        self.all_mosaic_gts_Gi_log_lik_array = []
        for germ_cells, mut_cells in self.mosaic_genotypes:
            Gi_log_lik = self.all_gts_log_lik_array[germ_cells]
            self.all_mosaic_gts_Gi_log_lik_array.append(Gi_log_lik)
        self.all_mosaic_gts_Gi_log_lik_array = np.array(
            self.all_mosaic_gts_Gi_log_lik_array
        )

    def cal_all_gts_trans_mosaic_log_posterior(self):
        # 获得 self.all_mosaic_gts_log_lik_prior_array
        # 获得 self.selected_mosaic_gts_log_lik_array
        # XXX 此段代码逻辑写得很奇怪，但是只是单纯为了获得一些变量属性
        if POSTERIOR_METHOD == "Gi-Given":
            self.all_mosaic_gts_log_lik_prior_array = (
                self.all_mosaic_gts_Gi_log_lik_array
                + self.all_mosaic_gts_log_lik_array
                + self.log_mutation_rate_prior_with_germ2germ_rate_array
            )
        elif POSTERIOR_METHOD == "Gi-free":
            self.all_mosaic_gts_log_lik_prior_array = (
                self.all_mosaic_gts_log_lik_array
                + self.log_mutation_rate_prior_with_germ2germ_rate_array
            )
        elif POSTERIOR_METHOD == "Likelihood":
            self.all_mosaic_gts_log_lik_prior_array = (
                self.all_mosaic_gts_log_lik_array
            )
        selected_mosaic_index = self.select_mosaic_genotypes_index()
        self.mosaic_posterior_order = selected_mosaic_index
        self.selected_mosaic_gts_log_lik_prior_array = (
            self.all_mosaic_gts_log_lik_prior_array[selected_mosaic_index]
        )
        self.all_mosaic_gts_total_log_posterior = logsumexp(
            self.selected_mosaic_gts_log_lik_prior_array
        )
        self.all_mosaic_gts_log_posterior_array = (
            self.selected_mosaic_gts_log_lik_prior_array
            - self.all_mosaic_gts_total_log_posterior
        )
        if POSTERIOR_METHOD == "Likelihood":
            self.selected_mosaic_gts_log_lik_array = (
                self.selected_mosaic_gts_log_lik_prior_array
            )
            self.selected_all_mosaic_gts_total_log_lik = (
                self.all_mosaic_gts_total_log_posterior
            )
            self.selected_all_mosaic_gts_log_lik_fraction_array = (
                self.all_mosaic_gts_log_posterior_array
            )
        else:
            self.selected_mosaic_gts_log_lik_array = (
                self.all_mosaic_gts_log_lik_array[selected_mosaic_index]
            )
            self.selected_all_mosaic_gts_total_log_lik = logsumexp(
                self.selected_mosaic_gts_log_lik_array
            )
            self.selected_all_mosaic_gts_log_lik_fraction_array = (
                self.selected_mosaic_gts_log_lik_array
                - self.selected_all_mosaic_gts_total_log_lik
            )

    def select_mosaic_genotypes_index(self):
        all_mosaic_index = list(range(len(self.mosaic_genotypes)))
        het2hom_mosaic_index = [
            index
            for index, value in enumerate(self.mosaic_gts_types)
            if value in ["het2hom"]
        ]
        germ2germ_mosaic_index = [
            index
            for index, value in enumerate(self.mosaic_gts_types)
            if value == "germ2germ"
        ]
        mosaic_index = [
            index
            for index, value in enumerate(self.mosaic_fraction_type)
            if value == "mosaic"
        ]
        germ_1_mut = [
            index
            for index, value in enumerate(self.mosaic_gts_types)
            if value in ["germ2germ", "het2hom", "homhet2het"]
        ]
        if POSTERIOR_GERM_INCLUDED:
            if POSTERIOR_MUT_TYPE == "all":
                if POSTERIOR_ALLOW_MOSAIC_FRACTION == "all":
                    if ALLOWABLE_HET2HOM:
                        selected_mosaic_index = all_mosaic_index
                    else:
                        selected_mosaic_index = list(
                            set(all_mosaic_index) - set(het2hom_mosaic_index)
                        )
                else:
                    if ALLOWABLE_HET2HOM:
                        selected_mosaic_index = list(
                            set(germ2germ_mosaic_index + mosaic_index)
                        )
                    else:
                        selected_mosaic_index = list(
                            set(germ2germ_mosaic_index + mosaic_index)
                            - set(het2hom_mosaic_index)
                        )
            elif POSTERIOR_MUT_TYPE == "1mut":
                if POSTERIOR_ALLOW_MOSAIC_FRACTION == "all":
                    if ALLOWABLE_HET2HOM:
                        selected_mosaic_index = germ_1_mut
                    else:
                        selected_mosaic_index = list(
                            set(germ_1_mut) - set(het2hom_mosaic_index)
                        )
                else:
                    if ALLOWABLE_HET2HOM:
                        selected_mosaic_index1 = list(
                            set(germ_1_mut) - set(germ2germ_mosaic_index)
                        )
                    else:
                        selected_mosaic_index1 = list(
                            set(germ_1_mut)
                            - set(germ2germ_mosaic_index)
                            - set(het2hom_mosaic_index)
                        )
                    selected_mosaic_index2 = list(
                        set(selected_mosaic_index1) & set(mosaic_index)
                    )
                    selected_mosaic_index = list(
                        set(germ2germ_mosaic_index + selected_mosaic_index2)
                    )
        else:
            if POSTERIOR_MUT_TYPE == "all":
                if POSTERIOR_ALLOW_MOSAIC_FRACTION == "all":
                    if ALLOWABLE_HET2HOM:
                        selected_mosaic_index = list(
                            set(all_mosaic_index) - set(germ2germ_mosaic_index)
                        )
                    else:
                        selected_mosaic_index = list(
                            set(all_mosaic_index)
                            - set(germ2germ_mosaic_index)
                            - set(het2hom_mosaic_index)
                        )
                else:
                    if ALLOWABLE_HET2HOM:
                        selected_mosaic_index1 = list(
                            set(all_mosaic_index) - set(germ2germ_mosaic_index)
                        )
                    else:
                        selected_mosaic_index1 = list(
                            set(all_mosaic_index)
                            - set(germ2germ_mosaic_index)
                            - set(het2hom_mosaic_index)
                        )
                    selected_mosaic_index = list(
                        set(selected_mosaic_index1) & set(mosaic_index)
                    )
            elif POSTERIOR_MUT_TYPE == "1mut":
                if POSTERIOR_ALLOW_MOSAIC_FRACTION == "all":
                    if ALLOWABLE_HET2HOM:
                        if (
                            CONSIDER_GERM2GERM_IN_POSTERIOR
                        ):  # TODO: 其他条件下还没有添加该参数，需要后续添加一下
                            selected_mosaic_index = [
                                index
                                for index, value in enumerate(
                                    self.mosaic_gts_types
                                )
                                if value
                                in ["het2hom", "homhet2het", "germ2germ"]
                            ]
                        else:
                            selected_mosaic_index = [
                                index
                                for index, value in enumerate(
                                    self.mosaic_gts_types
                                )
                                if value in ["het2hom", "homhet2het"]
                            ]
                    else:
                        if CONSIDER_GERM2GERM_IN_POSTERIOR:
                            selected_mosaic_index = [
                                index
                                for index, value in enumerate(
                                    self.mosaic_gts_types
                                )
                                if value in ["homhet2het", "germ2germ"]
                            ]
                        else:
                            selected_mosaic_index = [
                                index
                                for index, value in enumerate(
                                    self.mosaic_gts_types
                                )
                                if value in ["homhet2het"]
                            ]
                else:
                    if ALLOWABLE_HET2HOM:
                        selected_mosaic_index1 = [
                            index
                            for index, value in enumerate(
                                self.mosaic_gts_types
                            )
                            if value in ["het2hom", "homhet2het"]
                        ]
                        selected_mosaic_index = list(
                            set(selected_mosaic_index1) & set(mosaic_index)
                        )
                    else:
                        selected_mosaic_index1 = [
                            index
                            for index, value in enumerate(
                                self.mosaic_gts_types
                            )
                            if value in ["homhet2het"]
                        ]
                        selected_mosaic_index = list(
                            set(selected_mosaic_index1) & set(mosaic_index)
                        )
        return selected_mosaic_index

    def cal_different_levels_log_mosaic_posterior(self, posterior_level):
        # XXX: maybe exist some GT should be selected but has been filtered in selecting mosaic index
        # XXX: and contribution inconsistent posterior calculation between unordered GT
        # for merge GT,because use mosaic index, so all_mosaic_gts_log_lik_prior_array include all mosaic genotypes and mosaic index will be used
        # for calculation merge GT posterior, all_mosaic_gts_total_log_posterior will be used
        mosaic_posterior_order = self.mosaic_posterior_order
        log_lik_log_posterior_dict, mosaicGT_dict_dict = {}, {}
        log_lik_and_log_posterior_dict = {
            "lik": {
                "all_mosaic_gts_log_posterior_array": self.selected_all_mosaic_gts_log_lik_fraction_array,
                "all_mosaic_gts_log_lik_prior_array": self.all_mosaic_gts_log_lik_array,
                "all_mosaic_gts_total_log_posterior": self.selected_all_mosaic_gts_total_log_lik,
            },
            "posterior": {
                "all_mosaic_gts_log_posterior_array": self.all_mosaic_gts_log_posterior_array,
                "all_mosaic_gts_log_lik_prior_array": self.all_mosaic_gts_log_lik_prior_array,
                "all_mosaic_gts_total_log_posterior": self.all_mosaic_gts_total_log_posterior,
            },
        }
        for lik_or_post, data_dict in log_lik_and_log_posterior_dict.items():
            if POSTERIOR_METHOD == "Likelihood":
                if lik_or_post == "lik":
                    continue
            all_mosaic_gts_log_posterior_array = data_dict[
                "all_mosaic_gts_log_posterior_array"
            ]
            all_mosaic_gts_log_lik_prior_array = data_dict[
                "all_mosaic_gts_log_lik_prior_array"
            ]
            all_mosaic_gts_total_log_posterior = data_dict[
                "all_mosaic_gts_total_log_posterior"
            ]
            if posterior_level == "orderedGT":
                log_posterior = all_mosaic_gts_log_posterior_array
                mosaicGT_dict = {"AllMosaicGT": mosaic_posterior_order}
            elif posterior_level == "unorderedGT":
                unorderedGT_log_posterior_dict = {}
                mosaicGT_dict = {}
                for mosaic_index in mosaic_posterior_order:
                    cells = self.mosaic_genotypes[mosaic_index]
                    germ_cells_index = cells[0]
                    mut_cells_index = cells[1]
                    germ_cells_gt = self.genotypes[germ_cells_index]
                    mut_cells_gt = self.genotypes[mut_cells_index]
                    germ_cells_unordered_gt = tuple(sorted(germ_cells_gt))
                    mut_cells_unordered_gt = tuple(sorted(mut_cells_gt))
                    unorderedGT_log_posterior_dict.setdefault(
                        (germ_cells_unordered_gt, mut_cells_unordered_gt), []
                    ).append(all_mosaic_gts_log_lik_prior_array[mosaic_index])
                    mosaicGT_dict.setdefault(
                        (germ_cells_unordered_gt, mut_cells_unordered_gt), []
                    ).append(mosaic_index)
                log_posterior = {}
                for (
                    unordered_mosaic_gt,
                    log_posterior_list,
                ) in unorderedGT_log_posterior_dict.items():
                    log_posterior[unordered_mosaic_gt] = (
                        logsumexp(log_posterior_list)
                        - all_mosaic_gts_total_log_posterior
                    )
            elif posterior_level == "unorderedGTunorderedMutDirection":
                unorderedGTunorderedMutDirection_log_posterior_dict = {}
                mosaicGT_dict = {}
                for mosaic_index in mosaic_posterior_order:
                    cells = self.mosaic_genotypes[mosaic_index]
                    germ_cells_index = cells[0]
                    mut_cells_index = cells[1]
                    germ_cells_gt = self.genotypes[germ_cells_index]
                    mut_cells_gt = self.genotypes[mut_cells_index]
                    germ_cells_unordered_gt = tuple(sorted(germ_cells_gt))
                    mut_cells_unordered_gt = tuple(sorted(mut_cells_gt))
                    mosaic_unordered_gt = tuple(
                        sorted(
                            [germ_cells_unordered_gt, mut_cells_unordered_gt]
                        )
                    )
                    unorderedGTunorderedMutDirection_log_posterior_dict.setdefault(
                        mosaic_unordered_gt, []
                    ).append(
                        all_mosaic_gts_log_lik_prior_array[mosaic_index]
                    )
                    mosaicGT_dict.setdefault(mosaic_unordered_gt, []).append(
                        mosaic_index
                    )
                log_posterior = {}
                for (
                    unorderedGT_unorderedMutDirection_mosaic_gt,
                    log_posterior_list,
                ) in (
                    unorderedGTunorderedMutDirection_log_posterior_dict.items()
                ):
                    log_posterior[
                        unorderedGT_unorderedMutDirection_mosaic_gt
                    ] = (
                        logsumexp(log_posterior_list)
                        - all_mosaic_gts_total_log_posterior
                    )
            elif posterior_level == "AlleleTransformation":
                mosaicGT_dict = {}
                AlleleTransformation_log_posterior_dict = {}
                for mosaic_index in mosaic_posterior_order:
                    cells = self.mosaic_genotypes[mosaic_index]
                    germ_cells_index = cells[0]
                    mut_cells_index = cells[1]
                    germ_cells_gt = self.genotypes[germ_cells_index]
                    mut_cells_gt = self.genotypes[mut_cells_index]
                    if germ_cells_gt == mut_cells_gt:
                        AlleleTransformation_log_posterior_dict.setdefault(
                            (germ_cells_gt[0], mut_cells_gt[0]), []
                        ).append(
                            all_mosaic_gts_log_lik_prior_array[mosaic_index]
                        )
                        mosaicGT_dict.setdefault(
                            (germ_cells_gt[0], mut_cells_gt[0]), []
                        ).append(mosaic_index)
                    elif (mut_cells_gt[0] == germ_cells_gt[0]) and (
                        mut_cells_gt[1] != germ_cells_gt[1]
                    ):  # XXX
                        mut_allele = mut_cells_gt[1]
                        germ_allele = germ_cells_gt[1]
                        AlleleTransformation_log_posterior_dict.setdefault(
                            (germ_allele, mut_allele), []
                        ).append(
                            all_mosaic_gts_log_lik_prior_array[mosaic_index]
                        )
                        mosaicGT_dict.setdefault(
                            (germ_allele, mut_allele), []
                        ).append(mosaic_index)
                    elif (mut_cells_gt[0] != germ_cells_gt[0]) and (
                        mut_cells_gt[1] == germ_cells_gt[1]
                    ):  # XXX
                        mut_allele = mut_cells_gt[0]
                        germ_allele = germ_cells_gt[0]
                        AlleleTransformation_log_posterior_dict.setdefault(
                            (germ_allele, mut_allele), []
                        ).append(
                            all_mosaic_gts_log_lik_prior_array[mosaic_index]
                        )
                        mosaicGT_dict.setdefault(
                            (germ_allele, mut_allele), []
                        ).append(mosaic_index)
                    elif (mut_cells_gt[0] == germ_cells_gt[1]) and (
                        mut_cells_gt[1] != germ_cells_gt[0]
                    ):  # XXX
                        mut_allele = mut_cells_gt[1]
                        germ_allele = germ_cells_gt[0]
                        AlleleTransformation_log_posterior_dict.setdefault(
                            (germ_allele, mut_allele), []
                        ).append(
                            all_mosaic_gts_log_lik_prior_array[mosaic_index]
                        )
                        mosaicGT_dict.setdefault(
                            (germ_allele, mut_allele), []
                        ).append(mosaic_index)
                    elif (mut_cells_gt[0] != germ_cells_gt[1]) and (
                        mut_cells_gt[1] == germ_cells_gt[0]
                    ):  # XXX
                        mut_allele = mut_cells_gt[0]
                        germ_allele = germ_cells_gt[1]
                        AlleleTransformation_log_posterior_dict.setdefault(
                            (germ_allele, mut_allele), []
                        ).append(
                            all_mosaic_gts_log_lik_prior_array[mosaic_index]
                        )
                        mosaicGT_dict.setdefault(
                            (germ_allele, mut_allele), []
                        ).append(mosaic_index)
                    else:
                        AlleleTransformation_log_posterior_dict.setdefault(
                            ("2mut", "2mut"), []
                        ).append(
                            all_mosaic_gts_log_lik_prior_array[mosaic_index]
                        )
                        mosaicGT_dict.setdefault(("2mut", "2mut"), []).append(
                            mosaic_index
                        )
                log_posterior = {}
                for (
                    allele_transformation,
                    log_posterior_list,
                ) in AlleleTransformation_log_posterior_dict.items():
                    log_posterior[allele_transformation] = (
                        logsumexp(log_posterior_list)
                        - all_mosaic_gts_total_log_posterior
                    )
            elif posterior_level == "allmosaic":
                germ2germ = list(
                    set(mosaic_posterior_order) & set(self.germ2germ_list)
                )
                non_germ2germ = list(
                    set(mosaic_posterior_order) - set(germ2germ)
                )
                log_posterior = np.log(
                    1
                    - np.exp(
                        logsumexp(
                            all_mosaic_gts_log_lik_prior_array[germ2germ]
                            - all_mosaic_gts_total_log_posterior
                        )
                    )
                )
                mosaicGT_dict = {"non_germ2germ": non_germ2germ}
            elif posterior_level == "homhet":
                homhet = list(
                    set(mosaic_posterior_order) & set(self.homhet_list)
                )
                log_posterior = logsumexp(
                    all_mosaic_gts_log_lik_prior_array[homhet]
                    - all_mosaic_gts_total_log_posterior
                )
                mosaicGT_dict = {"homhet": homhet}
            elif posterior_level == "hethet":
                hethet = list(
                    set(mosaic_posterior_order) & set(self.hethet_list)
                )
                log_posterior = logsumexp(
                    all_mosaic_gts_log_lik_prior_array[hethet]
                    - all_mosaic_gts_total_log_posterior
                )
                mosaicGT_dict = {"hethet": hethet}
            elif posterior_level == "hom2hom_two_mut":
                hom2hom_two_mut = list(
                    set(mosaic_posterior_order) & set(self.hom2hom_two_mut)
                )
                log_posterior = logsumexp(
                    all_mosaic_gts_log_lik_prior_array[hom2hom_two_mut]
                    - all_mosaic_gts_total_log_posterior
                )
                mosaicGT_dict = {"hom2hom_two_mut": hom2hom_two_mut}
            log_lik_log_posterior_dict[lik_or_post] = log_posterior
            mosaicGT_dict_dict[lik_or_post] = mosaicGT_dict
        return log_lik_log_posterior_dict, mosaicGT_dict_dict

    def get_mosaic_gt(self, mosaic_index):
        cells = self.mosaic_genotypes[mosaic_index]
        germ_cells_index = cells[0]
        mut_cells_index = cells[1]
        germ_cells_gt = self.genotypes[germ_cells_index]
        mut_cells_gt = self.genotypes[mut_cells_index]
        return tuple([germ_cells_gt, mut_cells_gt])  # tuple

    def get_final_gt_index(self, gt_output):
        new_gt = []
        for current_sample_allele_index in gt_output:
            allele = self.alleles_used_for_gt[current_sample_allele_index]
            new_index = self.all_alleles_list_for_gt_output.index(allele)
            new_gt.append(new_index)
        return tuple(new_gt)  # tuple

    def get_final_mosaic_index(self, mosaic_output):
        new_mosaic_output = []
        for current_sample_gt_index in mosaic_output:
            new_gt = self.get_final_gt_index(current_sample_gt_index)
            new_mosaic_output.append(tuple(new_gt))
        return tuple(new_mosaic_output)  # tuple

    def gt_output_format(self, ref_index, gt_output):
        collate_gt = []
        for allele_index in gt_output:
            if allele_index == ref_index:
                collate_gt.append(0)
            else:
                if allele_index < ref_index:
                    collate_gt.append(allele_index + 1)
                elif allele_index > ref_index:
                    collate_gt.append(allele_index)
        return tuple(collate_gt)

    def collate_allele_output_format(self, ref_index):
        self.alleles_used_for_output = self.alleles_used_for_gt.copy()
        if ref_index < self.alleles_number_used_for_gt:
            _ = self.alleles_used_for_output.pop(ref_index)
        else:
            pass

    def cal_mosaic_log_posterior_given_Gi(self, Gi_index):
        selected_mosaic_index_given_Gi = []
        all_mosaic_gts_log_lik_prior_array = (
            self.all_mosaic_gts_Gi_log_lik_array
        )
        selected_mosaic_index = self.mosaic_posterior_order
        for mosaic_index, cells in enumerate(self.mosaic_genotypes):
            germ_cells_index = cells[0]
            if germ_cells_index == Gi_index:
                if mosaic_index in selected_mosaic_index:
                    selected_mosaic_index_given_Gi.append(mosaic_index)
        mosaic_posterior_order_given_Gi = selected_mosaic_index_given_Gi
        selected_mosaic_gts_log_lik_prior_given_Gi_array = (
            all_mosaic_gts_log_lik_prior_array[mosaic_posterior_order_given_Gi]
        )
        selected_mosaic_gts_log_posterior_given_Gi_array = (
            selected_mosaic_gts_log_lik_prior_given_Gi_array
            - logsumexp(selected_mosaic_gts_log_lik_prior_given_Gi_array)
        )
        return selected_mosaic_gts_log_posterior_given_Gi_array


# reads_list,
# reads_accuracy_list,
# alleles_STR_length_list,
# alleles_depth_dict,
# all_alleles_list_for_gt_output,  # alleles_sequence_depth_dict,
# stutter_model_params,
# in_frame_stutter_model,
# out_frame_stutter_model,
# locus_infos,
# other_params,
# import pandas as pd
# def main31():
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(
#     4,
#     8,
#     figsize=(30, 10),
#     sharex=False,
#     sharey=False,
# )
#     demo1_mosaic_fraction = []
#     demo1_reads_accuracy_list = []
#     demo1_alleles_STR_length_list = [12,14]
#     demo1_all_alleles_list_for_gt_output=[12,14]
#     demo1_mosaic_genotype_lik = {}
#     for i in range(31):
#         row = i // 8
#         col = i % 8
#         demo1_reads_list = [12 for _ in range(i)] + [14 for _ in range(30 - i)]
#         demo1_alleles_depth_dict = {12: i, 14: 30 - i}
#         demo1_locus_infos = {
#             "chr": "chr1",
#             "str_zero_based_start_included": 26453,
#             "str_zero_based_end_excluded": 26465,
#             "motif_length": 2,
#             "period": 6,
#             "STR_id": "Human_STR_3",
#             "motif": "GT",
#             "ploidy": 2,
#         }
#         demo1_other_params = {
#             "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim31_correct_weixiang3_cell_based_log_onlylik.txt",
#             "phase_mode": False,
#         }
#         demo1_stutter_model = {
#             "inframe_single_step_prob": 0.9,
#             "inframe_ins_prob": 0.01,
#             "inframe_del_prob": 0.01,
#             "outframe_single_step_prob": 0.9,
#             "outframe_ins_prob": 0.01,
#             "outframe_del_prob": 0.01,
#         }
#         stutter_model_params = demo1_stutter_model
#         inframe_stutter_model = myutils.stutter_model(
#             stutter_model_params["inframe_single_step_prob"],
#             stutter_model_params["inframe_ins_prob"],
#             stutter_model_params["inframe_del_prob"],
#             2,
#             )
#         outframe_stutter_model = myutils.stutter_model(
#             stutter_model_params["outframe_single_step_prob"],
#             stutter_model_params["outframe_ins_prob"],
#             stutter_model_params["outframe_del_prob"],
#             2,
#             )
#         demo1_mosaic_fraction_estimation = MosaicFractionPerSampleEstimator(
#             demo1_reads_list,
#             demo1_reads_accuracy_list,
#             demo1_alleles_STR_length_list,
#             demo1_alleles_depth_dict,
#             demo1_all_alleles_list_for_gt_output,
#             demo1_stutter_model,
#             inframe_stutter_model,
#             outframe_stutter_model,
#             demo1_locus_infos,
#             demo1_other_params,
#         )
#         if demo1_mosaic_fraction_estimation.callable:
#             demo1_mosaic_fraction_estimation.mosaic_fraction_estimation()
#         print(ITERATION_MODE)
#         # demo1_mosaic_fraction_estimation.train()
#         print(demo1_mosaic_fraction_estimation.locus_state)
#         demo1_mosaic_fraction.append(
#             demo1_mosaic_fraction_estimation.mosaic_fraction
#         )
#         for index, value in enumerate(demo1_mosaic_fraction_estimation.all_mosaic_log_prior_lik_list):
#             demo1_mosaic_genotype_lik.setdefault(index, []).append(value)
#         for germ_index, germ_lik in enumerate(demo1_mosaic_fraction_estimation.all_gts_log_lik_array):
#             demo1_mosaic_genotype_lik.setdefault(germ_index+len(demo1_mosaic_fraction_estimation.all_mosaic_log_prior_lik_list), []).append(germ_lik)
#         if PLOT:
#             import matplotlib.pyplot as plt
#             log_ll = (
#                 demo1_mosaic_fraction_estimation.total_mosaic_log_prior_lik_list
#             )
#             epochs = range(1, len(log_ll) + 1)
#             axs[row, col].plot(epochs, log_ll, label="Log-Likelihood")
#             axs[row, col].set_xlabel("Epochs")
#             axs[row, col].set_ylabel("Log-Likelihood")
#     plt.subplots_adjust(wspace=0.6, hspace=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#     plt.savefig(
#         "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim31_correct_weixiang3_converge_curve_test_cell_based_onlylik.png",
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()
#     if PLOT:
#         import matplotlib.pyplot as plt

#         plt.plot(
#             list(range(31)),
#             demo1_mosaic_fraction,
#             "b",
#             label="Mosaic Fraction",
#         )
#         print(demo1_mosaic_fraction)
#         plt.title("Mosaic Fraction")
#         plt.xlabel("Mosaic Allele Depth")
#         plt.ylabel("Mosaic Fraction")
#         plt.legend()
#         plt.savefig(
#             "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim_31_correct_weixiang3_cell_based_onlylik.png",
#             bbox_inches="tight",
#             dpi=200,
#         )
#         plt.show()
#         print(demo1_mosaic_genotype_lik)
#         print(demo1_mosaic_fraction_estimation.all_mosaic_gt_list)
#         print(demo1_mosaic_fraction_estimation.mosaic_genotypes)
#         demo1_mosaic_all_genotype_lik = pd.DataFrame(demo1_mosaic_genotype_lik)
#         for index, row in demo1_mosaic_all_genotype_lik.iterrows():
#             x = row[demo1_mosaic_all_genotype_lik.columns].values
#             x_norm = np.exp(x - logsumexp(x))
#             demo1_mosaic_all_genotype_lik.loc[index, demo1_mosaic_all_genotype_lik.columns] = x_norm
#         plt.figure(figsize=(10, 5))

#         # 绘制每个产品的数据
#         for feature in demo1_mosaic_all_genotype_lik.columns:
#             plt.plot(list(range(31)), demo1_mosaic_all_genotype_lik[feature], label=feature, marker='o')

#         plt.title('Mosaic Genotype Likelihood')
#         plt.xlabel('ALT allele depth')
#         plt.ylabel('Likelihood')
#         plt.legend()
#         plt.grid(False)
#         plt.savefig(
#             "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim_31_correct_weixiang3_cell_based_onlylik_mosaiclik.png",
#             bbox_inches="tight",
#             dpi=200,
#         )
#         plt.show()
# # Sim32 Test cell-based
# # cell-based hom2het stutter 0.1 with stutter reads
# def main32():
#     import matplotlib.pyplot as plt
#     demo1_mosaic_fraction = []
#     fig, axs = plt.subplots(
#     4,
#     8,
#     figsize=(30, 10),
#     sharex=False,
#     sharey=False,
# )
#     demo1_mosaic_fraction = []
#     demo1_reads_accuracy_list = []
#     demo1_alleles_STR_length_list = [12,14,10,16]
#     demo1_all_alleles_list_for_gt_output=[12,14,10,16]
#     demo1_mosaic_genotype_lik = {}
#     for i in range(31):
#         row = i // 8
#         col = i % 8
#         demo1_reads_list = [12 for _ in range(i)] + [14 for _ in range(30 - i)] + [10,10,16]
#         demo1_alleles_depth_dict = {12: i, 14: 30 - i,10:2,16:1}
#         demo1_locus_infos = {
#             "chr": "chr1",
#             "str_zero_based_start_included": 26453,
#             "str_zero_based_end_excluded": 26465,
#             "motif_length": 2,
#             "period": 6,
#             "STR_id": "Human_STR_3",
#             "motif": "GT",
#             "ploidy": 2,
#         }
#         demo1_other_params = {
#             "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim32_correct_weixiang3_cell_based_onlylik.txt",
#             "phase_mode": False,
#         }
#         demo1_stutter_model = {
#             "inframe_single_step_prob": 0.9,
#             "inframe_ins_prob": 0.01,
#             "inframe_del_prob": 0.01,
#             "outframe_single_step_prob": 0.9,
#             "outframe_ins_prob": 0.01,
#             "outframe_del_prob": 0.01,
#         }
#         stutter_model_params = demo1_stutter_model
#         inframe_stutter_model = myutils.stutter_model(
#             stutter_model_params["inframe_single_step_prob"],
#             stutter_model_params["inframe_ins_prob"],
#             stutter_model_params["inframe_del_prob"],
#             6,
#             )
#         outframe_stutter_model = myutils.stutter_model(
#             stutter_model_params["outframe_single_step_prob"],
#             stutter_model_params["outframe_ins_prob"],
#             stutter_model_params["outframe_del_prob"],
#             6,
#             )
#         demo1_mosaic_fraction_estimation = MosaicFractionPerSampleEstimator(
#             demo1_reads_list,
#             demo1_reads_accuracy_list,
#             demo1_alleles_STR_length_list,
#             demo1_alleles_depth_dict,
#             demo1_all_alleles_list_for_gt_output,
#             demo1_stutter_model,
#             inframe_stutter_model,
#             outframe_stutter_model,
#             demo1_locus_infos,
#             demo1_other_params,
#         )
#         if demo1_mosaic_fraction_estimation.callable:
#             demo1_mosaic_fraction_estimation.mosaic_fraction_estimation()
#         demo1_mosaic_fraction.append(
#             demo1_mosaic_fraction_estimation.mosaic_fraction
#         )
#         print(ITERATION_MODE)
#         # demo1_mosaic_fraction_estimation.train()
#         print(demo1_mosaic_fraction_estimation.locus_state)
#         for index, value in enumerate(demo1_mosaic_fraction_estimation.all_mosaic_log_prior_lik_list):
#             demo1_mosaic_genotype_lik.setdefault(index, []).append(value)
#         for germ_index, germ_lik in enumerate(demo1_mosaic_fraction_estimation.all_gts_log_lik_array):
#             demo1_mosaic_genotype_lik.setdefault(germ_index+len(demo1_mosaic_fraction_estimation.all_mosaic_log_prior_lik_list), []).append(germ_lik)
#         if PLOT:
#             log_ll = (
#                 demo1_mosaic_fraction_estimation.total_mosaic_log_prior_lik_list
#             )
#             epochs = range(1, len(log_ll) + 1)
#             axs[row, col].plot(epochs, log_ll, label="Log-Likelihood")
#             axs[row, col].set_xlabel("Epochs")
#             axs[row, col].set_ylabel("Log-Likelihood")
#     plt.subplots_adjust(wspace=0.6, hspace=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#     plt.savefig(
#         "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim32_correct_weixiang3_converge_curve_test_cell_based_onlylik.png",
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()
#     if PLOT:
#         import matplotlib.pyplot as plt

#         plt.plot(
#             list(range(31)),
#             demo1_mosaic_fraction,
#             "b",
#             label="Mosaic Fraction",
#         )
#         print(demo1_mosaic_fraction)
#         plt.title("Mosaic Fraction")
#         plt.xlabel("Mosaic Allele Depth")
#         plt.ylabel("Mosaic Fraction")
#         plt.legend()
#         plt.savefig(
#             "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim_32_correct_weixiang3_cell_based_onlylik.png",
#             bbox_inches="tight",
#             dpi=200,
#         )
#         plt.show()
#         print(demo1_mosaic_genotype_lik)
#         print(demo1_mosaic_fraction_estimation.all_mosaic_gt_list)
#         print(demo1_mosaic_fraction_estimation.mosaic_genotypes)
#         demo1_mosaic_all_genotype_lik = pd.DataFrame(demo1_mosaic_genotype_lik)
#         for index, row in demo1_mosaic_all_genotype_lik.iterrows():
#             x = row[demo1_mosaic_all_genotype_lik.columns].values
#             x_norm = np.exp(x - logsumexp(x))
#             demo1_mosaic_all_genotype_lik.loc[index, demo1_mosaic_all_genotype_lik.columns] = x_norm
#         plt.figure(figsize=(10, 5))

#         # 绘制每个产品的数据
#         colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(demo1_mosaic_all_genotype_lik.columns)))
#         k = 0
#         for feature, color in zip(demo1_mosaic_all_genotype_lik.columns, colors):
#             if sum(demo1_mosaic_all_genotype_lik[feature]) < 0.001:
#                 continue
#             plt.plot(list(range(31)), demo1_mosaic_all_genotype_lik[feature], label=feature, marker='o', color=colors[k])
#             k += 1
#         plt.title('Mosaic Genotype Likelihood')
#         plt.xlabel('ALT allele depth')
#         plt.ylabel('Likelihood')
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
#         plt.tight_layout()  # 自动调整布局以防止标签被剪切
#         plt.grid(False)
#         plt.savefig(
#             "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim_32_correct_weixiang3_cell_based_onlylik_mosaiclik.png",
#             bbox_inches="tight",
#             dpi=200,
#         )
#         plt.show()

# # Sim33 Test cell-based
# # cell-based het2het stutter 0.1 without stutter reads
# def main33():
#     import matplotlib.pyplot as plt
#     demo1_mosaic_fraction = []
#     fig, axs = plt.subplots(
#         4,
#         4,
#         figsize=(16, 10),
#         sharex=False,
#         sharey=False,
#     )
#     demo1_reads_accuracy_list = []
#     demo1_alleles_STR_length_list = [10,12,14]
#     demo1_all_alleles_list_for_gt_output=[10,12,14]
#     demo1_mosaic_genotype_lik = {}
#     for i in range(16):
#         row = i // 4
#         col = i % 4
#         demo1_reads_list = (
#             [10] * 15 + [12 for _ in range(i)] + [14 for _ in range(15 - i)]
#         )
#         demo1_alleles_depth_dict = {10: 15, 12: i, 14: 15 - i}
#         demo1_locus_infos = {
#             "chr": "chr1",
#             "str_zero_based_start_included": 26453,
#             "str_zero_based_end_excluded": 26465,
#             "motif_length": 2,
#             "period": 6,
#             "STR_id": "Human_STR_3",
#             "motif": "GT",
#             "ploidy": 2,
#         }
#         demo1_other_params = {
#             "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim33_correct_weixiang3_cell_based_log_onlylik.txt",
#             "phase_mode": False,
#         }
#         demo1_stutter_model = {
#             "inframe_single_step_prob": 0.9,
#             "inframe_ins_prob": 0.01,
#             "inframe_del_prob": 0.01,
#             "outframe_single_step_prob": 0.9,
#             "outframe_ins_prob": 0.01,
#             "outframe_del_prob": 0.01,
#         }
#         stutter_model_params = demo1_stutter_model
#         inframe_stutter_model = myutils.stutter_model(
#             stutter_model_params["inframe_single_step_prob"],
#             stutter_model_params["inframe_ins_prob"],
#             stutter_model_params["inframe_del_prob"],
#             4,
#             )
#         outframe_stutter_model = myutils.stutter_model(
#             stutter_model_params["outframe_single_step_prob"],
#             stutter_model_params["outframe_ins_prob"],
#             stutter_model_params["outframe_del_prob"],
#             4,
#             )
#         demo1_mosaic_fraction_estimation = MosaicFractionPerSampleEstimator(
#             demo1_reads_list,
#             demo1_reads_accuracy_list,
#             demo1_alleles_STR_length_list,
#             demo1_alleles_depth_dict,
#             demo1_all_alleles_list_for_gt_output,
#             demo1_stutter_model,
#             inframe_stutter_model,
#             outframe_stutter_model,
#             demo1_locus_infos,
#             demo1_other_params,
#         )
#         if demo1_mosaic_fraction_estimation.callable:
#             demo1_mosaic_fraction_estimation.mosaic_fraction_estimation()
#         print(ITERATION_MODE)
#         #demo1_mosaic_fraction_estimation.train()
#         print(demo1_mosaic_fraction_estimation.locus_state)
#         demo1_mosaic_fraction.append(
#             demo1_mosaic_fraction_estimation.mosaic_fraction
#         )
#         for index, value in enumerate(demo1_mosaic_fraction_estimation.all_mosaic_log_prior_lik_list):
#             demo1_mosaic_genotype_lik.setdefault(index, []).append(value)
#         for germ_index, germ_lik in enumerate(demo1_mosaic_fraction_estimation.all_gts_log_lik_array):
#             demo1_mosaic_genotype_lik.setdefault(germ_index+len(demo1_mosaic_fraction_estimation.all_mosaic_log_prior_lik_list), []).append(germ_lik)
#         if PLOT:
#             import matplotlib.pyplot as plt

#             log_ll = (
#                 demo1_mosaic_fraction_estimation.total_mosaic_log_prior_lik_list
#             )
#             epochs = range(1, len(log_ll) + 1)
#             axs[row, col].plot(epochs, log_ll, label="Log-Likelihood")
#             axs[row, col].set_xlabel("Epochs")
#             axs[row, col].set_ylabel("Log-Likelihood")
#     plt.subplots_adjust(wspace=0.4, hspace=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#     plt.savefig(
#         "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim33_correct_weixiang3_converge_curve_test_cell_based_onlylik.png",
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()
#     if PLOT:
#         import matplotlib.pyplot as plt

#         plt.plot(
#             list(range(16)),
#             demo1_mosaic_fraction,
#             "b",
#             label="Mosaic Fraction",
#         )
#         plt.title("Mosaic Fraction")
#         plt.xlabel("Mosaic Allele Depth")
#         plt.ylabel("Mosaic Fraction")
#         plt.legend()
#         plt.savefig(
#             "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim33_correct_weixiang3_cell_based_onlylik.png",
#             bbox_inches="tight",
#             dpi=200,
#         )
#         plt.show()
#         print(demo1_mosaic_genotype_lik)
#         print(demo1_mosaic_fraction_estimation.all_mosaic_gt_list)
#         print(demo1_mosaic_fraction_estimation.mosaic_genotypes)
#         demo1_mosaic_all_genotype_lik = pd.DataFrame(demo1_mosaic_genotype_lik)
#         for index, row in demo1_mosaic_all_genotype_lik.iterrows():
#             x = row[demo1_mosaic_all_genotype_lik.columns].values
#             x_norm = np.exp(x - logsumexp(x))
#             demo1_mosaic_all_genotype_lik.loc[index, demo1_mosaic_all_genotype_lik.columns] = x_norm
#         plt.figure(figsize=(10, 5))

#         # 绘制每个产品的数据
#         # colors = plt.cm.viridis(np.linspace(0, 1, len(demo1_mosaic_all_genotype_lik.columns)))
#         # colors = plt.cm.tab10(np.linspace(0, 1, len(demo1_mosaic_all_genotype_lik.columns)))
#         colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(demo1_mosaic_all_genotype_lik.columns)))
#         k = 0
#         for feature, color in zip(demo1_mosaic_all_genotype_lik.columns, colors):
#             if sum(demo1_mosaic_all_genotype_lik[feature]) < 0.001:
#                 continue
#             plt.plot(list(range(16)), demo1_mosaic_all_genotype_lik[feature], label=feature, marker='o', color=colors[k])
#             k += 1
#         plt.title('Mosaic Genotype Likelihood')
#         plt.xlabel('ALT allele depth')
#         plt.ylabel('Likelihood')
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
#         plt.tight_layout()  # 自动调整布局以防止标签被剪切
#         plt.grid(False)
#         plt.savefig(
#             "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim_33_correct_weixiang3_cell_based_onlylik_mosaiclik.png",
#             bbox_inches="tight",
#             dpi=200,
#         )
#         plt.show()

# # Sim34 Test cell-based
# # cell-based het2het stutter 0.1 with stutter reads
# def main34():
#     import matplotlib.pyplot as plt
#     demo1_mosaic_fraction = []
#     fig, axs = plt.subplots(
#         4,
#         4,
#         figsize=(16, 10),
#         sharex=False,
#         sharey=False,
#     )
#     demo1_mosaic_fraction = []
#     demo1_reads_accuracy_list = []
#     demo1_alleles_STR_length_list = [12,14,10,16,8,9,11,13]
#     demo1_all_alleles_list_for_gt_output=[12,14,10,16,8,9,11,13]
#     demo1_mosaic_genotype_lik = {}
#     for i in range(16):
#         row = i // 4
#         col = i % 4
#         demo1_reads_list = (
#             [10] * 15
#             + [12 for _ in range(i)]
#             + [14 for _ in range(15 - i)]
#             + [8]
#             + [9]
#             + [11]
#             + [13]
#         )
#         demo1_alleles_depth_dict = {
#             10: 15,
#             12: i,
#             14: 15 - i,
#             8: 1,
#             9: 1,
#             11: 1,
#             13: 1,
#         }
#         demo1_locus_infos = {
#             "chr": "chr1",
#             "str_zero_based_start_included": 26453,
#             "str_zero_based_end_excluded": 26465,
#             "motif_length": 2,
#             "period": 6,
#             "STR_id": "Human_STR_3",
#             "motif": "GT",
#             "ploidy": 2,
#         }
#         demo1_other_params = {
#             "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim34_correct_weixiang3_cell_based_log_onlylik.txt",
#             "phase_mode": False,
#         }
#         demo1_stutter_model = {
#             "inframe_single_step_prob": 0.9,
#             "inframe_ins_prob": 0.01,
#             "inframe_del_prob": 0.01,
#             "outframe_single_step_prob": 0.9,
#             "outframe_ins_prob": 0.01,
#             "outframe_del_prob": 0.01,
#         }

#         stutter_model_params = demo1_stutter_model
#         inframe_stutter_model = myutils.stutter_model(
#             stutter_model_params["inframe_single_step_prob"],
#             stutter_model_params["inframe_ins_prob"],
#             stutter_model_params["inframe_del_prob"],
#             8,
#             )
#         outframe_stutter_model = myutils.stutter_model(
#             stutter_model_params["outframe_single_step_prob"],
#             stutter_model_params["outframe_ins_prob"],
#             stutter_model_params["outframe_del_prob"],
#             8,
#             )
#         demo1_mosaic_fraction_estimation = MosaicFractionPerSampleEstimator(
#             demo1_reads_list,
#             demo1_reads_accuracy_list,
#             demo1_alleles_STR_length_list,
#             demo1_alleles_depth_dict,
#             demo1_all_alleles_list_for_gt_output,
#             demo1_stutter_model,
#             inframe_stutter_model,
#             outframe_stutter_model,
#             demo1_locus_infos,
#             demo1_other_params,
#         )
#         if demo1_mosaic_fraction_estimation.callable:
#             demo1_mosaic_fraction_estimation.mosaic_fraction_estimation()
#         print(ITERATION_MODE)
#         #demo1_mosaic_fraction_estimation.train()
#         print(demo1_mosaic_fraction_estimation.locus_state)
#         demo1_mosaic_fraction.append(
#             demo1_mosaic_fraction_estimation.mosaic_fraction
#         )
#         for index, value in enumerate(demo1_mosaic_fraction_estimation.all_mosaic_log_prior_lik_list):
#             demo1_mosaic_genotype_lik.setdefault(index, []).append(value)
#         for germ_index, germ_lik in enumerate(demo1_mosaic_fraction_estimation.all_gts_log_lik_array):
#             demo1_mosaic_genotype_lik.setdefault(germ_index+len(demo1_mosaic_fraction_estimation.all_mosaic_log_prior_lik_list), []).append(germ_lik)
#         if PLOT:
#             import matplotlib.pyplot as plt

#             log_ll = (
#                 demo1_mosaic_fraction_estimation.total_mosaic_log_prior_lik_list
#             )
#             epochs = range(1, len(log_ll) + 1)
#             axs[row, col].plot(epochs, log_ll, label="Log-Likelihood")
#             axs[row, col].set_xlabel("Epochs")
#             axs[row, col].set_ylabel("Log-Likelihood")
#     plt.subplots_adjust(wspace=0.4, hspace=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#     plt.savefig(
#         "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim34_correct_weixiang3_converge_curve_test_cell_based_onlylik.png",
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()
#     if PLOT:
#         import matplotlib.pyplot as plt

#         plt.plot(
#             list(range(16)),
#             demo1_mosaic_fraction,
#             "b",
#             label="Mosaic Fraction",
#         )
#         plt.title("Mosaic Fraction")
#         plt.xlabel("Mosaic Allele Depth")
#         plt.ylabel("Mosaic Fraction")
#         plt.legend()
#         plt.savefig(
#             "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim_34_correct_weixiang3_cell_based_onlylik.png",
#             bbox_inches="tight",
#             dpi=200,
#         )
#         plt.show()
#         print(demo1_mosaic_genotype_lik)
#         print(demo1_mosaic_fraction_estimation.all_mosaic_gt_list)
#         print(demo1_mosaic_fraction_estimation.mosaic_genotypes)
#         demo1_mosaic_all_genotype_lik = pd.DataFrame(demo1_mosaic_genotype_lik)
#         for index, row in demo1_mosaic_all_genotype_lik.iterrows():
#             x = row[demo1_mosaic_all_genotype_lik.columns].values
#             x_norm = np.exp(x - logsumexp(x))
#             demo1_mosaic_all_genotype_lik.loc[index, demo1_mosaic_all_genotype_lik.columns] = x_norm
#         plt.figure(figsize=(10, 5))

#         # 绘制每个产品的数据
#         # colors = plt.cm.viridis(np.linspace(0, 1, len(demo1_mosaic_all_genotype_lik.columns)))
#         # colors = plt.cm.tab10(np.linspace(0, 1, len(demo1_mosaic_all_genotype_lik.columns)))
#         colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(demo1_mosaic_all_genotype_lik.columns)))
#         k = 0
#         for feature, color in zip(demo1_mosaic_all_genotype_lik.columns, colors):
#             if sum(demo1_mosaic_all_genotype_lik[feature]) < 0.001:
#                 continue
#             plt.plot(list(range(16)), demo1_mosaic_all_genotype_lik[feature], label=feature, marker='o', color=colors[k])
#             k += 1
#         plt.title('Mosaic Genotype Likelihood')
#         plt.xlabel('ALT allele depth')
#         plt.ylabel('Likelihood')
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
#         plt.tight_layout()  # 自动调整布局以防止标签被剪切
#         plt.grid(False)
#         plt.savefig(
#             "/Users/lid/Github/MosaicSTR/test/test_mosaic_fraction_estimate/sim_34_correct_weixiang3_cell_based_onlylik_mosaiclik.png",
#             bbox_inches="tight",
#             dpi=200,
#         )
#         plt.show()

# if DEBUG:
#     import matplotlib.pyplot as plt
#     MAX_ITERATION = 500
#     MIN_ITERATION = 10
#     ALLOWABLE_HET2HOM = True
#     ITERATION_MODE = "cell-based"
#     UNORDER_MOSAIC_GT = True
#     if __name__ == "__main__":
#         print(
#             "Start the debugging of mosaic_fraction_estimation.py", flush=True
#         )
#         main31()
#         main32()
#         main33()
#         main34()
#         print("End the debugging of mosaic_fraction_estimation.py", flush=True)
# HACK: Need DEBUG because MaxIterationReached # TODO TO DEBUG
# Element 250x
# chr10   47523661        Human_STR_156101
# CAGTGCGGCCCCGGGCACCAGCCCTGGCCCCG
# GCCCCGGCCCCGGCCCCGGCTAGGG       CAGTGCGGCCCCGGGCACCAGCCCTGGCCCCGGCCCCGGCCCCAGCTA
# GGG,CAGTGCGGCCCCGGGCCCCAGCCCTGGCCCCGGCCCCGGCCCCGGCCCCGGCTAGGG,CAGGGCGGCCCCGGGCAC
# CAGCCCTGGCCCCGGCCCCGGCCCCGGCCCCGGCTAGGG,CAGTGCCGCCCCGGGCACCAGCCCTGGCCCCGGCCCAGGC
# CCCGGCCCCGGCTAAGG,CAGTGCGGCCCCAGGCACCAGCCCTGGCCCCGGCCCCGGCCCCGGCCCCGGCCCCGGCTAGG
# G,CAGTCCGGCCCCGGGCACCAGCCCTGGCCCCAGCCCCCGCCCCAGCCCCGGCTAGGG,CAGTGCAGCCCCGAGCACCA
# GCCCTTGCACCGGCCCCGGCCCCGGCCCCGGCTAGGG

# CAGGGCGGCCCCGGGCACCAGCCCTGGCCCCGGCCCCGGCCCCGGCCCCGGCTAGGG
# CAGTGC C GCCCC G GGCACCAGCCCTGGCCCCGGCCC A GGCCCCGGCCCCGGC        TA A GG,
# CAGTGC G GCCCC A GGCACCAGCCCTGGCCCCGGCCC C GGCCCCGGCCCCGGC CCCGGC TA G GG

# .:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:
# .:.:.:8:5/0:1.0:2465.406962234718:0.0:MaxIterationReached:.:.:.:.:.:.:.:.:.:.:.:
# .:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:NO:UnKnownNearbySNP:.:.:.:.:.:.:.:.:.:.:.:.:.:.:
# .:.:.:.:.:None:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:.:0,0,0,4,136,74,0:0.038,0.
# 795:0.038,0.962:0.0:.:.:.:.:.:.:.:.:.:.:.:not_all_singleton_obs_allele:.:.:.:.:.
# :.:.:.:.:.:.:.:.:.:.:.
# MAX_ITERATION = 500
# 在 Element 250x 的十几个位点碰到这种情况，其他数据和时候没遇到这种情况，可能存在少数时候没法收敛的情况，跑那么多数据只有这几个位点出现这种情况，因此暂先不管这种情况，暂先忽略不计哈
# HACK: Need DEBUG because MaxIterationReached # TODO TO DEBUG
