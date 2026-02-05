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
]  # 1 - 10**-4
MS_MUTATION_RATE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "ms_mutation_rate"
]  # 10**-4
INITIAL_MOSAIC_FRACTION = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "initial_mosaic_fraction"
]  # 0.1
USE_MINOR_CELLS_POP_AS_MUT_CELLS = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "use_minor_cells_pop_as_mut_cells"
]  # False
PHASING_ALLELE_FILTER = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "phasing_allele_filter"
]  # True
MAX_ALLOWABLE_ALLELE_NUM = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "max_allowable_allele_num"
]  # 5
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
]  # Likelihood
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
MIN_PHASING_AVERAGE_DEPTH = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "min_phasing_average_depth"
]  # 1
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
FILTER_BIG_STUTTER_SIZE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "FILTER_BIG_STUTTER_SIZE"
]  # True
BIG_STUTTER_SIZE = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "BIG_STUTTER_SIZE"
]  # {1: 20, 2: 10, 3: 10, 4: 9, 5: 8, 6: 7}
BIG_STUTTER_DEPTH_CUTOFF = MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "BIG_STUTTER_DEPTH_CUTOFF"
]  # 1
USE_UNPHASE_USED_ALLELE_ORDER = True
# HAP_IDENTIFY_POA = True
MIN_LOG_PROB = -100
# DEFAULT_MAX_STEP = 100
# HACK: For default steps and max step limitation for KeyErrors


class PhasedMosaicFractionPerSampleEstimator:
    def __init__(
        self,
        reads_list,
        reads_accuracy_list,
        alleles_STR_length_list,
        alleles_depth_dict,
        all_alleles_list_for_gt_output,
        unphase_alleles_used_for_gt,
        # alleles_sequence_depth_dict,
        nearby_snp,
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
        if ALLELE_SORT:
            alleles_depth_dict = dict(
                sorted(alleles_depth_dict.items(), key=lambda x: x[1])
            )
        logger_config.configure_logger(other_params["log_file_path"])
        self.reads_list = reads_list
        self.reads_accuracy_list = reads_accuracy_list
        self.reads_number = len(self.reads_list)
        self.reads_alleles_list = list(set(self.reads_list))
        self.reads_alleles_number = len(self.reads_alleles_list)
        self.reads_alleles_str_list = []
        self.reads_alleles_snp_list = []
        for reads_alleles in self.reads_alleles_list:
            self.reads_alleles_str_list.append(reads_alleles[0])
            self.reads_alleles_snp_list.append(reads_alleles[1])
        self.alleles_STR_length_list = alleles_STR_length_list
        self.all_alleles_list_for_gt_output = all_alleles_list_for_gt_output
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
        if USE_UNPHASE_USED_ALLELE_ORDER:
            self.alleles_used_for_gt = unphase_alleles_used_for_gt
        else:
            if PHASING_ALLELE_FILTER:
                self.allele_filter()
            else:
                self.alleles_used_for_gt = self.alleles_order
                # self.alleles_seq_used_for_gt = self.alleles_seq_order
        self.alleles_number_used_for_gt = len(self.alleles_used_for_gt)
        self.alleles_linking_order = []
        self.nearby_snp = nearby_snp
        for snp in nearby_snp:
            for str in self.alleles_used_for_gt:
                self.alleles_linking_order.append((str, snp))
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
                myutils.get_all_gt_ordered_permutations_with_replacement(
                    list(range(self.alleles_number_used_for_gt)), 2
                )
            )
            self.gt_number = len(self.genotypes)
            self.mosaic_genotypes = (
                myutils.get_all_gt_ordered_permutations_with_replacement(
                    list(range(self.gt_number)), 2
                )
            )

            self.all_reads_all_alleles_linking_log_lik()
            self.all_reads_all_phased_gts_log_lik()
            self.cal_mutation_rate_prior_phase()
            self.get_different_levels_mosaic_list()
            if DEBUG:
                if ESTIMATION_MODE == "double":
                    self.hom2het_total_mosaic_log_prior_lik_list = []
                    self.het2het_total_mosaic_log_prior_lik_list = []
                elif ESTIMATION_MODE == "unique":
                    self.total_mosaic_log_prior_lik_list = []
                else:
                    pass
        else:
            logging.info(
                f"This locus is not callable because {self.loci_calling}."
            )
            pass  # TODO: Need format information for output

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

    def allele_filter(self):
        # binominal test to filter noise
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

        if len(alleles_used_for_gt) <= MAX_ALLOWABLE_ALLELE_NUM:
            self.alleles_used_for_gt = alleles_used_for_gt
            return
        if BINOMINAL_FILTER:
            for allele, allele_depth in self.alleles_depth_dict.items():
                pvalue = stats.binomtest(
                    allele_depth,
                    n=self.reads_number,
                    p=error_rate,
                    alternative="less",
                ).pvalue
                if pvalue > 0.05:
                    continue
                    # break
                else:
                    alleles_used_for_gt.remove(allele)
        # filter alleles to max allowable allele number
        # But keep all allele reads
        # alleles_used_for_gt = self.alleles_order[binominal_filter_number:]
        # remove_allele = self.alleles_order[:binominal_filter_number]
        # alleles_seq_used_for_gt = self.alleles_seq_used_for_gt[binominal_filter_number:]
        if len(alleles_used_for_gt) > MAX_ALLOWABLE_ALLELE_NUM:
            self.alleles_used_for_gt = alleles_used_for_gt[
                len(alleles_used_for_gt) - MAX_ALLOWABLE_ALLELE_NUM :
            ]
            # self.alleles_seq_used_for_gt = alleles_seq_used_for_gt[
            #    len(alleles_seq_used_for_gt) - MAX_ALLOWABLE_ALLELE_NUM :
            # ]
        else:
            self.alleles_used_for_gt = alleles_used_for_gt
            # self.alleles_seq_used_for_gt = alleles_seq_used_for_gt

    # def allele_filter(self):
    #     # binominal test to filter noise
    #     binominal_filter_number = 0
    #     error_rate = (
    #         self.stutter_model_params["inframe_del_prob"]
    #         + self.stutter_model_params["inframe_ins_prob"]
    #         + self.stutter_model_params["outframe_del_prob"]
    #         + self.stutter_model_params["outframe_ins_prob"]
    #     )
    #     if BINOMINAL_FILTER:
    #         for allele_depth in self.alleles_depth:
    #             pvalue = stats.binomtest(
    #                 allele_depth,
    #                 n=self.reads_number,
    #                 p=error_rate,
    #                 alternative="less",
    #             ).pvalue
    #             if pvalue > 0.05:
    #                 break
    #             else:
    #                 binominal_filter_number += 1
    #     # filter alleles to max allowable allele number
    #     # But keep all allele reads
    #     alleles_used_for_gt = self.alleles_order[binominal_filter_number:]
    #     # alleles_seq_used_for_gt = self.alleles_seq_used_for_gt[binominal_filter_number:]
    #     if len(alleles_used_for_gt) > MAX_ALLOWABLE_ALLELE_NUM:
    #         self.alleles_used_for_gt = alleles_used_for_gt[
    #             len(alleles_used_for_gt) - MAX_ALLOWABLE_ALLELE_NUM :
    #         ]
    #         # self.alleles_seq_used_for_gt = alleles_seq_used_for_gt[
    #         #     len(alleles_seq_used_for_gt) - MAX_ALLOWABLE_ALLELE_NUM :
    #         # ]
    #     else:
    #         self.alleles_used_for_gt = alleles_used_for_gt
    #         # self.alleles_seq_used_for_gt = alleles_seq_used_for_gt

    def check_loci_callable(self):
        coverage_state = self.reads_number < MIN_AVERAGE_COVERAGE
        alleles_number_state = self.alleles_number_used_for_gt <= 1
        reads_snp_alleles_number = len(self.reads_alleles_snp_list)
        reads_str_alleles_number = len(self.reads_alleles_str_list)
        if coverage_state:
            self.loci_calling = "CoverageFail"
            self.callable = False
        elif alleles_number_state:
            self.loci_calling = "Germline"
            self.callable = False
        elif reads_snp_alleles_number < 2 and reads_str_alleles_number < 2:
            self.loci_calling = "NoEnoughPhasingReadsSTRSNPFail"
            self.callable = False
        elif reads_str_alleles_number < 2 and reads_snp_alleles_number >= 2:
            self.loci_calling = "NoEnoughPhasingReadsSTRFail"
            self.callable = False
        elif reads_snp_alleles_number < 2 and reads_str_alleles_number >= 2:
            self.loci_calling = "NoEnoughPhasingReadsSNPFail"
            self.callable = False
        else:
            self.loci_calling = "CandidateMosaic"
            self.callable = True

    def all_reads_all_alleles_linking_log_lik(self):
        self.all_reads_all_alleles_linking_log_lik_array = np.zeros(
            (
                len(self.reads_list),
                len(self.alleles_linking_order),
            ),
            dtype=np.float64,
        )
        if LIKELIHOOD_MODE == "length-based":
            phasable_read_index = 0
            for read_linking_seq, read_linking_accuracy in zip(
                self.reads_list, self.reads_accuracy_list
            ):
                read_str_hap = read_linking_seq[0]
                read_snv_hap = read_linking_seq[1]
                read_str_hap_accuracy = read_linking_accuracy[0]
                read_snv_hap_accuracy = read_linking_accuracy[1]
                for (
                    allele_linking_order_index,
                    allele_linking_order,
                ) in enumerate(self.alleles_linking_order):
                    allele_str_hap = allele_linking_order[0]
                    allele_snv_hap = allele_linking_order[1]
                    self.all_reads_all_alleles_linking_log_lik_array[
                        phasable_read_index, allele_linking_order_index
                    ] = self.cal_per_read_linking_log_likelihood_given_allele_len_based(
                        allele_str_hap,
                        allele_snv_hap,
                        read_str_hap,
                        read_snv_hap,
                        read_str_hap_accuracy,
                        read_snv_hap_accuracy,
                    )
                phasable_read_index += 1
        else:
            phasable_read_index = 0
            for read_linking_seq, read_linking_accuracy in zip(
                self.reads_list, self.reads_accuracy_list
            ):
                read_str_hap = read_linking_seq[0]
                read_snv_hap = read_linking_seq[1]
                read_str_hap_accuracy = read_linking_accuracy[0]
                read_snv_hap_accuracy = read_linking_accuracy[1]
                for (
                    allele_linking_order_index,
                    allele_linking_order,
                ) in enumerate(self.alleles_linking_order):
                    allele_str_hap = allele_linking_order[0]
                    allele_snv_hap = allele_linking_order[1]
                    self.all_reads_all_alleles_linking_log_lik_array[
                        phasable_read_index, allele_linking_order_index
                    ] = self.cal_per_read_linking_log_likelihood_given_allele_seq_based(
                        allele_str_hap,
                        allele_snv_hap,
                        read_str_hap,
                        read_snv_hap,
                        read_str_hap_accuracy,
                        read_snv_hap_accuracy,
                    )
                phasable_read_index += 1
            pass  # TODO

    def cal_per_read_linking_log_likelihood_given_allele_len_based(
        self,
        allele_str_hap,
        allele_snv_hap,
        read_str_hap,
        read_snv_hap,
        read_str_hap_accuracy,
        read_snv_hap_accuracy,
    ):
        if allele_snv_hap == read_snv_hap:
            per_read_snv_log_ll = np.log(read_snv_hap_accuracy)
        else:
            per_read_snv_log_ll = np.log((1 - read_snv_hap_accuracy) / 3)

        per_read_str_log_ll = (
            self.cal_per_read_log_likelihood_given_allele_len_based(
                allele_str_hap, read_str_hap
            )
        )
        per_read_log_ll = per_read_str_log_ll + per_read_snv_log_ll
        return per_read_log_ll

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
                if bp_diff > 0:
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

    def cal_per_read_linking_log_likelihood_given_allele_seq_based(
        self,
        allele_str_hap,
        allele_snv_hap,
        read_str_hap,
        read_snv_hap,
        read_str_hap_accuracy,
        read_snv_hap_accuracy,
    ):
        if allele_snv_hap == read_snv_hap:
            per_read_snv_log_ll = np.log(read_snv_hap_accuracy)
        else:
            if read_snv_hap_accuracy == 1:
                per_read_snv_log_ll = MIN_LOG_PROB
            else:
                per_read_snv_log_ll = np.log((1 - read_snv_hap_accuracy) / 3)
        per_read_str_log_ll = (
            self.cal_per_read_log_likelihood_given_allele_seq_based(
                allele_str_hap, read_str_hap, read_str_hap_accuracy
            )
        )
        per_read_log_ll = per_read_str_log_ll + per_read_snv_log_ll
        return per_read_log_ll

    def cal_per_read_log_likelihood_given_allele_seq_based(
        self, allele_str_hap, read_str_hap, read_str_hap_accuracy
    ):
        bp_diff = len(read_str_hap[1]) - len(allele_str_hap[1])
        if bp_diff == 0.0:
            log_str_align_prob = hap_alignment.no_stutter_str_alignment(
                allele_str_hap[1],
                read_str_hap[1],
                read_str_hap_accuracy[1],
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
                        allele_str_hap[1],
                        read_str_hap[1],
                        read_str_hap_accuracy[1],
                        steps,
                        self.inframe_stutter_model_dict
                        # ins_rate,
                        # p_geom,
                        # self.locus_infos["motif_length"]
                    )
                else:
                    log_str_align_prob = hap_alignment.stutter_deletion(
                        # in_or_out_frame,
                        allele_str_hap[1],
                        read_str_hap[1],
                        read_str_hap_accuracy[1],
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
                        allele_str_hap[1],
                        read_str_hap[1],
                        read_str_hap_accuracy[1],
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
                        allele_str_hap[1],
                        read_str_hap[1],
                        read_str_hap_accuracy[1],
                        steps,
                        self.outframe_stutter_model_dict
                        # del_rate,
                        # p_geom,
                        # self.locus_infos["motif_length"]
                    )
        if ADD_FLANKING_PROB:
            log_left_flk_align_prob = hap_alignment.align_flk(
                allele_str_hap[0],
                read_str_hap[0],
                tuple(read_str_hap_accuracy[0]),
            )
            log_right_flk_align_prob = hap_alignment.align_flk(
                allele_str_hap[2],
                read_str_hap[2],
                tuple(read_str_hap_accuracy[2]),
            )
        else:
            log_left_flk_align_prob = 0
            log_right_flk_align_prob = 0
        per_read_str_log_ll = (
            log_str_align_prob
            + log_left_flk_align_prob
            + log_right_flk_align_prob
        )
        return per_read_str_log_ll

    def all_reads_all_phased_gts_log_lik(self):
        self.all_reads_all_phased_gts_log_lik_array = np.zeros(
            (len(self.reads_list), len(self.genotypes)), dtype=np.float64
        )
        for read_index, read in enumerate(self.reads_list):
            for gt_index, gt in enumerate(self.genotypes):
                self.all_reads_all_phased_gts_log_lik_array[
                    read_index, gt_index
                ] = self.cal_per_read_given_phased_gt_log_lik(read_index, gt)
        self.all_phased_gts_log_lik_array = np.sum(
            self.all_reads_all_phased_gts_log_lik_array, axis=0
        )

    def cal_per_read_given_phased_gt_log_lik(self, read_index, gt):
        allele1_index = gt[0]
        allele2_index = gt[1]
        read_allele1_log_lik = (
            self.all_reads_all_alleles_linking_log_lik_array[
                read_index, allele1_index
            ]
        )
        read_allele2_log_lik = (
            self.all_reads_all_alleles_linking_log_lik_array[
                read_index, allele2_index + self.alleles_number_used_for_gt
            ]
        )
        read_gt_log_lik = logsumexp(
            [read_allele1_log_lik, read_allele2_log_lik], b=[0.5, 0.5]
        )
        return read_gt_log_lik

    def cal_mutation_rate_prior_phase(self):
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
                self.genotypes[germ_cells][0] == self.genotypes[mut_cells][0]
                or self.genotypes[germ_cells][1]
                == self.genotypes[mut_cells][1]
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
                    self.loci_calling = "Germline"
                    self.posterior_cal_logic = False
                    self.estimated_mosaic_fraction = 0
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
        self.mosaic_fraction = INITIAL_MOSAIC_FRACTION
        if coverage_state:
            logging.info(
                f"{self.locus_infos['STR_id']} has a sample depth less than"
                f" {MIN_AVERAGE_COVERAGE}"
            )
            self.locus_state = "CoverageFail"
            return
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
                    mut_allele = (
                        mut_cells_gt[1] + self.alleles_number_used_for_gt
                    )
                    mut_cell_ano_allele = mut_cells_gt[0]
                elif (mut_cells_gt[0] != germ_cells_gt[0]) and (
                    mut_cells_gt[1] == germ_cells_gt[1]
                ):  # XXX
                    mut_allele = mut_cells_gt[0]
                    mut_cell_ano_allele = (
                        mut_cells_gt[1] + self.alleles_number_used_for_gt
                    )
                if (
                    (mut_cells_gt[0] == germ_cells_gt[0])
                    and (mut_cells_gt[1] != germ_cells_gt[1])
                ) or (
                    (mut_cells_gt[0] != germ_cells_gt[0])
                    and (mut_cells_gt[1] == germ_cells_gt[1])
                ):  # XXX only allow one allele to mutate between gts
                    current_mut_allele_log_posterior = (
                        self.cal_per_read_mut_log_posterior(
                            germ_cells_gt[0],
                            germ_cells_gt[1] + self.alleles_number_used_for_gt,
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
                            self.all_phased_gts_log_lik_array[germ_cells_index]
                            + self.log_mutation_rate_prior[mosaic_index]
                            + self.cal_phased_mosaic_gt_log_lik(
                                cells, mosaic_fraction
                            )
                        )
                    elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GI":
                        log_mosaic_prior_lik = (
                            self.all_phased_gts_log_lik_array[germ_cells_index]
                            + self.cal_phased_mosaic_gt_log_lik(
                                cells, mosaic_fraction
                            )
                        )
                    elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "MutRate":
                        log_mosaic_prior_lik = self.log_mutation_rate_prior[
                            mosaic_index
                        ] + self.cal_phased_mosaic_gt_log_lik(
                            cells, mosaic_fraction
                        )
                    elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "OnlyLik":
                        log_mosaic_prior_lik = (
                            self.cal_phased_mosaic_gt_log_lik(
                                cells, mosaic_fraction
                            )
                        )
                    else:
                        logging.error("GI prior update mode error.")
                        raise ValueError("GI prior update mode error.")
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
                        mut_allele = (
                            mut_cells_gt[1] + self.alleles_number_used_for_gt
                        )
                        mut_cell_ano_allele = mut_cells_gt[0]
                    elif (mut_cells_gt[0] != germ_cells_gt[0]) and (
                        mut_cells_gt[1] == germ_cells_gt[1]
                    ):  # XXX
                        mut_allele = mut_cells_gt[0]
                        mut_cell_ano_allele = (
                            mut_cells_gt[1] + self.alleles_number_used_for_gt
                        )
                    if (
                        (mut_cells_gt[0] == germ_cells_gt[0])
                        and (mut_cells_gt[1] != germ_cells_gt[1])
                    ) or (
                        (mut_cells_gt[0] != germ_cells_gt[0])
                        and (mut_cells_gt[1] == germ_cells_gt[1])
                    ):  # XXX only allow one allele to mutate between gts
                        current_mut_allele_log_posterior = (
                            self.cal_per_read_mut_log_posterior(
                                germ_cells_gt[0],
                                germ_cells_gt[1]
                                + self.alleles_number_used_for_gt,
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
                                self.all_phased_gts_log_lik_array[
                                    germ_cells_index
                                ]
                                + self.log_mutation_rate_prior[mosaic_index]
                                + self.cal_phased_mosaic_gt_log_lik(
                                    cells, mosaic_fraction
                                )
                            )
                        elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GI":
                            log_mosaic_prior_lik = (
                                self.all_phased_gts_log_lik_array[
                                    germ_cells_index
                                ]
                                + self.cal_phased_mosaic_gt_log_lik(
                                    cells, mosaic_fraction
                                )
                            )
                        elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "MutRate":
                            log_mosaic_prior_lik = (
                                self.log_mutation_rate_prior[mosaic_index]
                                + self.cal_phased_mosaic_gt_log_lik(
                                    cells, mosaic_fraction
                                )
                            )
                        elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "OnlyLik":
                            log_mosaic_prior_lik = (
                                self.cal_phased_mosaic_gt_log_lik(
                                    cells, mosaic_fraction
                                )
                            )
                        else:
                            logging.error("GI prior update mode error.")
                            raise ValueError("GI prior update mode error.")
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
            per_read_germ1_log_lik = (
                self.all_reads_all_alleles_linking_log_lik_array[
                    read_index, germ_allele1
                ]
            )
            per_read_germ2_log_lik = (
                self.all_reads_all_alleles_linking_log_lik_array[
                    read_index, germ_allele2
                ]
            )
            per_read_mut_cell_ano_allele_log_lik = (
                self.all_reads_all_alleles_linking_log_lik_array[
                    read_index, germ_allele3
                ]
            )
            per_read_mut_log_lik = (
                self.all_reads_all_alleles_linking_log_lik_array[
                    read_index, mut_allele
                ]
            )
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
                if (mut_cells_gt[0] != germ_cells_gt[0]) and (
                    mut_cells_gt[1] != germ_cells_gt[1]
                ):
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
                per_read_germ_log_lik = (
                    self.all_reads_all_phased_gts_log_lik_array[
                        read_index, germ_cells_index
                    ]
                )
                per_read_mut_log_lik = (
                    self.all_reads_all_phased_gts_log_lik_array[
                        read_index, mut_cells_index
                    ]
                )
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
                    self.all_phased_gts_log_lik_array[germ_cells_index]
                    + self.log_mutation_rate_prior[mosaic_index]
                    + self.cal_phased_mosaic_gt_log_lik(cells, mosaic_fraction)
                )
            elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "GI":
                log_mosaic_prior_lik = self.all_phased_gts_log_lik_array[
                    germ_cells_index
                ] + self.cal_phased_mosaic_gt_log_lik(cells, mosaic_fraction)
            elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "MutRate":
                log_mosaic_prior_lik = self.log_mutation_rate_prior[
                    mosaic_index
                ] + self.cal_phased_mosaic_gt_log_lik(cells, mosaic_fraction)
            elif USE_GI_PRIOR_UPDATE_MOSAIC_FRACTION == "OnlyLik":
                log_mosaic_prior_lik = self.cal_phased_mosaic_gt_log_lik(
                    cells, mosaic_fraction
                )
            else:
                logging.error("GI prior update mode error.")
                raise ValueError("GI prior update mode error.")
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

    def cal_phased_mosaic_gt_log_lik(self, mosaic_gt, mosaic_fraction):
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
                self.all_reads_all_phased_gts_log_lik_array[
                    read_index, germ_cells_index
                ],
                self.all_reads_all_phased_gts_log_lik_array[
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
            per_read_germ_log_lik = (
                self.all_reads_all_phased_gts_log_lik_array[
                    read_index, germ_cells_index
                ]
            )
            per_read_mut_log_lik = self.all_reads_all_phased_gts_log_lik_array[
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

    def cal_somatic_log_posterior(self):
        if self.posterior_cal_logic:
            self.all_reads_all_phased_mosaic_gts_log_lik(
                self.estimated_mosaic_fraction
            )
            if ONLY_USE_MLE_MOSAIC_FRACTION:
                mle_mosaic_fraction_mosaic_index = np.argmax(
                    self.all_phased_mosaic_gts_log_lik_array
                )
                mle_mosaic_fraction = self.estimated_mosaic_fraction[
                    mle_mosaic_fraction_mosaic_index
                ]
                mle_mosaic_fraction_list = [mle_mosaic_fraction] * len(
                    self.estimated_mosaic_fraction
                )
                self.all_reads_all_phased_mosaic_gts_log_lik(
                    mle_mosaic_fraction_list
                )
            else:
                pass
            if POSTERIOR_METHOD == "Gi-Given":
                self.Gi_phased_log_lik_with_mosaic_gt_order()
                self.cal_all_gts_trans_phased_mosaic_log_posterior()
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
                GermlineLogLik = self.all_phased_mosaic_gts_log_lik_array[
                    Germline_MosaicIndex
                ]
                germ_cells_gt = MaxMosaicGT_for_output[0]
                mut_cells_gt = MaxMosaicGT_for_output[1]
                germ_cells_gt_p2_index = self.genotypes.index(
                    tuple([germ_cells_gt[1], germ_cells_gt[0]])
                )
                mut_cells_gt_p2_index = self.genotypes.index(
                    tuple([mut_cells_gt[1], mut_cells_gt[0]])
                )
                MaxPostIndex_p2_index = self.mosaic_genotypes.index(
                    tuple([germ_cells_gt_p2_index, mut_cells_gt_p2_index])
                )
                MaxPostIndex_p2 = AllMosaicGTOrder.index(MaxPostIndex_p2_index)
                MaxMosaicLogPosterior_p2 = AllMosaicGTLogPosterior[
                    MaxPostIndex_p2
                ]
                MaxMosaicLogLik_p2 = AllMosaicGTLogLik[MaxPostIndex_p2]
                MaxMosaicLogPosterior_unphase = logsumexp(
                    [MaxMosaicLogPosterior, MaxMosaicLogPosterior_p2]
                )
                MaxMosaicLogLik_unphase = logsumexp(
                    [MaxMosaicLogLik, MaxMosaicLogLik_p2]
                )
                MaxMosaicPhaseLogPostPosterior = (
                    MaxMosaicLogPosterior - MaxMosaicLogPosterior_unphase
                )
                MaxMosaicPhaseLogLikPosterior = (
                    MaxMosaicLogLik - MaxMosaicLogLik_unphase
                )
            else:
                # XXX: unique and double mosaic fraction estimation method don't support for Gi-free currently
                self.cal_all_gts_trans_phased_mosaic_log_posterior()
                (
                    AllMosaicGTLogLikLogPosterior_dict,
                    AllMosaicGTDict_dict,
                ) = self.cal_different_levels_log_mosaic_posterior(
                    "orderedGTunorderedMutDirection"
                )
                AllMosaicGTDict = AllMosaicGTDict_dict["posterior"]
                if POSTERIOR_METHOD == "Gi-free":
                    AllMosaicGTLogPosterior = (
                        AllMosaicGTLogLikLogPosterior_dict["posterior"]
                    )
                    AllMosaicGTLogLik = AllMosaicGTLogLikLogPosterior_dict[
                        "lik"
                    ]
                elif POSTERIOR_METHOD == "Likelihood":
                    AllMosaicGTLogPosterior = (
                        AllMosaicGTLogLikLogPosterior_dict["posterior"]
                    )
                    AllMosaicGTLogLik = AllMosaicGTLogPosterior
                AllMosaicGTOrder = AllMosaicGTDict[
                    "orderedGTunorderedMutDirection"
                ]
                AllMosaicGTLogPosterior_sorted = sorted(
                    AllMosaicGTLogPosterior.items(), key=lambda item: item[1]
                )
                MaxMosaicGT_for_output = AllMosaicGTLogPosterior_sorted[-1][0]
                MaxMosaicLogPosterior = AllMosaicGTLogPosterior_sorted[-1][1]
                MaxMosaicLogLik = AllMosaicGTLogLik[MaxMosaicGT_for_output]
                SecondMosaicGT_for_output = AllMosaicGTLogPosterior_sorted[-2][
                    0
                ]
                SecondMosaicLogPosterior = AllMosaicGTLogPosterior_sorted[-2][
                    1
                ]
                SecondMosaicLogLik = AllMosaicGTLogLik[
                    SecondMosaicGT_for_output
                ]
                GermlineGT1_for_output = tuple(
                    [MaxMosaicGT_for_output[0], MaxMosaicGT_for_output[0]]
                )
                GermlineGT2_for_output = tuple(
                    [MaxMosaicGT_for_output[1], MaxMosaicGT_for_output[1]]
                )
                GermlineLogPosterior1 = AllMosaicGTLogPosterior[
                    GermlineGT1_for_output
                ]
                GermlineLogPosterior2 = AllMosaicGTLogPosterior[
                    GermlineGT2_for_output
                ]
                if GermlineLogPosterior1 > GermlineLogPosterior2:
                    GermlineGT_for_output = GermlineGT1_for_output
                    GermlineLogPosterior = GermlineLogPosterior1
                    GermlineLogLik = AllMosaicGTLogLik[GermlineGT_for_output]
                else:
                    GermlineGT_for_output = GermlineGT2_for_output
                    GermlineLogPosterior = GermlineLogPosterior2
                    GermlineLogLik = AllMosaicGTLogLik[GermlineGT_for_output]
                germ_cells_gt = MaxMosaicGT_for_output[0]
                mut_cells_gt = MaxMosaicGT_for_output[1]
                germ_cells_gt_p2_index = self.genotypes.index(
                    tuple([germ_cells_gt[1], germ_cells_gt[0]])
                )
                mut_cells_gt_p2_index = self.genotypes.index(
                    tuple([mut_cells_gt[1], mut_cells_gt[0]])
                )
                MaxPostIndex_p2_index = self.mosaic_genotypes.index(
                    tuple([germ_cells_gt_p2_index, mut_cells_gt_p2_index])
                )
                MaxPostIndex_p2 = AllMosaicGTOrder.index(MaxPostIndex_p2_index)
                MaxMosaicLogPosterior_p2 = AllMosaicGTLogPosterior[
                    MaxPostIndex_p2
                ]
                MaxMosaicLogLik_p2 = AllMosaicGTLogLik[MaxPostIndex_p2]
                MaxMosaicLogPosterior_unphase = logsumexp(
                    [MaxMosaicLogPosterior, MaxMosaicLogPosterior_p2]
                )
                MaxMosaicLogLik_unphase = logsumexp(
                    [MaxMosaicLogLik, MaxMosaicLogLik_p2]
                )
                MaxMosaicPhaseLogPostPosterior = (
                    MaxMosaicLogPosterior - MaxMosaicLogPosterior_unphase
                )
                MaxMosaicPhaseLogLikPosterior = (
                    MaxMosaicLogLik - MaxMosaicLogLik_unphase
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
                MaxMosaicLogPosterior_unphase,
                MaxMosaicLogLik_unphase,
                MaxMosaicPhaseLogPostPosterior,
                MaxMosaicPhaseLogLikPosterior,
                SecondMosaicGT_for_output,
                SecondMosaicLogPosterior,
                SecondMosaicLogLik,
                GermlineGT_for_output,
                GermlineLogPosterior,
                GermlineLogLik,
                Non_Germ2germLogPosterior,
                Non_Germ2germLogLikFraction,
            )
        else:
            return tuple([False] * 15)
            pass  # TODO: Need format information for output

    def all_reads_all_phased_mosaic_gts_log_lik(self, mosaic_fraction):
        self.all_reads_all_phased_mosaic_gts_log_lik_array = np.zeros(
            (len(self.reads_list), len(self.mosaic_genotypes)),
            dtype=np.float64,
        )
        for read_index, read in enumerate(self.reads_list):
            for mosaic_index, cells in enumerate(self.mosaic_genotypes):
                germ_cells_index = cells[0]
                mut_cells_index = cells[1]
                per_read_germ_log_lik = (
                    self.all_reads_all_phased_gts_log_lik_array[
                        read_index, germ_cells_index
                    ]
                )
                per_read_mut_log_lik = (
                    self.all_reads_all_phased_gts_log_lik_array[
                        read_index, mut_cells_index
                    ]
                )

                per_read_mosaic_log_lik = logsumexp(
                    [per_read_germ_log_lik, per_read_mut_log_lik],
                    b=[
                        1 - mosaic_fraction[mosaic_index],
                        mosaic_fraction[mosaic_index],
                    ],
                )

                self.all_reads_all_phased_mosaic_gts_log_lik_array[
                    read_index, mosaic_index
                ] = per_read_mosaic_log_lik
        self.all_phased_mosaic_gts_log_lik_array = np.sum(
            self.all_reads_all_phased_mosaic_gts_log_lik_array, axis=0
        )

    def Gi_phased_log_lik_with_mosaic_gt_order(self):
        self.all_mosaic_gts_phased_Gi_log_lik_array = []
        for germ_cells, mut_cells in self.mosaic_genotypes:
            Gi_log_lik = self.all_phased_gts_log_lik_array[germ_cells]
            self.all_mosaic_gts_phased_Gi_log_lik_array.append(Gi_log_lik)
        self.all_mosaic_gts_phased_Gi_log_lik_array = np.array(
            self.all_mosaic_gts_phased_Gi_log_lik_array
        )

    def cal_all_gts_trans_phased_mosaic_log_posterior(self):
        if POSTERIOR_METHOD == "Gi-Given":
            self.all_phased_mosaic_gts_log_lik_prior_array = (
                self.all_mosaic_gts_phased_Gi_log_lik_array
                + self.all_phased_mosaic_gts_log_lik_array
                + self.log_mutation_rate_prior_with_germ2germ_rate_array
            )
        elif POSTERIOR_METHOD == "Gi-free":
            self.all_phased_mosaic_gts_log_lik_prior_array = (
                self.all_phased_mosaic_gts_log_lik_array
                + self.log_mutation_rate_prior_with_germ2germ_rate_array
            )
        elif POSTERIOR_METHOD == "Likelihood":
            self.all_phased_mosaic_gts_log_lik_prior_array = (
                self.all_phased_mosaic_gts_log_lik_array
            )
        selected_mosaic_index = self.select_mosaic_genotypes_index()
        self.phased_mosaic_posterior_order = selected_mosaic_index
        self.selected_phased_mosaic_gts_log_lik_prior_array = (
            self.all_phased_mosaic_gts_log_lik_prior_array[
                selected_mosaic_index
            ]
        )
        self.all_phased_mosaic_gts_total_log_posterior = logsumexp(
            self.selected_phased_mosaic_gts_log_lik_prior_array
        )
        self.all_phased_mosaic_gts_log_posterior_array = (
            self.selected_phased_mosaic_gts_log_lik_prior_array
            - self.all_phased_mosaic_gts_total_log_posterior
        )
        if POSTERIOR_METHOD == "Likelihood":
            self.selected_phased_mosaic_gts_log_lik_array = (
                self.selected_phased_mosaic_gts_log_lik_prior_array
            )
            self.selected_all_phased_mosaic_gts_total_log_lik = (
                self.all_phased_mosaic_gts_total_log_posterior
            )
            self.selected_all_phased_mosaic_gts_log_lik_fraction_array = (
                self.all_phased_mosaic_gts_log_posterior_array
            )
        else:
            self.selected_phased_mosaic_gts_log_lik_array = (
                self.all_phased_mosaic_gts_log_lik_array[selected_mosaic_index]
            )
            self.selected_all_phased_mosaic_gts_total_log_lik = logsumexp(
                self.selected_phased_mosaic_gts_log_lik_array
            )
            self.selected_all_phased_mosaic_gts_log_lik_fraction_array = (
                self.selected_phased_mosaic_gts_log_lik_array
                - self.selected_all_phased_mosaic_gts_total_log_lik
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
                        selected_mosaic_index = [
                            index
                            for index, value in enumerate(
                                self.mosaic_gts_types
                            )
                            if value in ["het2hom", "homhet2het"]
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
        all_mosaic_gts_log_posterior_array = (
            self.all_phased_mosaic_gts_log_posterior_array
        )
        mosaic_posterior_order = self.phased_mosaic_posterior_order
        all_mosaic_gts_log_lik_prior_array = (
            self.all_phased_mosaic_gts_log_lik_prior_array
        )
        all_mosaic_gts_total_log_posterior = (
            self.all_phased_mosaic_gts_total_log_posterior
        )
        # XXX: imbalance probability calculation because hom have one 0/0, but het have two 1/0 and 0/1 in ordered transform to unordered
        # XXX: unordered probability with more ordered GT combinations will add up the ordered GT combinations probability
        # XXX: so own higher probability, in fact they should own higher probability like 孟德尔自由组合定律
        # BUG/TODO: if we think higher probability, I should change the unordered uphase model to make the heterzygosty own higher probability like 孟德尔自由组合定律
        # XXX: ordered phase and unordered phase, mutation direction, different combinations number, and ordered phase to unordered unphased transformation
        # XXX: maybe exist some GT should be selected but has been filtered in selecting mosaic index
        # XXX: and contribution inconsistent posterior calculation between unordered GT
        # for merge GT,because use mosaic index,so all_mosaic_gts_log_lik_prior_array include all mosaic genotypes and mosaic index will be used
        # for calculation merge GT posterior, all_mosaic_gts_total_log_posterior will be used
        mosaic_posterior_order = self.phased_mosaic_posterior_order
        log_lik_log_posterior_dict, mosaicGT_dict_dict = {}, {}
        log_lik_and_log_posterior_dict = {
            "lik": {
                "all_mosaic_gts_log_posterior_array": self.selected_all_phased_mosaic_gts_log_lik_fraction_array,
                "all_mosaic_gts_log_lik_prior_array": self.all_phased_mosaic_gts_log_lik_array,
                "all_mosaic_gts_total_log_posterior": self.selected_all_phased_mosaic_gts_total_log_lik,
            },
            "posterior": {
                "all_mosaic_gts_log_posterior_array": self.all_phased_mosaic_gts_log_posterior_array,
                "all_mosaic_gts_log_lik_prior_array": self.all_phased_mosaic_gts_log_lik_prior_array,
                "all_mosaic_gts_total_log_posterior": self.all_phased_mosaic_gts_total_log_posterior,
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

            elif posterior_level == "orderedGTunorderedMutDirection":
                # XXX: I worry about that unorderedMutDirection don't have the mosaic fraction and 1-mosaic fraction mle and
                # XXX: don't own the same Log-likelihood value
                # BUG: Is this possible ?
                # TODO: Do I need always use ordered GT whatever phase or unphase
                orderedGTunorderedMutDirection_log_posterior_dict = {}
                mosaicGT_dict = {}
                for mosaic_index in mosaic_posterior_order:
                    cells = self.mosaic_genotypes[mosaic_index]
                    germ_cells_index = cells[0]
                    mut_cells_index = cells[1]
                    germ_cells_gt = self.genotypes[germ_cells_index]
                    mut_cells_gt = self.genotypes[mut_cells_index]
                    mosaic_ordered_gt = tuple(
                        sorted([germ_cells_gt, mut_cells_gt])
                    )
                    orderedGTunorderedMutDirection_log_posterior_dict.setdefault(
                        mosaic_ordered_gt, []
                    ).append(
                        all_mosaic_gts_log_lik_prior_array[mosaic_index]
                    )
                    mosaicGT_dict.setdefault(mosaic_ordered_gt, []).append(
                        mosaic_index
                    )
                log_posterior = {}
                for (
                    orderedGT_unorderedMutDirection_mosaic_gt,
                    log_posterior_list,
                ) in orderedGTunorderedMutDirection_log_posterior_dict.items():
                    log_posterior[
                        orderedGT_unorderedMutDirection_mosaic_gt
                    ] = (
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
                            (germ_allele, mut_allele, 1), []
                        ).append(
                            all_mosaic_gts_log_lik_prior_array[mosaic_index]
                        )
                        mosaicGT_dict.setdefault(
                            (germ_allele, mut_allele, 1), []
                        ).append(mosaic_index)
                    elif (mut_cells_gt[0] != germ_cells_gt[0]) and (
                        mut_cells_gt[1] == germ_cells_gt[1]
                    ):  # XXX
                        mut_allele = mut_cells_gt[0]
                        germ_allele = germ_cells_gt[0]
                        AlleleTransformation_log_posterior_dict.setdefault(
                            (germ_allele, mut_allele, 0), []
                        ).append(
                            all_mosaic_gts_log_lik_prior_array[mosaic_index]
                        )
                        mosaicGT_dict.setdefault(
                            (germ_allele, mut_allele, 0), []
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
        return tuple([germ_cells_gt, mut_cells_gt])

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
            self.all_mosaic_gts_phased_Gi_log_lik_array
        )
        selected_mosaic_index = self.phased_mosaic_posterior_order
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

    def cal_phased_mosaic_gt_log_lik_given_p1p2(self):
        pass  # TODO: use p1 and p2 as prior probabilities to calculate the log likelihood of each unphased mosaic genotypes

    def mosaic_genotype_linking_log_lik_posteriors(
        self, mosaic_genotype, snp_h1_h2, mosaic_fraction
    ):
        germ_cells_index = mosaic_genotype[0]
        mut_cells_index = mosaic_genotype[1]
        germ_cells_gt = self.genotypes[germ_cells_index]
        mut_cells_gt = self.genotypes[mut_cells_index]
        germ_cell_allele1 = self.alleles_used_for_gt[germ_cells_gt[0]]
        germ_cell_allele2 = self.alleles_used_for_gt[germ_cells_gt[1]]
        mut_cell_allele1 = self.alleles_used_for_gt[mut_cells_gt[0]]
        mut_cell_allele2 = self.alleles_used_for_gt[mut_cells_gt[1]]
        snp_h1 = snp_h1_h2[0]
        snp_h2 = snp_h1_h2[1]
        germ_cell_allele1_h1 = self.alleles_linking_order.index(
            (germ_cell_allele1, snp_h1)
        )
        germ_cell_allele2_h2 = self.alleles_linking_order.index(
            (germ_cell_allele2, snp_h2)
        )
        mut_cell_allele1_h1 = self.alleles_linking_order.index(
            (mut_cell_allele1, snp_h1)
        )
        mut_cell_allele2_h2 = self.alleles_linking_order.index(
            (mut_cell_allele2, snp_h2)
        )
        p1_log_lik = sum(
            logsumexp(
                [
                    np.log((1 - mosaic_fraction) / 2)
                    + self.all_reads_all_alleles_linking_log_lik_array[
                        :, germ_cell_allele1_h1
                    ],
                    np.log((1 - mosaic_fraction) / 2)
                    + self.all_reads_all_alleles_linking_log_lik_array[
                        :, germ_cell_allele2_h2
                    ],
                    np.log(mosaic_fraction / 2)
                    + self.all_reads_all_alleles_linking_log_lik_array[
                        :, mut_cell_allele1_h1
                    ],
                    np.log(mosaic_fraction / 2)
                    + self.all_reads_all_alleles_linking_log_lik_array[
                        :, mut_cell_allele2_h2
                    ],
                ],
                axis=0,
            )
        )
        germ_cell_allele1_h2 = self.alleles_linking_order.index(
            (germ_cell_allele1, snp_h2)
        )
        germ_cell_allele2_h1 = self.alleles_linking_order.index(
            (germ_cell_allele2, snp_h1)
        )
        mut_cell_allele1_h2 = self.alleles_linking_order.index(
            (mut_cell_allele1, snp_h2)
        )
        mut_cell_allele2_h1 = self.alleles_linking_order.index(
            (mut_cell_allele2, snp_h1)
        )
        p2_log_lik = sum(
            logsumexp(
                [
                    np.log((1 - mosaic_fraction) / 2)
                    + self.all_reads_all_alleles_linking_log_lik_array[
                        :, germ_cell_allele1_h2
                    ],
                    np.log((1 - mosaic_fraction) / 2)
                    + self.all_reads_all_alleles_linking_log_lik_array[
                        :, germ_cell_allele2_h1
                    ],
                    np.log(mosaic_fraction / 2)
                    + self.all_reads_all_alleles_linking_log_lik_array[
                        :, mut_cell_allele1_h2
                    ],
                    np.log(mosaic_fraction / 2)
                    + self.all_reads_all_alleles_linking_log_lik_array[
                        :, mut_cell_allele2_h1
                    ],
                ],
                axis=0,
            )
        )
        p1_log_posterior = p1_log_lik - logsumexp([p1_log_lik, p2_log_lik])
        p2_log_posterior = p2_log_lik - logsumexp([p1_log_lik, p2_log_lik])
        return p1_log_lik, p2_log_lik, p1_log_posterior, p2_log_posterior

    def get_final_gt_index(self, gt_output):
        new_gt = []
        for current_sample_allele_index in gt_output:
            allele = self.alleles_used_for_gt[current_sample_allele_index]
            new_index = self.all_alleles_list_for_gt_output.index(allele)
            new_gt.append(new_index)
        return new_gt

    def get_final_mosaic_index(self, mosaic_output):
        new_mosaic_output = []
        for current_sample_gt_index in mosaic_output:
            new_gt = self.get_final_gt_index(current_sample_gt_index)
            new_mosaic_output.append(tuple(new_gt))
        return new_mosaic_output
