import pandas as pd
import numpy as np
from scipy.stats import geom
from scipy.special import logsumexp
import logging

# import os
import myutils

# from ..configs import config_params
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from configs import config_params
import config_params

DEBUG = False  # False
PLOT = False  # False
LOG_ONE_HALF = config_params.CONSTANT["LOG_ONE_HALF"]  # np.log(0.5)

EM_PARAMS = config_params.EM_PARAMS
MIN_SAMPLES_PASS_COVERAGE_NUM = EM_PARAMS["loci_filter"][
    "MIN_SAMPLES_PASS_COVERAGE_NUM"
]  # 5
MIN_SAMPLES_PASS_COVERAGE_FRACTION = EM_PARAMS["loci_filter"][
    "MIN_SAMPLES_PASS_COVERAGE_FRACTION"
]  # 0.1

FILTER_BIG_STUTTER_SIZE = EM_PARAMS["loci_filter"][
    "FILTER_BIG_STUTTER_SIZE"
]  # True
BIG_STUTTER_SIZE = EM_PARAMS["loci_filter"][
    "BIG_STUTTER_SIZE"
]  # 1 20, 2 10,3 10,4 9,5 8,6 7
BIG_STUTTER_DEPTH_CUTOFF = EM_PARAMS["loci_filter"][
    "BIG_STUTTER_DEPTH_CUTOFF"
]  # 1

MIN_AVERAGE_COVERAGE = EM_PARAMS["loci_filter"]["MIN_AVERAGE_COVERAGE"]  # 5
MIN_COVERAGE = EM_PARAMS["loci_filter"]["MIN_COVERAGE"]  # 100
SAMPLE_COVERAGE_CUTOFF = EM_PARAMS["loci_filter"][
    "SAMPLE_COVERAGE_CUTOFF"
]  # 5
SAMPLE_NUMBER_CUTOFF = EM_PARAMS["loci_filter"]["SAMPLE_NUMBER_CUTOFF"]  # 1
FILTER_OUT_SAMPLE_COVERAGE_CUTOFF = EM_PARAMS["loci_filter"][
    "FILTER_OUT_SAMPLE_COVERAGE_CUTOFF"
]  # 5
FILTER_LOW_COVERAGE_SAMPLE = EM_PARAMS["loci_filter"][
    "FILTER_LOW_COVERAGE_SAMPLE"
]  # False
MAX_ITERATION = EM_PARAMS["iteration_params"]["MAX_ITERATION"]  # 500
MIN_ITERATION = EM_PARAMS["iteration_params"]["MIN_ITERATION"]  # 20
CHECK_CONVERGE_PERIOD = EM_PARAMS["iteration_params"][
    "CHECK_CONVERGE_PERIOD"
]  # 2
DIFFERENCE_CONVERGE = EM_PARAMS["iteration_params"][
    "DIFFERENCE_CONVERGE"
]  # 0.0001
MAX_TOLERANCE = EM_PARAMS["iteration_params"][
    "MAX_TOLERANCE"
]  # np.log(1 / 10)
INIT_PRIOR_LOG_LL = EM_PARAMS["init_params"][
    "INIT_PRIOR_LOG_LL"
]  # -100000000.0
MIN_STUTTER_INS = EM_PARAMS["init_params"]["MIN_STUTTER_INS"]  # 0.000001
MIN_STUTTER_DEL = EM_PARAMS["init_params"]["MIN_STUTTER_DEL"]  # 0.000001
MAX_SINGLE_STEP_PROB = EM_PARAMS["init_params"][
    "MAX_SINGLE_STEP_PROB"
]  # 0.999
MAX_STUTTER_INS = EM_PARAMS["init_params"]["MAX_STUTTER_INS"]  # 0.9
MAX_STUTTER_DEL = EM_PARAMS["init_params"]["MAX_STUTTER_DEL"]  # 0.9
MIN_SINGLE_STEP_PROB = EM_PARAMS["init_params"][
    "MIN_SINGLE_STEP_PROB"
]  # 0.001
# 需要注意不同情况下默认的 stutter , homopolymer 的 out-frame 和为了防止 log 值异常的极值处理
# 情况分为没有证据（不训练，使用初始模型，homopolymer 为最小概率）
# 有证据没有 stutter（in-frame 和 out-frame 都没有 stutter）时，使用 Q30-Q40, homopolymer 为最小概率
# 为了输出, 如果单一 model 没有 stutter，则单一 model 使用最小 stutter 概率，homopolymer 也为最小概率
# 为了数值计算非零或者非一，使用最小 stutter 概率，homopolymer 也为最小概率
MIN_LOG_ZERO_PROB = EM_PARAMS["init_params"][
    "MIN_LOG_ZERO_PROB"
]  # HACK: small enough ? -100 没用到
MIN_GEOM_PROBS_SCALE = EM_PARAMS["init_params"][
    "MIN_GEOM_PROBS_SCALE"
]  # np.log(1e-6),  # 没用到
UPDATE_POP_LOG_AF = EM_PARAMS["bool_params"][
    "UPDATE_POP_LOG_AF"
]  # TODO: update pop log af and use this params if low allele dropout rate and clean data: HAS DONE True
ADD_PSEUDO_COUNT = EM_PARAMS["bool_params"]["ADD_PSEUDO_COUNT"]  # True
ORDERED_REPLACEMENT_GT = EM_PARAMS["other_params"][
    "ORDERED_REPLACEMENT_GT"
]  # True
HARDYWEINBERG_AF = EM_PARAMS["other_params"]["HARDYWEINBERG_AF"]  # True

INIT_STUTTER_MODEL = config_params.INIT_STUTTER_MODEL  # 0.9 0.05 0.01 0.000001
DEFAULT_STUTTER_MODEL = (
    config_params.DEFAULT_STUTTER_MODEL
)  # 0.9 0.05 0.01 0.000001
ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL = (
    config_params.ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL
)  # 0.000001, 0.999
MIN_NO_STUTTER_RATE = EM_PARAMS["init_params"]["MIN_NO_STUTTER_RATE"]  # 0.0001
NORMALIZATION_FACTOR_FOR_POP_AF = EM_PARAMS["init_params"][
    "NORMALIZATION_FACTOR_FOR_POP_AF"
]  # 10000

ALLOW_NUMPY_MIN_VALUE = np.finfo(float).tiny  # 1e-300


class NoiseEstimator:
    """Estimate the noise using EM algorithm
        Max Likelihood priority rather than posterior if prior is not accurate
        Give a uniform distribution for parameters we don't have confidence estimations

    Raises:
        ValueError: Ploidy should be haploid or diploid

    Returns:
        NoiseEstimator instance with noise parameters and methods
    """

    def __init__(
        self,
        all_samples_all_alleles_depth_dict,
        locus_infos,
        other_params,
    ):
        self.locus_infos = locus_infos
        self.other_params = other_params
        self.raw_samples_number = len(
            all_samples_all_alleles_depth_dict.keys()
        )  # XXX: primary samples number
        raw_alleles_order = list(
            pd.DataFrame(all_samples_all_alleles_depth_dict).fillna(0.0).index
        )
        self.raw_alleles_number = len(raw_alleles_order)
        if (
            self.raw_alleles_number <= 1
        ):  # XXX: solve ValueError: max() arg is an empty sequence
            self.max_raw_stutter_step = 1
        else:
            self.max_raw_stutter_step = max(raw_alleles_order) - min(
                raw_alleles_order
            )
        if FILTER_LOW_COVERAGE_SAMPLE:
            filtered_samples_number = 0
            filtered_samples_depth = 0
            samples_name = ""
            new_all_samples_all_alleles_depth_dict = {}
            for sample, depth in all_samples_all_alleles_depth_dict.items():
                if sum(depth.values()) >= FILTER_OUT_SAMPLE_COVERAGE_CUTOFF:
                    new_all_samples_all_alleles_depth_dict[sample] = depth
                else:
                    samples_name = samples_name + sample + "_"
                    filtered_samples_number += 1
                    filtered_samples_depth += sum(depth.values())
            all_samples_all_alleles_depth_dict = (
                new_all_samples_all_alleles_depth_dict
            )
            filtered_samples_name_samples_num = samples_name + str(
                filtered_samples_number
            )
            self.other_params[
                "filtered_samples_name_samples_num"
            ] = filtered_samples_name_samples_num
            self.other_params[
                "filtered_low_depth_samples_reads_num"
            ] = filtered_samples_depth
            # XXX: Filter low coverage, samples sum({}.values()) = 0
        (
            self.all_alleles_all_samples_depth_array,
            self.samples_order,
            self.alleles_order,
        ) = self.__dict2array(all_samples_all_alleles_depth_dict)
        self.alleles_number = len(self.alleles_order)
        self.samples_number = len(self.samples_order)
        # assert (
        #     self.samples_number > 0
        # ), "The number of samples should be greater than zero."
        # log_level = getattr(logging, other_params["loglevel"].upper(), None)
        # logfile = other_params["log_file"]
        # if not isinstance(log_level, int):
        #     raise ValueError(f"Invalid log level: {other_params['loglevel']}")
        # logger_stutter = logging.getLogger("StutterEstimator")
        # logger_stutter.setLevel(log_level)
        # if other_params["log_to_file"]:
        #     logger_stutter_file_handler = myutils.LockedFileHandler(logfile)
        # else:
        #     logger_stutter_file_handler = logging.StreamHandler()
        # # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
        # formatter = logging.Formatter(
        #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # )
        # logger_stutter_file_handler.setFormatter(formatter)
        # if not logger_stutter.handlers:  # 避免重复添加handler
        #     logger_stutter.addHandler(logger_stutter_file_handler)
        # self.other_params["logger_object"] = logger_stutter
        if ORDERED_REPLACEMENT_GT:
            self.gt_combinations_list = (
                myutils.get_all_gt_ordered_permutations_with_replacement(
                    list(range(self.alleles_number)),
                    self.locus_infos["ploidy"],
                )
            )
        else:
            self.gt_combinations_list = (
                myutils.get_all_gt_unordered_combinations_with_replacement(
                    list(range(self.alleles_number)),
                    self.locus_infos["ploidy"],
                )
            )
        self.gt_number = len(self.gt_combinations_list)
        if locus_infos["ploidy"] == 1:
            if self.alleles_number != 0:
                self.gts_pop_log_af = np.log(
                    np.array(
                        [float(1 / self.alleles_number)] * self.alleles_number
                    )
                )
            else:
                self.gts_pop_log_af = np.array([])
        elif locus_infos["ploidy"] == 2:
            if self.gt_number != 0:
                self.gts_pop_log_af = np.log(
                    np.array([float(1 / self.gt_number)] * self.gt_number)
                )
            else:
                self.gts_pop_log_af = np.array([])
            self.diploid_allele_imbalance_af_h1 = np.array(
                [1 / 2] * self.samples_number
            )
            self.diploid_allele_imbalance_af_h2 = np.array(
                [1 / 2] * self.samples_number
            )
        else:
            raise ValueError("Ploidy should be 1 or 2")
        if self.alleles_number > 1:
            (
                self.bp_difference_of_alleles_all_params_array,
                self.max_inframe_stutter_steps,
                self.max_outframe_stutter_steps,
            ) = self.__cal_bp_difference_of_alleles()
            if UPDATE_POP_LOG_AF:
                self.alleles_pop_log_af = np.log(
                    np.array(
                        [float(1 / self.alleles_number)] * self.alleles_number
                    )
                )
                if self.locus_infos["ploidy"] == 2:
                    self.alleles_gts_indication_array = (
                        self.__get_alleles_gts_indication_array()
                    )
        else:
            self.bp_difference_of_alleles_all_params_array = np.zeros(
                (self.alleles_number, self.alleles_number, 7)
            ).astype(np.float64)
            self.max_inframe_stutter_steps = 0
            self.max_outframe_stutter_steps = 0
        self.depth_per_sample = self.all_alleles_all_samples_depth_array.sum(
            axis=0
        )
        self.total_depth = np.sum(self.all_alleles_all_samples_depth_array)
        if self.total_depth == 0:
            self.min_stutter = MIN_STUTTER_INS
        else:
            if self.locus_infos["motif_length"] == 1:
                self.min_stutter = 1 / (
                    self.total_depth * 2
                )  # XXX:minus stutter params
            else:
                self.min_stutter = 1 / (
                    self.total_depth * 4
                )  # XXX:minus stutter params
        if self.samples_number == 0:
            self.average_depth = 0
            self.pass_depth_sample_number = 0
            self.pass_depth_sample_fraction = 0
        else:
            self.average_depth = self.total_depth / self.samples_number
            self.pass_depth_sample_number = np.sum(
                (self.depth_per_sample >= SAMPLE_COVERAGE_CUTOFF)
            )
            self.pass_depth_sample_fraction = (
                self.pass_depth_sample_number
                / self.raw_samples_number  # XXX: primary samples number
            )
        self.all_samples_all_gt_log_posterior = np.array([], dtype=np.float64)
        self.trained = False
        self.total_niter = 0
        self.n_train = 0
        self.converged = False
        self.converged_list = [False, False, False]
        self.locus_state = "WaitTrain"
        if DEBUG:
            print(all_samples_all_alleles_depth_dict)
            print(locus_infos)
            print(other_params)
            self.total_prior_log_ll_list = []

    def __dict2array(self, all_samples_all_alleles_depth_dict):
        all_samples_all_alleles_depth_df = pd.DataFrame(
            all_samples_all_alleles_depth_dict
        ).fillna(0.0)
        if (
            FILTER_BIG_STUTTER_SIZE
        ):  # HACK: stutter 过滤了 big stutter size reads，那么 calling mosaic 时也要过滤这条 read，否则容易造成 hom calling 成 het
            # HACK: 因为 calling mosaic 的时候，只考虑到单个样本的 reads 信息，所以无法过滤掉 all samples low depth big stutter size reads，因此当前倾向于 FILTER_BIG_STUTTER_SIZE = False
            big_stutter_size_reads_number = 0
            big_stutter_size_alleles_number = 0
            max_stutter_size = (
                BIG_STUTTER_SIZE[self.locus_infos["motif_length"]]
                * self.locus_infos["motif_length"]
            )
            for index, row in all_samples_all_alleles_depth_df.iterrows():
                bp_diff = abs(index - self.locus_infos["ref_allele_length"])
                if (
                    bp_diff > max_stutter_size
                    and row.sum() <= BIG_STUTTER_DEPTH_CUTOFF
                ):
                    all_samples_all_alleles_depth_df = (
                        all_samples_all_alleles_depth_df.drop(index)
                    )
                    big_stutter_size_reads_number += row.sum()
                    big_stutter_size_alleles_number += 1
            self.other_params[
                "big_stutter_size_reads_number"
            ] = big_stutter_size_reads_number
            self.other_params[
                "big_stutter_size_alleles_number"
            ] = big_stutter_size_alleles_number
        samples_order = list(all_samples_all_alleles_depth_df.columns)
        alleles_order = np.array(
            all_samples_all_alleles_depth_df.index
        ).astype(np.float64)
        # XXX When estimate noise, we don't care which is reference allele.
        all_alleles_all_samples_depth_array = np.array(
            all_samples_all_alleles_depth_df
        )
        all_alleles_all_samples_depth_array = (
            all_alleles_all_samples_depth_array.astype(np.float64)
        )
        return (
            all_alleles_all_samples_depth_array,
            samples_order,
            alleles_order,
        )

    def __cal_bp_difference_of_alleles(self):
        # XXX: bp difference for pseudo_reads - alleles, Need .T transpose
        bp_difference_of_alleles_array = (
            np.subtract(
                self.alleles_order[:, np.newaxis], self.alleles_order
            ).astype(np.float64)
        ).T
        bp_difference_of_alleles_all_params_array = np.zeros(
            (self.alleles_number, self.alleles_number, 7)
        ).astype(np.float64)
        bp_difference_of_alleles_all_params_array[:, :, 0] = np.where(
            bp_difference_of_alleles_array == 0, 1.0, 0.0
        )
        bp_difference_of_alleles_all_params_array[:, :, 1] = np.where(
            (bp_difference_of_alleles_array > 0)
            & (
                (
                    bp_difference_of_alleles_array
                    % self.locus_infos["motif_length"]
                )
                == 0
            ),
            1.0,
            0.0,
        )
        bp_difference_of_alleles_all_params_array[:, :, 2] = np.where(
            (bp_difference_of_alleles_array < 0)
            & (
                (
                    bp_difference_of_alleles_array
                    % self.locus_infos["motif_length"]
                )
                == 0
            ),
            1.0,
            0.0,
        )
        bp_difference_of_alleles_all_params_array[:, :, 4] = np.where(
            (bp_difference_of_alleles_array > 0)
            & (
                (
                    bp_difference_of_alleles_array
                    % self.locus_infos["motif_length"]
                )
                != 0
            ),
            1.0,
            0.0,
        )
        bp_difference_of_alleles_all_params_array[:, :, 5] = np.where(
            (bp_difference_of_alleles_array < 0)
            & (
                (
                    bp_difference_of_alleles_array
                    % self.locus_infos["motif_length"]
                )
                != 0
            ),
            1.0,
            0.0,
        )
        inframe_bp_difference_of_alleles_array = (
            np.abs(bp_difference_of_alleles_array)
            / self.locus_infos["motif_length"]
        )
        outframe_bp_difference_of_alleles_array = np.abs(
            bp_difference_of_alleles_array
        ) - (
            np.abs(bp_difference_of_alleles_array)
            // self.locus_infos["motif_length"]
        )
        bp_difference_of_alleles_all_params_array[:, :, 3] = np.where(
            (bp_difference_of_alleles_array % self.locus_infos["motif_length"])
            == 0,
            inframe_bp_difference_of_alleles_array,
            0.0,
        )
        bp_difference_of_alleles_all_params_array[:, :, 6] = np.where(
            (bp_difference_of_alleles_array % self.locus_infos["motif_length"])
            != 0,
            outframe_bp_difference_of_alleles_array,
            0.0,
        )
        max_inframe_stutter_steps = np.max(
            bp_difference_of_alleles_all_params_array[:, :, 3]
        )
        max_outframe_stutter_steps = np.max(
            bp_difference_of_alleles_all_params_array[:, :, 6]
        )
        return (
            bp_difference_of_alleles_all_params_array,
            max_inframe_stutter_steps,
            max_outframe_stutter_steps,
        )

    def __get_alleles_gts_indication_array(self):
        alleles_gts_indication_array = np.zeros(
            (self.alleles_number, self.gt_number), dtype=np.float64
        )
        for gt_index, gt in enumerate(self.gt_combinations_list):
            allele_1_index = gt[0]
            allele_2_index = gt[1]
            alleles_gts_indication_array[allele_1_index, gt_index] += 1
            alleles_gts_indication_array[allele_2_index, gt_index] += 1
        return alleles_gts_indication_array

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
        stutter_single_step_prob = max(
            [stutter_single_step_prob, MIN_SINGLE_STEP_PROB]
        )
        stutter_ins_prob = max([stutter_ins_prob, self.min_stutter])
        stutter_del_prob = max([stutter_del_prob, self.min_stutter])
        stutter_ins_prob = min([stutter_ins_prob, MAX_STUTTER_INS])
        stutter_del_prob = min([stutter_del_prob, MAX_STUTTER_DEL])
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
        no_stutter_rate = max(
            [
                1.0 - stutter_del_prob - stutter_ins_prob,
                MIN_NO_STUTTER_RATE,
            ]
        )
        # XXX: for 388 estimate_noise: RuntimeWarning: divide by zero encountered in log
        stutter_probs = (
            list(np.log(stutter_del_prob) + np.log(step_probs[::-1]))
            + [np.log(no_stutter_rate)]
            + list(np.log(stutter_ins_prob) + np.log(step_probs))
        )
        log_stutter_model_dic = dict(zip(stutter_sizes, stutter_probs))
        return log_stutter_model_dic

    def calc_per_read_log_likelihood_given_allele(
        self, allele_str_length, read_str_length
    ):
        bp_diff = read_str_length - allele_str_length
        if bp_diff == 0.0:
            per_read_log_ll = np.log(
                np.exp(self.inframe_stutter_model_dict[0])
                + np.exp(self.outframe_stutter_model_dict[0])
                - 1
            )
        else:
            quotient, remainder = divmod(
                abs(bp_diff), self.locus_infos["motif_length"]
            )
            if remainder == 0.0:
                per_read_log_ll = self.inframe_stutter_model_dict[
                    int((bp_diff) / self.locus_infos["motif_length"])
                ]
            else:
                if bp_diff > 0.0:
                    bp_diff_2_continuous_value = bp_diff - quotient
                else:
                    bp_diff_2_continuous_value = bp_diff + quotient
                per_read_log_ll = self.outframe_stutter_model_dict[
                    bp_diff_2_continuous_value
                ]
        return per_read_log_ll

    def calc_all_reads_log_likelihood_given_allele(self):
        # BUG: Notice that (A+B)**a > (A**a+B**a): HAS SOLVED
        # BUG: Notice that needs add read assignment probability calculation: HAS SOLVED
        if self.locus_infos["ploidy"] == 1:
            all_samples_all_alleles_log_ll = np.zeros(
                (self.samples_number, self.alleles_number), dtype=np.float64
            )  # Dimension tuple as parameter
            for allele_index, allele_str_length in enumerate(
                self.alleles_order
            ):  # enumerate is 0-based index
                per_pseudo_read_log_ll = np.array(
                    [
                        self.calc_per_read_log_likelihood_given_allele(
                            allele_str_length, pseudo_read_str_length
                        )
                        for pseudo_read_str_length in self.alleles_order
                    ]
                )
                per_pseudo_read_log_ll = per_pseudo_read_log_ll.reshape(
                    len(per_pseudo_read_log_ll), 1
                )
                all_samples_all_alleles_log_ll[:, allele_index] += np.sum(
                    self.all_alleles_all_samples_depth_array
                    * per_pseudo_read_log_ll,
                    axis=0,
                )
            return all_samples_all_alleles_log_ll
        elif self.locus_infos["ploidy"] == 2:
            all_samples_all_gts_log_ll = np.zeros(
                (self.samples_number, self.gt_number), dtype=np.float64
            )  # Dimension tuple as parameter
            for gt_index, gt in enumerate(
                self.gt_combinations_list
            ):  # enumerate is 0-based index
                allele1_str_length = self.alleles_order[gt[0]]
                allele2_str_length = self.alleles_order[gt[1]]
                per_pseudo_read_log_ll_given_allele1 = np.array(
                    [
                        [
                            self.calc_per_read_log_likelihood_given_allele(
                                allele1_str_length, pseudo_read_str_length
                            )
                            for pseudo_read_str_length in self.alleles_order
                        ]
                    ]
                ).T
                per_pseudo_read_log_ll_given_allele2 = np.array(
                    [
                        [
                            self.calc_per_read_log_likelihood_given_allele(
                                allele2_str_length, pseudo_read_str_length
                            )
                            for pseudo_read_str_length in self.alleles_order
                        ]
                    ]
                ).T
                per_samples_per_pseudo_read_log_ll = logsumexp(
                    [
                        per_pseudo_read_log_ll_given_allele1,
                        per_pseudo_read_log_ll_given_allele2,
                    ],
                    axis=0,
                    b=[
                        self.diploid_allele_imbalance_af_h1[np.newaxis, :],
                        self.diploid_allele_imbalance_af_h2[np.newaxis, :],
                    ],
                )
                all_samples_all_gts_log_ll[:, gt_index] += np.sum(
                    self.all_alleles_all_samples_depth_array
                    * per_samples_per_pseudo_read_log_ll,
                    axis=0,
                )
            return all_samples_all_gts_log_ll
        else:
            raise ValueError("Ploidy should be 1 or 2")

    def calc_unphased_unordered_gt_posterior(self):
        if self.locus_infos["ploidy"] == 1:
            all_samples_all_gt_prior_log_ll = (
                self.gts_pop_log_af + self.all_samples_all_gts_log_ll
            )
        elif self.locus_infos["ploidy"] == 2:
            all_samples_all_gt_prior_log_ll = (
                self.gts_pop_log_af + self.all_samples_all_gts_log_ll
            )
        else:
            raise ValueError("Ploidy should be 1 or 2")
        p_data_log_sum_gt_prior_log_ll_per_sample = logsumexp(
            all_samples_all_gt_prior_log_ll, axis=1
        )[np.newaxis].T
        all_samples_all_gt_log_posterior = (
            all_samples_all_gt_prior_log_ll
            - p_data_log_sum_gt_prior_log_ll_per_sample
        )
        all_samples_all_gt_total_prior_log_ll = np.sum(
            p_data_log_sum_gt_prior_log_ll_per_sample
        )
        return (
            all_samples_all_gt_log_posterior,
            all_samples_all_gt_total_prior_log_ll,
        )

    def update_pop_log_AF(self):
        # TODO: Need Test updating pop AF when (update or not/ploidy/OrderedOrNot/HARDY-WeinBerg) HAS DONE
        if (
            self.locus_infos["ploidy"] == 1
        ):  # HACK: STILL NO DEBUG FOR RuntimeWarning: divide by zero encountered in log
            gts_pop_log_af = logsumexp(
                self.all_samples_all_gt_log_posterior, axis=0
            ) - np.log(self.samples_number)
            alleles_pop_log_af = gts_pop_log_af
            return gts_pop_log_af, alleles_pop_log_af
        elif self.locus_infos["ploidy"] == 2:
            if HARDYWEINBERG_AF:
                diploid_pop_gt_log_af = logsumexp(
                    self.all_samples_all_gt_log_posterior, axis=0
                ) - np.log(self.samples_number)
                diploid_pop_gt_af = np.exp(diploid_pop_gt_log_af)
                alleles_gts_posterior_array = (
                    self.alleles_gts_indication_array * diploid_pop_gt_af
                )
                alleles_total_popAF = np.sum(
                    alleles_gts_posterior_array, axis=1
                )
                diploid_pop_alleles_AF = alleles_total_popAF / (
                    alleles_total_popAF.sum()
                )
                if DEBUG:
                    print("Before popAF: ")
                    print(diploid_pop_alleles_AF)
                if diploid_pop_alleles_AF.min() == 0.0:
                    diploid_pop_alleles_AF[diploid_pop_alleles_AF == 0.0] = (
                        diploid_pop_alleles_AF[
                            diploid_pop_alleles_AF.nonzero()
                        ].min()
                        / NORMALIZATION_FACTOR_FOR_POP_AF
                    )
                # XXX: for 567 570 estimate_noise: RuntimeWarning: divide by zero encountered in log
                if DEBUG:
                    print("After popAF: ")
                    print(diploid_pop_alleles_AF)
                if ORDERED_REPLACEMENT_GT:
                    unnorm_diploid_pop_af = np.array(
                        [
                            diploid_pop_alleles_AF[gt[0]]
                            * diploid_pop_alleles_AF[gt[1]]
                            for gt in self.gt_combinations_list
                        ]
                    )
                else:
                    unnorm_diploid_pop_af = np.array(
                        [
                            diploid_pop_alleles_AF[gt[0]]
                            * diploid_pop_alleles_AF[gt[1]]
                            if gt[0] == gt[1]
                            else 2
                            * diploid_pop_alleles_AF[gt[0]]
                            * diploid_pop_alleles_AF[gt[1]]
                            for gt in self.gt_combinations_list
                        ]
                    )
                gts_pop_log_af = np.log(
                    unnorm_diploid_pop_af / (unnorm_diploid_pop_af.sum())
                    + ALLOW_NUMPY_MIN_VALUE
                )  # HACK: ALLOW_NUMPY_MIN_VALUE to debug zero division error Done
                # /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/code/estimate_noise.py:690: RuntimeWarning: divide by zero encountered in log # gts_pop_log_af = np.log(
                # HACK: 不解决也没事，因为有些 popAF 非常小，小到几乎为零，而这些 af 总和为 1，所以有另外一些 af 比较大的值占比较大的权重，而几乎为零的这些 af 在计算中权重较小可以忽略不计，所以即使是 0 或者 -inf 也没事，概率的计算仍旧是对的，不用为了这个 warning 而担心去特地修改代码或者设置最小或者最大阀值在这里哈
                alleles_pop_log_af = np.log(
                    diploid_pop_alleles_AF + ALLOW_NUMPY_MIN_VALUE
                )  # HACK: ALLOW_NUMPY_MIN_VALUE to debug zero division error Done
                return gts_pop_log_af, alleles_pop_log_af
            else:  # HACK: STILL NO DEBUG FOR RuntimeWarning: divide by zero encountered in log
                gts_pop_log_af = logsumexp(
                    self.all_samples_all_gt_log_posterior, axis=0
                ) - np.log(self.samples_number)
                alleles_pop_log_af = np.log(
                    np.array([1 / self.alleles_number] * self.alleles_number)
                )
                return gts_pop_log_af, alleles_pop_log_af
        else:
            raise ValueError("Ploidy should be 1 or 2")

    def update_stutter_model(self):
        if self.locus_infos["ploidy"] == 1:
            bp_difference_of_dip_gt_samples_all_params_add_depth_array = (
                np.einsum(
                    "abc,bd->adc",
                    self.bp_difference_of_alleles_all_params_array,
                    self.all_alleles_all_samples_depth_array,
                )
            )
        elif self.locus_infos["ploidy"] == 2:
            bp_difference_of_dip_gt_samples_all_params_add_depth_array = (
                np.zeros((self.gt_number, self.samples_number, 7)).astype(
                    np.float64
                )
            )
            for gt_index, gt in enumerate(self.gt_combinations_list):
                allele_1_index, allele_2_index = gt
                # BUG: Notice that I needs add read assignment probability calculation: HAS SOLVED
                allele1_str_length = self.alleles_order[allele_1_index]
                allele2_str_length = self.alleles_order[allele_2_index]
                per_pseudo_read_log_ll_given_allele1 = np.array(
                    [
                        self.calc_per_read_log_likelihood_given_allele(
                            allele1_str_length, pseudo_read_str_length
                        )
                        for pseudo_read_str_length in self.alleles_order
                    ]
                )
                per_pseudo_read_log_ll_given_allele2 = np.array(
                    [
                        self.calc_per_read_log_likelihood_given_allele(
                            allele2_str_length, pseudo_read_str_length
                        )
                        for pseudo_read_str_length in self.alleles_order
                    ]
                )
                samples_alleles_per_pseudo_read_prior_ll_given_allele1 = (
                    self.diploid_allele_imbalance_af_h1[:, np.newaxis]
                    * np.exp(per_pseudo_read_log_ll_given_allele1)
                )
                samples_alleles_per_pseudo_read_prior_ll_given_allele2 = (
                    self.diploid_allele_imbalance_af_h2[:, np.newaxis]
                    * np.exp(per_pseudo_read_log_ll_given_allele2)
                )
                samples_alleles_per_pseudo_read_prior_ll_given_gt = (
                    samples_alleles_per_pseudo_read_prior_ll_given_allele1
                    + samples_alleles_per_pseudo_read_prior_ll_given_allele2
                )
                samples_alleles_per_pseudo_read_posterior_given_allele1 = (
                    samples_alleles_per_pseudo_read_prior_ll_given_allele1
                    / samples_alleles_per_pseudo_read_prior_ll_given_gt
                )
                samples_alleles_per_pseudo_read_posterior_given_allele2 = (
                    samples_alleles_per_pseudo_read_prior_ll_given_allele2
                    / samples_alleles_per_pseudo_read_prior_ll_given_gt
                )
                all_samples_all_alleles_depth_array = (
                    self.all_alleles_all_samples_depth_array.T
                )
                samples_alleles_per_pseudo_read_posterior_add_depth_given_allele1 = (
                    all_samples_all_alleles_depth_array
                    * samples_alleles_per_pseudo_read_posterior_given_allele1
                )
                samples_alleles_per_pseudo_read_posterior_add_depth_given_allele2 = (
                    all_samples_all_alleles_depth_array
                    * samples_alleles_per_pseudo_read_posterior_given_allele2
                )
                samples_per_pseudo_read_posterior_add_depth_add_all_params_given_allele1 = (
                    samples_alleles_per_pseudo_read_posterior_add_depth_given_allele1
                    @ self.bp_difference_of_alleles_all_params_array[
                        allele_1_index, :, :
                    ]
                )
                samples_per_pseudo_read_posterior_add_depth_add_all_params_given_allele2 = (
                    samples_alleles_per_pseudo_read_posterior_add_depth_given_allele2
                    @ self.bp_difference_of_alleles_all_params_array[
                        allele_2_index, :, :
                    ]
                )
                samples_per_pseudo_read_posterior_add_depth_add_all_params_given_gt = (
                    samples_per_pseudo_read_posterior_add_depth_add_all_params_given_allele1
                    + samples_per_pseudo_read_posterior_add_depth_add_all_params_given_allele2
                )
                bp_difference_of_dip_gt_samples_all_params_add_depth_array[
                    gt_index, :, :
                ] = samples_per_pseudo_read_posterior_add_depth_add_all_params_given_gt
        else:
            raise ValueError("Ploidy should be 1 or 2")
        gt_samples_all_params_add_depth_add_posterior_array = (
            bp_difference_of_dip_gt_samples_all_params_add_depth_array
            * (
                np.exp(self.all_samples_all_gt_log_posterior.T)[
                    :, :, np.newaxis
                ]
            )
        )
        total_pseudo_reads_num = np.sum(
            gt_samples_all_params_add_depth_add_posterior_array[
                :, :, [0, 1, 2, 4, 5]
            ]
        )
        if total_pseudo_reads_num == 0:
            self.other_params["logger_object"].info(
                "The total number of pseudo reads is 0,"
                " and we will use the previous set of parameters."
            )
            return (
                1
                - self.inframe_ins_prob
                - self.inframe_del_prob
                - self.outframe_ins_prob
                - self.outframe_del_prob,
                self.inframe_ins_prob,
                self.inframe_del_prob,
                self.inframe_single_step_prob,
                self.outframe_ins_prob,
                self.outframe_del_prob,
                self.outframe_single_step_prob,
            )  # XXX: When the total number of reads is zero, it is advisable to use the default stutter parameters.
        if ADD_PSEUDO_COUNT:
            if self.locus_infos["motif_length"] == 1:
                no_stutter_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 0
                        ]
                    )
                    + 1
                ) / (total_pseudo_reads_num + 3)
                inframe_ins_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 1
                        ]
                    )
                    + 1
                ) / (total_pseudo_reads_num + 3)
                inframe_del_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 2
                        ]
                    )
                    + 1
                ) / (total_pseudo_reads_num + 3)
                outframe_ins_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 4
                        ]
                    )
                    + 0
                ) / (total_pseudo_reads_num + 3)
                outframe_del_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 5
                        ]
                    )
                    + 0
                ) / (total_pseudo_reads_num + 3)
                inframe_pseudo_reads_num = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, [1, 2]
                        ]
                    )
                    + 2
                )
                outframe_pseudo_reads_num = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, [4, 5]
                        ]
                    )
                    + 0
                )
                inframe_all_steps_num = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, [3]
                        ]
                    )
                    + 1
                    + 1.1
                )
                if inframe_all_steps_num == 0:
                    # self.other_params["logger_object"].info(
                    #     "The total number of inframe stutter steps is 0,"
                    #     " and we will use the previous set of parameters."
                    # )  # XXX: When the total number of reads is zero, it is advisable to use the MAX_SINGLE_STEP_PROB.
                    inframe_single_step_prob = MAX_SINGLE_STEP_PROB
                else:
                    inframe_single_step_prob = (
                        inframe_pseudo_reads_num / inframe_all_steps_num
                    )
                outframe_all_steps_num = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, [6]
                        ]
                    )
                    + 0
                    + 0
                )
                if outframe_all_steps_num == 0:
                    # self.other_params["logger_object"].info(
                    #     "The total number of outframe stutter steps is 0,"
                    #     " and we will use the previous set of parameters."
                    # )  # XXX: When the total number of reads is zero, it is advisable to use the MAX_SINGLE_STEP_PROB.
                    outframe_single_step_prob = MAX_SINGLE_STEP_PROB
                else:
                    outframe_single_step_prob = (
                        outframe_pseudo_reads_num / outframe_all_steps_num
                    )
            else:
                no_stutter_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 0
                        ]
                    )
                    + 1
                ) / (total_pseudo_reads_num + 5)
                inframe_ins_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 1
                        ]
                    )
                    + 1
                ) / (total_pseudo_reads_num + 5)
                inframe_del_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 2
                        ]
                    )
                    + 1
                ) / (total_pseudo_reads_num + 5)
                outframe_ins_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 4
                        ]
                    )
                    + 1
                ) / (total_pseudo_reads_num + 5)
                outframe_del_prob = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, 5
                        ]
                    )
                    + 1
                ) / (total_pseudo_reads_num + 5)
                inframe_pseudo_reads_num = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, [1, 2]
                        ]
                    )
                    + 2
                )
                outframe_pseudo_reads_num = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, [4, 5]
                        ]
                    )
                    + 2
                )
                inframe_all_steps_num = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, [3]
                        ]
                    )
                    + 1
                    + 1.1
                )
                if inframe_all_steps_num == 0:
                    # self.other_params["logger_object"].info(
                    #     "The total number of inframe stutter steps is 0,"
                    #     " and we will use the previous set of parameters."
                    # )  # XXX: When the total number of reads is zero, it is advisable to use the MAX_SINGLE_STEP_PROB.
                    inframe_single_step_prob = MAX_SINGLE_STEP_PROB
                else:
                    inframe_single_step_prob = (
                        inframe_pseudo_reads_num / inframe_all_steps_num
                    )
                outframe_all_steps_num = (
                    np.sum(
                        gt_samples_all_params_add_depth_add_posterior_array[
                            :, :, [6]
                        ]
                    )
                    + 1
                    + 1.1
                )
                if outframe_all_steps_num == 0:
                    # self.other_params["logger_object"].info(
                    #     "The total number of outframe stutter steps is 0,"
                    #     " and we will use the previous set of parameters."
                    # )  # XXX: When the total number of reads is zero, it is advisable to use the MAX_SINGLE_STEP_PROB.
                    outframe_single_step_prob = MAX_SINGLE_STEP_PROB
                else:
                    outframe_single_step_prob = (
                        outframe_pseudo_reads_num / outframe_all_steps_num
                    )
        else:
            no_stutter_prob = (
                np.sum(
                    gt_samples_all_params_add_depth_add_posterior_array[
                        :, :, 0
                    ]
                )
                / total_pseudo_reads_num
            )
            inframe_ins_prob = (
                np.sum(
                    gt_samples_all_params_add_depth_add_posterior_array[
                        :, :, 1
                    ]
                )
                / total_pseudo_reads_num
            )
            inframe_del_prob = (
                np.sum(
                    gt_samples_all_params_add_depth_add_posterior_array[
                        :, :, 2
                    ]
                )
                / total_pseudo_reads_num
            )
            outframe_ins_prob = (
                np.sum(
                    gt_samples_all_params_add_depth_add_posterior_array[
                        :, :, 4
                    ]
                )
                / total_pseudo_reads_num
            )
            outframe_del_prob = (
                np.sum(
                    gt_samples_all_params_add_depth_add_posterior_array[
                        :, :, 5
                    ]
                )
                / total_pseudo_reads_num
            )
            inframe_pseudo_reads_num = np.sum(
                gt_samples_all_params_add_depth_add_posterior_array[
                    :, :, [1, 2]
                ]
            )
            outframe_pseudo_reads_num = np.sum(
                gt_samples_all_params_add_depth_add_posterior_array[
                    :, :, [4, 5]
                ]
            )
            inframe_all_steps_num = np.sum(
                gt_samples_all_params_add_depth_add_posterior_array[:, :, [3]]
            )
            if inframe_all_steps_num == 0:
                # self.other_params["logger_object"].info(
                #     "The total number of inframe stutter steps is 0,"
                #     " and we will use the previous set of parameters."
                # )  # XXX: When the total number of reads is zero, it is advisable to use the MAX_SINGLE_STEP_PROB.
                inframe_single_step_prob = MAX_SINGLE_STEP_PROB
            else:
                inframe_single_step_prob = (
                    inframe_pseudo_reads_num / inframe_all_steps_num
                )
            outframe_all_steps_num = np.sum(
                gt_samples_all_params_add_depth_add_posterior_array[:, :, [6]]
            )
            if outframe_all_steps_num == 0:
                # self.other_params["logger_object"].info(
                #     "The total number of outframe stutter steps is 0,"
                #     " and we will use the previous set of parameters."
                # )  # XXX: When the total number of reads is zero, it is advisable to use the MAX_SINGLE_STEP_PROB.
                outframe_single_step_prob = MAX_SINGLE_STEP_PROB
            else:
                outframe_single_step_prob = (
                    outframe_pseudo_reads_num / outframe_all_steps_num
                )
        return (
            no_stutter_prob,
            inframe_ins_prob,
            inframe_del_prob,
            inframe_single_step_prob,
            outframe_ins_prob,
            outframe_del_prob,
            outframe_single_step_prob,
        )

    def train(self):
        if (
            self.locus_state == "CoverageFail"
            or self.locus_state == "PassCovSampleNumOrFractionFail"
            or self.locus_state == "NoStutterFail"
            or self.locus_state == "NoEnoughSampleNumberFail"
        ):
            self.other_params["logger_object"].info(
                "This locus will not be trained due to"
                " insufficient coverage or number of alleles or samples,"
                " and the train function was invoked previously."
            )
            return
        elif self.trained and self.converged:
            self.other_params["logger_object"].info(
                "We invoked the train function previously,"
                " and this locus has now converged."
            )
            return
        elif self.trained and (1 - self.converged):
            self.niter = MIN_ITERATION
            self.other_params["logger_object"].info(
                "We invoked the train function previously,"
                " but this locus did not converge."
                " We will continue the training"
                " using the previous set of parameters."
            )
        else:
            self.other_params["logger_object"].info(
                "The train function has not been invoked before,"
                " and we will train this locus from the beginning."
            )
            self.niter = 0
            self.total_prior_log_ll = INIT_PRIOR_LOG_LL
            coverage1 = self.total_depth < MIN_COVERAGE
            coverage2 = self.average_depth < MIN_AVERAGE_COVERAGE
            coverage3 = (
                self.pass_depth_sample_number < MIN_SAMPLES_PASS_COVERAGE_NUM
            )
            coverage4 = (
                self.pass_depth_sample_fraction
                < MIN_SAMPLES_PASS_COVERAGE_FRACTION
            )
            self.coverage_state_list = [
                coverage1,
                coverage2,
                coverage3,
                coverage4,
            ]
            self.inframe_del_prob = INIT_STUTTER_MODEL["inframe_del_prob"]
            self.inframe_ins_prob = INIT_STUTTER_MODEL["inframe_ins_prob"]
            self.inframe_single_step_prob = INIT_STUTTER_MODEL[
                "inframe_single_step_prob"
            ]
            self.outframe_single_step_prob = INIT_STUTTER_MODEL[
                "outframe_single_step_prob"
            ]
            self.inframe_stutter_model_dict = self.stutter_model(
                INIT_STUTTER_MODEL["inframe_single_step_prob"],
                INIT_STUTTER_MODEL["inframe_ins_prob"],
                INIT_STUTTER_MODEL["inframe_del_prob"],
                self.max_inframe_stutter_steps,
            )
            if self.locus_infos["motif_length"] == 1:
                self.outframe_ins_prob = INIT_STUTTER_MODEL[
                    "homopolymer_outframe_ins_prob"
                ]
                self.outframe_del_prob = INIT_STUTTER_MODEL[
                    "homopolymer_outframe_del_prob"
                ]
                self.outframe_stutter_model_dict = self.stutter_model(
                    INIT_STUTTER_MODEL["outframe_single_step_prob"],
                    INIT_STUTTER_MODEL["homopolymer_outframe_ins_prob"],
                    INIT_STUTTER_MODEL["homopolymer_outframe_del_prob"],
                    self.max_outframe_stutter_steps,
                )
            else:
                self.outframe_ins_prob = INIT_STUTTER_MODEL[
                    "outframe_ins_prob"
                ]
                self.outframe_del_prob = INIT_STUTTER_MODEL[
                    "outframe_del_prob"
                ]
                self.outframe_stutter_model_dict = self.stutter_model(
                    INIT_STUTTER_MODEL["outframe_single_step_prob"],
                    INIT_STUTTER_MODEL["outframe_ins_prob"],
                    INIT_STUTTER_MODEL["outframe_del_prob"],
                    self.max_outframe_stutter_steps,
                )
            self.no_stutter_prob = (
                1
                - self.inframe_del_prob
                - self.inframe_ins_prob
                - self.outframe_del_prob
                - self.outframe_ins_prob
            )
            # samples number depth and stutter
            # samples number and depth fails prior,if fails,use default stutter model
            # then success,if no stutter,use low stutter model
            if self.samples_number < SAMPLE_NUMBER_CUTOFF:
                self.other_params["logger_object"].info(
                    f"{self.locus_infos['STR_id']} has a sample number less"
                    f" than {SAMPLE_NUMBER_CUTOFF}"
                )
                self.locus_state = "NoEnoughSampleNumberFail"
                return
            elif coverage1 and coverage2:
                self.other_params["logger_object"].info(
                    f"{self.locus_infos['STR_id']} has a total depth less than"
                    f" {MIN_COVERAGE} and an average depth less than"
                    f" {MIN_AVERAGE_COVERAGE}"
                )
                self.locus_state = "CoverageFail"
                return
            elif (
                coverage3 and coverage4
            ):  # XXX: at least 0.1 samples fraction pass coverage cutoff and 5 samples pass coverage cutoff
                self.other_params["logger_object"].info(
                    f"For {self.locus_infos['STR_id']},"
                    " the number of samples with sufficient coverage"
                    f" (depth not less than {SAMPLE_COVERAGE_CUTOFF})"
                    " is below the required threshold of"
                    f" {MIN_SAMPLES_PASS_COVERAGE_NUM},"
                    " and the fraction of samples with sufficient coverage"
                    " is below the required threshold of"
                    f" {MIN_SAMPLES_PASS_COVERAGE_FRACTION}."
                )
                self.locus_state = "PassCovSampleNumOrFractionFail"
                return
            if (
                self.max_inframe_stutter_steps == 0
                and self.max_outframe_stutter_steps == 0
            ):
                self.other_params["logger_object"].info(
                    f"{self.locus_infos['STR_id']} shows no stutter and"
                    " only a length-based allele in all samples."
                )
                self.locus_state = "NoStutterFail"
                self.inframe_del_prob = (
                    ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL["inframe_del_prob"]
                )
                self.inframe_ins_prob = (
                    ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL["inframe_ins_prob"]
                )
                self.inframe_single_step_prob = (
                    ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL[
                        "inframe_single_step_prob"
                    ]
                )
                self.outframe_single_step_prob = (
                    ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL[
                        "outframe_single_step_prob"
                    ]
                )
                if self.locus_infos["motif_length"] == 1:
                    self.outframe_del_prob = (
                        ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL[
                            "homopolymer_outframe_del_prob"
                        ]
                    )
                    self.outframe_ins_prob = (
                        ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL[
                            "homopolymer_outframe_ins_prob"
                        ]
                    )
                else:
                    self.outframe_del_prob = (
                        ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL[
                            "outframe_del_prob"
                        ]
                    )
                    self.outframe_ins_prob = (
                        ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL[
                            "outframe_ins_prob"
                        ]
                    )
                self.no_stutter_prob = (
                    1
                    - self.inframe_del_prob
                    - self.inframe_ins_prob
                    - self.outframe_del_prob
                    - self.outframe_ins_prob
                )
                return
        while (self.niter < MIN_ITERATION) or (
            (not self.converged) and (self.niter < MAX_ITERATION)
        ):
            self.all_samples_all_gts_log_ll = (
                self.calc_all_reads_log_likelihood_given_allele()
            )
            prev_total_prior_log_ll = self.total_prior_log_ll
            (
                self.all_samples_all_gt_log_posterior,
                self.total_prior_log_ll,
            ) = self.calc_unphased_unordered_gt_posterior()
            total_prior_log_ll_diff = (
                self.total_prior_log_ll - prev_total_prior_log_ll
            )
            converged0 = total_prior_log_ll_diff < MAX_TOLERANCE
            if converged0:
                self.other_params["logger_object"].info(
                    "The total prior log-likelihood of"
                    f" {self.locus_infos['STR_id']} unexpectedly decreased to"
                    " less than 1/10."
                )
                self.converged = "LL_Decreased"
                self.converged_list = [converged0, "unknown", "unknown"]
                break
            (
                prev_inframe_ins_prob,
                prev_inframe_del_prob,
                prev_inframe_single_step_prob,
                prev_outframe_ins_prob,
                prev_outframe_del_prob,
                prev_outframe_single_step_prob,
            ) = (
                self.inframe_ins_prob,
                self.inframe_del_prob,
                self.inframe_single_step_prob,
                self.outframe_ins_prob,
                self.outframe_del_prob,
                self.outframe_single_step_prob,
            )
            if DEBUG:
                self.total_prior_log_ll_list.append(self.total_prior_log_ll)
                print(f"iter:{self.niter}")
                print(self.inframe_ins_prob)
                print(self.inframe_del_prob)
                print(self.inframe_single_step_prob)
                print(self.outframe_ins_prob)
                print(self.outframe_del_prob)
                print(self.outframe_single_step_prob)
                print(self.no_stutter_prob)
                print(self.total_prior_log_ll)
                print(np.exp(self.all_samples_all_gt_log_posterior))
                if UPDATE_POP_LOG_AF:
                    print(self.gts_pop_log_af)
                    print(self.alleles_pop_log_af)
            if UPDATE_POP_LOG_AF:
                (
                    self.gts_pop_log_af,
                    self.alleles_pop_log_af,
                ) = self.update_pop_log_AF()
                # XXX: popAF does not serve as a constraint for judging convergence
                # XXX: HipSTR/MUTEA/ME think so
            (
                self.no_stutter_prob,
                self.inframe_ins_prob,
                self.inframe_del_prob,
                self.inframe_single_step_prob,
                self.outframe_ins_prob,
                self.outframe_del_prob,
                self.outframe_single_step_prob,
            ) = self.update_stutter_model()
            if (
                (self.inframe_ins_prob < 0)
                or (self.inframe_ins_prob > 1)
                or (self.inframe_del_prob < 0)
                or (self.inframe_del_prob > 1)
                or (self.inframe_single_step_prob < 0)
                or (self.inframe_single_step_prob > 1)
                or (self.outframe_ins_prob < 0)
                or (self.outframe_ins_prob > 1)
                or (self.outframe_del_prob < 0)
                or (self.outframe_del_prob > 1)
                or (self.outframe_single_step_prob < 0)
                or (self.outframe_single_step_prob > 1)
                or (self.no_stutter_prob < 0)
                or (self.no_stutter_prob > 1)
            ):
                self.other_params["logger_object"].info(
                    "The stutter model parameters for"
                    f" {self.locus_infos['STR_id']} are out of range and"
                    " will be fixed to the maximum or minimum value."
                )
            self.inframe_stutter_model_dict = self.stutter_model(
                self.inframe_single_step_prob,
                self.inframe_ins_prob,
                self.inframe_del_prob,
                self.max_inframe_stutter_steps,
            )
            self.outframe_stutter_model_dict = self.stutter_model(
                self.outframe_single_step_prob,
                self.outframe_ins_prob,
                self.outframe_del_prob,
                self.max_outframe_stutter_steps,
            )
            if self.niter % CHECK_CONVERGE_PERIOD == 0:
                inframe_ins_diff = abs(
                    self.inframe_ins_prob - prev_inframe_ins_prob
                )
                inframe_del_diff = abs(
                    self.inframe_del_prob - prev_inframe_del_prob
                )
                inframe_single_step_diff = abs(
                    self.inframe_single_step_prob
                    - prev_inframe_single_step_prob
                )
                outframe_ins_diff = abs(
                    self.outframe_ins_prob - prev_outframe_ins_prob
                )
                outframe_del_diff = abs(
                    self.outframe_del_prob - prev_outframe_del_prob
                )
                outframe_single_step_diff = abs(
                    self.outframe_single_step_prob
                    - prev_outframe_single_step_prob
                )
                converged1 = (
                    ((-total_prior_log_ll_diff) / prev_total_prior_log_ll)
                    < DIFFERENCE_CONVERGE
                ) and (total_prior_log_ll_diff < DIFFERENCE_CONVERGE)
                converged2 = (
                    (inframe_del_diff < DIFFERENCE_CONVERGE)
                    and (inframe_ins_diff < DIFFERENCE_CONVERGE)
                    and (inframe_single_step_diff < DIFFERENCE_CONVERGE)
                    and (outframe_del_diff < DIFFERENCE_CONVERGE)
                    and (outframe_ins_diff < DIFFERENCE_CONVERGE)
                    and (outframe_single_step_diff < DIFFERENCE_CONVERGE)
                )
                # BUG: Check converge condition and converged0 should be in the before of stutter params updating and break in time
                # HAS SOLVED
                self.converged = converged1 or converged2
                self.converged_list = [converged0, converged1, converged2]
            self.niter += 1
            self.total_niter += 1
        self.n_train += 1
        self.trained = True
        if self.converged == "LL_Decreased":
            self.other_params["logger_object"].info(
                f"{self.locus_infos['STR_id']}: The total prior log-likelihood"
                " unexpectedly decreased to less than 1/10."
            )
            self.locus_state = "LLDecreased"
        elif (
            self.converged == True
        ):  # BUG numpy.bool_ don't support 'is' operator: HAS SOLVED
            self.other_params["logger_object"].info(
                f"{self.locus_infos['STR_id']} has converged."
            )
            self.locus_state = "HasConverged"
        else:
            self.other_params["logger_object"].info(
                f"{self.locus_infos['STR_id']}: Maximum iteration reached."
            )
            self.locus_state = "MaxIterationReached"

    def stutter_output(self, stutter_model_out_file):
        if (
            self.locus_state == "CoverageFail"
            or self.locus_state == "PassCovSampleNumOrFractionFail"
            or self.locus_state == "NoEnoughSampleNumberFail"
        ):
            pass
        elif self.locus_state == "NoStutterFail":
            self.inframe_del_prob = self.min_stutter
            self.inframe_ins_prob = self.min_stutter
            self.outframe_del_prob = self.min_stutter
            self.outframe_ins_prob = self.min_stutter
        else:
            self.inframe_del_prob = max(
                [self.min_stutter, self.inframe_del_prob]
            )
            self.inframe_del_prob = min(
                [MAX_STUTTER_DEL, self.inframe_del_prob]
            )
            self.inframe_ins_prob = max(
                [self.min_stutter, self.inframe_ins_prob]
            )
            self.inframe_ins_prob = min(
                [MAX_STUTTER_INS, self.inframe_ins_prob]
            )
            self.inframe_single_step_prob = max(
                [MIN_SINGLE_STEP_PROB, self.inframe_single_step_prob]
            )
            self.inframe_single_step_prob = min(
                [MAX_SINGLE_STEP_PROB, self.inframe_single_step_prob]
            )
            self.outframe_del_prob = max(
                [self.min_stutter, self.outframe_del_prob]
            )
            self.outframe_del_prob = min(
                [MAX_STUTTER_DEL, self.outframe_del_prob]
            )
            self.outframe_ins_prob = max(
                [self.min_stutter, self.outframe_ins_prob]
            )
            self.outframe_ins_prob = min(
                [MAX_STUTTER_INS, self.outframe_ins_prob]
            )
            self.outframe_single_step_prob = max(
                [MIN_SINGLE_STEP_PROB, self.outframe_single_step_prob]
            )
            self.outframe_single_step_prob = min(
                [MAX_SINGLE_STEP_PROB, self.outframe_single_step_prob]
            )
        self.no_stutter_prob = (
            1
            - self.inframe_del_prob
            - self.inframe_ins_prob
            - self.outframe_del_prob
            - self.outframe_ins_prob
        )
        # if (self.trained) and (
        #     self.max_inframe_stutter_steps == 0
        # ):  # self.trained = True 代表已经训练过了即 out-frame model 有 stutter
        #     self.inframe_del_prob = MIN_STUTTER_DEL
        #     self.inframe_ins_prob = MIN_STUTTER_INS
        #     self.inframe_single_step_prob = MAX_SINGLE_STEP_PROB
        #     self.no_stutter_prob = (
        #         1
        #         - self.inframe_del_prob
        #         - self.inframe_ins_prob
        #         - self.outframe_del_prob
        #         - self.outframe_ins_prob
        #     )
        # elif (self.trained) and (
        #     self.max_outframe_stutter_steps == 0
        # ):  # self.trained = True 代表已经训练过了即 in-frame model 有 stutter
        #     self.outframe_del_prob = MIN_STUTTER_DEL
        #     self.outframe_ins_prob = MIN_STUTTER_INS
        #     self.outframe_single_step_prob = MAX_SINGLE_STEP_PROB
        #     self.no_stutter_prob = (
        #         1
        #         - self.inframe_del_prob
        #         - self.inframe_ins_prob
        #         - self.outframe_del_prob
        #         - self.outframe_ins_prob
        #     )
        locus_stutter = (
            f"{self.locus_infos['chr']}\t"
            f"{self.locus_infos['str_zero_based_start_included']}\t"
            f"{self.locus_infos['str_zero_based_end_excluded']}\t"
            f"{self.locus_infos['motif_length']}\t"
            f"{self.locus_infos['period']}\t"
            f"{self.locus_infos['STR_id']}\t"
            f"{self.locus_infos['motif']}\t"
            f"{self.locus_infos['ploidy']}\t"
            f"{self.samples_number}\t"
            f"{self.alleles_number}\t"
            f"{self.gt_number}\t"
            f"{self.total_depth}\t"
            f"{self.average_depth}\t"
            f"{self.pass_depth_sample_number}\t"
            f"{self.pass_depth_sample_fraction}\t"
            f"{self.locus_state}\t"
            f"{self.converged}\t"
            f"{self.trained}\t"
            f"{self.n_train}\t"
            f"{self.total_niter}\t"
            f"{self.niter}\t"
            f"{self.no_stutter_prob}\t"
            f"{self.inframe_ins_prob}\t"  # 23
            f"{self.inframe_del_prob}\t"
            f"{self.inframe_single_step_prob}\t"
            f"{self.outframe_ins_prob}\t"
            f"{self.outframe_del_prob}\t"
            f"{self.outframe_single_step_prob}\t"
            f"{self.total_prior_log_ll}\t"
            f"{UPDATE_POP_LOG_AF}\t"
            f"{self.max_inframe_stutter_steps}\t"  # 31
            f"{self.max_outframe_stutter_steps}\t"
            f"{self.coverage_state_list[0]}\t"
            f"{self.coverage_state_list[1]}\t"
            f"{self.coverage_state_list[2]}\t"
            f"{self.coverage_state_list[3]}\t"
            f"{self.converged_list[0]}\t"
            f"{self.converged_list[1]}\t"
            f"{self.converged_list[2]}\t"
            f"{self.other_params['total_unfiltered_depth']}\t"  # XXX: total_unfiltered_depth means the total depth of all samples and we don't filter out any reads for this parameter
            f"{self.other_params['fail_reason_reads_number'].get('unmap_issue',0)}\t"
            f"{self.other_params['fail_reason_reads_number'].get('library_issue',0)}\t"
            f"{self.other_params['fail_reason_reads_number'].get('spanning_issue',0)}\t"
            f"{self.other_params['fail_reason_reads_number'].get('map_issue',0)}\t"
            f"{self.other_params['fail_reason_reads_number'].get('SegmentConditionFail',0)}\t"
            f"{self.other_params['fail_reason_reads_number'].get('ReadN',0)}\t"
            f"{self.other_params['fail_reason_reads_number'].get('SegmentResultFail',0)}\t"
            f"{self.raw_samples_number}\t"
            f"{self.raw_alleles_number}\t"
            f"{self.other_params.get('big_stutter_size_reads_number',0)}\t"
            f"{self.other_params.get('big_stutter_size_alleles_number',0)}\t"
            f"{self.other_params.get('filtered_samples_name_samples_num',0)}\t"
            f"{self.other_params.get('filtered_low_depth_samples_reads_num',0)}\t"
            f"{self.max_raw_stutter_step}\n"
        )
        with open(stutter_model_out_file, "a") as stutter_out:
            if myutils.acquire_lock(stutter_out):
                stutter_out.write(locus_stutter)
                myutils.release_lock(stutter_out)
        if DEBUG:
            print("Locus stutter parameters: ")
            print(locus_stutter)


def main():
    # demo1: haploid and motif_length = 1
    demo1_all_samples_all_alleles_depth_dict = {
        "Sample_A": {21: 20, 22: 2, 20: 2, 19: 1, 23: 1},
        "Sample_B": {21: 20, 22: 1, 20: 2},
        "Sample_C": {21: 10, 20: 9, 22: 2, 23: 1, 19: 1},
    }
    demo1_locus_infos = {
        "chr": "chrY",
        "str_zero_based_start_included": 57070227,
        "str_zero_based_end_excluded": 57070248,
        "motif_length": 1,
        "period": 21,
        "STR_id": "Human_STR_1619105",
        "motif": "T",
        "ploidy": 1,
    }
    demo1_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo1_log.txt"
    }
    demo1_noise_estimation = NoiseEstimator(
        demo1_all_samples_all_alleles_depth_dict,
        demo1_locus_infos,
        demo1_other_params,
    )
    # demo2: haploid and motif_length = 3
    demo2_all_samples_all_alleles_depth_dict = {
        "Sample_A": {18: 1, 22: 1, 17: 12, 12: 1, 16: 1},
        "Sample_B": {18: 2, 22: 1, 17: 14, 12: 1, 16: 2},
        "Sample_C": {18: 1, 22: 12, 17: 12, 12: 1, 27: 1, 16: 1, 21: 1, 23: 1},
    }
    demo2_locus_infos = {
        "chr": "chrY",
        "str_zero_based_start_included": 57067388,
        "str_zero_based_end_excluded": 57067405,
        "motif_length": 3,
        "period": 5.66667,
        "STR_id": "Human_STR_1619104",
        "motif": "ATT",
        "ploidy": 1,
    }
    demo2_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo2_log.txt"
    }
    demo2_noise_estimation = NoiseEstimator(
        demo2_all_samples_all_alleles_depth_dict,
        demo2_locus_infos,
        demo2_other_params,
    )
    # demo3: diploid and motif_length = 1
    demo3_all_samples_all_alleles_depth_dict = {
        "Sample_A": {12: 10, 14: 11, 10: 2, 15: 1},
        "Sample_B": {14: 20, 13: 2, 16: 2},
        "Sample_C": {15: 12, 14: 1, 13: 1},
    }
    demo3_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 28588,
        "str_zero_based_end_excluded": 28603,
        "motif_length": 1,
        "period": 15,
        "STR_id": "Human_STR_4",
        "motif": "T",
        "ploidy": 2,
    }
    demo3_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo3_log.txt"
    }
    demo3_noise_estimation = NoiseEstimator(
        demo3_all_samples_all_alleles_depth_dict,
        demo3_locus_infos,
        demo3_other_params,
    )
    # demo4: diploid and motif_length = 2
    demo4_all_samples_all_alleles_depth_dict = {
        "Sample_A": {12: 10, 14: 11, 10: 2, 15: 1},
        "Sample_B": {14: 20, 13: 2, 16: 2},
        "Sample_C": {15: 12, 14: 1, 13: 1},
    }
    demo4_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo4_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo4_log.txt"
    }
    demo4_noise_estimation = NoiseEstimator(
        demo4_all_samples_all_alleles_depth_dict,
        demo4_locus_infos,
        demo4_other_params,
    )
    # demo5: diploid and motif_length = 2 and include zero depth samples
    demo5_all_samples_all_alleles_depth_dict = {
        "Sample_A": {12: 10, 14: 11, 10: 2, 15: 1},
        "Sample_B": {14: 20, 13: 2, 16: 2},
        "Sample_C": {15: 0, 14: 0, 13: 0},
    }
    demo5_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo5_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo5_log.txt"
    }
    demo5_noise_estimation = NoiseEstimator(
        demo5_all_samples_all_alleles_depth_dict,
        demo5_locus_infos,
        demo5_other_params,
    )
    # demo6: diploid and motif_length = 2 and total depth is zero
    demo6_all_samples_all_alleles_depth_dict = {
        "Sample_A": {},
        "Sample_B": {},
        "Sample_C": {},
    }
    demo6_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo6_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo6_log.txt"
    }
    demo6_noise_estimation = NoiseEstimator(
        demo6_all_samples_all_alleles_depth_dict,
        demo6_locus_infos,
        demo6_other_params,
    )
    # demo7: diploid and motif_length = 2 and only one allele
    demo7_all_samples_all_alleles_depth_dict = {
        "Sample_A": {12: 3},
        "Sample_B": {},
        "Sample_C": {},
    }
    demo7_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo7_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo7_log.txt"
    }
    demo7_noise_estimation = NoiseEstimator(
        demo7_all_samples_all_alleles_depth_dict,
        demo7_locus_infos,
        demo7_other_params,
    )
    # demo8: diploid and motif_length = 2 and two alleles and test AF
    demo8_all_samples_all_alleles_depth_dict = {
        str(i): {12: i, 14: 30 - i} for i in range(31)
    }
    demo8_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo8_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo8_log.txt"
    }
    demo8_noise_estimation = NoiseEstimator(
        demo8_all_samples_all_alleles_depth_dict,
        demo8_locus_infos,
        demo8_other_params,
    )
    # demo9: haploid and motif_length = 2 and two alleles and test AF
    demo9_all_samples_all_alleles_depth_dict = {
        str(i): {12: i, 14: 30 - i} for i in range(31)
    }
    demo9_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 1,
    }
    demo9_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo9_log.txt"
    }
    demo9_noise_estimation = NoiseEstimator(
        demo9_all_samples_all_alleles_depth_dict,
        demo9_locus_infos,
        demo9_other_params,
    )
    # demo10: diploid and motif_length = 1 and two alleles and test == 0.25 stutter errors
    demo10_all_samples_all_alleles_depth_dict = {
        "Sample_A": {12: 10, 14: 10},
        "Sample_B": {14: 20},
    }
    demo10_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 28588,
        "str_zero_based_end_excluded": 28603,
        "motif_length": 1,
        "period": 15,
        "STR_id": "Human_STR_4",
        "motif": "T",
        "ploidy": 2,
    }
    demo10_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo10_log.txt"
    }
    demo10_noise_estimation = NoiseEstimator(
        demo10_all_samples_all_alleles_depth_dict,
        demo10_locus_infos,
        demo10_other_params,
    )
    # demo11: diploid and motif_length = 1 and two alleles and test AF
    demo11_all_samples_all_alleles_depth_dict = {
        str(i): {12: i, 14: 30 - i} for i in range(31)
    }
    demo11_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 1,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo11_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo11_log.txt"
    }
    demo11_noise_estimation = NoiseEstimator(
        demo11_all_samples_all_alleles_depth_dict,
        demo11_locus_infos,
        demo11_other_params,
    )
    # demo12: diploid and motif_length = 2 and three alleles and test AF and imbalance AF and out-frame stutter errors
    demo12_all_samples_all_alleles_depth_dict = {
        str(i): {12: i, 13: 30 - i, 14: 10} for i in range(31)
    }
    demo12_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo12_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo12_log.txt"
    }
    demo12_noise_estimation = NoiseEstimator(
        demo12_all_samples_all_alleles_depth_dict,
        demo12_locus_infos,
        demo12_other_params,
    )
    # demo13: haploid and motif_length = 2 and three alleles and test AF and imbalance AF and out-frame stutter errors
    demo13_all_samples_all_alleles_depth_dict = {
        str(i): {12: i, 13: 30 - i, 14: 10} for i in range(31)
    }
    demo13_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 1,
    }
    demo13_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo13_log.txt"
    }
    demo13_noise_estimation = NoiseEstimator(
        demo13_all_samples_all_alleles_depth_dict,
        demo13_locus_infos,
        demo13_other_params,
    )
    # demo14: diploid and motif_length = 1 and three alleles and test AF and imbalance AF and out-frame stutter errors
    # Motif_length = 1
    demo14_all_samples_all_alleles_depth_dict = {
        str(i): {12: i, 13: 30 - i, 14: 10} for i in range(31)
    }
    demo14_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 1,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo14_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo14_log.txt"
    }
    demo14_noise_estimation = NoiseEstimator(
        demo14_all_samples_all_alleles_depth_dict,
        demo14_locus_infos,
        demo14_other_params,
    )
    # demo15: diploid and motif_length = 1 and three alleles and test AF and imbalance AF and out-frame stutter errors
    # Motif_length = 1 Test step size
    demo15_all_samples_all_alleles_depth_dict = {
        str(i): {11: i, 13: 30 - i, 15: 10} for i in range(31)
    }
    demo15_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 1,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo15_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo15_log.txt"
    }
    demo15_noise_estimation = NoiseEstimator(
        demo15_all_samples_all_alleles_depth_dict,
        demo15_locus_infos,
        demo15_other_params,
    )
    # demo16: Haploid and motif_length = 1 and three alleles and test AF and imbalance AF and out-frame stutter errors
    # Haploid Motif_length = 1 Test step size
    demo16_all_samples_all_alleles_depth_dict = {
        str(i): {11: i, 13: 30 - i, 15: 10} for i in range(31)
    }
    demo16_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 1,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 1,
    }
    demo16_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo16_log.txt"
    }
    demo16_noise_estimation = NoiseEstimator(
        demo16_all_samples_all_alleles_depth_dict,
        demo16_locus_infos,
        demo16_other_params,
    )
    # demo17: diploid and motif_length = 2 and only two alleles and Test empty depth samples
    demo17_all_samples_all_alleles_depth_dict = {
        "Sample_A": {12: 20, 14: 10},
        "Sample_B": {},
        "Sample_C": {},
    }
    demo17_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 26453,
        "str_zero_based_end_excluded": 26465,
        "motif_length": 2,
        "period": 6,
        "STR_id": "Human_STR_3",
        "motif": "GT",
        "ploidy": 2,
    }
    demo17_other_params = {
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/demo17_log.txt"
    }
    demo17_noise_estimation = NoiseEstimator(
        demo17_all_samples_all_alleles_depth_dict,
        demo17_locus_infos,
        demo17_other_params,
    )
    demos_noise_estimation_list = [
        demo1_noise_estimation,
        demo2_noise_estimation,
        demo3_noise_estimation,
        demo4_noise_estimation,
        demo5_noise_estimation,
        demo6_noise_estimation,
        demo7_noise_estimation,
        demo8_noise_estimation,
        demo9_noise_estimation,
        demo10_noise_estimation,
        demo11_noise_estimation,
        demo12_noise_estimation,
        demo13_noise_estimation,
        demo14_noise_estimation,
        demo15_noise_estimation,
        demo16_noise_estimation,
        demo17_noise_estimation,
    ]
    for demo_index, demo in enumerate(demos_noise_estimation_list):
        demo_id = demo_index + 1
        print(demo.other_params["log_file_path"])
        demo.train()
        demo.stutter_output(
            "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/stutter_model.txt"
        )
        print(np.exp(demo.all_samples_all_gt_log_posterior))
        if PLOT:
            import matplotlib.pyplot as plt

            log_ll = demo.total_prior_log_ll_list
            epochs = range(1, len(log_ll) + 1)
            plt.plot(epochs, log_ll, "b", label="Log-Likelihood")
            plt.title("Convergence Curve")
            plt.xlabel("Epochs")
            plt.ylabel("Log-Likelihood")
            plt.legend()
            plt.savefig(
                f"/Users/lid/Github/MosaicSTR/Results/stutter_sim_test/Convergence_Curve_demo{demo_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()
            if demo_id > 7 and demo_id != 10 and demo_id != 17:
                x = np.arange(31) / demo.average_depth
                if demo.locus_infos["ploidy"] == 1:
                    if demo.alleles_number == 2:
                        allele1 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 0]
                        )
                        allele2 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 1]
                        )
                        plt.plot(x, allele1, label="Alternative Haploid")
                        plt.plot(x, allele2, label="Reference Haploid")
                        plt.scatter(x, allele1, c="blue", marker="o")
                        plt.scatter(x, allele2, c="orange", marker="s")
                    elif demo.alleles_number == 3:
                        GT1 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 0]
                        )
                        GT2 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 1]
                        )
                        GT3 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 2]
                        )
                        plt.plot(x, GT1, label="GT1: 0")
                        plt.plot(x, GT2, label="GT2: 1")
                        plt.plot(x, GT3, label="GT3: 2")
                        plt.scatter(x, GT1, c="blue", marker="o")
                        plt.scatter(x, GT2, c="orange", marker="s")
                        plt.scatter(x, GT3, c="green", marker="^")
                    else:
                        raise ValueError(
                            "The number of alleles is not 2 or 3."
                        )
                elif demo.locus_infos["ploidy"] == 2:
                    if demo.alleles_number == 2:
                        alt_hom = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 0]
                        )
                        hom = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 2]
                        )
                        het = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 1]
                        )
                        plt.plot(x, hom, label="Reference Homozygous")
                        plt.plot(x, het, label="Heterozygous")
                        plt.plot(x, alt_hom, label="Alternative Homozygous")
                        plt.scatter(x, hom, c="blue", marker="o")
                        plt.scatter(x, het, c="orange", marker="s")
                        plt.scatter(x, alt_hom, c="green", marker="^")
                    elif demo.alleles_number == 3:
                        GT1 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 0]
                        )
                        GT2 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 1]
                        )
                        GT3 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 2]
                        )
                        GT4 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 3]
                        )
                        GT5 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 4]
                        )
                        GT6 = np.exp(
                            demo.all_samples_all_gt_log_posterior[:, 5]
                        )
                        plt.plot(x, GT1, label="GT1: 0/0")
                        plt.plot(x, GT2, label="GT2: 0/1")
                        plt.plot(x, GT3, label="GT3: 0/2")
                        plt.plot(x, GT4, label="GT4: 1/1")
                        plt.plot(x, GT5, label="GT5: 1/2")
                        plt.plot(x, GT6, label="GT6: 2/2")
                        plt.scatter(x, GT1, c="blue", marker="o")
                        plt.scatter(x, GT2, c="orange", marker="s")
                        plt.scatter(x, GT3, c="green", marker="^")
                        plt.scatter(x, GT4, c="red", marker="*")
                        plt.scatter(x, GT5, c="purple", marker=">")
                        plt.scatter(x, GT6, c="brown", marker="<")
                    else:
                        raise ValueError(
                            "The number of alleles is not 2 or 3."
                        )
                else:
                    raise ValueError("The ploidy is not 1 or 2.")
                plt.legend()
                plt.savefig(
                    f"/Users/lid/Github/MosaicSTR/Results/stutter_sim_test/Simulation_AF_LL_demo{demo_id}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.show()
    # XXX: Test Results:
    # demo13 看出 alleles reads number 比较多的并不一定是真实的 haploid
    # 如果 stutter error 很多，ins or del rate 很高的话，更小 reads number 的 allele 也可能是其真实的 haploid
    # 因为 stutter errors 比 no stutter errors 概率可能更高
    # 观察到的 errors 比预计的多的原因：
    # （1）somatic mutations （2）system errors including mapping issues
    # XXX: 当迭代出较高的 stutter errors parameters 时需要注意一下,可能所有的样本都是 mosaic 样本,
    # XXX: 也可能是系统性 recurrent 错误比如 mapping 问题或者系统性的扩增错误或者测序错误
    # Test params difference when update popAF
    # Test1
    # "bool_params": {
    #     "UPDATE_POP_LOG_AF": False},
    # "other_params": {
    #     "ORDERED_REPLACEMENT_GT": False,
    #     "HARDYWEINBERG_AF": False
    # Test2
    # "bool_params": {
    #     "UPDATE_POP_LOG_AF": False},
    # "other_params": {
    #     "ORDERED_REPLACEMENT_GT": True,
    #     "HARDYWEINBERG_AF": False
    # Test3
    # "bool_params": {
    #     "UPDATE_POP_LOG_AF": True},
    # "other_params": {
    #     "ORDERED_REPLACEMENT_GT": False,
    #     "HARDYWEINBERG_AF": False
    # Test4
    # "bool_params": {
    #     "UPDATE_POP_LOG_AF": True},
    # "other_params": {
    #     "ORDERED_REPLACEMENT_GT": True,
    #     "HARDYWEINBERG_AF": False
    # Test5
    # "bool_params": {
    #     "UPDATE_POP_LOG_AF": True},
    # "other_params": {
    #     "ORDERED_REPLACEMENT_GT": False,
    #     "HARDYWEINBERG_AF": True
    # Test6
    # "bool_params": {
    #     "UPDATE_POP_LOG_AF": True},
    # "other_params": {
    #     "ORDERED_REPLACEMENT_GT": True,
    #     "HARDYWEINBERG_AF": True
    # XXX: Test Results
    # Test1 几乎等于 Test2
    # Test3 几乎等于 Test2，除了 demo2 差别较大 以及 demo 12 和 13 的 out-frame
    # demo2 in-out frame ins 都提高了，因为 allele length 为 17 的 allele 的 popAF 在迭代中增加了，符合预期
    # Test4 几乎等于 Test3
    # Test5 几乎等于 Test4 除了 demo8 和 11 和 14 的 inframe
    # 使用 Hardy-Weinberg AF 使杂合的概率减少了，杂合的先验减少，stutter errors 增多
    # 0/0:0.1;0/1:0.8;1/1:0.1,则 0: 0.5;1: 0.5，使用 HWE 后 0/0:0.25;0/1:0.5;1/1:0.25
    # 可见纯合的概率减少了，杂合的概率增加了，则 stutter errors 会增多，符合预期
    # Test6 几乎等于 Test5
    # XXX: Conclusions:
    # 有序和无序的GT对参数迭代差别不大，
    # 使用 HARDYWEINBERG_AF 结果轻微不同
    # 迭代不迭代 popAF 结果也稍微不同


## DEBUG 在群体数据:
# 0. 每次 Debug 需要一份程序的 debug 代码和一份打包的代码，这样方便后续 debug 和 单元测试，尽量保持已有目录的文件来避免下次单元测试时仍可以正常进行
# 1. 找出 hipstr 和 我的数据中差异较大的条目，去分割获得样本数据
# 2. 检测通过测试脚本检测迭代的过程是否存在问题
# 3. 检测 IGV 来判断分割及 stutter 是否存在问题
# 4. 我们的假设是 no stutter 的概率是最大的，假如 stutter ins 或者 del 的概率更大，则所看非所真，这样的位点我们没办法从肉眼上观测是否为真位点，因此很大的可能性是一个假位点
# (最大期望和观察一致，最大期望和观察不一致，这两种情况需要分别进行对待，小心进行处理)
# 5. 比较我的代码和 hipstr/mutea 和 wenxuan 代码的异同，找出问题所在
# 6. SVision and SVision-pro 代码和文章进行学习
# 7. 代码 公式 测试：模拟测试和真实数据测试（结果）进行整理 ppt 和展示
# python3 stutter_model_estimation.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_data.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l stutter_log -lf -q stutter_result -c 17 -s 9769 -e 9786
# bug1: 文件名传递错误 bam_name 传递错误，导致最终只有一个样本被 stutter estimation
# 更新 af 和不更新 af 的结果相差不多，所以 af 的更新不会很大的影响
# 观察下分割结果和 IGV 结果的一致性，以及与 HipSTR 不一致的原因查看
# 观察下分割结果与 stutter 计算时的一致性
# 为啥有时候 -4 有时候 -6，为啥呢 ？
# python3 stutter_model_estimation.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_male_hasHipSTR_data.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l stutter_log -lf -q stutter_result -c 17 -s 9769 -e 9786
# 和 HipSTR 同样的数据，衡量出来的 stutter model 不一致
# HipSTR
# IGEOM	IDOWN	IUP	OGEOM	ODOWN	OUP
# 0.671077
# 0.015211100
# 0.017321000
# 0.953461
# 7.71357e-04
# 7.71357e-04
# # my
# f"{self.no_stutter_prob}\t"
# f"{self.inframe_ins_prob}\t" # 22
# f"{self.inframe_del_prob}\t"
# f"{self.inframe_single_step_prob}\t"
# f"{self.outframe_ins_prob}\t"
# f"{self.outframe_del_prob}\t"
# f"{self.outframe_single_step_prob}\t"
# 0.9029072853733151      0.08646429160774448  0.010626423018940337     0.6907307544591069      1e-06   1e-06   0.999
# {'SRR19794513': {17: 12, 19: 1, 12: 1, 16: 4, 18: 5}, 'SRR19794514': {17: 16, 16: 5, 18: 5, 19: 1}, 'SRR19794515': {17: 31, 19: 3, 18: 5, 16: 2, 15: 3}, 'SRR6349057': {17: 21, 18: 3}, 'SRR6349062': {17: 29, 18: 2, 19: 1}, 'SRR6349065': {16: 1, 17: 1}, 'SRR13989893': {17: 132, 19: 3, 18: 15, 16: 2}, 'SRR13989894': {17: 106, 19: 3, 18: 8, 26: 1}, 'SRR13989895': {17: 146, 16: 2, 18: 3}, 'SRR13989896': {17: 149, 18: 11, 19: 1}, 'SRR13989897': {17: 155, 16: 2, 21: 1, 18: 7, 19: 1, 20: 1}, 'SRR6349120': {17: 25, 18: 2}, 'SRR6349131': {16: 2, 17: 4, 19: 1}, 'SRR6349136': {17: 20, 19: 1, 18: 1}, 'SRR6349167': {17: 40, 19: 1, 18: 2}, 'SRR6349193': {17: 26, 23: 1}, 'SRR19794567': {17: 17, 18: 1}, 'SRR6349203': {18: 2, 17: 25}, 'SRR6349227': {17: 42, 19: 1, 18: 1}, 'SRR19794568': {17: 16}, 'SRR6349220': {17: 22, 18: 2, 23: 1}}
# IGV 检查，问题来源：分割问题对于 stutter 的判断存在模棱两可，如果要用 hmm 分割，那么前后应该一致，不然前面算出来的 stutter 比较小，后面又有很多 stutter 就容易把纯合 calling 成杂和
# 我的 stutter 高估的原因总结：一，分割模棱两可问题，二，样本名低级错误 bug，其他问题，没有了哈
# 分割一致，stutter 使用一致的话问题不会太大，但是如果分割不一致，stutter 使用一致的话问题就会很大
# 另外分割模棱两可，如果 flanking 错位比对，则造成非常小的错误率，本来 stutter 估计偏小对于坐标分割，然后你再去让flanking去错位比对，那么这个错误率就会更小
# 稍微有一两条 reads 就会导致 GT 变成杂和，导致 Gi 变成杂和，有可能降低 mosaic 检出灵敏度，也有可能产生非常多样的 mosaic calling，不同细胞容易产生相对不同的 mosaic GT
# 所以最优解决方案是，不用 flanking，前后都用 hmm，然后 sequence-based model 使用最大似然的比对去计算比对碱基的 base quality 比对概率
# 设置最小边界根据 HipSTR 的默认值和估计结果，
# rho in out max 0.999 min 0.03 up max 0.62 min 0.00009
# Other Issues Loci
# 2
# 118546013
# 118546027
# 3
# 6
# Human_STR_766223
# ATT
# #
# 7
# 67029896
# 67029913
# 6
# 3.66667
# Human_STR_1331214
# AAAAAT
# python3 stutter_model_estimation.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_data.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l stutter_log2 -lf -q stutter_result2 -c 2 -s 118546013 -e 118546027
# python3 stutter_model_estimation.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_data.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l stutter_log3 -lf -q stutter_result3 -c 7 -s 67029896 -e 67029913
# Panel 使用的坐标更改过了嘛？为啥会有这个问题呢？
# python3 stutter_model_estimation.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_data_two_samples.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l stutter_log4 -lf -q stutter_result4 -c 7 -s 67029896 -e 67029913 > ../debug/baseq_chr7.txt
# python3 stutter_model_estimation.py -i /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/1485_v3_bulk_data_two_samples.csv -r /storage/douyanmeiLab/wangweixiang/data/GATK_b37_bundles/bundle/b37/bundle/b37/human_g1k_v37_decoy.fasta -b /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/metadata/human_g1k_v37_decoy.hipstr_reference_startminusone_zerobased_leftcloserightopen_5bp_hmm_revised_sorted.bed.gz -p /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/stutter_model/debug -l stutter_log5 -lf -q stutter_result5 -c 2 -s 118546013 -e 118546027 > ../debug/baseq_chr2.txt


def main2():
    demo1_all_samples_all_alleles_depth_dict = {
        "Sample_A": {18: 28, 25: 2},
        "Sample_B": {18: 29, 23: 1},
        "Sample_C": {18: 27, 25: 2},
    }
    demo1_locus_infos = {
        "chr": "chr1",
        "str_zero_based_start_included": 28588,
        "str_zero_based_end_excluded": 28606,
        "motif_length": 6,
        "period": 3,
        "STR_id": "Human_STR_4",
        "motif": "TTTTTT",
        "ploidy": 2,
    }
    logger_stutter = logging.getLogger("StutterEstimator")
    demo1_other_params = {
        "logger_object": logger_stutter,
        "total_unfiltered_depth": 90,
        "fail_reason_reads_number": {},
        "log_file_path": "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/new_rho_test_log2.txt",
    }
    demo1_noise_estimation = NoiseEstimator(
        demo1_all_samples_all_alleles_depth_dict,
        demo1_locus_infos,
        demo1_other_params,
    )
    demos_noise_estimation_list = [
        demo1_noise_estimation,
    ]
    for demo_index, demo in enumerate(demos_noise_estimation_list):
        demo_id = demo_index + 1
        print(demo.other_params["log_file_path"])
        demo.train()
        demo.stutter_output(
            "/Users/lid/Github/MosaicSTR/test/test_noise_estimate_py/new_rho_test_stutter_model.txt"
        )
        print(np.exp(demo.all_samples_all_gt_log_posterior))
        if PLOT:
            import matplotlib.pyplot as plt

            log_ll = demo.total_prior_log_ll_list
            epochs = range(1, len(log_ll) + 1)
            plt.plot(epochs, log_ll, "b", label="Log-Likelihood")
            plt.title("Convergence Curve")
            plt.xlabel("Epochs")
            plt.ylabel("Log-Likelihood")
            plt.legend()
            plt.savefig(
                f"/Users/lid/Github/MosaicSTR/Results/stutter_sim_test/new_rho_test_Convergence_Curve_demo{demo_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()


# if DEBUG:
#     MAX_ITERATION = 50
#     MIN_ITERATION = 10
#     if __name__ == "__main__":
#         print("Start the debugging of noise_estimate2.py", flush=True)
#         # main()
#         main2()
