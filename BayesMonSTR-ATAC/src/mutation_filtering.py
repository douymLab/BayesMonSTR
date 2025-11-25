from scipy.stats import chi2, ks_2samp, binomtest
import numpy as np
from scipy.special import logsumexp

# import os
DEBUG = False
PLOT = False
import config_params

# if DEBUG:
#     import sys

#     sys.path.append(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     )
#     from configs import config_params
# else:
#     from ..configs import config_params

MAX_ALLOWABLE_ALLELE_NUM = config_params.MOSAIC_FRACTION_ESTIMATION_PARAMS[
    "max_allowable_allele_num"
]  # 5
MIN_PHASING_READS_DEPTH = config_params.PHASING_PARAMS[
    "min_phasing_reads_depth"
]  # 5
HETEROZYGOSITY_PRIOR = config_params.PRIOR[
    "heterozygosity_prior"
]  # 0.25,  # For snp, the value is 10^(-3)
SNP_SEQUENCING_ERROR_RATE_PRIOR = config_params.PRIOR[
    "snp_sequencing_error_rate_prior"
]  # 10**-3,  # Q30 samples


def ht_log_likelihood_ratio_test(
    bigger_mle_unconstraint_alter_lnlik,
    smaller_mle_constraint_null_lnlik,
    params_number_freedom,
):
    #  the non-log likelihood ratio is bounded between zero and one(smaller/bigger).
    LR = -2 * (
        smaller_mle_constraint_null_lnlik - bigger_mle_unconstraint_alter_lnlik
    )
    if LR <= 0:
        pvalue = 1.0
    else:
        pvalue = chi2.sf(LR, params_number_freedom)
    return LR, pvalue


def ht_log_posterior_odds_test(
    bigger_mle_unconstraint_alter_lnpost,
    smaller_mle_constraint_null_lnpost,
    params_number_freedom,
):
    #  the non-log likelihood ratio is bounded between zero and one(smaller/bigger).
    posteriorodds = -2 * (
        smaller_mle_constraint_null_lnpost
        - bigger_mle_unconstraint_alter_lnpost
    )
    if posteriorodds <= 0:
        pvalue = 1.0
    else:
        pvalue = chi2.sf(posteriorodds, params_number_freedom)
    return posteriorodds, pvalue


def cal_relative_likelihood_ratio(
    bigger_mle_unconstraint_alter_lnlik, smaller_mle_constraint_null_lnlik
):
    relative_likelihood_ratio = np.exp(
        bigger_mle_unconstraint_alter_lnlik - smaller_mle_constraint_null_lnlik
    )
    return relative_likelihood_ratio


def cal_relative_posterior_odds(
    bigger_mle_unconstraint_alter_lnpost, smaller_mle_constraint_null_lnpost
):
    relative_posterior_ratio = np.exp(
        bigger_mle_unconstraint_alter_lnpost
        - smaller_mle_constraint_null_lnpost
    )
    return relative_posterior_ratio


def cal_AIC_loglik(loglikelihood, params_num):
    AIC = 2 * params_num - 2 * loglikelihood
    return AIC


def cal_AIC_logpost(logpost, params_num):
    AIC = 2 * params_num - 2 * logpost
    return AIC


def AIC_posterior_odds(
    model_1_params_num, model_2_params_num, model_1_lnpost, model_2_lnpost
):
    AIC_1 = cal_AIC_loglik(model_1_lnpost, model_1_params_num)
    AIC_2 = cal_AIC_loglik(model_2_lnpost, model_2_params_num)
    AIC_odds = AIC_1 - AIC_2
    return AIC_odds


def AIC_lik_ratio(
    model_1_params_num, model_2_params_num, model_1_lnlik, model_2_lnik
):
    AIC_1 = cal_AIC_loglik(model_1_lnlik, model_1_params_num)
    AIC_2 = cal_AIC_loglik(model_2_lnik, model_2_params_num)
    AIC_ratio = AIC_1 - AIC_2
    return AIC_ratio


def KS_test_for_different_length_mosaic_allele_with_exp_germ(
    germline_stutter_model_len_dis_array,
    candidate_mosaic_observed_len_dis_array,
):
    # KS test
    [ks_vla, ks_p] = ks_2samp(
        germline_stutter_model_len_dis_array,
        candidate_mosaic_observed_len_dis_array,
        alternative="two-sided",
        method="auto",
        axis=0,
        nan_policy="omit",
        keepdims=False,
    )
    return ks_vla, ks_p


def KS_test_for_different_length_mosaic_allele_with_exp_mosaic(
    mosaic_stutter_model_len_dis_array, candidate_mosaic_observed_len_dis_array
):
    # KS test
    [ks_vla, ks_p] = ks_2samp(
        mosaic_stutter_model_len_dis_array,
        candidate_mosaic_observed_len_dis_array,
        alternative="two-sided",
        method="auto",
        axis=0,
        nan_policy="omit",
        keepdims=False,
    )
    return ks_vla, ks_p


def binominal_test_observed_mosaic_allele_over_stutter(
    total_depth,
    sum_stutter_rate_from_either_germ_allele,
    observed_mosaic_allele_num,
):
    p_value = binomtest(
        observed_mosaic_allele_num,
        n=total_depth,
        p=sum_stutter_rate_from_either_germ_allele,
        alternative="greater",
    ).pvalue
    return p_value


def binominal_test_expected_mosaic_allele_over_stutter(
    total_depth,
    sum_stutter_rate_from_either_germ_allele,
    expected_mosaic_allele_num,
):
    p_value = binomtest(
        expected_mosaic_allele_num,
        n=total_depth,
        p=sum_stutter_rate_from_either_germ_allele,
        alternative="greater",
    ).pvalue
    return p_value


def cal_p1_posterior_for_phasing_confidence(
    p1_log_lik, p2_log_lik, total_phasing_depth
):
    if total_phasing_depth >= MIN_PHASING_READS_DEPTH:
        p1_log_posterior = p1_log_lik - logsumexp([p1_log_lik, p2_log_lik])
    else:
        p1_log_posterior = 0.5
    return p1_log_posterior


def cal_p1_fraction_for_phasing_confidence(p1_phasing_depth, p2_phasing_depth):
    total_phasing_depth = p1_phasing_depth + p2_phasing_depth
    if total_phasing_depth >= MIN_PHASING_READS_DEPTH:
        p1_fraction = p1_phasing_depth / (p1_phasing_depth + p2_phasing_depth)
    else:
        p1_fraction = 0.5
    return p1_fraction


def cal_stutter_errors_discordant_rate_LRT_for_phasing_confidence(
    # XXX: Need Thinks
    alter_mapping_errors_lnlik_or_posterior,
    null_stutter_errors_lnlik_or_posterior,
    params_number_freedom,
):
    LR = -2 * (
        null_stutter_errors_lnlik_or_posterior
        - alter_mapping_errors_lnlik_or_posterior
    )
    if LR <= 0:
        pvalue = 1.0
    else:
        pvalue = chi2.sf(LR, params_number_freedom)
    return LR, pvalue


def binominal_test_less_observed_concordant_reads_for_phasing_confidence(
    concordant_rate, total_depth, observed_concordant_reads
):
    # XXX: lik(posterior)-based counts, Max-Lik based counts and observed counts
    # XXX: 假设观察到的 phased hap 即是 MLE 的 phased hap 而不考虑观察到的 reads 的测序base-calling及建库pcr stutter 等测序的错误率
    p_value = binomtest(
        observed_concordant_reads,
        n=total_depth,
        p=concordant_rate,
        alternative="less",
    ).pvalue
    return p_value


def binominal_test_less_mle_concordant_reads_for_phasing_confidence(
    concordant_rate, total_depth, mle_concordant_reads
):
    # XXX: lik(posterior)-based counts, Max-Lik based counts and observed counts
    p_value = binomtest(
        mle_concordant_reads,
        n=total_depth,
        p=concordant_rate,
        alternative="less",
    ).pvalue
    return p_value


def binominal_test_less_posterior_concordant_reads_for_phasing_confidence(
    concordant_rate, total_depth, posterior_concordant_reads
):
    # XXX: lik(posterior)-based counts, Max-Lik based counts and observed counts
    p_value = binomtest(
        posterior_concordant_reads,
        n=total_depth,
        p=concordant_rate,
        alternative="less",
    ).pvalue
    return p_value


def allele_filter(
    total_depth,
    alleles_order,
    alleles_depth_list,
    inframe_del_prob,
    outframe_del_prob,
    inframe_ins_prob,
    outframe_ins_prob,
):
    # binominal test to filter noise
    binominal_filter_number = 0
    error_rate = (
        inframe_del_prob
        + outframe_del_prob
        + inframe_ins_prob
        + outframe_ins_prob
    )
    for allele_depth in alleles_depth_list:
        pvalue = binomtest(
            allele_depth,
            n=total_depth,
            p=error_rate,
            alternative="less",
        ).pvalue
        if pvalue > 0.05:
            break
        else:
            binominal_filter_number += 1
    # filter alleles to max allowable allele number
    # But keep all allele reads
    filtered_alleles_used_for_gt = alleles_order[binominal_filter_number:]
    if len(filtered_alleles_used_for_gt) > MAX_ALLOWABLE_ALLELE_NUM:
        alleles_used_for_gt = filtered_alleles_used_for_gt[
            len(filtered_alleles_used_for_gt) - MAX_ALLOWABLE_ALLELE_NUM :
        ]
    else:
        alleles_used_for_gt = filtered_alleles_used_for_gt
    return alleles_used_for_gt


def mapQ_filter(low_mapQ_reads_fraction, threshold):
    if low_mapQ_reads_fraction <= threshold:
        return True
    else:
        return False


def background_stutter_loci_filter(stutter_error_rate_sum, threshold):
    if stutter_error_rate_sum <= threshold:
        return True
    else:
        return False
