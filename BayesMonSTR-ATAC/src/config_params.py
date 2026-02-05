import numpy as np

CONSTANT = {"LOG_ONE_HALF": np.log(0.5), "LOG_TWO": np.log(2)}

EM_PARAMS = {
    "loci_filter": {
        "MIN_SAMPLES_PASS_COVERAGE_NUM": 5,  # 100,
        "MIN_SAMPLES_PASS_COVERAGE_FRACTION": 0.1,
        "MIN_AVERAGE_COVERAGE": 5,
        "MIN_COVERAGE": 100,  # 500,
        "SAMPLE_COVERAGE_CUTOFF": 5,
        "SAMPLE_NUMBER_CUTOFF": 1,
        "FILTER_LOW_COVERAGE_SAMPLE": False,
        "FILTER_OUT_SAMPLE_COVERAGE_CUTOFF": 5,
        "FILTER_BIG_STUTTER_SIZE": False,
        "BIG_STUTTER_SIZE": {1: 20, 2: 10, 3: 10, 4: 9, 5: 8, 6: 7},
        "BIG_STUTTER_DEPTH_CUTOFF": 1,
    },
    "iteration_params": {
        "MAX_ITERATION": 500,
        "MIN_ITERATION": 20,
        "CHECK_CONVERGE_PERIOD": 2,
        "DIFFERENCE_CONVERGE": 0.0001,
        "MAX_TOLERANCE": np.log(1 / 10),
    },
    "init_params": {
        "INIT_PRIOR_LOG_LL": -100000000.0,
        "MIN_STUTTER_INS": 0.000001,
        "MIN_STUTTER_DEL": 0.000001,
        "MAX_SINGLE_STEP_PROB": 0.999,
        "MAX_STUTTER_INS": 0.9,
        "MAX_STUTTER_DEL": 0.9,
        "MIN_SINGLE_STEP_PROB": 0.001,
        "MIN_LOG_ZERO_PROB": -100,  # 没用到
        "MIN_GEOM_PROBS_SCALE": np.log(1e-6),  # 没用到
        "MIN_NO_STUTTER_RATE": 0.0001,  # XXX: for 388 estimate_noise: RuntimeWarning: divide by zero encountered in log
        "NORMALIZATION_FACTOR_FOR_POP_AF": 10000,  # XXX: for 567 570 estimate_noise: RuntimeWarning: divide by zero encountered in log
    },
    "bool_params": {"UPDATE_POP_LOG_AF": True, "ADD_PSEUDO_COUNT": True},
    "other_params": {
        "ORDERED_REPLACEMENT_GT": True,
        "HARDYWEINBERG_AF": True,
    },
}
# XXX HipSTR TUTORIAL -> For each locus, use a stutter model with PGEOM=0.9 and UP=DOWN=0.05 for in-frame artifacts and PGEOM=0.9 and UP=DOWN=0.01 for out-of-frame artifacts
# XXX CODE -> stutter_model_ = new StutterModel(0.9, 0.1, 0.1, 0.8, 0.01, 0.01, motif_len_)
INIT_STUTTER_MODEL = {
    "inframe_single_step_prob": 0.9,
    "inframe_ins_prob": 0.05,
    "inframe_del_prob": 0.05,
    "outframe_single_step_prob": 0.9,
    "outframe_ins_prob": 0.01,
    "outframe_del_prob": 0.01,
    "homopolymer_outframe_ins_prob": 0.000001,
    "homopolymer_outframe_del_prob": 0.000001,
}

# HipSTR
# For each locus, use a stutter model with PGEOM=0.9 and UP=DOWN=0.05 for in-frame artifacts and PGEOM=0.9 and UP=DOWN=0.01 for out-of-frame artifacts
# https://hipstr-tool.github.io/HipSTR/

DEFAULT_STUTTER_MODEL = INIT_STUTTER_MODEL

ZERO_STUTTER_ENOUGH_DEPTH_STUTTER_MODEL = {
    "inframe_single_step_prob": 0.999,
    "inframe_ins_prob": 0.000001,
    "inframe_del_prob": 0.000001,
    "outframe_single_step_prob": 0.999,
    "outframe_ins_prob": 0.000001,
    "outframe_del_prob": 0.000001,
    "homopolymer_outframe_ins_prob": 0.000001,
    "homopolymer_outframe_del_prob": 0.000001,
}

MOSAIC_FRACTION_ESTIMATION_PARAMS = {
    "MAX_ITERATION": 500,
    "MIN_ITERATION": 10,
    "ms_mutation_rate": 10**-4,  # np.log(1/2)
    "non_mutation_rate": 1 - 10**-4,  # np.log(1/2)
    "initial_mosaic_fraction": 0.1,  # 0.1,  # XXX: or 0.01
    "use_minor_cells_pop_as_mut_cells": False,
    "allele_filter": True,
    "allele_order": "fixed",
    "max_allowable_allele_num": 8,
    "allowable_het2hom": True,
    "iteration_mode": "cell-based",
    "init_total_mosaic_log_prior_lik": -100000000.0,
    "likelihood_mode": "seq-based",  # "length-based",
    # "STR_specific_alignment_probs": "MAX-LIK",
    "ADD_FLANKING_PROB": False,
    # "FLANKING_ALLELE_LENGTH": 5,
    "het2hom_set_zero_mut_reads": False,
    "mf_more_than_halt_set_zero": False,
    "max_lik_assignment": False,
    "correct_yanmei": True,
    "initial_haploid_mosaic_fraction": 10**-4,
    "germ_2_germ_mutation_rate": 10**-4,  # -np.inf,  # XXX np.log(1 - 10**-4)  结果不对
    "allowable_two_mut_alleles": False,
    "hom2het_and_het2het_two_mosaic_fraction_mode": False,
    "het2het_two_alleles_mosaic_fraction": 0,
    "one_allele_mosaic_fraction": 0,
    "all_gt_transformation_mosaic_fraction": True,
    "mle_optimize_method": "SLSQP",  # L-BFGS-B
    "posterior_level": "orderedGT",
    "allmle_or_unique_mosaic_fraction": "unique_mosaic_fraction",
    "mosaic_fraction_bounds": (0, 1),
    "posterior_method": "Gi-free",  # "Likelihood","Gi-free","Gi-Given"
    "posterior_germ_included": False,
    "posterior_mut_type": "1mut",  # "all",
    "posterior_allow_mosaic_fraction": "all",
    "estimation_mode": "unique",
    "numeric_calculation_tolerance": 1e-3,
    "phase_mode": True,
    "min_phasing_average_depth": 1,
    "phasing_allele_filter": True,
    "log_lik_and_log_posterior_meantime": True,
    "only_use_mle_mosaic_fraction": False,
    "ms_mutation_rate_dict": {
        1: 10**-4,
        2: 10**-4,
        3: 10**-4,
        4: 10**-4,
        5: 10**-4,
        6: 10**-4,
    },  # XXX: different ms mutation rate prior, using  constant variable to reduce calculations later
    "allele_balance": 1 / 2,
    "binominal_filter": False,
    "allele_sort": True,  # False,
    "use_gi_prior_update_mosaic_fraction": "OnlyLik",  # GI, MutRate, OnlyLik, GIandMutRate
    "uniform_mosaic_fraction_prior": False,
    "consider_germ2germ_in_params_update": False,
    "consider_germ2germ_in_posterior": True,
    "unorder_mosaic_gt": True,
    "initial_guess_number": 50,
    "use_initial_guess": True,
    "FILTER_BIG_STUTTER_SIZE": False,
    "BIG_STUTTER_SIZE": {1: 20, 2: 10, 3: 10, 4: 9, 5: 8, 6: 7},
    "BIG_STUTTER_DEPTH_CUTOFF": 1,
    "double_het_comba_2x_prob": False,  # TODO: I need calculation heteryzygous with a 2x lik than hom 1x lik ???
}

PRIOR = {
    "heterozygosity_prior": 0.25,  # For snp, the value is 10^(-3)
    "snp_sequencing_error_rate_prior": 10**-3,  # Q30 samples
}

PHASING_PARAMS = {
    "min_phasing_reads_depth": 5,
    "SNP_SPANNING_RANGE": 1000,
    "EXCLUDED_SNP_FLANKING_LENGTH": 15,  # just like HipSTR
    "PHASING_STRATEDY": "READ_NAME",
    "hSNP_VAF_MIN": 0.4,  # 0.2
    "hSNP_VAF_MAX": 0.6,  # 0.8
    "hSNP_MIN_DEPTH": 10,  # 20,
    "AVOID_TERMINAL_SNPs_FOR_PADDING_LENGTH": 5,
    "SPANNING_hSNP_VAF_MIN": 0.2,  # 0.4
    "SPANNING_hSNP_VAF_MAX": 0.8,  # 0.6
    "SPANNING_DEPTH_CUTOFF": 10,  # 5
    "AF_DIFFERENCE_CUTOFF": 10,
    "COUNT_HAP_NUM_DEPTH_CUTOFF": 1,
    "LRT_FREEDOM": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2},  # {1: 2, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5},
    "PHASE_POSTERIOR_CUTOFF": 0.9,
    "MIN_PER_READ_LIKELIHOOD": 1e-10,
    "ANYWAY_LOOK_NSNP": False,
    "COUNT_MLE_HAP_NUM_DEPTH_CUTOFF": 1,
}

ALLELE_EXTRACT = {
    "MAPQ_CUTOFF": 20,
    "BASEQ_CUTOFF": 20,
    "MISMATCHES_CUTOFF": 15,
    "READS_FLANKING_LENGTH_CUTOFF": 5,  # 10,  # 15, # revise_padding
    "READS_SOFT_CLIPPING_CUTOFF": 100000000,
    "READS_HARD_CLIPPING_CUTOFF": 100000000,
    "USE_SEQUENCING_LENGTH": False,
    "FLANKING_ALLELE_LENGTH": 5,  # 没用到
    "SPANNING_DEPTH_CUTOFF": 5,  # 没用到
    "AF_DIFFERENCE_CUTOFF": 10,  # 没用到
    "FLANKING_INDEL_CHECK_RANGE": 15,
    "SegMethods": "Coordinates",
    "PCRLibrary": "PCR",  # PCR
    "PADDING_BPS": 5,  # revise_padding
    "USE_MIS_FRAC": True,
    "MIS_FRAC_CUTOFF": 0.1,
}

COORDINATE_SEG = {
    "GLOBAL_FOR_STR_AND_FLANKING": True,
    "PADDING_BPS": 5,  # revise_padding 10
    "DYNAMIC_PADDING": False,
    "DYNAMIC_PADDING_DICT": {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
    },
}

X_PAR1_start_zero_based_37_19 = 60000
X_PAR1_end_zero_based_37_19 = 2699520
X_PAR2_start_zero_based_37_19 = 154931043
X_PAR2_end_zero_based_37_19 = 155260560
Y_PAR1_start_zero_based_37_19 = 10000
Y_PAR1_end_zero_based_37_19 = 2649520
Y_PAR2_start_zero_based_37_19 = 59034049
Y_PAR2_end_zero_based_37_19 = 59363566
X_PAR1_start_zero_based_38 = 10000
X_PAR1_end_zero_based_38 = 2781479
X_PAR2_start_zero_based_38 = 155701382
X_PAR2_end_zero_based_38 = 156030895
Y_PAR1_start_zero_based_38 = 10000
Y_PAR1_end_zero_based_38 = 2781479
Y_PAR2_start_zero_based_38 = 56887902
Y_PAR2_end_zero_based_38 = 57217415

HAP_ALIGNMENT = {
    # "ADD_FLANKING_ALIGNMENT": True,
    "MIN_NON_STUTTER_RATE": np.log(0.000001),
    "MAX_LIK_ALIGNMENT": True,  # False,
    "MAX_PGEOM": 0.999,
    "MIN_INDEL_RATE": 0.000001,
    "IMPOSSIBLE_INS_RATE": np.log(10**-9),
    "CONSIDER_INS_BASEQ": True,
}

MUTATION_FILTER = {
    "ALLELE_BALANCE": 1 / 2,
    "SNP_SEQUENCING_ERROR_RATE_PRIOR": 0.001,
    "MAX_MOSAIC_POSTERIOR_CUTOFF": 0.9,
    "ALL_MOSAIC_POSTERIOR_CUTOFF": 0.9,
    "LRT_PVALUE_CUTOFF": 0.05,
    "BINOMIAL_TEST_PVALUE_CUTOFF": 0.05,
    "PHASE_POSTERIOR_CUTOFF": 0.25,
    "PROB_BASED_GERMLINE": True,
}

UNPHASE_PHASE_OUTPUT = {
    "PS": ".",
    "NSNP": ".",
    "HVAF": ".",
    "PP": ".",
    "PQ": ".",
    "PGT": ".",
    "PMGT": ".",
    "PVARIANTTYPE": ".",
    "PMUTP": ".",
    "PFRAME": ".",
    "PMN": ".",
    "PREGION": ".",
    "PMP": ".",
    "PAMP": ".",
    "PGQ": ".",
    "PGG": ".",
    "PMF": ".",
    "PPSTR": ".",
    "PDP": ".",
    "PFDP": ".",
    "PEAF": ".",
    "PEAD": ".",
    "PAAD": ".",
    "PNSTUTTER": ".",
    "PDSTUTTER": ".",
    "PFILTER": ".",
}
