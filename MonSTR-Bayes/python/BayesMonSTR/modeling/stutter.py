# -*- coding: utf-8 -*-
import itertools
import math
import warnings
from functools import lru_cache
from operator import itemgetter

import numpy as np

# import pysnooper
from numpy import exp, log
from scipy.special import comb, logsumexp
from scipy.stats import betabinom, chi2

from ..processing import bam_processor
from ..utils import FLK_BP, tools
from . import align


class LengthStutterEM:
    """Length-only MS stutter error model."""

    def __init__(
        self,
        args,
        reads_data_len,
        motif_len,
        af=0.5,
    ) -> None:
        # preprocessing params
        assert 0 < af < 1
        # assert motif_len in [1, 2, 3, 4, 5, 6] # maybe not
        # non-log values
        self.args = args
        self.af = af
        self.stutter = {
            "rho_bulk_in": 0.9,
            "up_bulk_in": 0.01,
            "down_bulk_in": 0.01,
            "rho_sc_in": 0.9,
            "up_sc_in": 0.01,
            "down_sc_in": 0.01,
            "rho_bulk_out": 0.9,
            "up_bulk_out": 0.01,
            "down_bulk_out": 0.01,
            "rho_sc_out": 0.9,
            "up_sc_out": 0.01,
            "down_sc_out": 0.01,
        }
        self.motif_len = motif_len
        self.reads_data = reads_data_len  # length-based
        self.num_ind = len(self.reads_data)
        self.num_sc_sample = [len(ind[1]) for ind in self.reads_data]
        self.allele_pool = [
            sorted(set((tools.flatten(ind)))) for ind in self.reads_data
        ]
        flatten_reads = list(tools.flatten(self.reads_data))
        self.total_num_reads = len(flatten_reads)
        self.total_num_bulk_reads = sum([len(ind[0]) for ind in self.reads_data])
        self.total_num_sc_reads = self.total_num_reads - self.total_num_bulk_reads
        self.total_allele_pool = sorted(set(flatten_reads))
        self.num_total_allele = len(self.total_allele_pool)
        if self.num_total_allele:
            self.freq = dict(
                zip(self.total_allele_pool, itertools.repeat(1 / self.num_total_allele))
            )
        self.mu_msi = 10e-4
        self.threshold = None
        # genotype pools for each individual
        # ind * gt
        self.genotypes = [tools.cartesian_prod(pool) for pool in self.allele_pool]
        self.old_L = float("-inf")
        self.trained = False

    # @tools.timing
    def train(self):
        r"""Estimation of MS stutter error model using EM-algorithm"""
        for _ in range(10):
            # ---------------------- E-step ----------------------
            # region E-step
            # Calculation of likelihoods for each candidate genotypes
            # num_ind * num_sample * num_gt
            self.sample_likelihoods = []  # bulk is the 1st, e.g. bulk + sc1 + sc2
            for idx_ind, bam_ind in enumerate(self.reads_data):
                ind_sample_likelihood = []
                # pooled bulk
                ind_sample_likelihood.append(
                    [
                        self.gt_likelihoods(gt, bam_ind[0], 0.5, "bulk")
                        for gt in self.genotypes[idx_ind]
                    ]
                )
                # single cells
                for reads in bam_ind[1]:
                    ind_sample_likelihood.append(
                        [
                            self.gt_likelihoods(gt, reads, 0.5, "sc")
                            for gt in self.genotypes[idx_ind]
                        ]
                    )
                self.sample_likelihoods.append(ind_sample_likelihood)
            self.sample_likelihoods = [
                [tools.normalize_log(ll) for ll in ind]
                for ind in self.sample_likelihoods
            ]
            # new_total_sample_LL = logsumexp(
            #     list(tools.flatten(self.sample_likelihoods))
            # )

            # Calculation of germline priors
            # num_ind * num_gt
            self.germ_priors = [
                [self.gt_germ_priors(gt, idx_ind) for gt in self.genotypes[idx_ind]]
                for idx_ind in range(self.num_ind)
            ]
            new_total_sample_LL = sum([max(ind) for ind in self.germ_priors if ind])
            # normalization
            self.germ_priors = [tools.normalize_log(ll) for ll in self.germ_priors]

            # Calculation of single cell sample posteriors
            # num_ind * num_sample * num_gt
            self.sample_posteriors = []
            for idx_ind in range(self.num_ind):
                smpl_posteriors = []
                for idx_sample in range(1, self.num_sc_sample[idx_ind] + 1):
                    smpl_posteriors.append(
                        [
                            self.gt_sc_posteriors(gt, idx_ind, idx_sample)
                            for gt in self.genotypes[idx_ind]
                        ]
                    )
                self.sample_posteriors.append(smpl_posteriors)
            new_total_sample_LL += sum(
                [max(s) for ind in self.sample_posteriors for s in ind if s]
            )

            # normalization
            self.sample_posteriors = [
                [tools.normalize_log(ss) for ss in ind]
                for ind in self.sample_posteriors
            ]
            # endregion

            # ---------------------- M-step ----------------------
            # region M-step
            # in-frame, (read - allele) % self.motif_len == 0
            up_bulk_in_list = []
            up_sc_in_list = []
            down_bulk_in_list = []
            down_sc_in_list = []
            rho_bulk_n_in_list = []
            rho_bulk_d_in_list = []
            rho_sc_n_in_list = []
            rho_sc_d_in_list = []
            up_bulk_out_list = []
            up_sc_out_list = []
            down_bulk_out_list = []
            down_sc_out_list = []
            rho_bulk_n_out_list = []
            rho_bulk_d_out_list = []
            rho_sc_n_out_list = []
            rho_sc_d_out_list = []
            for idx_ind, ind in enumerate(self.reads_data):
                # bulk
                bulk_read_data = ind[0]
                for read in bulk_read_data:  # bulk is germ_priors
                    up_bulk_in_list.append(
                        self.read_assign_prob(
                            idx_ind, None, read, tools.greater, "bulk", "in"
                        )
                    )
                    down_bulk_in_list.append(
                        self.read_assign_prob(
                            idx_ind, None, read, tools.less, "bulk", "in"
                        )
                    )
                    rho_bulk_n_in_list.append(
                        self.read_assign_prob(
                            idx_ind, None, read, tools.not_equal, "bulk", "in"
                        )
                    )
                    rho_bulk_d_in_list.append(
                        self.read_assign_prob(
                            idx_ind, None, read, tools.length, "bulk", "in"
                        )
                    )
                    up_bulk_out_list.append(
                        self.read_assign_prob(
                            idx_ind, None, read, tools.greater, "bulk", "out"
                        )
                    )
                    down_bulk_out_list.append(
                        self.read_assign_prob(
                            idx_ind, None, read, tools.less, "bulk", "out"
                        )
                    )
                    rho_bulk_n_out_list.append(
                        self.read_assign_prob(
                            idx_ind, None, read, tools.not_equal, "bulk", "out"
                        )
                    )
                    rho_bulk_d_out_list.append(
                        self.read_assign_prob(
                            idx_ind, None, read, tools.length, "bulk", "out"
                        )
                    )
                # single cell sample
                for idx_sample, sample in enumerate(ind[1]):  # sc starts from zero
                    for read in sample:
                        up_sc_in_list.append(
                            self.read_assign_prob(
                                idx_ind, idx_sample, read, tools.greater, "sc", "in"
                            )
                        )
                        down_sc_in_list.append(
                            self.read_assign_prob(
                                idx_ind, idx_sample, read, tools.less, "sc", "in"
                            )
                        )
                        rho_sc_n_in_list.append(
                            self.read_assign_prob(
                                idx_ind, idx_sample, read, tools.not_equal, "sc", "in"
                            )
                        )
                        rho_sc_d_in_list.append(
                            self.read_assign_prob(
                                idx_ind, idx_sample, read, tools.length, "sc", "in"
                            )
                        )
                        up_sc_out_list.append(
                            self.read_assign_prob(
                                idx_ind, idx_sample, read, tools.greater, "sc", "out"
                            )
                        )
                        down_sc_out_list.append(
                            self.read_assign_prob(
                                idx_ind, idx_sample, read, tools.less, "sc", "out"
                            )
                        )
                        rho_sc_n_out_list.append(
                            self.read_assign_prob(
                                idx_ind, idx_sample, read, tools.not_equal, "sc", "out"
                            )
                        )
                        rho_sc_d_out_list.append(
                            self.read_assign_prob(
                                idx_ind, idx_sample, read, tools.length, "sc", "out"
                            )
                        )
            up_sc_in = (
                logsumexp(up_sc_in_list) - log(self.total_num_sc_reads)
                if up_sc_in_list
                else log(0.0001)
            )
            up_bulk_in = (
                logsumexp(up_bulk_in_list) - log(self.total_num_bulk_reads)
                if up_bulk_in_list
                else log(0.0001)
            )
            down_sc_in = (
                logsumexp(down_sc_in_list) - log(self.total_num_sc_reads)
                if down_sc_in_list
                else log(0.0001)
            )
            down_bulk_in = (
                logsumexp(down_bulk_in_list) - log(self.total_num_bulk_reads)
                if down_bulk_in_list
                else log(0.0001)
            )
            up_sc_out = (
                logsumexp(up_sc_out_list) - log(self.total_num_sc_reads)
                if up_sc_out_list
                else log(0.0001)
            )
            up_bulk_out = (
                logsumexp(up_bulk_out_list) - log(self.total_num_bulk_reads)
                if up_bulk_out_list
                else log(0.0001)
            )
            down_sc_out = (
                logsumexp(down_sc_out_list) - log(self.total_num_sc_reads)
                if down_sc_out_list
                else log(0.0001)
            )
            down_bulk_out = (
                logsumexp(down_bulk_out_list) - log(self.total_num_bulk_reads)
                if down_bulk_out_list
                else log(0.0001)
            )
            rho_sc_in = (
                logsumexp(rho_sc_n_in_list) - logsumexp(rho_sc_d_in_list)
                if rho_sc_n_in_list
                else log(0.999)
            )
            rho_bulk_in = (
                logsumexp(rho_bulk_n_in_list) - logsumexp(rho_bulk_d_in_list)
                if rho_bulk_n_in_list
                else log(0.999)
            )
            rho_sc_out = (
                logsumexp(rho_sc_n_out_list) - logsumexp(rho_sc_d_out_list)
                if rho_sc_n_out_list
                else log(0.999)
            )
            rho_bulk_out = (
                logsumexp(rho_bulk_n_out_list) - logsumexp(rho_bulk_d_out_list)
                if rho_bulk_n_out_list
                else log(0.999)
            )
            stutter = {
                "rho_bulk_in": exp(rho_bulk_in),
                "up_bulk_in": exp(up_bulk_in),
                "down_bulk_in": exp(down_bulk_in),
                "rho_sc_in": exp(rho_sc_in),
                "up_sc_in": exp(up_sc_in),
                "down_sc_in": exp(down_sc_in),
                "rho_bulk_out": exp(rho_bulk_out),
                "up_bulk_out": exp(up_bulk_out),
                "down_bulk_out": exp(down_bulk_out),
                "rho_sc_out": exp(rho_sc_out),
                "up_sc_out": exp(up_sc_out),
                "down_sc_out": exp(down_sc_out),
            }
            # breakpoint()
            # HACK: rho <= 0.999 HipSTR
            for param in stutter:
                if "rho" in param:
                    stutter[param] = stutter[param] if stutter[param] < 0.999 else 0.999
                else:
                    stutter[param] = (
                        stutter[param] if stutter[param] > 0.0001 else 0.0001
                    )

            # print(self.stutter, stutter)
            new_L = sum([abs(stutter[key] - self.stutter[key]) for key in stutter])
            self.stutter = stutter
            # freq
            freq = {}
            for allele in self.total_allele_pool:
                freq_list = []
                for idx_ind in range(self.num_ind):
                    freq_list.append(self.get_allele_germ_priors(idx_ind, allele))
                freq[allele] = exp(logsumexp(freq_list) + log(1 / (2 * self.num_ind)))
            self.freq = freq
            self.read_allele_prob.cache_clear()
            # endregion

            # determination of convergence
            if self.threshold is None:
                self.threshold = abs(new_L * 0.001)
            diff = self.old_L - new_L
            self.old_L = new_L
            if not self.args.quiet:
                tools.logger.info("%s %s %s", new_L, diff, new_total_sample_LL)
                # breakpoint()
            if abs(diff) < self.threshold:
                break
            # print(self.stutter)
        self.trained = True

    @lru_cache(maxsize=2048)
    def read_allele_prob(self, read, allele, prot=None, af=0.5) -> float:
        r"""The conditional probability of reads originating from either allele."""
        assert prot in ["bulk", "sc"]

        if (read - allele) % self.motif_len == 0:  # in-frame
            if prot == "sc":
                if read == allele:
                    error = log(
                        1 - self.stutter["up_sc_in"] - self.stutter["down_sc_in"]
                    )

                elif read > allele:
                    if self.stutter["rho_sc_in"] == 1:
                        error = log(
                            self.stutter["up_sc_in"] * self.stutter["rho_sc_in"]
                        )
                    else:
                        error = log(
                            self.stutter["up_sc_in"]
                            * self.stutter["rho_sc_in"]
                            * (1 - self.stutter["rho_sc_in"])
                            ** ((read - allele) / self.motif_len - 1)
                        )

                else:
                    if self.stutter["rho_sc_in"] == 1:
                        error = log(
                            self.stutter["down_sc_in"] * self.stutter["rho_sc_in"]
                        )
                    else:
                        error = log(
                            self.stutter["down_sc_in"]
                            * self.stutter["rho_sc_in"]
                            * (1 - self.stutter["rho_sc_in"])
                            ** ((allele - read) / self.motif_len - 1)
                        )

            elif prot == "bulk":
                if read == allele:
                    error = log(
                        1 - self.stutter["up_bulk_in"] - self.stutter["down_bulk_in"]
                    )

                elif read > allele:
                    if self.stutter["rho_bulk_in"] == 1:
                        error = log(
                            self.stutter["up_bulk_in"] * self.stutter["rho_bulk_in"]
                        )
                    else:
                        error = log(
                            self.stutter["up_bulk_in"]
                            * self.stutter["rho_bulk_in"]
                            * (1 - self.stutter["rho_bulk_in"])
                            ** ((read - allele) / self.motif_len - 1)
                        )

                else:
                    if self.stutter["rho_bulk_in"] == 1:
                        error = log(
                            self.stutter["down_bulk_in"] * self.stutter["rho_bulk_in"]
                        )
                    else:
                        error = log(
                            self.stutter["down_bulk_in"]
                            * self.stutter["rho_bulk_in"]
                            * (1 - self.stutter["rho_bulk_in"])
                            ** ((allele - read) / self.motif_len - 1)
                        )

        else:  # out-frame
            if prot == "sc":
                if read == allele:
                    error = log(
                        1 - self.stutter["up_sc_out"] - self.stutter["down_sc_out"]
                    )

                elif read > allele:
                    if self.stutter["rho_sc_out"] == 1:
                        error = log(
                            self.stutter["up_sc_out"] * self.stutter["rho_sc_out"]
                        )
                    else:
                        error = log(
                            self.stutter["up_sc_out"]
                            * self.stutter["rho_sc_out"]
                            * (1 - self.stutter["rho_sc_out"])
                            ** (read - allele - ((read - allele) // self.motif_len) - 1)
                        )

                else:
                    if self.stutter["rho_sc_out"] == 1:
                        error = log(
                            self.stutter["down_sc_out"] * self.stutter["rho_sc_out"]
                        )
                    else:
                        error = log(
                            self.stutter["down_sc_out"]
                            * self.stutter["rho_sc_out"]
                            * (1 - self.stutter["rho_sc_out"])
                            ** (allele - read - ((allele - read) // self.motif_len) - 1)
                        )

            elif prot == "bulk":
                if read == allele:
                    error = log(
                        1 - self.stutter["up_bulk_out"] - self.stutter["down_bulk_out"]
                    )

                elif read > allele:
                    if self.stutter["rho_bulk_out"] == 1:
                        error = log(
                            self.stutter["up_bulk_out"] * self.stutter["rho_bulk_out"]
                        )
                    else:
                        error = log(
                            self.stutter["up_bulk_out"]
                            * self.stutter["rho_bulk_out"]
                            * (1 - self.stutter["rho_bulk_out"])
                            ** (read - allele - ((read - allele) // self.motif_len) - 1)
                        )

                else:
                    if self.stutter["rho_bulk_out"] == 1:
                        error = log(
                            self.stutter["down_bulk_out"] * self.stutter["rho_bulk_out"]
                        )
                    else:
                        error = log(
                            self.stutter["down_bulk_out"]
                            * self.stutter["rho_bulk_out"]
                            * (1 - self.stutter["rho_bulk_out"])
                            ** (allele - read - ((allele - read) // self.motif_len) - 1)
                        )

        return log(af) + error

    def gt_sc_posteriors(self, gt_sc, idx_ind, idx_sample) -> float:
        """Joint probability of germline and sample's genotype posteriors."""
        gt_sc_posteriors = logsumexp(
            (
                self.get_gt_germ_priors(idx_ind, gt_sc, 0) + log(1 - self.mu_msi),
                self.get_gt_germ_priors(idx_ind, gt_sc, 1) + log(0.5 * self.mu_msi),
                self.get_gt_germ_priors(idx_ind, gt_sc, 2) + log(0.5 * self.mu_msi),
            )
        )
        gt_sc_posteriors += self.get_gt_likelihoods(idx_ind, idx_sample, gt_sc)
        return gt_sc_posteriors

    def gt_germ_priors(self, gt_germ, idx_ind) -> float:
        r"""Germline genotype priors."""
        j_allele, k_allele = gt_germ
        if self.freq.get(j_allele) == 0 or self.freq.get(k_allele) == 0:
            return float("-inf")
        else:
            gt_germ_priors = log(self.freq.get(j_allele)) + log(self.freq.get(k_allele))
        if j_allele != k_allele:
            gt_germ_priors += log(2)
        # assume 0 is bulk
        gt_germ_priors += self.get_gt_likelihoods(idx_ind, 0, gt_germ)
        # assume rest is sc
        for idx_sample in range(1, len(self.sample_likelihoods[idx_ind])):
            gt_germ_priors += logsumexp(
                (
                    log(1 - self.mu_msi)
                    + self.get_gt_likelihoods(idx_ind, idx_sample, gt_germ, 0),
                    log(0.5 * self.mu_msi)
                    + self.get_gt_likelihoods(idx_ind, idx_sample, gt_germ, 1),
                    log(0.5 * self.mu_msi)
                    + self.get_gt_likelihoods(idx_ind, idx_sample, gt_germ, 2),
                )
            )
        return gt_germ_priors

    def get_gt_germ_priors(self, idx_ind, genotype, nor=0) -> float:
        """Get the germline genotype(s) priors for an individual."""
        assert nor in [0, 1, 2]
        idx_gt = []

        if nor == 0:
            idx_gt = self.genotypes[idx_ind].index(genotype)
            priors = self.germ_priors[idx_ind][idx_gt]

        elif nor == 1:
            for idx, gt in enumerate(self.genotypes[idx_ind]):
                if (gt != genotype) and (gt[1] in genotype):
                    idx_gt.append(idx)
            if not idx_gt:
                return float("-inf")
                raise ValueError("no candidate gt in gt pool.")
            priors = itemgetter(*idx_gt)(self.germ_priors[idx_ind])
            priors = logsumexp(priors)

        elif nor == 2:
            for idx, gt in enumerate(self.genotypes[idx_ind]):
                if (gt != genotype) and (gt[0] in genotype):
                    idx_gt.append(idx)
            if not idx_gt:
                return float("-inf")
                raise ValueError("no candidate gt in gt pool.")
            priors = itemgetter(*idx_gt)(self.germ_priors[idx_ind])
            priors = logsumexp(priors)

        return priors

    def get_sample_posteriors(self, idx_ind, idx_sample, genotype) -> float:
        """Get the sample genotype(s) posteriors for a sample in an individual."""
        idx_gt = self.genotypes[idx_ind].index(genotype)
        return self.sample_posteriors[idx_ind][idx_sample][idx_gt]

    def get_allele_germ_priors(self, idx_ind, allele) -> float:
        """Used for calculation of allele freqency in M-step"""
        priors = []
        for idx_gt, gt in enumerate(self.genotypes[idx_ind]):
            if allele in gt:
                if gt[0] == gt[1]:  # hom
                    priors.append(self.germ_priors[idx_ind][idx_gt] + log(2))
                else:
                    priors.append(self.germ_priors[idx_ind][idx_gt])
        if priors:
            return logsumexp(priors)
        else:
            return float("-inf")

    def gt_likelihoods(self, genotype, reads_data, af=0.5, prot=None) -> float:
        r"""Reads probabilities given genotype for bulk or single cell sample."""
        j_allele, k_allele = genotype
        r = [
            logsumexp(
                (
                    self.read_allele_prob(read, j_allele, prot, af),
                    self.read_allele_prob(read, k_allele, prot, 1 - af),
                )
            )
            for read in reads_data
        ]
        return sum(r)

    def get_gt_likelihoods(self, idx_ind, idx_sample, genotype, nor=0) -> float:
        r"""Get the genotype(s) likelihood given the sample for an individual.

        Return one of the following genotype(s) likelihoods

        .. math::

            P(D_{i, c} \mid G_{cell}) = \left\{
                \begin{array}{ll}
                P(D_{i, c} \mid G_{cell} = (j, k)), & \text{same as germline event} \\
                P(D_{i, c} \mid G_{cell} = (\neg j, k)), & \text{somatic mutation on first allele} \\
                P(D_{i, c} \mid G_{cell} = (j, \neg k)), & \text{somatic mutation on second allele}
                \end{array}
            \right.

        Parameters
        ----------
        idx_ind : `int`
            Individual index.
        idx_sample : `int`
            Sample index, 0 is bulk data for each individual.
        genotype : tuple[`int`, `int`]
            Genotype(s).
        nor : {0, 1, 2}, optional
            NOR symbol position, 0 indicates [j, k] scenario,
            1 indicates [¬j, k] scenario, and 2 indicates [j, ¬k] scenario, by default 0.

        Returns
        -------
        `float`
            The genotype log likelihood.
        """

        assert nor in [0, 1, 2]
        idx_gt = []

        if nor == 0:
            # simple form genotype (e.g. [j, k])
            idx_gt = self.genotypes[idx_ind].index(genotype)
            likelihoods = self.sample_likelihoods[idx_ind][idx_sample][idx_gt]

        elif nor == 1:
            # form with NOR symbol genotype (e.g. [¬j, k])
            for idx, gt in enumerate(self.genotypes[idx_ind]):
                if (gt != genotype) and (gt[1] in genotype):
                    idx_gt.append(idx)
            if not idx_gt:
                return float("-inf")
                raise ValueError("no candidate gt in gt pool.")
            likelihoods = itemgetter(*idx_gt)(
                self.sample_likelihoods[idx_ind][idx_sample]
            )
            likelihoods = logsumexp(likelihoods)

        elif nor == 2:
            # form with NOR symbol genotype (e.g. [j, ¬k])
            for idx, gt in enumerate(self.genotypes[idx_ind]):
                if (gt != genotype) and (gt[0] in genotype):
                    idx_gt.append(idx)
            if not idx_gt:
                return float("-inf")
                raise ValueError("no candidate gt in gt pool.")
            likelihoods = itemgetter(*idx_gt)(
                self.sample_likelihoods[idx_ind][idx_sample]
            )
            likelihoods = logsumexp(likelihoods)

        return likelihoods

    def phasing_prob(self, idx_ind, reads_data, reads_spn_data, genotype, het_site):
        """Calculate the haplotype phasing prbability given heterozygous genotype."""

        j_allele, k_allele = genotype
        assert j_allele != k_allele

        # product
        hap_prob_1 = self.freq.get(j_allele) + self.freq.get(k_allele)
        hap_prob_2 = self.freq.get(j_allele) + self.freq.get(k_allele)

        for read in reads_data:
            hap_prob_1 += logsumexp(
                (
                    self.read_allele_prob(read, j_allele, "sc")
                    + self.SNP_prob(read, 30, het_site[0]),
                    self.read_allele_prob(read, k_allele, "sc")
                    + self.SNP_prob(read, 30, het_site[1]),
                )
            )
        for read in reads_data:
            hap_prob_1 += logsumexp(
                (
                    self.read_allele_prob(read, k_allele, "sc")
                    + self.SNP_prob(read, 30, het_site[0]),
                    self.read_allele_prob(read, j_allele, "sc")
                    + self.SNP_prob(read, 30, het_site[1]),
                )
            )

        # normalization
        hap_prob_sum = logsumexp((hap_prob_1, hap_prob_2))
        hap_prob_1 = hap_prob_1 - hap_prob_sum
        hap_prob_2 = hap_prob_2 - hap_prob_sum
        # phasing_log_ratio = hap_prob_1 - hap_prob_sum

        return hap_prob_1, hap_prob_2

    def SNP_prob(self, read, baseQ, allele) -> float:
        r"""Calculate read-allele probability based on SNP error model."""
        error_rate = 10 ** (-baseQ / 10)
        if read == allele:
            return log(1 - error_rate)
        else:
            return log(error_rate / 3)

    def read_assign_prob(
        self, idx_ind, idx_sample, read, ind_func, prot=None, frame=None
    ):
        r"""Read assignment log probability."""

        read_assign_prob = []
        read_assign_indic = []
        genotypes = self.genotypes[idx_ind]

        if prot == "sc":
            for gt in genotypes:
                j_allele, k_allele = gt
                read_assign_prob.append(
                    (
                        self.get_sample_posteriors(idx_ind, idx_sample, gt)
                        + self.read_allele_prob(read, j_allele, prot, self.af)
                    )
                )
                read_assign_indic.append(
                    ind_func(read, j_allele, self.motif_len, frame)
                )
                read_assign_prob.append(
                    (
                        self.get_sample_posteriors(idx_ind, idx_sample, gt)
                        + self.read_allele_prob(read, k_allele, prot, self.af)
                    )
                )
                read_assign_indic.append(
                    ind_func(read, k_allele, self.motif_len, frame)
                )
        else:
            for gt in genotypes:
                j_allele, k_allele = gt
                read_assign_prob.append(
                    (
                        self.get_gt_germ_priors(idx_ind, gt)
                        + self.read_allele_prob(read, j_allele, prot, self.af)
                    )
                )
                read_assign_indic.append(
                    ind_func(read, j_allele, self.motif_len, frame)
                )
                read_assign_prob.append(
                    (
                        self.get_gt_germ_priors(idx_ind, gt)
                        + self.read_allele_prob(read, k_allele, prot, self.af)
                    )
                )
                read_assign_indic.append(
                    ind_func(read, k_allele, self.motif_len, frame)
                )

        # normalization
        read_assign_prob = tools.normalize_log(read_assign_prob)
        res = [a + b for a, b in zip(read_assign_prob, read_assign_indic)]
        return logsumexp(res)


class HapStutterEM:
    """Sequence-based MS genotyping model."""

    def __init__(
        self,
        args,
        region,
        stutter,
        reads_data_seq,
        reads_data_qual,
        sc_prot_list,
        hSNP_info=None,
        reads_data_spn_seq=None,
        reads_data_spn_qual=None,
        hap_freq=None,
    ) -> None:
        # preprocessing params
        # assert motif_len in [1, 2, 3, 4, 5, 6]
        # non-log values
        self.args = args
        self.af = []
        self.ab_data = []
        self.stutter = stutter
        self.region = region
        self.sc_prot_list = sc_prot_list
        self.hap_freq = hap_freq
        if not args.ra:
            ref_allele = bam_processor.fa_reader(args.ref, region, flank_size=FLK_BP)
            ref_allele_seg = (
                ref_allele[:FLK_BP],
                ref_allele[FLK_BP:-FLK_BP],
                ref_allele[-FLK_BP:],
            )
        # allele pool gen
        self.allele_pool = []
        for idx_ind, ind in enumerate(reads_data_seq):
            # bulk
            ind_allele_pool = bam_processor.allele_gen(
                {allele: ind[0].count(allele) for allele in set(ind[0])}
            )
            # sc
            for sc_reads in ind[1]:
                ind_allele_pool.extend(
                    bam_processor.allele_gen(
                        {allele: sc_reads.count(allele) for allele in set(sc_reads)}
                    )
                )
            if args.ra:
                self.allele_pool.append(sorted(set(ind_allele_pool)))
            else:
                self.allele_pool.append(sorted(set(ind_allele_pool + [ref_allele_seg])))
        self.total_allele_pool = sorted(set(tools.flatten_list(self.allele_pool)))
        self.allele_flk_size = (
            (
                min(len(s[0]) for s in self.total_allele_pool),
                min(len(s[2]) for s in self.total_allele_pool),
            )
            if self.total_allele_pool
            else (5, 5)
        )
        # trim allele and reads
        self.allele_pool = tools.map_nested_list(
            tools.trim_flk, self.allele_pool, self.allele_flk_size
        )
        self.allele_pool = [sorted(set(ind)) for ind in self.allele_pool]
        self.total_allele_pool = sorted(set(tools.flatten_list(self.allele_pool)))
        self.num_allele_pool = [len(ind) for ind in self.allele_pool]
        self.num_total_allele = len(self.total_allele_pool)
        # for ref_allele
        if not args.ra:
            self.ref_allele = (
                ref_allele[FLK_BP - self.allele_flk_size[0] : FLK_BP],
                ref_allele[FLK_BP:-FLK_BP],
                ref_allele[
                    -FLK_BP : self.allele_flk_size[1] - FLK_BP
                    if self.allele_flk_size[1] - FLK_BP
                    else None
                ],
            )
            if self.ref_allele not in self.total_allele_pool:
                self.total_allele_pool.insert(0, self.ref_allele)
            else:
                self.total_allele_pool.insert(
                    0,
                    self.total_allele_pool.pop(
                        self.total_allele_pool.index(self.ref_allele)
                    ),
                )
        self.reads_data_seq = tools.map_nested_filter(
            tools.trim_flk, reads_data_seq, self.allele_flk_size
        )
        self.reads_data_qual = tools.map_nested_filter(
            tools.trim_flk, reads_data_qual, self.allele_flk_size
        )
        # phase related
        if reads_data_spn_seq and reads_data_spn_qual:
            self.reads_data_spn_seq = tools.map_nested_filter(
                tools.trim_flk, reads_data_spn_seq, self.allele_flk_size
            )
            self.reads_data_spn_qual = tools.map_nested_filter(
                tools.trim_flk, reads_data_spn_qual, self.allele_flk_size
            )
            if hSNP_info[0] != "auto":
                _, best_records, best_genotypes, best_distances = hSNP_info
                # self.best_records = best_records
                # self.best_genotypes = best_genotypes
                # h_1, h_2
                self.hSNP = [
                    (
                        (r.ref, r.alts[0])
                        if best_genotypes[idx_ind] == (0, 1)
                        else (
                            (r.alts[0], r.ref)
                            if best_genotypes[idx_ind] == (1, 0)
                            else best_genotypes[idx_ind]
                        )
                    )
                    for idx_ind, r in enumerate(best_records)
                ]
                self.hSNP_start = [r.start if r else None for r in best_records]
            else:
                _, best_records, best_hSNPs, best_distances = hSNP_info
                self.hSNP = best_hSNPs
                self.hSNP_start = [r.start if r else None for r in best_records]
                # self.best_genotypes = None
            self.best_distances = best_distances
            self.phasable = [
                len(tuple(tools.flatten_list(r))) > 0 for r in reads_data_spn_seq
            ]  # self.phasable = [bool(v) for v in best_records]
            self.num_spn = [
                (
                    [len(self.reads_data_spn_seq[idx_ind][0])]
                    + [
                        len(self.reads_data_spn_seq[idx_ind][1][i])
                        for i in range(len(self.reads_data_spn_seq[idx_ind][1]))
                    ]
                    if ind
                    else None
                )
                for idx_ind, ind in enumerate(self.phasable)
            ]
            # hap pool gen
            self.hap_pool = []
            for idx_ind, ind in enumerate(reads_data_spn_seq):
                # bulk
                ind_hap_pool = bam_processor.allele_gen(
                    {allele: ind[0].count(allele) for allele in set(ind[0])}
                )
                # sc
                for sc_reads in ind[1]:
                    ind_hap_pool.extend(
                        bam_processor.allele_gen(
                            {allele: sc_reads.count(allele) for allele in set(sc_reads)}
                        )
                    )
                self.hap_pool.append(sorted(set(ind_hap_pool)))
            self.num_hap_pool = [len(ind) for ind in self.hap_pool]
            self.num_haps_phased = [
                len(
                    bam_processor.allele_gen(
                        {
                            allele: list(tools.flatten_list(v)).count(allele)
                            for allele in set(tools.flatten_list(v))
                        }
                    )
                )
                if v
                else 0
                for v in reads_data_spn_seq
            ]
            self.phasing_check = []
            self.dis_prop = []
            self.con = []
            self.lr_phase = []
        else:
            self.reads_data_spn_seq = None
            self.reads_data_spn_qual = None
            # self.best_genotypes = None
            self.best_distances = None
            self.phasable = None
            self.hSNP_start = None
            self.hSNP = None
            self.num_spn = None
            self.hap_pool = None
            self.num_hap_pool = None
            self.num_haps_phased = None
            self.phasing_check = None
            self.dis_prop = None
            self.con = None
            self.lr_phase = None
        self.bb_likelihoods = []
        self.bb_likelihoods_p = []
        self.prod_likelihoods = []
        self.prod_likelihoods_p = []
        self.mo_posterior_bbs = []
        self.mo_posterior_bbs_p = []
        self.phy_llbb = []
        self.phy_llbb_p = []
        # genotypes per individual
        self.genotypes = [tools.cartesian_prod(pool) for pool in self.allele_pool]
        if self.phasable:
            self.phase_probs = [
                (
                    {gt: log(0.5) for gt in gt_ind if gt[0] != gt[1]}
                    if self.phasable[idx_ind]
                    else {}
                )
                for idx_ind, gt_ind in enumerate(self.genotypes)
            ]
        # sample metadata
        self.num_ind = len(self.reads_data_seq)
        self.num_sc_sample = [len(ind[1]) for ind in self.reads_data_seq]
        flatten_reads = list(tools.flatten_list(self.reads_data_seq))
        self.total_num_reads = len(flatten_reads)
        self.total_num_bulk_reads = sum([len(ind[0]) for ind in self.reads_data_seq])
        self.total_num_sc_reads = self.total_num_reads - self.total_num_bulk_reads
        self.samples_depth = []
        for ind in self.reads_data_seq:
            ind_samples_depth = []
            ind_samples_depth.append(len(ind[0]))
            ind_samples_depth.extend([len(ss) for ss in ind[1]])
            self.samples_depth.append(ind_samples_depth)
        # params
        if self.num_total_allele:
            if hap_freq and sum(hap_freq.values()):
                self.freq = dict(
                    zip(
                        self.total_allele_pool,
                        itertools.repeat(1e-4),
                    )
                )
                for hap in self.total_allele_pool:
                    if hap in self.hap_freq.keys():
                        self.freq[hap] = hap_freq[hap] / sum(hap_freq.values())
                self.freq = tools.normalize_dict(self.freq)
            else:
                self.freq = dict(
                    zip(
                        self.total_allele_pool,
                        itertools.repeat(1 / self.num_total_allele),
                    )
                )  # non-log
        self.mut_rate = 1e-4  # mu_mosaic
        self.mu = [
            dict(
                zip(
                    tools.cartesian_product(ind_allele_pool),
                    itertools.repeat(log(self.mut_rate)),
                )
            )
            for ind_allele_pool in self.allele_pool
        ]  # list of dict, seq-based, log value
        for idx_m, m in enumerate(self.mu):  # ******
            for k in m.keys():
                if k[0] == k[1]:
                    self.mu[idx_m][k] = log(1 - self.mut_rate)
        self.threshold = None
        self.total_sample_likelihood = float("-inf")
        self.total_params = float("-inf")
        self.total_params_refine = float("-inf")
        self.mode = args.mode
        # breakpoint()

    # @tools.timing
    def train(self):
        """phasable"""
        termination = False
        if not self.phasable:  # unphasble
            self.sample_likelihoods = []
            for idx_ind, bam_ind in enumerate(self.reads_data_seq):
                ind_sample_likelihood = []
                # pooled bulk
                ind_sample_likelihood.append(
                    [
                        self.gt_likelihoods(
                            gt,
                            bam_ind[0],
                            self.reads_data_qual[idx_ind][0],
                            "bulk",
                        )
                        for gt in self.genotypes[idx_ind]
                    ]
                )
                # single cells
                for idx_reads, reads in enumerate(bam_ind[1]):
                    ind_sample_likelihood.append(
                        [
                            self.gt_likelihoods(
                                gt,
                                reads,
                                self.reads_data_qual[idx_ind][1][idx_reads],
                                self.sc_prot_list[idx_ind][idx_reads],
                            )
                            for gt in self.genotypes[idx_ind]
                        ]
                    )
                self.sample_likelihoods.append(ind_sample_likelihood)

            self.sample_likelihoods = [
                [tools.normalize_log(ll) for ll in ind]
                for ind in self.sample_likelihoods
            ]
        for _ in range(10):
            # ---------------------- E-step ----------------------
            if self.phasable:
                # Calculation of genotype likelihoods for all reads
                # num_ind * num_sample * num_gt
                self.sample_likelihoods = []
                for idx_ind, bam_ind in enumerate(self.reads_data_seq):
                    if self.phasable[idx_ind]:
                        ind_sample_likelihood = []
                        # pooled bulk
                        if self.num_spn[idx_ind][0]:
                            ind_sample_likelihood.append(
                                [
                                    self.gt_phase_likelihoods(
                                        idx_ind,
                                        gt,
                                        bam_ind[0],
                                        self.reads_data_qual[idx_ind][0],
                                        "bulk",
                                    )
                                    for gt in self.genotypes[idx_ind]
                                ]
                            )
                        else:
                            ind_sample_likelihood.append(
                                [
                                    self.gt_likelihoods(
                                        gt,
                                        bam_ind[0],
                                        self.reads_data_qual[idx_ind][0],
                                        "bulk",
                                    )
                                    for gt in self.genotypes[idx_ind]
                                ]
                            )
                        # single cells
                        for idx_reads, reads in enumerate(bam_ind[1]):
                            if (
                                self.num_spn[idx_ind][idx_reads + 1]
                                and self.af[idx_ind][idx_reads]
                            ):
                                ind_sample_likelihood.append(
                                    [
                                        self.gt_phase_likelihoods(
                                            idx_ind,
                                            gt,
                                            reads,
                                            self.reads_data_qual[idx_ind][1][idx_reads],
                                            self.sc_prot_list[idx_ind][idx_reads],
                                            self.af[idx_ind][idx_reads],
                                        )
                                        for gt in self.genotypes[idx_ind]
                                    ]
                                )
                            else:
                                ind_sample_likelihood.append(
                                    [
                                        self.gt_likelihoods(
                                            gt,
                                            reads,
                                            self.reads_data_qual[idx_ind][1][idx_reads],
                                            self.sc_prot_list[idx_ind][idx_reads],
                                        )
                                        for gt in self.genotypes[idx_ind]
                                    ]
                                )

                        self.sample_likelihoods.append(ind_sample_likelihood)
                    else:
                        ind_sample_likelihood = []
                        # pooled bulk
                        ind_sample_likelihood.append(
                            [
                                self.gt_likelihoods(
                                    gt,
                                    bam_ind[0],
                                    self.reads_data_qual[idx_ind][0],
                                    "bulk",
                                )
                                for gt in self.genotypes[idx_ind]
                            ]
                        )
                        # single cells
                        for idx_reads, reads in enumerate(bam_ind[1]):
                            ind_sample_likelihood.append(
                                [
                                    self.gt_likelihoods(
                                        gt,
                                        reads,
                                        self.reads_data_qual[idx_ind][1][idx_reads],
                                        self.sc_prot_list[idx_ind][idx_reads],
                                    )
                                    for gt in self.genotypes[idx_ind]
                                ]
                            )
                        self.sample_likelihoods.append(ind_sample_likelihood)

                self.sample_likelihoods = [
                    [tools.normalize_log(ll) for ll in ind]
                    for ind in self.sample_likelihoods
                ]
            # new_total_sample_LL = sum(
            #     list(tools.flatten(self.sample_likelihoods))
            # )  # before or after normalize, use diff of params

            # Calculation of each individuals' germline priors
            # for each candidate genotypes
            # num_ind * num_gt
            if self.mode == "sp":
                self.germ_priors_unnorm = [
                    [self.gt_germ_priors(gt, idx_ind) for gt in self.genotypes[idx_ind]]
                    for idx_ind in range(self.num_ind)
                ]
                # normalization
                self.germ_priors = [
                    tools.normalize_log(ll) for ll in self.germ_priors_unnorm
                ]
            else:
                # Calculate mu-based bulk sample likelihoods
                self.bulk_mu_likelihoods = []
                for idx_ind, bam_ind in enumerate(self.reads_data_seq):
                    self.bulk_mu_likelihoods.append(
                        [
                            self.gt_mu_likelihoods(
                                gt,
                                bam_ind[0],
                                self.reads_data_qual[idx_ind][0],
                                idx_ind,
                            )
                            for gt in self.genotypes[idx_ind]
                        ]
                    )
                self.bulk_mu_likelihoods = [
                    tools.normalize_log(ind) for ind in self.bulk_mu_likelihoods
                ]
                # Calculation of each individuals' mu-based germline priors
                # for each candidate genotypes
                # num_ind * num_gt
                self.mu_germ_priors_unnorm = [
                    [
                        self.gt_germ_priors(gt, idx_ind, mu_based=True)
                        for gt in self.genotypes[idx_ind]
                    ]
                    for idx_ind in range(self.num_ind)
                ]
                # normalization
                self.mu_germ_priors = [
                    tools.normalize_log(ll) for ll in self.mu_germ_priors_unnorm
                ]

            # region calculate phasing probabilities
            if self.phasable:
                for idx_ind in range(self.num_ind):
                    if not self.phasable[idx_ind]:
                        continue
                    for phase_gt in self.phase_probs[idx_ind]:
                        j_allele, k_allele = phase_gt
                        # if phase_gt ==:
                        #     print(self.phase_probs[idx_ind][phase_gt])
                        # breakpoint()
                        hap_prob_1 = self.phasing_prob(idx_ind, (j_allele, k_allele))
                        hap_prob_2 = self.phasing_prob(idx_ind, (k_allele, j_allele))
                        hap_prob_sum = tools.normalize_log(hap_prob_1 + hap_prob_2)
                        self.phase_probs[idx_ind][phase_gt] = logsumexp(
                            hap_prob_sum[:3]
                        ) - logsumexp(hap_prob_sum)
            # endregion calculate phasing probabilities

            # Calculation of each single cell sample's posteriors for given germline
            # num_ind * germ_gt * num_sc_sample * num_gt, all sc, idx_sc starts from 0
            self.sc_posteriors = []
            for idx_ind in range(self.num_ind):
                germ_posteriors = []
                for gt_germ in self.genotypes[idx_ind]:
                    # idx_sample starts from 1 cuz sc in `get_gt_likelihoods`
                    germ_smpl_posteriors = []
                    for idx_sample in range(1, self.num_sc_sample[idx_ind] + 1):
                        germ_smpl_posteriors.append(
                            tools.normalize_log(
                                [
                                    self.gt_sc_posteriors(
                                        gt_sc, gt_germ, idx_ind, idx_sample
                                    )
                                    for gt_sc in self.genotypes[idx_ind]
                                ]
                            )
                        )
                    germ_posteriors.append(germ_smpl_posteriors)
                self.sc_posteriors.append(germ_posteriors)

            # Calc of sc sample's posteriors
            # ind * sample * gt
            self.sample_posteriors = [
                [
                    [
                        self.gt_posteriors(idx_ind, idx_sample, gt)
                        for gt in self.genotypes[idx_ind]
                    ]
                    for idx_sample in range(self.num_sc_sample[idx_ind])
                ]
                for idx_ind in range(self.num_ind)
            ]
            # normalization
            self.sample_posteriors = [
                [tools.normalize_log(ss) for ss in ind]
                for ind in self.sample_posteriors
            ]

            # ---------------------- M-step ----------------------
            # mu
            # warnings.simplefilter("error")  # HACK:
            with warnings.catch_warnings(action="error"):
                for idx_ind in range(self.num_ind):
                    for j_allele in self.allele_pool[idx_ind]:
                        for k_allele in self.allele_pool[idx_ind]:
                            if j_allele == k_allele:
                                continue
                            # calc of mu_{i, j->k}
                            n_vals = []
                            for gt_germ in self.genotypes[idx_ind]:
                                if j_allele not in gt_germ:
                                    continue
                                for gt_cell in self.genotypes[idx_ind]:
                                    if k_allele not in gt_cell:
                                        continue
                                    if not (
                                        (
                                            gt_germ[0] == gt_germ[1]
                                            and j_allele in gt_cell
                                        )
                                        or (
                                            (
                                                (
                                                    gt_germ[0]
                                                    if gt_germ[0] != j_allele
                                                    else gt_germ[1]
                                                )
                                                == (
                                                    gt_cell[0]
                                                    if gt_cell[0] != k_allele
                                                    else gt_cell[1]
                                                )
                                            )
                                            and (
                                                (
                                                    gt_germ[0]
                                                    if gt_germ[0] != j_allele
                                                    else gt_germ[1]
                                                )
                                                not in (j_allele, k_allele)
                                            )
                                        )
                                    ):
                                        continue  # elimate ADO and multi-mut and jk->jk, remain jj->jk, jm->km
                                    val = self.get_gt_germ_priors(
                                        idx_ind,
                                        gt_germ,
                                        mu_based=(False if self.mode == "sp" else True),
                                    )
                                    val += logsumexp(
                                        [
                                            self.get_sc_posteriors(
                                                idx_ind, idx_sample, gt_germ, gt_cell
                                            )
                                            for idx_sample in range(
                                                0, self.num_sc_sample[idx_ind]
                                            )
                                        ]
                                    )
                                    n_vals.append(val)

                            d_vals = []
                            for gt_germ in self.genotypes[idx_ind]:
                                if j_allele not in gt_germ:
                                    continue
                                val = self.get_gt_germ_priors(
                                    idx_ind,
                                    gt_germ,
                                    mu_based=(False if self.mode == "sp" else True),
                                ) + log(self.num_sc_sample[idx_ind])
                                if gt_germ[0] == gt_germ[1]:
                                    val += log(2)
                                d_vals.append(val)
                            try:
                                self.mu[idx_ind][(j_allele, k_allele)] = logsumexp(
                                    n_vals
                                ) - logsumexp(d_vals)
                            except:
                                self.mu[idx_ind][(j_allele, k_allele)] = -100
                # j->j, should have non_j_allele.
                for idx_ind in range(self.num_ind):
                    for j_allele in self.allele_pool[idx_ind]:
                        try:
                            l_factor = logsumexp(
                                [
                                    self.mu[idx_ind][(j_allele, k_allele)]
                                    for k_allele in self.allele_pool[idx_ind]
                                    if j_allele != k_allele
                                ]
                            )
                            self.mu[idx_ind][(j_allele, j_allele)] = logsumexp(
                                [0, l_factor], b=[1, -1]
                            )
                        except:
                            self.mu[idx_ind][(j_allele, j_allele)] = 0
                # HACK: replace -inf -100, -100000
                self.mu = [
                    {key: value if value >= -100 else -100 for key, value in d.items()}
                    for d in self.mu
                ]
                # normalization
                # mu = []
                # for idx_ind in range(self.num_ind):
                #     mu_dict = {}
                #     for j_allele in self.allele_pool[idx_ind]:
                #         l_factor = []
                #         for k_allele in self.allele_pool[idx_ind]:
                #             l_factor.append(self.mu[idx_ind][(j_allele, k_allele)])
                #         factor = logsumexp(l_factor)
                #         if factor == float("-inf"):
                #             for k_allele in self.allele_pool[idx_ind]:
                #                 mu_dict[(j_allele, k_allele)] = float("-inf")
                #         else:
                #             for k_allele in self.allele_pool[idx_ind]:
                #                 mu_dict[(j_allele, k_allele)] = (
                #                     self.mu[idx_ind][(j_allele, k_allele)] - factor
                #                 )
                #     mu.append(mu_dict)
                # self.mu = mu # remove to-do, not normalize
            # warnings.resetwarnings()  # HACK:

            # freq
            if not self.hap_freq:
                freq = {}
                for allele in self.total_allele_pool:
                    freq_list = [
                        self.get_allele_germ_priors(idx_ind, allele)
                        for idx_ind in range(self.num_ind)
                    ]
                    freq[allele] = exp(
                        logsumexp(freq_list) + log(1 / (2 * self.num_ind))
                    )
                self.freq = freq  # DEBUG:

            # determination of convergence based on LL. no LL, params ok, germ maybe ok.
            # if self.threshold == None:
            #     self.threshold = abs(new_total_sample_LL * 0.001)
            # diff = new_total_sample_LL - self.total_sample_likelihood
            # self.total_sample_likelihood = new_total_sample_LL
            # if not self.args.quiet:
            #     print(f"{new_total_sample_LL}, {diff}")
            # if diff < self.threshold:
            #     break

            # determination of convergence based on params
            mu = [dict((k, exp(v)) for k, v in ind.items()) for ind in self.mu]
            new_total_params = sum([abs(v) for ind in mu for v in ind.values()])
            if self.threshold is None:
                self.threshold = abs(new_total_params * 0.0001)
            diff = new_total_params - self.total_params
            self.total_params = new_total_params
            if not self.args.quiet:
                tools.logger.info("%s %s", new_total_params, diff)
                # print([dict((k, exp(v)) for k, v in ind.items()) for ind in self.mu])
            # break
            if termination:
                break
            if diff < self.threshold:
                break
                termination = True

        self.mu = [dict((k, exp(v)) for k, v in ind.items()) for ind in self.mu]
        self.trained = True

    def read_aln(self, read, read_qualities, haps, prot=None):
        lls = tools.normalize_log(
            [self.read_allele_prob(read, h, read_qualities, prot) for h in haps]
        )
        if len(haps) == 2:
            return 0 if lls[0] >= lls[1] else 1
        return 0 if lls[0] == max(lls) else 1

    def read_allele_prob(
        self, read, allele, read_qualities, prot=None, hsnp=None
    ) -> float:
        """read prob given allele
        bulk or sc params
        germline or mosaic allele
        phasable or not
        """
        assert prot in ["bulk", "sc", "mda", "pta", "scc", "ts"]
        if (len(read[1]) - len(allele[1])) % self.region["motif_len"] == 0:
            up = self.stutter[f"up_{prot}_in"]
            down = self.stutter[f"down_{prot}_in"]
            rho = self.stutter[f"rho_{prot}_in"]
        else:
            up = self.stutter[f"up_{prot}_out"]
            down = self.stutter[f"down_{prot}_out"]
            rho = self.stutter[f"rho_{prot}_out"]
        return (
            align.align_flk(read[0], allele[0], read_qualities[0])
            + align.align_flk(read[2], allele[2], read_qualities[2])
            + align.align_ms(
                allele[1],
                read[1],
                read_qualities[1],
                (up, down, rho, self.region["motif_len"]),
            )
            + (log(align._emiss(read[3], hsnp, read_qualities[3])) if hsnp else 0)
        )
        # if (len(read[1]) - len(allele[1])) % self.region["motif_len"] == 0:
        #     prob = (
        #         align.align_flk(read[0], allele[0], read_qualities[0])
        #         + align.align_flk(read[2], allele[2], read_qualities[2])
        #         + align.align_ms(
        #             allele[1],
        #             read[1],
        #             read_qualities[1],
        #             (
        #                 (
        #                     self.stutter["up_sc_in"]
        #                     if prot == "sc"
        #                     else self.stutter["up_bulk_in"]
        #                 ),
        #                 (
        #                     self.stutter["down_sc_in"]
        #                     if prot == "sc"
        #                     else self.stutter["down_bulk_in"]
        #                 ),
        #                 (
        #                     self.stutter["rho_sc_in"]
        #                     if prot == "sc"
        #                     else self.stutter["rho_bulk_in"]
        #                 ),
        #                 self.region["motif_len"],
        #             ),
        #         )
        #         + (log(align._emiss(read[3], hsnp, read_qualities[3])) if hsnp else 0)
        #     )
        # else:
        #     prob = (
        #         align.align_flk(read[0], allele[0], read_qualities[0])
        #         + align.align_flk(read[2], allele[2], read_qualities[2])
        #         + align.align_ms(
        #             allele[1],
        #             read[1],
        #             read_qualities[1],
        #             (
        #                 (
        #                     self.stutter["up_sc_out"]
        #                     if prot == "sc"
        #                     else self.stutter["up_bulk_out"]
        #                 ),
        #                 (
        #                     self.stutter["down_sc_out"]
        #                     if prot == "sc"
        #                     else self.stutter["down_bulk_out"]
        #                 ),
        #                 (
        #                     self.stutter["rho_sc_in"]
        #                     if prot == "sc"
        #                     else self.stutter["rho_bulk_out"]
        #                 ),
        #                 self.region["motif_len"],
        #             ),
        #         )
        #         + (log(align._emiss(read[3], hsnp, read_qualities[3])) if hsnp else 0)
        #     )
        # return prob

    def read_gt_prob(
        self, read, gt, read_qualities, prot=None, hsnp=None, af=0.5
    ) -> float:
        """read prob given germline genotype"""

        j_allele, k_allele = gt  # opt for hom to-do

        # read_flk_l, read_ms, read_flk_r = read
        # read_q_flk_l, read_q_ms, read_q_flk_r = read_qualities
        # allele_flk_l, allele_ms, allele_flk_r = allele

        # read_flk_l, read_ms, read_flk_r, read_snp = read
        # read_q_flk_l, read_q_ms, read_q_flk_r, read_snp_q = read_qualities
        # allele_flk_l, allele_ms, allele_flk_r = allele # no SNP!
        if hsnp and af is not None:
            a = self.read_allele_prob(read, j_allele, read_qualities, prot, hsnp[0])
            b = self.read_allele_prob(read, k_allele, read_qualities, prot, hsnp[1])
            p = logsumexp([a, b], b=[af, 1 - af])
        elif hsnp:
            a = self.read_allele_prob(read, j_allele, read_qualities, prot, hsnp[0])
            b = self.read_allele_prob(read, k_allele, read_qualities, prot, hsnp[1])
            p = logsumexp([a, b], b=[0.5, 0.5])
        else:
            a = self.read_allele_prob(read, j_allele, read_qualities, prot)
            b = self.read_allele_prob(read, k_allele, read_qualities, prot)
            p = logsumexp([a, b], b=[0.5, 0.5])
        return p

    def gt_likelihoods(
        self,
        genotype,
        reads_data_seq,
        reads_qualities,
        prot,
        hsnp=None,
        af=0.5,
    ) -> float:
        r"""Reads probabilities given germline genotype for bulk or single cell sample."""

        r = [
            self.read_gt_prob(
                reads_data_seq[idx],
                genotype,
                reads_qualities[idx],
                prot,
                hsnp,
                af,
            )
            for idx in range(len(reads_data_seq))
        ]
        return sum(r)

    def gt_mu_phase_likelihoods(
        self,
        idx_ind,
        genotype,
        reads_data_seq,
        reads_qualities,
        hSNP,  # mut hap h-0 or h-1
        idx_hSNP=None,
    ) -> float:
        r"""Reads probabilities given both germline and mosaic allele for bulk sample."""

        j_allele, k_allele = genotype  # germline allele
        r = []
        if j_allele == k_allele:  # hom
            for idx, _ in enumerate(reads_data_seq):
                r_a_probs = []
                for mo_allele in self.allele_pool[idx_ind]:  # mosaic allele
                    if mo_allele == j_allele:
                        continue
                    r_a_probs.append(
                        self.mu[idx_ind][(j_allele, mo_allele)]
                        + self.read_allele_prob(
                            reads_data_seq[idx],
                            mo_allele,
                            reads_qualities[idx],
                            "bulk",
                            hSNP[idx_hSNP],
                        )
                    )
                r_a_probs.append(  # germline allele
                    self.mu[idx_ind][(j_allele, j_allele)]
                    + self.read_allele_prob(
                        reads_data_seq[idx], j_allele, reads_qualities[idx], "bulk"
                    )
                )
                r.append(logsumexp(r_a_probs))
        else:  # het
            for idx, _ in enumerate(reads_data_seq):
                r_a_probs = []
                for mo_allele in self.allele_pool[idx_ind]:  # mosaic allele
                    if mo_allele in genotype:
                        continue
                    r_a_probs.append(
                        log(0.5)
                        + self.mu[idx_ind][(j_allele, mo_allele)]
                        + self.read_allele_prob(
                            reads_data_seq[idx],
                            mo_allele,
                            reads_qualities[idx],
                            "bulk",
                            hSNP[0],
                        )
                    )
                    r_a_probs.append(
                        log(0.5)
                        + self.mu[idx_ind][(k_allele, mo_allele)]
                        + self.read_allele_prob(
                            reads_data_seq[idx],
                            mo_allele,
                            reads_qualities[idx],
                            "bulk",
                            hSNP[1],
                        )
                    )
                r_a_probs.append(  # germline allele
                    log(0.5)
                    + self.mu[idx_ind][(j_allele, j_allele)]
                    + self.read_allele_prob(
                        reads_data_seq[idx],
                        j_allele,
                        reads_qualities[idx],
                        "bulk",
                        hSNP[0],
                    )
                )
                r_a_probs.append(  # germline allele
                    log(0.5)
                    + self.mu[idx_ind][(k_allele, k_allele)]
                    + self.read_allele_prob(
                        reads_data_seq[idx],
                        k_allele,
                        reads_qualities[idx],
                        "bulk",
                        hSNP[1],
                    )
                )
                r.append(logsumexp(r_a_probs))

        return sum(r)

    def gt_phase_likelihoods(
        self, idx_ind, genotype, reads_data_seq, reads_qualities, prot, af=0.5
    ):
        """phase genotype likelihoods for (un)phasable reads"""
        j_allele, k_allele = genotype
        # hom
        if j_allele == k_allele:
            r = [
                self.read_allele_prob(
                    reads_data_seq[idx], j_allele, reads_qualities[idx], prot
                )
                for idx in range(len(reads_data_seq))
            ]
        else:  # het
            r = []
            p1 = self.phase_probs[idx_ind][genotype]
            p1 = p1 if p1 < -1e-12 else -1e-12
            p1 = p1 if p1 > -1e12 else -1e12
            p2 = logsumexp([p1, 0], b=[-1, 1])
            for idx, _ in enumerate(reads_data_seq):
                r.append(
                    logsumexp(
                        [
                            log(af)
                            + p1
                            + self.read_allele_prob(
                                reads_data_seq[idx],
                                j_allele,
                                reads_qualities[idx],
                                prot,
                            ),
                            log(1 - af)
                            + p1
                            + self.read_allele_prob(
                                reads_data_seq[idx],
                                k_allele,
                                reads_qualities[idx],
                                prot,
                            ),
                            log(af)
                            + p2
                            + self.read_allele_prob(
                                reads_data_seq[idx],
                                k_allele,
                                reads_qualities[idx],
                                prot,
                            ),
                            log(1 - af)
                            + p2
                            + self.read_allele_prob(
                                reads_data_seq[idx],
                                j_allele,
                                reads_qualities[idx],
                                prot,
                            ),
                        ]
                    )
                )
        return sum(r)

    def gt_mu_likelihoods(
        self,
        genotype,
        reads_data_seq,
        reads_qualities,
        idx_ind,
    ) -> float:
        r"""Reads probabilities given both germline and mosaic allele for bulk sample."""

        j_allele, k_allele = genotype  # germline allele
        r = []

        # hom
        if j_allele == k_allele:
            for idx, _ in enumerate(reads_data_seq):
                r_a_probs = []  # for a read

                for mo_allele in self.allele_pool[idx_ind]:  # mosaic allele
                    if mo_allele == j_allele:
                        continue
                    r_a_probs.append(
                        self.mu[idx_ind][(j_allele, mo_allele)]
                        + self.read_allele_prob(
                            reads_data_seq[idx], mo_allele, reads_qualities[idx], "bulk"
                        )
                    )
                r_a_probs.append(  # germline allele
                    self.mu[idx_ind][(j_allele, j_allele)]
                    + self.read_allele_prob(
                        reads_data_seq[idx], j_allele, reads_qualities[idx], "bulk"
                    )
                )
                r.append(logsumexp(r_a_probs))

        # het
        else:
            for idx, _ in enumerate(reads_data_seq):
                r_a_probs = []  # for a read

                for mo_allele in self.allele_pool[idx_ind]:  # mosaic allele
                    if mo_allele in genotype:
                        continue
                    r_a_probs.append(
                        log(0.5)
                        + self.mu[idx_ind][(j_allele, mo_allele)]
                        + self.read_allele_prob(
                            reads_data_seq[idx], mo_allele, reads_qualities[idx], "bulk"
                        )
                    )
                    r_a_probs.append(
                        log(0.5)
                        + self.mu[idx_ind][(k_allele, mo_allele)]
                        + self.read_allele_prob(
                            reads_data_seq[idx], mo_allele, reads_qualities[idx], "bulk"
                        )
                    )
                r_a_probs.append(  # germline allele
                    log(0.5)
                    + self.mu[idx_ind][(j_allele, j_allele)]
                    + self.read_allele_prob(
                        reads_data_seq[idx], j_allele, reads_qualities[idx], "bulk"
                    )
                )
                r_a_probs.append(  # germline allele
                    log(0.5)
                    + self.mu[idx_ind][(k_allele, k_allele)]
                    + self.read_allele_prob(
                        reads_data_seq[idx], k_allele, reads_qualities[idx], "bulk"
                    )
                )
                r.append(logsumexp(r_a_probs))

        return sum(r)

    def gt_sc_posteriors(self, gt_sc, gt_germ, idx_ind, idx_sample) -> float:
        """Single cell sample's joint genotype posteriors."""
        # idx_sample 0 is bulk, starting from 1 actually
        gt_sc_posteriors = self.get_gt_likelihoods(idx_ind, idx_sample, gt_sc)
        if gt_sc == gt_germ:
            return (
                gt_sc_posteriors
                + (
                    log(2)
                    if ((not self.phasable) or (not self.phasable[idx_ind]))
                    else 0
                )
                + self.mu[idx_ind][(gt_germ[0], gt_sc[0])]
                + self.mu[idx_ind][(gt_germ[1], gt_sc[1])]
            )
        elif gt_sc == (gt_germ[1], gt_germ[0]):
            return (
                gt_sc_posteriors
                + (
                    log(2)
                    if ((not self.phasable) or (not self.phasable[idx_ind]))
                    else 0
                )
                + self.mu[idx_ind][(gt_germ[0], gt_sc[1])]
                + self.mu[idx_ind][(gt_germ[0], gt_sc[1])]
            )
        elif (gt_germ[0] == gt_germ[1]) and (gt_germ[0] in gt_sc):  # jj->jk
            mut_allele = gt_sc[0] if gt_sc[0] != gt_germ[0] else gt_sc[1]
            return (
                gt_sc_posteriors
                + (
                    log(2)
                    if ((not self.phasable) or (not self.phasable[idx_ind]))
                    else 0
                )
                + self.mu[idx_ind][(gt_germ[0], gt_germ[0])]
                + self.mu[idx_ind][(gt_germ[0], mut_allele)]
            )
        elif len(tuple(set(gt_sc) & set(gt_germ))) == 1:  # jm->km
            m_allele = tuple(set(gt_sc) & set(gt_germ))[0]
            j_allele = gt_germ[0] if gt_germ[0] != m_allele else gt_germ[1]
            k_allele = gt_sc[0] if gt_sc[0] != m_allele else gt_sc[1]
            return (
                gt_sc_posteriors
                + self.mu[idx_ind][(j_allele, k_allele)]
                + self.mu[idx_ind][(m_allele, m_allele)]
            )
        else:
            return float("-inf")

    def gt_sc_posteriors_final(
        self,
        idx_ind,
        idx_sample,
        gt_germ,
        gt_mut,
        ref_allele,
        germ_allele,
        mosaic_allele,
    ):
        """Single cell sample's joint genotype posteriors."""
        # idx_sample 0 is bulk, starting from 1
        gt_sc_posteriors_germ = (
            self.get_gt_germ_priors(idx_ind, gt_germ)
            + self.get_gt_likelihoods(idx_ind, idx_sample, gt_germ)
            + log(
                self.mu[idx_ind][(germ_allele, germ_allele)]
                * self.mu[idx_ind][(ref_allele, ref_allele)]
            )
        )
        gt_sc_posteriors_mut = (
            self.get_gt_germ_priors(idx_ind, gt_germ)
            + self.get_gt_likelihoods(idx_ind, idx_sample, gt_mut)
            + log(
                self.mu[idx_ind][(germ_allele, mosaic_allele)]
                * self.mu[idx_ind][(ref_allele, ref_allele)]
            )
        )
        # print(idx_sample, gt_sc_posteriors_germ, gt_sc_posteriors_mut, sep="\n")
        return tools.normalize_log([gt_sc_posteriors_germ, gt_sc_posteriors_mut])

    def get_gt_likelihoods(self, idx_ind, idx_sample, genotype) -> float:
        r"""Get the genotype(s) likelihood given the sample for an individual."""
        # simple form genotype (e.g. [j, k])
        idx_gt = self.genotypes[idx_ind].index(genotype)
        return self.sample_likelihoods[idx_ind][idx_sample][idx_gt]

    def get_mu_likelihoods(self, idx_ind, genotype):
        idx_gt = self.genotypes[idx_ind].index(genotype)
        return self.bulk_mu_likelihoods[idx_ind][idx_gt]

    # def get_sc_posteriors(self, idx_ind, genotype) -> float:
    #     idx_gt = self.genotypes[idx_ind].index(genotype)
    #     sc_posteriors = [lls[idx_gt] for lls in self.sample_posteriors[idx_ind]]
    #     return logsumexp(sc_posteriors)

    # def get_sc_likelihoods(self, idx_ind, genotype) -> float:
    #     idx_gt = self.genotypes[idx_ind].index(genotype)
    #     sc_likelihoods = [lls[idx_gt] for lls in self.sample_likelihoods[idx_ind][1:]]
    #     return logsumexp(sc_likelihoods)

    # def get_sample_posteriors(self, idx_ind, idx_sample, genotype) -> float:
    #     """Get the sample genotype(s) posteriors for a sample in an individual."""
    #     idx_gt = self.genotypes[idx_ind].index(genotype)
    #     return self.sample_posteriors[idx_ind][idx_sample][idx_gt]

    def get_sc_posteriors(self, idx_ind, idx_sample, gt_germ, gt_sc) -> float:
        """Joint"""
        idx_gt_germ = self.genotypes[idx_ind].index(gt_germ)
        idx_gt_sc = self.genotypes[idx_ind].index(gt_sc)
        return self.sc_posteriors[idx_ind][idx_gt_germ][idx_sample][idx_gt_sc]

    def gt_posteriors(self, idx_ind, idx_sample, gt_sc):
        """Single cell sample posteriors"""
        gt_posteriors = []
        for gt_germ in self.genotypes[idx_ind]:
            gt_posteriors.append(
                self.get_sc_posteriors(idx_ind, idx_sample, gt_germ, gt_sc)
                + self.get_gt_germ_priors(idx_ind, gt_germ)
            )
        return logsumexp(gt_posteriors)

    # @pysnooper.snoop("germ.log")
    def gt_germ_priors(self, gt_germ, idx_ind, mu_based=False) -> float:
        r"""Germline genotype priors."""

        j_allele, k_allele = gt_germ
        # print(j_allele, k_allele, j_allele == k_allele)
        if j_allele != k_allele:  # het
            # pop freq==0
            if self.freq.get(j_allele) == 0 or self.freq.get(k_allele) == 0:
                return float("-inf")
            gt_germ_priors = (
                log(2) + log(self.freq.get(j_allele)) + log(self.freq.get(k_allele))
                # + (log(2) if not self.phasable else 0)
            )
            # assume 0 is bulk
            if mu_based:
                gt_germ_priors += self.get_mu_likelihoods(idx_ind, gt_germ)  # se mode
            else:
                # sp mode
                gt_germ_priors += self.get_gt_likelihoods(idx_ind, 0, gt_germ)
            # assume rest is sc
            for idx_sample in range(1, len(self.sample_likelihoods[idx_ind])):
                cell_gts = []
                for gt_sc in self.genotypes[idx_ind]:
                    # ADO
                    if gt_sc == (j_allele, j_allele) or gt_sc == (k_allele, k_allele):
                        continue
                    if j_allele in gt_sc and k_allele not in gt_sc:  # [j / non_k] gt
                        non_k_allele = gt_sc[1] if gt_sc[1] != j_allele else gt_sc[0]
                        cell_gts.append(
                            self.mu[idx_ind].get((k_allele, non_k_allele))
                            + self.mu[idx_ind].get((j_allele, j_allele))
                            + self.get_gt_likelihoods(idx_ind, idx_sample, gt_sc)
                        )
                    elif k_allele in gt_sc and j_allele not in gt_sc:  # [non_j / k] gt
                        non_j_allele = gt_sc[0] if gt_sc[0] != k_allele else gt_sc[1]
                        cell_gts.append(
                            self.mu[idx_ind].get((j_allele, non_j_allele))
                            + self.mu[idx_ind].get((k_allele, k_allele))
                            + self.get_gt_likelihoods(idx_ind, idx_sample, gt_sc)
                        )
                # [j, k]
                cell_gts.append(
                    self.mu[idx_ind].get((j_allele, j_allele))
                    + self.mu[idx_ind].get((k_allele, k_allele))
                    + self.get_gt_likelihoods(idx_ind, idx_sample, gt_germ)
                )
                gt_germ_priors += logsumexp(cell_gts)
        else:  # hom
            if self.freq.get(j_allele) == 0:
                return float("-inf")
            gt_germ_priors = 2 * log(self.freq.get(j_allele))
            # assume 0 is bulk
            if mu_based:
                gt_germ_priors += self.get_mu_likelihoods(idx_ind, gt_germ)  # se mode
            else:
                # sp mode
                gt_germ_priors += self.get_gt_likelihoods(idx_ind, 0, gt_germ)
            # assume rest is sc
            for idx_sample in range(1, len(self.sample_likelihoods[idx_ind])):
                # print(idx_sample)
                cell_gts = []
                # [j / non_j]
                for gt_sc in self.genotypes[idx_ind]:
                    if j_allele in gt_sc and gt_sc[0] != gt_sc[1]:  # [j / non_k] gt
                        non_j_allele = gt_sc[1] if gt_sc[1] != j_allele else gt_sc[0]
                        cell_gts.append(
                            log(2)
                            + self.mu[idx_ind].get((j_allele, j_allele))
                            + self.mu[idx_ind].get((j_allele, non_j_allele))
                            + self.get_gt_likelihoods(idx_ind, idx_sample, gt_sc)
                        )
                # [j, j]
                cell_gts.append(
                    # log(2 * exp(self.mu[idx_ind].get((j_allele, j_allele))) - 1)
                    self.mu[idx_ind].get((j_allele, j_allele)) * 2
                    + self.get_gt_likelihoods(idx_ind, idx_sample, gt_germ)
                )
                gt_germ_priors += logsumexp(cell_gts)
        return gt_germ_priors

    def get_gt_germ_priors(self, idx_ind, genotype, mu_based=False, norm=True) -> float:
        """Get the germline genotype(s) priors for an individual."""

        idx_gt = self.genotypes[idx_ind].index(genotype)
        if norm:
            if mu_based or self.mode == "se":  # for mu bulk
                priors = self.mu_germ_priors[idx_ind][idx_gt]
            else:
                priors = self.germ_priors[idx_ind][idx_gt]
        else:  # for LR test
            if mu_based or self.mode == "se":  # for mu bulk
                priors = self.mu_germ_priors_unnorm[idx_ind][idx_gt]
            else:
                priors = self.germ_priors_unnorm[idx_ind][idx_gt]
        return priors

    def get_allele_germ_priors(self, idx_ind, allele) -> float:
        """Used for calculation of allele freqency in M-step"""
        priors = []
        if self.mode == "se":
            for idx_gt, gt in enumerate(self.genotypes[idx_ind]):
                if allele in gt:
                    if gt[0] == gt[1]:  # hom
                        priors.append(self.mu_germ_priors[idx_ind][idx_gt] + log(2))
                    else:
                        priors.append(self.mu_germ_priors[idx_ind][idx_gt])
        else:
            for idx_gt, gt in enumerate(self.genotypes[idx_ind]):
                if allele in gt:
                    if gt[0] == gt[1]:  # hom
                        priors.append(self.germ_priors[idx_ind][idx_gt] + log(2))
                    else:
                        priors.append(self.germ_priors[idx_ind][idx_gt])
        if priors:
            return logsumexp(priors)
        else:
            return float("-inf")

    # @pysnooper.snoop("phasing_2.log", normalize=True, overwrite=False)
    def phasing_prob(self, idx_ind, genotype):
        r"""Phasing probabilities"""
        j_allele, k_allele = genotype
        phase_gt = (
            genotype
            if genotype in self.phase_probs[idx_ind].keys()
            else (k_allele, j_allele)
        )
        mut_jj_cell_list = [
            self.gt_likelihoods(
                (j_allele, j_allele),
                self.reads_data_seq[idx_ind][1][idx_sample],
                self.reads_data_qual[idx_ind][1][idx_sample],
                self.sc_prot_list[idx_ind][idx_sample],
            )
            < self.gt_likelihoods(
                (j_allele, k_allele),
                self.reads_data_seq[idx_ind][1][idx_sample],
                self.reads_data_qual[idx_ind][1][idx_sample],
                self.sc_prot_list[idx_ind][idx_sample],
            )
            for idx_sample in range(self.num_sc_sample[idx_ind])
        ]
        mut_kk_cell_list = [
            self.gt_likelihoods(
                (k_allele, k_allele),
                self.reads_data_seq[idx_ind][1][idx_sample],
                self.reads_data_qual[idx_ind][1][idx_sample],
                self.sc_prot_list[idx_ind][idx_sample],
            )
            < self.gt_likelihoods(
                (j_allele, k_allele),
                self.reads_data_seq[idx_ind][1][idx_sample],
                self.reads_data_qual[idx_ind][1][idx_sample],
                self.sc_prot_list[idx_ind][idx_sample],
            )
            for idx_sample in range(self.num_sc_sample[idx_ind])
        ]
        # mut_jj_cell_list = [True] * self.num_sc_sample[idx_ind]
        # mut_kk_cell_list = [True] * self.num_sc_sample[idx_ind]
        jj = (
            self.phase_probs[idx_ind][phase_gt]
            + self.get_gt_germ_priors(idx_ind, (j_allele, j_allele), self.mode == "se")
            + self.gt_likelihoods(
                (j_allele, j_allele),
                self.reads_data_spn_seq[idx_ind][0],
                self.reads_data_spn_qual[idx_ind][0],
                "bulk",
                self.hSNP[idx_ind],
            )
            if self.mode == "sp"
            else self.gt_mu_phase_likelihoods(
                idx_ind,
                (j_allele, j_allele),
                self.reads_data_spn_seq[idx_ind][0],
                self.reads_data_spn_qual[idx_ind][0],
                self.hSNP[idx_ind],
                1,
            )
        )
        # + 2 * log(self.freq.get(j_allele))
        # + self.gt_likelihoods(
        #     (j_allele, j_allele),
        #     self.reads_data_spn_seq[idx_ind][0],
        #     self.reads_data_spn_qual[idx_ind][0],
        #     "bulk",
        #     self.hSNP[idx_ind],
        # )
        jk = (
            self.phase_probs[idx_ind][phase_gt]
            + self.get_gt_germ_priors(idx_ind, phase_gt, self.mode == "se")
            + self.gt_likelihoods(
                (j_allele, k_allele),
                self.reads_data_spn_seq[idx_ind][0],
                self.reads_data_spn_qual[idx_ind][0],
                "bulk",
                self.hSNP[idx_ind],
            )
            if self.mode == "sp"
            else self.gt_mu_phase_likelihoods(
                idx_ind,
                (j_allele, k_allele),
                self.reads_data_spn_seq[idx_ind][0],
                self.reads_data_spn_qual[idx_ind][0],
                self.hSNP[idx_ind],
            )
        )
        # + log(2)
        # + log(self.freq.get(j_allele))
        # + log(self.freq.get(k_allele))
        # + self.gt_likelihoods(
        #     (j_allele, k_allele),
        #     self.reads_data_spn_seq[idx_ind][0],
        #     self.reads_data_spn_qual[idx_ind][0],
        #     "bulk",
        #     self.hSNP[idx_ind],
        # )
        kk = (
            self.phase_probs[idx_ind][phase_gt]
            + self.get_gt_germ_priors(idx_ind, (k_allele, k_allele), self.mode == "se")
            + self.gt_likelihoods(
                (k_allele, k_allele),
                self.reads_data_spn_seq[idx_ind][0],
                self.reads_data_spn_qual[idx_ind][0],
                "bulk",
                self.hSNP[idx_ind],
            )
            if self.mode == "sp"
            else self.gt_mu_phase_likelihoods(
                idx_ind,
                (k_allele, k_allele),
                self.reads_data_spn_seq[idx_ind][0],
                self.reads_data_spn_qual[idx_ind][0],
                self.hSNP[idx_ind],
                0,
            )
        )
        # + 2 * log(self.freq.get(k_allele))
        # + self.gt_likelihoods(
        #     (k_allele, k_allele),
        #     self.reads_data_spn_seq[idx_ind][0],
        #     self.reads_data_spn_qual[idx_ind][0],
        #     "bulk",
        #     self.hSNP[idx_ind],
        # )
        jj_ = self.gt_likelihoods(
            (j_allele, j_allele),
            self.reads_data_spn_seq[idx_ind][0],
            self.reads_data_spn_qual[idx_ind][0],
            "bulk",
        )
        jk_ = self.gt_likelihoods(
            genotype,
            self.reads_data_spn_seq[idx_ind][0],
            self.reads_data_spn_qual[idx_ind][0],
            "bulk",
        )
        kk_ = self.gt_likelihoods(
            (k_allele, k_allele),
            self.reads_data_spn_seq[idx_ind][0],
            self.reads_data_spn_qual[idx_ind][0],
            "bulk",
        )
        for idx_sample in range(self.num_sc_sample[idx_ind]):
            if mut_jj_cell_list[idx_sample]:
                jj += logsumexp(
                    [
                        self.gt_likelihoods(
                            (j_allele, k_allele),
                            self.reads_data_spn_seq[idx_ind][1][idx_sample],
                            self.reads_data_spn_qual[idx_ind][1][idx_sample],
                            self.sc_prot_list[idx_ind][idx_sample],
                            self.hSNP[idx_ind],
                            self.af[idx_ind][idx_sample],
                        )
                        + self.mu[idx_ind].get((j_allele, j_allele))
                        + self.mu[idx_ind].get((j_allele, k_allele)),
                        self.gt_likelihoods(
                            (j_allele, j_allele),
                            self.reads_data_spn_seq[idx_ind][1][idx_sample],
                            self.reads_data_spn_qual[idx_ind][1][idx_sample],
                            self.sc_prot_list[idx_ind][idx_sample],
                            self.hSNP[idx_ind],
                            self.af[idx_ind][idx_sample],
                        )
                        + self.mu[idx_ind].get((j_allele, j_allele))
                        + self.mu[idx_ind].get((j_allele, j_allele)),
                    ]
                )
            jk += (
                self.gt_likelihoods(
                    genotype,
                    self.reads_data_spn_seq[idx_ind][1][idx_sample],
                    self.reads_data_spn_qual[idx_ind][1][idx_sample],
                    self.sc_prot_list[idx_ind][idx_sample],
                    self.hSNP[idx_ind],
                    self.af[idx_ind][idx_sample],
                )
                + self.mu[idx_ind].get((j_allele, j_allele))
                + self.mu[idx_ind].get((k_allele, k_allele))
            )
            if mut_kk_cell_list[idx_sample]:
                kk += logsumexp(
                    [
                        self.gt_likelihoods(
                            genotype,
                            self.reads_data_spn_seq[idx_ind][1][idx_sample],
                            self.reads_data_spn_qual[idx_ind][1][idx_sample],
                            self.sc_prot_list[idx_ind][idx_sample],
                            self.hSNP[idx_ind],
                            self.af[idx_ind][idx_sample],
                        )
                        + self.mu[idx_ind].get((k_allele, j_allele))
                        + self.mu[idx_ind].get((k_allele, k_allele)),
                        self.gt_likelihoods(
                            (k_allele, k_allele),
                            self.reads_data_spn_seq[idx_ind][1][idx_sample],
                            self.reads_data_spn_qual[idx_ind][1][idx_sample],
                            self.sc_prot_list[idx_ind][idx_sample],
                            self.hSNP[idx_ind],
                            self.af[idx_ind][idx_sample],
                        )
                        + self.mu[idx_ind].get((k_allele, k_allele))
                        + self.mu[idx_ind].get((k_allele, k_allele)),
                    ]
                )
            if mut_jj_cell_list[idx_sample]:
                jj_ += logsumexp(
                    [
                        self.gt_likelihoods(
                            genotype,
                            self.reads_data_spn_seq[idx_ind][1][idx_sample],
                            self.reads_data_spn_qual[idx_ind][1][idx_sample],
                            self.sc_prot_list[idx_ind][idx_sample],
                        )
                        + self.mu[idx_ind].get((j_allele, k_allele))
                        + self.mu[idx_ind].get((j_allele, j_allele)),
                        self.gt_likelihoods(
                            (j_allele, j_allele),
                            self.reads_data_spn_seq[idx_ind][1][idx_sample],
                            self.reads_data_spn_qual[idx_ind][1][idx_sample],
                            self.sc_prot_list[idx_ind][idx_sample],
                        )
                        + self.mu[idx_ind].get((j_allele, j_allele))
                        + self.mu[idx_ind].get((j_allele, j_allele)),
                    ]
                )
            jk_ += (
                self.gt_likelihoods(
                    genotype,
                    self.reads_data_spn_seq[idx_ind][1][idx_sample],
                    self.reads_data_spn_qual[idx_ind][1][idx_sample],
                    self.sc_prot_list[idx_ind][idx_sample],
                )
                + self.mu[idx_ind].get((k_allele, k_allele))
                + self.mu[idx_ind].get((j_allele, j_allele))
            )
            if mut_kk_cell_list[idx_sample]:
                kk_ += logsumexp(
                    [
                        self.gt_likelihoods(
                            genotype,
                            self.reads_data_spn_seq[idx_ind][1][idx_sample],
                            self.reads_data_spn_qual[idx_ind][1][idx_sample],
                            self.sc_prot_list[idx_ind][idx_sample],
                        )
                        + self.mu[idx_ind].get((k_allele, k_allele))
                        + self.mu[idx_ind].get((k_allele, j_allele)),
                        self.gt_likelihoods(
                            (k_allele, k_allele),
                            self.reads_data_spn_seq[idx_ind][1][idx_sample],
                            self.reads_data_spn_qual[idx_ind][1][idx_sample],
                            self.sc_prot_list[idx_ind][idx_sample],
                        )
                        + self.mu[idx_ind].get((k_allele, k_allele))
                        + self.mu[idx_ind].get((k_allele, k_allele)),
                    ]
                )
        # gi_jj_lls = [
        #     tools.normalize_log(
        #         (
        #             self.gt_likelihoods(
        #                 (j_allele, j_allele),
        #                 self.reads_data_seq[idx_ind][1][idx_sample],
        #                 self.reads_data_qual[idx_ind][1][idx_sample],
        #                 self.sc_prot_list[idx_ind][idx_sample],
        #             ),
        #             self.gt_likelihoods(
        #                 (j_allele, k_allele),
        #                 self.reads_data_seq[idx_ind][1][idx_sample],
        #                 self.reads_data_qual[idx_ind][1][idx_sample],
        #                 self.sc_prot_list[idx_ind][idx_sample],
        #             ),
        #         )
        #     )
        #     for idx_sample in range(self.num_sc_sample[idx_ind])
        # ]
        # gi_kk_lls = [
        #     tools.normalize_log(
        #         (
        #             self.gt_likelihoods(
        #                 (k_allele, k_allele),
        #                 self.reads_data_seq[idx_ind][1][idx_sample],
        #                 self.reads_data_qual[idx_ind][1][idx_sample],
        #                 self.sc_prot_list[idx_ind][idx_sample],
        #             ),
        #             self.gt_likelihoods(
        #                 (j_allele, k_allele),
        #                 self.reads_data_seq[idx_ind][1][idx_sample],
        #                 self.reads_data_qual[idx_ind][1][idx_sample],
        #                 self.sc_prot_list[idx_ind][idx_sample],
        #             ),
        #         )
        #     )
        #     for idx_sample in range(self.num_sc_sample[idx_ind])
        # ]
        # if genotype == (
        #     ("TAATT", "AAAAGGAAAAAAAAGGAAGAAAGAAAAAGAAAAGCAAAAAGAAAAA", "TTGTA"),
        #     ("TAATT", "AAAAGGAAAAAAAAGGAAGAAAGAAAAAGAAAAGCAAAAAGAAAAA", "ATTGT"),
        # ):
        #     print("jj")
        #     print(mut_jj_cell_list)
        #     print(gi_jj_lls)
        #     print([jj, jk, kk])
        #     print([jj_, jk_, kk_])
        #     print([jj - jj_, jk - jk_, kk - kk_])
        #     breakpoint()
        # if genotype == (
        #     ("TAATT", "AAAAGGAAAAAAAAGGAAGAAAGAAAAAGAAAAGCAAAAAGAAAAA", "ATTGT"),
        #     ("TAATT", "AAAAGGAAAAAAAAGGAAGAAAGAAAAAGAAAAGCAAAAAGAAAAA", "TTGTA"),
        # ):
        #     print("kk")
        #     print(mut_kk_cell_list)
        #     print(gi_kk_lls)
        #     print([jj, jk, kk])
        #     print([jj_, jk_, kk_])
        #     print([jj - jj_, jk - jk_, kk - kk_])
        #     breakpoint()
        return [jj - jj_, jk - jk_, kk - kk_]

    def output_collate(self):
        # gt output
        # germline_gts = []  # used only for mu_mosaic_calc
        self.gts = []
        self.gt_lls = []
        self.gt_posts = []
        for idx_ind, ind in enumerate(self.sample_posteriors):
            gts_ind = []
            gt_lls_ind = []
            gt_posts_ind = []

            # germline
            if not self.germ_priors[idx_ind]:
                self.gts.append([])
                self.gt_lls.append([])
                self.gt_posts.append([])
                continue
            if self.mode == "sp":
                max_val, max_idx = max(
                    (val, idx) for idx, val in enumerate(self.germ_priors[idx_ind])
                )
            else:
                max_val, max_idx = max(
                    (val, idx) for idx, val in enumerate(self.mu_germ_priors[idx_ind])
                )
            gt_lls_ind.append(exp(self.sample_likelihoods[idx_ind][0][max_idx]))
            allele_1, allele_2 = self.genotypes[idx_ind][max_idx]
            gt_posts_ind.append(exp(max_val))
            gts_ind.append(
                (
                    self.total_allele_pool.index(allele_1),
                    self.total_allele_pool.index(allele_2),
                )
            )

            # sc
            for idx_sample, sample in enumerate(ind, start=1):
                gt_qual = max(sample)
                max_idx = sample.index(gt_qual)
                gt_posts_ind.append(exp(gt_qual))
                gt_lls_ind.append(
                    exp(self.sample_likelihoods[idx_ind][idx_sample][max_idx])
                )
                allele_1, allele_2 = self.genotypes[idx_ind][max_idx]
                gts_ind.append(
                    (
                        self.total_allele_pool.index(allele_1),
                        self.total_allele_pool.index(allele_2),
                    )
                )
            self.gts.append(gts_ind)
            self.gt_lls.append(gt_lls_ind)
            self.gt_posts.append(gt_posts_ind)

    def explicit_collate(self, idx_ind, idx_sample):
        ll_dict = {
            f"{self.total_allele_pool.index(gt[0])}/{self.total_allele_pool.index(gt[1])}": f"{exp(self.sample_likelihoods[idx_ind][idx_sample][self.genotypes[idx_ind].index(gt)]):.3f}"
            for gt in self.genotypes[idx_ind]
        }
        ll_dict = dict(sorted(ll_dict.items(), key=lambda item: item[1], reverse=True))
        posts_dict = (
            {
                f"{self.total_allele_pool.index(gt[0])}/{self.total_allele_pool.index(gt[1])}": f"{exp(self.sample_posteriors[idx_ind][idx_sample-1][self.genotypes[idx_ind].index(gt)]):.3f}"
                for gt in self.genotypes[idx_ind]
            }
            if idx_sample
            else {
                f"{self.total_allele_pool.index(gt[0])}/{self.total_allele_pool.index(gt[1])}": f"{exp(self.germ_priors[idx_ind][self.genotypes[idx_ind].index(gt)]):.3f}"
                for gt in self.genotypes[idx_ind]
            }
        )
        posts_dict = dict(
            sorted(posts_dict.items(), key=lambda item: item[1], reverse=True)
        )
        # print(ll_dict, posts_dict, sep="\t")
        # breakpoint()
        return ll_dict, posts_dict

        # mu_mosaic_calc
        # self.mu_mosaic = []
        # max_mus = []
        # for idx_ind, germline in enumerate(germline_gts):
        #     allele_1, allele_2 = germline
        #     if allele_1 == allele_2:  # germline event is hom
        #         mosaic_mu = float("-inf")
        #         target_allele = None
        #         for b in self.allele_pool[idx_ind]:
        #             if allele_1 == b:
        #                 continue
        #             if self.mu[idx_ind][(allele_1, b)] > mosaic_mu:
        #                 mosaic_mu = self.mu[idx_ind][(allele_1, b)]
        #                 target_allele = b
        #         if (
        #             target_allele != None
        #             and 10e-4 < self.mu[idx_ind][(allele_1, target_allele)] < 0.5
        #         ):  # mu threshold
        #             self.mu_mosaic.append(
        #                 "{}->{}({:.3f})".format(
        #                     self.total_allele_pool.index(allele_1),
        #                     self.total_allele_pool.index(target_allele),
        #                     self.mu[idx_ind][(allele_1, target_allele)],
        #                 )
        #             )
        #             max_mus.append((allele_1, target_allele))
        #         else:
        #             self.mu_mosaic.append("NA")
        #             max_mus.append("NA")
        #     else:
        #         self.mu_mosaic.append("NA")
        #         max_mus.append("NA")

        # # mosaic_calling
        # # DP-based mosaic posterior calc
        # self.mosaic_posteriors = []
        # for idx_ind, mut in enumerate(max_mus):
        #     if mut != "NA":
        #         mosaic_p = 0
        #         j_allele, k_allele = mut
        #         l_hom = [
        #             self.get_sample_posteriors(
        #                 idx_ind, idx_sample + 1, (j_allele, j_allele)
        #             )
        #             for idx_sample in range(self.num_sc_sample[idx_ind])
        #         ]
        #         if (j_allele, k_allele) in self.genotypes[idx_ind]:
        #             l_het = [
        #                 self.get_sample_posteriors(
        #                     idx_ind, idx_sample + 1, (j_allele, k_allele)
        #                 )
        #                 for idx_sample in range(self.num_sc_sample[idx_ind])
        #             ]
        #         else:
        #             l_het = [
        #                 self.get_sample_posteriors(
        #                     idx_ind, idx_sample + 1, (k_allele, j_allele)
        #                 )
        #                 for idx_sample in range(self.num_sc_sample[idx_ind])
        #             ]
        #         for a, b in zip(l_hom, l_het):
        #             mosaic_p += a - logsumexp([a, b])
        #         self.mosaic_posteriors.append("{:.3f}".format(1 - exp(mosaic_p)))
        #     else:
        #         self.mosaic_posteriors.append("NA")

    @staticmethod
    def mut_cell_prob(L, num_cell, mu):
        return comb(num_cell, L) * mu**L * (1 - mu) ** (num_cell - L)

    def mosaic_posteriors(self):
        """Collate results + DP-based + Bayes' mosaic posteriors"""
        self.site_type = []
        self.mo_posteriori = []  # raw data
        self.mo_posteriors = []  # final values
        # a list of tuples (ref_allele, mo_allele, mu_value, gt_germ, gt_mo)
        self.mu_mosaic = []
        self.sc_posteriors_final = []
        self.mut_cell_list_final = []
        self.gts = []
        self.gt_lls = []
        self.gt_posts = []
        for idx_ind in range(self.num_ind):
            # determine germline and mosaic
            gts_ind = []
            gt_lls_ind = []
            gt_posts_ind = []
            if self.mode == "sp":
                max_val, max_idx = max(
                    (val, idx) for idx, val in enumerate(self.germ_priors[idx_ind])
                )
            else:
                max_val, max_idx = max(
                    (val, idx) for idx, val in enumerate(self.mu_germ_priors[idx_ind])
                )
            gt_lls_ind.append(exp(self.sample_likelihoods[idx_ind][0][max_idx]))
            j_germ, k_germ = self.genotypes[idx_ind][max_idx]
            gt_posts_ind.append(exp(max_val))
            gts_ind.append(
                (
                    self.total_allele_pool.index(j_germ),
                    self.total_allele_pool.index(k_germ),
                )
            )
            self.site_type.append("no_type")
            # if not self.gts[idx_ind]:
            #     self.mo_posteriori.append("NA")
            #     self.mo_posteriors.append(0)
            #     self.mu_mosaic.append((None, None, None, None, None, None))
            #     continue

            # find greatest germ -> mosaic rate
            mosaic_mut_rate = float("-inf")  # mosaic rate
            ref_g_allele = None  # reference germ allele
            mu_g_allele = None  # mutant germ allele
            mu_m_allele = None  # mosaic allele
            for g_allele in {j_germ, k_germ}:
                for m_allele in self.allele_pool[idx_ind]:
                    if self.mu[idx_ind][(g_allele, m_allele)] > mosaic_mut_rate:
                        if m_allele not in (j_germ, k_germ):
                            mosaic_mut_rate = self.mu[idx_ind][(g_allele, m_allele)]
                            mu_g_allele = g_allele
                            mu_m_allele = m_allele
            # set mu
            if mosaic_mut_rate < 1e-12:
                mosaic_mut_rate = 0
            # DP-based mosaic posteriors, log-space, output exp.
            if None in (mu_g_allele, mu_m_allele):
                self.mo_posteriori.append("NA")
                self.mo_posteriors.append(0)
                self.sc_posteriors_final.append(None)
                self.mut_cell_list_final.append(None)
                if j_germ == k_germ:
                    self.mu_mosaic.append(
                        (
                            j_germ,
                            None,
                            None,
                            (j_germ, j_germ),
                            None,
                            j_germ,
                        )
                    )
                    self.site_type[idx_ind] = "hom"
                else:
                    gt_germline = (
                        (j_germ, k_germ)
                        if (j_germ, k_germ) in self.genotypes[idx_ind]
                        else (k_germ, j_germ)
                    )
                    self.mu_mosaic.append(
                        (
                            (
                                j_germ
                                if self.freq.get(j_germ, 0) >= self.freq.get(k_germ, 0)
                                else k_germ
                            ),
                            (
                                k_germ
                                if self.freq.get(j_germ, 0) >= self.freq.get(k_germ, 0)
                                else j_germ
                            ),
                            0,
                            gt_germline,
                            None,
                            (
                                j_germ
                                if self.freq.get(j_germ, 0) >= self.freq.get(k_germ, 0)
                                else k_germ
                            ),
                        )
                    )
                    self.site_type[idx_ind] = "het"
                gts_ind.extend(
                    (
                        (
                            self.total_allele_pool.index(j_germ),
                            self.total_allele_pool.index(k_germ),
                        ),
                    )
                    * self.num_sc_sample[idx_ind]
                )
                gt_lls_ind.extend(
                    exp(self.sample_likelihoods[idx_ind][idx + 1][max_idx])
                    for idx in range(self.num_sc_sample[idx_ind])
                )
                gt_posts_ind.extend(
                    exp(self.sample_posteriors[idx_ind][idx][max_idx])
                    for idx in range(self.num_sc_sample[idx_ind])
                )
                self.gts.append(gts_ind)
                self.gt_lls.append(gt_lls_ind)
                self.gt_posts.append(gt_posts_ind)
                continue
            else:
                self.site_type[idx_ind] = "mosaic"

            ref_g_allele = j_germ if j_germ != mu_g_allele else k_germ
            gt_germline = (
                (j_germ, k_germ)
                if (j_germ, k_germ) in self.genotypes[idx_ind]
                else (k_germ, j_germ)
            )
            gt_mosaic = (
                (ref_g_allele, mu_m_allele)
                if (ref_g_allele, mu_m_allele) in self.genotypes[idx_ind]
                else (mu_m_allele, ref_g_allele)
            )
            # region final gt calc
            # gt_germ_priors = log(self.freq.get(j_germ) * self.freq.get(k_germ))
            # # assume 0 is bulk
            # if self.mode == "se":
            #     # se mode
            #     gt_germ_priors += self.get_mu_likelihoods(idx_ind, gt_germline)
            # else:
            #     # sp mode
            #     gt_germ_priors += self.get_gt_likelihoods(idx_ind, 0, gt_germline)
            # # assume rest is sc
            # for idx_sample in range(1, len(self.sample_likelihoods[idx_ind])):
            #     cell_gts = []
            #     # [j / non_j]
            #     cell_gts.append(
            #         log(2)
            #         + self.mu[idx_ind].get((mu_g_allele, mu_g_allele))
            #         + self.mu[idx_ind].get((mu_g_allele, mu_m_allele))
            #         + self.get_gt_likelihoods(idx_ind, idx_sample, gt_mosaic)
            #     )
            #     # [j, j]
            #     cell_gts.append(
            #         self.mu[idx_ind].get((mu_g_allele, mu_g_allele)) * 2
            #         + self.get_gt_likelihoods(idx_ind, idx_sample, gt_germline)
            #     )
            #     gt_germ_priors += logsumexp(cell_gts)
            # breakpoint()
            self.sc_posteriors_final.append(
                [
                    self.gt_sc_posteriors_final(
                        idx_ind,
                        idx_sample,
                        gt_germline,
                        gt_mosaic,
                        ref_g_allele,
                        mu_g_allele,
                        mu_m_allele,
                    )
                    for idx_sample in range(1, self.num_sc_sample[idx_ind] + 1)
                ]
            )
            self.mut_cell_list_final.append(
                [int(i[0] < i[1]) for i in self.sc_posteriors_final[idx_ind]]
                if self.sc_posteriors_final[idx_ind]
                else None
            )
            idx_gt_germline = self.genotypes[idx_ind].index(gt_germline)
            idx_gt_mosaic = self.genotypes[idx_ind].index(gt_mosaic)
            for idx_sample in range(self.num_sc_sample[idx_ind]):
                gt_lls_ind.append(
                    exp(
                        tools.normalize_log(
                            (
                                self.sample_likelihoods[idx_ind][idx_sample + 1][
                                    idx_gt_germline
                                ],
                                self.sample_likelihoods[idx_ind][idx_sample + 1][
                                    idx_gt_mosaic
                                ],
                            )
                        )[1]
                    )
                    # if self.mut_cell_list_final[idx_ind][idx_sample]
                    # else exp(
                    #     self.sample_likelihoods[idx_ind][idx_sample][idx_gt_germline]
                    # )
                )
                gt_posts_ind.append(
                    exp(self.sc_posteriors_final[idx_ind][idx_sample][1])
                    # if self.mut_cell_list_final[idx_ind][idx_sample]
                    # else exp(self.mut_cell_list_final[idx_ind][0])
                )
                gts_ind.append(
                    (
                        self.total_allele_pool.index(gt_mosaic[0]),
                        self.total_allele_pool.index(gt_mosaic[1]),
                    )
                    if self.mut_cell_list_final[idx_ind][idx_sample]
                    else (
                        self.total_allele_pool.index(gt_germline[0]),
                        self.total_allele_pool.index(gt_germline[1]),
                    )
                )
            self.gts.append(gts_ind)
            self.gt_lls.append(gt_lls_ind)
            self.gt_posts.append(gt_posts_ind)
            # endregion final gt calc

            self.mu_mosaic.append(
                (
                    mu_g_allele,
                    mu_m_allele,
                    mosaic_mut_rate,
                    gt_germline,
                    gt_mosaic,
                    ref_g_allele,
                )
            )

            # DP
            M = np.full(
                (self.num_sc_sample[idx_ind] + 1, self.num_sc_sample[idx_ind] + 1),
                float("-inf"),
            )  # l*j, ALT allele * num_cell, first col is padding 0, ignore
            # init
            M[0, 1] = self.get_gt_likelihoods(idx_ind, 1, gt_germline)
            M[1, 1] = self.get_gt_likelihoods(idx_ind, 1, gt_mosaic)
            # recursion
            for j_col in range(2, self.num_sc_sample[idx_ind] + 1):
                for l_row in range(0, j_col + 1):
                    M[l_row, j_col] = logsumexp(
                        [
                            M[l_row, j_col - 1]
                            + self.get_gt_likelihoods(idx_ind, j_col, gt_germline),
                            M[l_row - 1, j_col - 1]
                            + self.get_gt_likelihoods(idx_ind, j_col, gt_mosaic),
                        ]
                    )
            mo_lls = exp(M[:, -1])
            mo_comb = [
                comb(self.num_sc_sample[idx_ind], l)
                for l in range(self.num_sc_sample[idx_ind] + 1)
            ]
            mo_priors = [
                self.mut_cell_prob(l, self.num_sc_sample[idx_ind], mosaic_mut_rate)
                for l in range(self.num_sc_sample[idx_ind] + 1)
            ]
            mo_posteriori = tools.normalize(mo_lls * mo_priors / mo_comb)
            self.mo_posteriori.append(mo_posteriori)
            if all(v == 0 for v in mo_posteriori):
                self.mo_posteriors.append(0)
            else:
                self.mo_posteriors.append(1 - mo_posteriori[0])

    def mosaic_posteriors_refine(self):
        """Collate results + DP-based + Bayes' mosaic posteriors"""
        self.site_type = []
        self.mo_posteriori = []  # raw data
        self.mo_posteriors = []  # final values
        # a list of tuples (ref_allele, mo_allele, mu_value, gt_germ, gt_mo)
        self.mu_mosaic = []
        self.sc_posteriors_final = []
        self.mut_cell_list_final = []
        self.gts = []
        self.gt_lls = []
        self.gt_posts = []
        for idx_ind in range(self.num_ind):
            # determine germline and mosaic
            gts_ind = []
            gt_lls_ind = []
            gt_posts_ind = []
            if self.mode == "sp":
                max_val, max_idx = max(
                    (val, idx)
                    for idx, val in enumerate(self.germ_priors_refine[idx_ind])
                )
            else:
                max_val, max_idx = max(
                    (val, idx)
                    for idx, val in enumerate(self.mu_germ_priors_refine[idx_ind])
                )
            gt_lls_ind.append(exp(self.sample_likelihoods_refine[idx_ind][0][max_idx]))
            j_germ, k_germ = self.genotypes_refine[idx_ind][max_idx]
            gt_posts_ind.append(exp(max_val))
            gts_ind.append(
                (
                    self.total_allele_pool.index(j_germ),
                    self.total_allele_pool.index(k_germ),
                )
            )
            self.site_type.append("no_type")
            # if not self.gts[idx_ind]:
            #     self.mo_posteriori.append("NA")
            #     self.mo_posteriors.append(0)
            #     self.mu_mosaic.append((None, None, None, None, None, None))
            #     continue

            # find greatest germ -> mosaic rate
            mosaic_mut_rate = float("-inf")  # mosaic rate
            ref_g_allele = None  # reference germ allele
            mu_g_allele = None  # mutant germ allele
            mu_m_allele = None  # mosaic allele
            for g_allele in {j_germ, k_germ}:
                for m_allele in self.allele_pool_refine[idx_ind]:
                    if self.mu_refine[idx_ind][(g_allele, m_allele)] > mosaic_mut_rate:
                        if m_allele not in (j_germ, k_germ):
                            mosaic_mut_rate = self.mu_refine[idx_ind][
                                (g_allele, m_allele)
                            ]
                            mu_g_allele = g_allele
                            mu_m_allele = m_allele
            # set mu
            if mosaic_mut_rate < 1e-12:
                mosaic_mut_rate = 0
            # DP-based mosaic posteriors, log-space, output exp.
            if None in (mu_g_allele, mu_m_allele):
                self.mo_posteriori.append("NA")
                self.mo_posteriors.append(0)
                self.sc_posteriors_final.append(None)
                self.mut_cell_list_final.append(None)
                if j_germ == k_germ:
                    self.mu_mosaic.append(
                        (
                            j_germ,
                            None,
                            None,
                            (j_germ, j_germ),
                            None,
                            j_germ,
                        )
                    )
                    self.site_type[idx_ind] = "hom"
                else:
                    gt_germline = (
                        (j_germ, k_germ)
                        if (j_germ, k_germ) in self.genotypes_refine[idx_ind]
                        else (k_germ, j_germ)
                    )
                    self.mu_mosaic.append(
                        (
                            (
                                j_germ
                                if self.freq.get(j_germ, 0) >= self.freq.get(k_germ, 0)
                                else k_germ
                            ),
                            (
                                k_germ
                                if self.freq.get(j_germ, 0) >= self.freq.get(k_germ, 0)
                                else j_germ
                            ),
                            0,
                            gt_germline,
                            None,
                            (
                                j_germ
                                if self.freq.get(j_germ, 0) >= self.freq.get(k_germ, 0)
                                else k_germ
                            ),
                        )
                    )
                    self.site_type[idx_ind] = "het"
                gts_ind.extend(
                    (
                        (
                            self.total_allele_pool.index(j_germ),
                            self.total_allele_pool.index(k_germ),
                        ),
                    )
                    * self.num_sc_sample[idx_ind]
                )
                gt_lls_ind.extend(
                    exp(self.sample_likelihoods_refine[idx_ind][idx + 1][max_idx])
                    for idx in range(self.num_sc_sample[idx_ind])
                )
                gt_posts_ind.extend(
                    exp(self.sample_posteriors_refine[idx_ind][idx][max_idx])
                    for idx in range(self.num_sc_sample[idx_ind])
                )
                self.gts.append(gts_ind)
                self.gt_lls.append(gt_lls_ind)
                self.gt_posts.append(gt_posts_ind)
                continue
            else:
                self.site_type[idx_ind] = "mosaic"

            ref_g_allele = j_germ if j_germ != mu_g_allele else k_germ
            gt_germline = (
                (j_germ, k_germ)
                if (j_germ, k_germ) in self.genotypes_refine[idx_ind]
                else (k_germ, j_germ)
            )
            gt_mosaic = (
                (ref_g_allele, mu_m_allele)
                if (ref_g_allele, mu_m_allele) in self.genotypes_refine[idx_ind]
                else (mu_m_allele, ref_g_allele)
            )
            self.sc_posteriors_final.append(
                [
                    self.gt_sc_posteriors_final_refine(
                        idx_ind,
                        idx_sample,
                        gt_germline,
                        gt_mosaic,
                        ref_g_allele,
                        mu_g_allele,
                        mu_m_allele,
                    )
                    for idx_sample in range(1, self.num_sc_sample[idx_ind] + 1)
                ]
            )
            self.mut_cell_list_final.append(
                [int(i[0] < i[1]) for i in self.sc_posteriors_final[idx_ind]]
                if self.sc_posteriors_final[idx_ind]
                else None
            )
            idx_gt_germline = self.genotypes_refine[idx_ind].index(gt_germline)
            idx_gt_mosaic = self.genotypes_refine[idx_ind].index(gt_mosaic)
            for idx_sample in range(self.num_sc_sample[idx_ind]):
                gt_lls_ind.append(
                    exp(
                        tools.normalize_log(
                            (
                                self.sample_likelihoods_refine[idx_ind][idx_sample + 1][
                                    idx_gt_germline
                                ],
                                self.sample_likelihoods_refine[idx_ind][idx_sample + 1][
                                    idx_gt_mosaic
                                ],
                            )
                        )[1]
                    )
                    # if self.mut_cell_list_final[idx_ind][idx_sample]
                    # else exp(
                    #     self.sample_likelihoods_refine[idx_ind][idx_sample][idx_gt_germline]
                    # )
                )
                gt_posts_ind.append(
                    exp(self.sc_posteriors_final[idx_ind][idx_sample][1])
                    # if self.mut_cell_list_final[idx_ind][idx_sample]
                    # else exp(self.mut_cell_list_final[idx_ind][0])
                )
                gts_ind.append(
                    (
                        self.total_allele_pool.index(gt_mosaic[0]),
                        self.total_allele_pool.index(gt_mosaic[1]),
                    )
                    if self.mut_cell_list_final[idx_ind][idx_sample]
                    else (
                        self.total_allele_pool.index(gt_germline[0]),
                        self.total_allele_pool.index(gt_germline[1]),
                    )
                )
            self.gts.append(gts_ind)
            self.gt_lls.append(gt_lls_ind)
            self.gt_posts.append(gt_posts_ind)
            # endregion final gt calc

            self.mu_mosaic.append(
                (
                    mu_g_allele,
                    mu_m_allele,
                    mosaic_mut_rate,
                    gt_germline,
                    gt_mosaic,
                    ref_g_allele,
                )
            )

            # DP
            M = np.full(
                (self.num_sc_sample[idx_ind] + 1, self.num_sc_sample[idx_ind] + 1),
                float("-inf"),
            )  # l*j, ALT allele * num_cell, first col is padding 0, ignore
            # init
            M[0, 1] = self.get_gt_likelihoods_refine(idx_ind, 1, gt_germline)
            M[1, 1] = self.get_gt_likelihoods_refine(idx_ind, 1, gt_mosaic)
            # recursion
            for j_col in range(2, self.num_sc_sample[idx_ind] + 1):
                for l_row in range(0, j_col + 1):
                    M[l_row, j_col] = logsumexp(
                        [
                            M[l_row, j_col - 1]
                            + self.get_gt_likelihoods_refine(
                                idx_ind, j_col, gt_germline
                            ),
                            M[l_row - 1, j_col - 1]
                            + self.get_gt_likelihoods_refine(idx_ind, j_col, gt_mosaic),
                        ]
                    )
            mo_lls = exp(M[:, -1])
            mo_comb = [
                comb(self.num_sc_sample[idx_ind], l)
                for l in range(self.num_sc_sample[idx_ind] + 1)
            ]
            mo_priors = [
                self.mut_cell_prob(l, self.num_sc_sample[idx_ind], mosaic_mut_rate)
                for l in range(self.num_sc_sample[idx_ind] + 1)
            ]
            mo_posteriori = tools.normalize(mo_lls * mo_priors / mo_comb)
            self.mo_posteriori.append(mo_posteriori)
            if all(v == 0 for v in mo_posteriori):
                self.mo_posteriors.append(0)
            else:
                self.mo_posteriors.append(1 - mo_posteriori[0])

    def lr_test(self, idx_ind):
        """LR tests with germline probabilities for phasable sites"""

        discordant_rate_bulk, discordant_rate = self.dis_prop[idx_ind]
        (
            mu_j_allele,
            mu_k_allele,
            mu_jk,
            gt_germline,
            gt_mosaic,
            ref_g_allele,
        ) = self.mu_mosaic[idx_ind]
        gt_mosaic_2 = (gt_mosaic[1], gt_mosaic[0])
        mut = (mu_j_allele, mu_k_allele)

        # region hat hypothesis assuming mapping obs.
        if discordant_rate_bulk:
            germline_hat = self.gt_hat_likelihoods(
                idx_ind,
                gt_germline,
                self.reads_data_spn_seq[idx_ind][0],
                self.reads_data_spn_qual[idx_ind][0],
                mut,
                "bulk",
                discordant_rate_bulk,
            )
            germline_null = self.gt_null_likelihoods(
                idx_ind,
                gt_germline,
                self.reads_data_spn_seq[idx_ind][0],
                self.reads_data_spn_qual[idx_ind][0],
                mut,
                "bulk",
            )
        else:
            germline_hat = 0
            germline_null = 0

        if discordant_rate:
            # mapping
            for idx_sc_sample in range(self.num_sc_sample[idx_ind]):
                cell_gts = []
                # [j / k] gt
                gt_hom_ll = self.gt_hat_likelihoods(
                    idx_ind,
                    gt_germline,
                    self.reads_data_spn_seq[idx_ind][1][idx_sc_sample],
                    self.reads_data_spn_qual[idx_ind][1][idx_sc_sample],
                    mut,
                    self.sc_prot_list[idx_ind][idx_sc_sample],
                    discordant_rate,
                    self.af[idx_ind][idx_sc_sample]
                    if self.af[idx_ind][idx_sc_sample]
                    else 0.5,
                )
                gt_het_ll = self.gt_hat_likelihoods(
                    idx_ind,
                    gt_mosaic,
                    self.reads_data_spn_seq[idx_ind][1][idx_sc_sample],
                    self.reads_data_spn_qual[idx_ind][1][idx_sc_sample],
                    mut,
                    self.sc_prot_list[idx_ind][idx_sc_sample],
                    discordant_rate,
                    self.af[idx_ind][idx_sc_sample]
                    if self.af[idx_ind][idx_sc_sample]
                    else 0.5,
                ) + self.gt_hat_likelihoods(
                    idx_ind,
                    gt_mosaic_2,
                    self.reads_data_spn_seq[idx_ind][1][idx_sc_sample],
                    self.reads_data_spn_qual[idx_ind][1][idx_sc_sample],
                    mut,
                    self.sc_prot_list[idx_ind][idx_sc_sample],
                    discordant_rate,
                    self.af[idx_ind][idx_sc_sample]
                    if self.af[idx_ind][idx_sc_sample]
                    else 0.5,
                )
                cell_gts.append(
                    log(self.mu[idx_ind].get((mu_j_allele, mu_j_allele)))
                    + log(self.mu[idx_ind].get((mu_j_allele, mu_k_allele)))
                    + gt_het_ll
                )
                # [j, j] gt
                cell_gts.append(
                    2 * log(self.mu[idx_ind].get((mu_j_allele, mu_j_allele)))
                    + gt_hom_ll
                )
                germline_hat += logsumexp(cell_gts)
            # stutter
            for idx_sc_sample in range(self.num_sc_sample[idx_ind]):
                cell_gts = []
                # [j / k] gt
                gt_het_ll = self.gt_null_likelihoods(
                    idx_ind,
                    gt_mosaic,
                    self.reads_data_spn_seq[idx_ind][1][idx_sc_sample],
                    self.reads_data_spn_qual[idx_ind][1][idx_sc_sample],
                    mut,
                    self.sc_prot_list[idx_ind][idx_sc_sample],
                    self.af[idx_ind][idx_sc_sample]
                    if self.af[idx_ind][idx_sc_sample]
                    else 0.5,
                ) + self.gt_null_likelihoods(
                    idx_ind,
                    gt_mosaic_2,
                    self.reads_data_spn_seq[idx_ind][1][idx_sc_sample],
                    self.reads_data_spn_qual[idx_ind][1][idx_sc_sample],
                    mut,
                    self.sc_prot_list[idx_ind][idx_sc_sample],
                    self.af[idx_ind][idx_sc_sample]
                    if self.af[idx_ind][idx_sc_sample]
                    else 0.5,
                )
                gt_hom_ll = self.gt_null_likelihoods(
                    idx_ind,
                    gt_germline,
                    self.reads_data_spn_seq[idx_ind][1][idx_sc_sample],
                    self.reads_data_spn_qual[idx_ind][1][idx_sc_sample],
                    mut,
                    self.sc_prot_list[idx_ind][idx_sc_sample],
                    self.af[idx_ind][idx_sc_sample]
                    if self.af[idx_ind][idx_sc_sample]
                    else 0.5,
                )
                cell_gts.append(
                    log(self.mu[idx_ind].get((mu_j_allele, mu_j_allele)))
                    + log(self.mu[idx_ind].get((mu_j_allele, mu_k_allele)))
                    + gt_het_ll
                )
                # [j, j] gt
                cell_gts.append(
                    2 * log(self.mu[idx_ind].get((mu_j_allele, mu_j_allele)))
                    + gt_hom_ll
                )
                germline_null += logsumexp(cell_gts)

        # endregion assuming stutter error
        statistic = 2 * (germline_hat - germline_null)
        if discordant_rate_bulk and discordant_rate:
            return statistic, chi2.sf(statistic, df=2)
        else:
            return statistic, chi2.sf(statistic, df=1)

    def gt_hat_likelihoods(
        self,
        idx_ind,
        genotype,
        reads_data_seq,
        reads_qualities,
        mut,
        prot,
        dis_rate=None,
        af=0.5,
    ) -> float:
        germline_allele, mosaic_allele = mut
        idx_mut_hap = 0 if self.con[idx_ind] == "j1k2" else 1
        r = []
        if prot != "bulk":
            for idx, _ in enumerate(reads_data_seq):
                if reads_data_seq[idx][:3] in mut:
                    probs = [
                        (
                            self.read_allele_prob_lrt(
                                reads_data_seq[idx],
                                allele,
                                reads_qualities[idx],
                                dis_rate,
                                self.hSNP[idx_ind][idx_hap],
                                prot,
                            )
                            if allele == germline_allele and idx_hap == idx_mut_hap
                            else self.read_allele_prob(
                                reads_data_seq[idx],
                                allele,
                                reads_qualities[idx],
                                prot,
                                self.hSNP[idx_ind][idx_hap],
                            )
                        )
                        for idx_hap, allele in enumerate(genotype)
                    ]
                    r.append(logsumexp(probs, b=[af, 1 - af]))
            return sum(r)
        else:
            assert germline_allele in genotype  # NOTE:
            if self.mode == "se":
                for idx, _ in enumerate(reads_data_seq):
                    if reads_data_seq[idx][:3] in mut:
                        probs = [
                            logsumexp(
                                [
                                    self.mu[idx_ind].get(
                                        (germline_allele, mosaic_allele)
                                    )
                                    + self.read_allele_prob(
                                        reads_data_seq[idx],
                                        mosaic_allele,
                                        reads_qualities[idx],
                                        "bulk",
                                        self.hSNP[idx_ind][idx_hap],
                                    ),
                                    self.mu[idx_ind].get(
                                        (germline_allele, germline_allele)
                                    )
                                    + (
                                        self.read_allele_prob_lrt(
                                            reads_data_seq[idx],
                                            allele,
                                            reads_qualities[idx],
                                            dis_rate,
                                            self.hSNP[idx_ind][idx_hap],
                                            prot,
                                        )
                                        if allele == germline_allele
                                        and idx_hap == idx_mut_hap
                                        else self.read_allele_prob(
                                            reads_data_seq[idx],
                                            allele,
                                            reads_qualities[idx],
                                            prot,
                                            self.hSNP[idx_ind][idx_hap],
                                        )
                                    ),
                                ]
                            )
                            for idx_hap, allele in enumerate(genotype)
                        ]
                        r.append(logsumexp(probs, b=[0.5, 0.5]))
                return sum(r)
            else:
                for idx, _ in enumerate(reads_data_seq):
                    if reads_data_seq[idx][:3] in mut:
                        probs = [
                            (
                                self.read_allele_prob_lrt(
                                    reads_data_seq[idx],
                                    allele,
                                    reads_qualities[idx],
                                    dis_rate,
                                    self.hSNP[idx_ind][idx_hap],
                                    prot,
                                )
                                if allele == germline_allele and idx_hap == idx_mut_hap
                                else self.read_allele_prob(
                                    reads_data_seq[idx],
                                    allele,
                                    reads_qualities[idx],
                                    prot,
                                    self.hSNP[idx_ind][idx_hap],
                                )
                            )
                            for idx_hap, allele in enumerate(genotype)
                        ]
                        r.append(logsumexp(probs, b=[0.5, 0.5]))
                return sum(r)

    def gt_null_likelihoods(
        self,
        idx_ind,
        genotype,
        reads_data_seq,
        reads_qualities,
        mut,
        prot,
        af=0.5,
    ):
        germline_allele, mosaic_allele = mut
        r = []
        if prot != "bulk":
            for idx, _ in enumerate(reads_data_seq):
                if reads_data_seq[idx][:3] in mut:
                    probs = [
                        self.read_allele_prob(
                            reads_data_seq[idx],
                            allele,
                            reads_qualities[idx],
                            prot,
                            self.hSNP[idx_ind][idx_hap],
                        )
                        for idx_hap, allele in enumerate(genotype)
                    ]
                    r.append(logsumexp(probs, b=[af, 1 - af]))
            return sum(r)
        else:
            assert germline_allele in genotype
            if self.mode == "se":
                for idx, _ in enumerate(reads_data_seq):
                    if reads_data_seq[idx][:3] in mut:
                        probs = [
                            logsumexp(
                                [
                                    self.mu[idx_ind].get(
                                        (germline_allele, mosaic_allele)
                                    )
                                    + self.read_allele_prob(
                                        reads_data_seq[idx],
                                        mosaic_allele,
                                        reads_qualities[idx],
                                        prot,
                                        self.hSNP[idx_ind][idx_hap],
                                    ),
                                    self.mu[idx_ind].get(
                                        (germline_allele, germline_allele)
                                    )
                                    + self.read_allele_prob(
                                        reads_data_seq[idx],
                                        allele,
                                        reads_qualities[idx],
                                        prot,
                                        self.hSNP[idx_ind][idx_hap],
                                    ),
                                ]
                            )
                            for idx_hap, allele in enumerate(genotype)
                        ]
                        r.append(logsumexp(probs, b=[0.5, 0.5]))
                return sum(r)
            else:
                for idx, _ in enumerate(reads_data_seq):
                    if reads_data_seq[idx][:3] in mut:
                        probs = [
                            self.read_allele_prob(
                                reads_data_seq[idx],
                                allele,
                                reads_qualities[idx],
                                prot,
                                self.hSNP[idx_ind][idx_hap],
                            )
                            for idx_hap, allele in enumerate(genotype)
                        ]
                        r.append(logsumexp(probs, b=[0.5, 0.5]))
                return sum(r)

    @staticmethod
    def read_allele_prob_lrt(
        read, allele, read_qualities, dis_rate=None, hsnp=None, prot=None
    ):
        """read prob given allele for lrt, replace the whole"""
        prob = (
            log(align._emiss(read[3], hsnp, read_qualities[3]))
            + align.align_flk(read[0], allele[0], read_qualities[0])
            + align.align_flk(read[2], allele[2], read_qualities[2])
            + align.align_ms_lrt(allele[1], read[1], read_qualities[1], dis_rate)
        )
        return prob

    def bb_train(self, idx_ind, supp, reads_data_seq, reads_data_qual):
        r"""Beta-binomial posteriors"""
        if not self.ab_data or not self.ab_data[idx_ind]:
            self.bb_likelihoods.append([])
            self.prod_likelihoods.append([])
            self.mo_posterior_bbs.append(None)
            self.phy_llbb.append([])
            return
        (
            mu_j_allele,
            mu_k_allele,
            mu_jk,
            gt_germline,
            gt_mosaic,
            ref_g_allele,
        ) = self.mu_mosaic[idx_ind]
        if mu_j_allele != ref_g_allele or not self.ab_data[idx_ind]:
            self.bb_likelihoods.append([])
            self.prod_likelihoods.append([])
            self.mo_posterior_bbs.append(None)
            self.phy_llbb.append([])
            return
        gts = (
            (mu_j_allele, mu_j_allele),
            (mu_j_allele, mu_k_allele),
            (mu_k_allele, mu_k_allele),
        )
        alt_list_f, dp_list_f, reads_seq, reads_qual = tools.filter_stutter(
            mu_j_allele,
            mu_k_allele,
            reads_data_seq,
            reads_data_qual,
        )
        if supp.get("mut_info", None):
            pos = supp.get("mut_info")[0]
            mut_type = supp.get("mut_info")[2]
        else:
            self.bb_likelihoods.append([])
            self.prod_likelihoods.append([])
            self.mo_posterior_bbs.append(None)
            self.phy_llbb.append([])
            return
        # ll
        phy_umutll = []
        phy_mutll = []
        phy_bb = []
        if len(mu_j_allele[1]) != len(mu_k_allele[1]):
            # bulk
            ll_bulk = [
                self.gt_likelihoods_bb_len(
                    gt,
                    reads_seq[0],
                    "bulk",
                )
                for gt in gts
            ]
            ind_sample_bb_likelihood = [ll_bulk]
            ind_sample_prod_likelihood = [ll_bulk]
            # single cells
            for idx_reads, reads in enumerate(reads_seq[1]):
                jj = self.gt_likelihoods_bb_len(
                    (mu_j_allele, mu_j_allele),
                    reads,
                    self.sc_prot_list[idx_ind][idx_reads],
                )
                jk = self.gt_likelihoods_bb_len(
                    (mu_j_allele, mu_k_allele),
                    reads,
                    self.sc_prot_list[idx_ind][idx_reads],
                )
                bb = self.bb_likelihood(
                    alt_list_f[idx_reads + 1],
                    dp_list_f[idx_reads + 1],
                    self.ab_data[idx_ind][idx_reads][0],
                    self.ab_data[idx_ind][idx_reads][1],
                )
                if math.isnan(bb):
                    bb = -1e7
                elif bb > 0:
                    bb = 0
                elif math.isinf(bb) and bb < 0:
                    bb = -1e50
                kk = self.gt_likelihoods_bb_len(
                    (mu_k_allele, mu_k_allele),
                    reads,
                    self.sc_prot_list[idx_ind][idx_reads],
                )
                ind_sample_bb_likelihood.append((jj, bb, kk))
                ind_sample_prod_likelihood.append((jj, jk, kk))
                phy_umutll.append(jj)
                phy_mutll.append(jk)
                phy_bb.append(bb)
            self.bb_likelihoods.append(
                [tools.normalize_log(ll) for ll in ind_sample_bb_likelihood]
            )
            self.prod_likelihoods.append(
                [tools.normalize_log(ll) for ll in ind_sample_prod_likelihood]
            )
        elif mut_type == "mismatch":
            # bulk
            ll_bulk = [
                self.gt_likelihoods_bb_mm(
                    gt,
                    reads_seq[0],
                    reads_qual[0],
                    "bulk",
                    pos,
                )
                for gt in gts
            ]
            ind_sample_bb_likelihood = [ll_bulk]
            ind_sample_prod_likelihood = [ll_bulk]
            # single cells
            for idx_reads, reads in enumerate(reads_seq[1]):
                jj = self.gt_likelihoods_bb_mm(
                    (mu_j_allele, mu_j_allele),
                    reads,
                    reads_qual[1][idx_reads],
                    self.sc_prot_list[idx_ind][idx_reads],
                    pos,
                )
                jk = self.gt_likelihoods_bb_mm(
                    (mu_j_allele, mu_k_allele),
                    reads,
                    reads_qual[1][idx_reads],
                    self.sc_prot_list[idx_ind][idx_reads],
                    pos,
                )
                bb = self.bb_likelihood(
                    alt_list_f[idx_reads + 1],
                    dp_list_f[idx_reads + 1],
                    self.ab_data[idx_ind][idx_reads][0],
                    self.ab_data[idx_ind][idx_reads][1],
                )
                kk = self.gt_likelihoods_bb_mm(
                    (mu_k_allele, mu_k_allele),
                    reads,
                    reads_qual[1][idx_reads],
                    self.sc_prot_list[idx_ind][idx_reads],
                    pos,
                )
                ind_sample_bb_likelihood.append((jj, bb, kk))
                ind_sample_prod_likelihood.append((jj, jk, kk))
                phy_umutll.append(jj)
                phy_mutll.append(jk)
                phy_bb.append(bb)
            self.bb_likelihoods.append(
                [tools.normalize_log(ll) for ll in ind_sample_bb_likelihood]
            )
            self.prod_likelihoods.append(
                [tools.normalize_log(ll) for ll in ind_sample_prod_likelihood]
            )
        else:
            self.bb_likelihoods.append([])
            self.prod_likelihoods.append([])
            self.mo_posterior_bbs.append(None)
            self.phy_llbb.append([])
            return
        self.phy_llbb.append([phy_umutll, phy_bb, phy_mutll])
        # G_i
        gt_germ_bbs = self.gt_germ_bb((mu_j_allele, mu_k_allele), idx_ind)
        if max(gt_germ_bbs) == gt_germ_bbs[1]:
            self.mo_posterior_bbs.append(0)
        else:
            M = np.full(
                (self.num_sc_sample[idx_ind] + 1, self.num_sc_sample[idx_ind] + 1),
                float("-inf"),
            )  # l*j, ALT allele * num_cell, first col is padding 0, ignore
            # init
            M[0, 1] = self.prod_likelihoods[idx_ind][1][0]
            M[1, 1] = self.prod_likelihoods[idx_ind][1][1]
            # recursion
            for j_col in range(2, self.num_sc_sample[idx_ind] + 1):
                for l_row in range(0, j_col + 1):
                    M[l_row, j_col] = logsumexp(
                        [
                            M[l_row, j_col - 1]
                            + self.prod_likelihoods[idx_ind][j_col][0],
                            # + self.get_gt_likelihoods(idx_ind, j_col, gt_germline),
                            M[l_row - 1, j_col - 1]
                            + self.prod_likelihoods[idx_ind][j_col][1],
                            # + self.get_gt_likelihoods(idx_ind, j_col, gt_mosaic),
                        ]
                    )
            mo_lls = exp(M[:, -1])
            mo_comb = [
                comb(self.num_sc_sample[idx_ind], l)
                for l in range(self.num_sc_sample[idx_ind] + 1)
            ]
            mo_priors = [
                self.mut_cell_prob(l, self.num_sc_sample[idx_ind], mu_jk)
                for l in range(self.num_sc_sample[idx_ind] + 1)
            ]
            # mo_posteriori = tools.normalize(mo_lls * mo_priors / mo_comb)
            mo_posteriori = [
                exp(v)
                for v in tools.normalize_log(M[:, -1] + log(mo_priors) - log(mo_comb))
            ]
            # breakpoint()
            if all(v == 0 for v in mo_posteriori):
                self.mo_posterior_bbs.append(0)
                return 0
            else:
                self.mo_posterior_bbs.append(1 - mo_posteriori[0])
                return 1 - mo_posteriori[0]
        # print([lst.index(max(lst)) for lst in self.bb_likelihoods[0]])
        # print([f"{a}/{dp}" for a, dp in zip(alt_list_f, dp_list_f)])
        # print(self.mo_posterior_bbs)

    def bb_train_p(self, idx_ind, supp, reads_data_seq, reads_data_qual):
        r"""Beta-binomial posteriors w/ phase"""
        if not self.ab_data or not self.ab_data[idx_ind]:
            self.bb_likelihoods_p.append([])
            self.prod_likelihoods_p.append([])
            self.mo_posterior_bbs_p.append(None)
            self.phy_llbb_p.append([])
            return
        (
            mu_j_allele,
            mu_k_allele,
            mu_jk,
            gt_germline,
            gt_mosaic,
            ref_g_allele,
        ) = self.mu_mosaic[idx_ind]
        if mu_j_allele != ref_g_allele or not self.ab_data[idx_ind]:
            self.bb_likelihoods_p.append([])
            self.prod_likelihoods_p.append([])
            self.mo_posterior_bbs_p.append(None)
            self.phy_llbb_p.append([])
            return
        gts = (
            (mu_j_allele, mu_j_allele),
            (mu_j_allele, mu_k_allele),
            (mu_k_allele, mu_k_allele),
        )
        alt_list_f, dp_list_f, reads_seq, reads_qual = tools.filter_stutter(
            mu_j_allele,
            mu_k_allele,
            reads_data_seq,
            reads_data_qual,
        )
        if supp.get("mut_info", None):
            pos = supp.get("mut_info")[0]
            mut_type = supp.get("mut_info")[2]
        else:
            self.bb_likelihoods_p.append([])
            self.prod_likelihoods_p.append([])
            self.mo_posterior_bbs_p.append(None)
            self.phy_llbb_p.append([])
            return
        # ll
        phy_umutll = []
        phy_mutll = []
        phy_bb = []
        gt_mut = (
            (mu_j_allele, mu_k_allele)
            if (mu_j_allele, mu_k_allele) in self.genotypes[idx_ind]
            else (mu_k_allele, mu_j_allele)
        )
        p1 = self.phase_probs[idx_ind][gt_mut]
        if len(mu_j_allele[1]) != len(mu_k_allele[1]):
            # bulk
            ll_bulk = [
                self.gt_phase_likelihoods_bb_len(gt, reads_seq[0], "bulk", p1)
                for gt in gts
            ]
            ind_sample_bb_likelihood = [ll_bulk]
            ind_sample_prod_likelihood = [ll_bulk]
            # single cells
            for idx_reads, reads in enumerate(reads_seq[1]):
                if (
                    self.af[idx_ind][idx_reads] is not None
                    and self.num_spn[idx_ind][idx_reads + 1]
                ):
                    jj = self.gt_phase_likelihoods_bb_len(
                        (mu_j_allele, mu_j_allele),
                        reads,
                        self.sc_prot_list[idx_ind][idx_reads],
                        p1,
                        af=self.af[idx_ind][idx_reads],
                    )
                    bb = self.bb_likelihood(
                        alt_list_f[idx_reads + 1],
                        dp_list_f[idx_reads + 1],
                        self.ab_data[idx_ind][idx_reads][0],
                        self.ab_data[idx_ind][idx_reads][1],
                    )
                    jk = self.gt_phase_likelihoods_bb_len(
                        gt_mut,
                        reads,
                        self.sc_prot_list[idx_ind][idx_reads],
                        p1,
                        af=self.af[idx_ind][idx_reads],
                    )
                    kk = self.gt_phase_likelihoods_bb_len(
                        (mu_k_allele, mu_k_allele),
                        reads,
                        self.sc_prot_list[idx_ind][idx_reads],
                        p1,
                        af=self.af[idx_ind][idx_reads],
                    )
                else:  # no spn
                    jj = self.gt_phase_likelihoods_bb_len(
                        (mu_j_allele, mu_j_allele),
                        reads,
                        self.sc_prot_list[idx_ind][idx_reads],
                    )
                    bb = self.bb_likelihood(
                        alt_list_f[idx_reads + 1],
                        dp_list_f[idx_reads + 1],
                        self.ab_data[idx_ind][idx_reads][0],
                        self.ab_data[idx_ind][idx_reads][1],
                    )
                    jk = bb
                    kk = self.gt_phase_likelihoods_bb_len(
                        (mu_k_allele, mu_k_allele),
                        reads,
                        self.sc_prot_list[idx_ind][idx_reads],
                    )
                ind_sample_bb_likelihood.append((jj, bb, kk))
                ind_sample_prod_likelihood.append((jj, jk, kk))
                phy_umutll.append(jj)
                phy_mutll.append(jk)
                phy_bb.append(bb)
            self.bb_likelihoods_p.append(
                [tools.normalize_log(ll) for ll in ind_sample_bb_likelihood]
            )
            self.prod_likelihoods_p.append(
                [tools.normalize_log(ll) for ll in ind_sample_prod_likelihood]
            )
        elif mut_type == "mismatch":
            # bulk
            ll_bulk = [
                self.gt_phase_likelihoods_bb_mm(
                    gt, reads_seq[0], reads_qual[0], pos, p1
                )
                for gt in gts
            ]
            ind_sample_bb_likelihood = [ll_bulk]
            ind_sample_prod_likelihood = [ll_bulk]
            # single cells
            for idx_reads, reads in enumerate(reads_seq[1]):
                if (
                    self.af[idx_ind][idx_reads] is not None
                    and self.num_spn[idx_ind][idx_reads + 1]
                ):
                    jj = self.gt_phase_likelihoods_bb_mm(
                        (mu_j_allele, mu_j_allele),
                        reads,
                        reads_qual[1][idx_reads],
                        pos,
                        p1,
                        af=self.af[idx_ind][idx_reads],
                    )
                    bb = self.bb_likelihood(
                        alt_list_f[idx_reads + 1],
                        dp_list_f[idx_reads + 1],
                        self.ab_data[idx_ind][idx_reads][0],
                        self.ab_data[idx_ind][idx_reads][1],
                    )
                    if self.num_spn[idx_ind][idx_reads + 1]:
                        jk = self.gt_phase_likelihoods_bb_mm(
                            gt_mut,
                            reads,
                            reads_qual[1][idx_reads],
                            pos,
                            p1,
                            af=self.af[idx_ind][idx_reads],
                        )
                    else:
                        jk = bb
                    kk = self.gt_phase_likelihoods_bb_mm(
                        (mu_k_allele, mu_k_allele),
                        reads,
                        reads_qual[1][idx_reads],
                        pos,
                        p1,
                        af=self.af[idx_ind][idx_reads],
                    )
                else:  # no spn
                    jj = self.gt_phase_likelihoods_bb_mm(
                        (mu_j_allele, mu_j_allele),
                        reads,
                        reads_qual[1][idx_reads],
                        pos,
                    )
                    bb = self.bb_likelihood(
                        alt_list_f[idx_reads + 1],
                        dp_list_f[idx_reads + 1],
                        self.ab_data[idx_ind][idx_reads][0],
                        self.ab_data[idx_ind][idx_reads][1],
                    )
                    jk = bb
                    kk = self.gt_phase_likelihoods_bb_mm(
                        (mu_k_allele, mu_k_allele),
                        reads,
                        reads_qual[1][idx_reads],
                        pos,
                    )
                ind_sample_bb_likelihood.append((jj, bb, kk))
                ind_sample_prod_likelihood.append((jj, jk, kk))
                phy_umutll.append(jj)
                phy_mutll.append(jk)
                phy_bb.append(bb)
            self.bb_likelihoods_p.append(
                [tools.normalize_log(ll) for ll in ind_sample_bb_likelihood]
            )
            self.prod_likelihoods_p.append(
                [tools.normalize_log(ll) for ll in ind_sample_prod_likelihood]
            )
        else:
            self.bb_likelihoods_p.append([])
            self.prod_likelihoods_p.append([])
            self.mo_posterior_bbs_p.append(None)
            self.phy_llbb_p.append([])
            return
        self.phy_llbb_p.append([phy_umutll, phy_bb, phy_mutll])
        # G_i
        gt_germ_bbs = self.gt_germ_bb_p((mu_j_allele, mu_k_allele), idx_ind)
        if max(gt_germ_bbs) == gt_germ_bbs[1]:
            self.mo_posterior_bbs_p.append(0)
        else:
            M = np.full(
                (self.num_sc_sample[idx_ind] + 1, self.num_sc_sample[idx_ind] + 1),
                float("-inf"),
            )  # l*j, ALT allele * num_cell, first col is padding 0, ignore
            # init
            M[0, 1] = self.prod_likelihoods_p[idx_ind][1][0]
            M[1, 1] = self.prod_likelihoods_p[idx_ind][1][1]
            # recursion
            for j_col in range(2, self.num_sc_sample[idx_ind] + 1):
                for l_row in range(0, j_col + 1):
                    M[l_row, j_col] = logsumexp(
                        [
                            M[l_row, j_col - 1]
                            + self.prod_likelihoods_p[idx_ind][j_col][0],
                            # + self.get_gt_likelihoods(idx_ind, j_col, gt_germline),
                            M[l_row - 1, j_col - 1]
                            + self.prod_likelihoods_p[idx_ind][j_col][1],
                            # + self.get_gt_likelihoods(idx_ind, j_col, gt_mosaic),
                        ]
                    )
            mo_lls = exp(M[:, -1])
            mo_comb = [
                comb(self.num_sc_sample[idx_ind], l)
                for l in range(self.num_sc_sample[idx_ind] + 1)
            ]
            mo_priors = [
                self.mut_cell_prob(l, self.num_sc_sample[idx_ind], mu_jk)
                for l in range(self.num_sc_sample[idx_ind] + 1)
            ]
            # mo_posteriori = tools.normalize(mo_lls * mo_priors / mo_comb)
            mo_posteriori = [
                exp(v)
                for v in tools.normalize_log(M[:, -1] + log(mo_priors) - log(mo_comb))
            ]
            # breakpoint()
            if all(v == 0 for v in mo_posteriori):
                self.mo_posterior_bbs_p.append(0)
                return 0
            else:
                self.mo_posterior_bbs_p.append(1 - mo_posteriori[0])
                return 1 - mo_posteriori[0]
        # print([lst.index(max(lst)) for lst in self.bb_likelihoods[0]])
        # print([f"{a}/{dp}" for a, dp in zip(alt_list_f, dp_list_f)])
        # print(self.mo_posterior_bbs_p)

    def bb_likelihood(
        self,
        alt,
        dp,
        alpha=1,
        beta=1,
    ):
        r"""Beta-binomial likelihood for single cells"""
        return log(betabinom.pmf(alt, dp, alpha, beta) / comb(dp, alt))

    def gt_likelihoods_bb_len(self, genotype, reads_data, prot=None) -> float:
        r"""Reads probabilities given genotype for bulk or single cell sample."""
        j_allele, k_allele = genotype
        r = [
            logsumexp(
                (
                    self.read_allele_prob_bb(read, j_allele, prot),
                    self.read_allele_prob_bb(read, k_allele, prot),
                )
            )
            for read in reads_data
        ]
        return sum(r)

    def gt_likelihoods_bb_mm(
        self,
        genotype,
        reads_data_seq,
        reads_qualities,
        prot,
        pos,
    ) -> float:
        r"""Reads probabilities given germline genotype for bulk or single cell sample."""
        j_allele, k_allele = genotype
        r = [
            log(
                0.5
                * (
                    align._emiss(
                        tools.desegmentation(reads_data_seq[idx])[pos - 1],
                        tools.desegmentation(j_allele)[pos - 1],
                        tools.desegmentation(reads_qualities[idx])[pos - 1],
                    )
                    + align._emiss(
                        tools.desegmentation(reads_data_seq[idx])[pos - 1],
                        tools.desegmentation(k_allele)[pos - 1],
                        tools.desegmentation(reads_qualities[idx])[pos - 1],
                    )
                )
            )
            for idx in range(len(reads_data_seq))
        ]
        return sum(r)

    def read_allele_prob_bb(self, read, allele, prot=None, af=0.5) -> float:
        r"""The conditional probability of reads originating from either allele."""
        assert prot in ["bulk", "sc", "mda", "pta", "scc"]
        read = len(read[1])
        allele = len(allele[1])
        if (read - allele) % self.region.get("motif_len") == 0:  # in-frame
            up = self.stutter[f"up_{prot}_in"]
            down = self.stutter[f"down_{prot}_in"]
            rho = self.stutter[f"rho_{prot}_in"]
            if read == allele:
                error = log(1 - up - down)
            elif read > allele:
                if rho == 1:
                    error = log(up * rho)
                else:
                    error = log(
                        up
                        * rho
                        * (1 - rho)
                        ** ((read - allele) / self.region.get("motif_len") - 1)
                    )
            else:
                if rho == 1:
                    error = log(down * rho)
                else:
                    error = log(
                        down
                        * rho
                        * (1 - rho)
                        ** ((allele - read) / self.region.get("motif_len") - 1)
                    )
        else:
            up = self.stutter[f"up_{prot}_out"]
            down = self.stutter[f"down_{prot}_out"]
            rho = self.stutter[f"rho_{prot}_out"]
            if read == allele:
                error = log(1 - up - down)
            elif read > allele:
                if rho == 1:
                    error = log(up * rho)
                else:
                    error = log(
                        up
                        * rho
                        * (1 - rho)
                        ** (
                            read
                            - allele
                            - ((read - allele) // self.region.get("motif_len"))
                            - 1
                        )
                    )
            else:
                if rho == 1:
                    error = log(down * rho)
                else:
                    error = log(
                        down
                        * rho
                        * (1 - rho)
                        ** (
                            allele
                            - read
                            - ((allele - read) // self.region.get("motif_len"))
                            - 1
                        )
                    )
        return log(af) + error

        # if (read - allele) % self.region.get("motif_len") == 0:  # in-frame
        #     if prot == "sc":
        #         if read == allele:
        #             error = log(
        #                 1 - self.stutter["up_sc_in"] - self.stutter["down_sc_in"]
        #             )

        #         elif read > allele:
        #             if self.stutter["rho_sc_in"] == 1:
        #                 error = log(
        #                     self.stutter["up_sc_in"] * self.stutter["rho_sc_in"]
        #                 )
        #             else:
        #                 error = log(
        #                     self.stutter["up_sc_in"]
        #                     * self.stutter["rho_sc_in"]
        #                     * (1 - self.stutter["rho_sc_in"])
        #                     ** ((read - allele) / self.region.get("motif_len") - 1)
        #                 )

        #         else:
        #             if self.stutter["rho_sc_in"] == 1:
        #                 error = log(
        #                     self.stutter["down_sc_in"] * self.stutter["rho_sc_in"]
        #                 )
        #             else:
        #                 error = log(
        #                     self.stutter["down_sc_in"]
        #                     * self.stutter["rho_sc_in"]
        #                     * (1 - self.stutter["rho_sc_in"])
        #                     ** ((allele - read) / self.region.get("motif_len") - 1)
        #                 )

        #     elif prot == "bulk":
        #         if read == allele:
        #             error = log(
        #                 1 - self.stutter["up_bulk_in"] - self.stutter["down_bulk_in"]
        #             )

        #         elif read > allele:
        #             if self.stutter["rho_bulk_in"] == 1:
        #                 error = log(
        #                     self.stutter["up_bulk_in"] * self.stutter["rho_bulk_in"]
        #                 )
        #             else:
        #                 error = log(
        #                     self.stutter["up_bulk_in"]
        #                     * self.stutter["rho_bulk_in"]
        #                     * (1 - self.stutter["rho_bulk_in"])
        #                     ** ((read - allele) / self.region.get("motif_len") - 1)
        #                 )

        #         else:
        #             if self.stutter["rho_bulk_in"] == 1:
        #                 error = log(
        #                     self.stutter["down_bulk_in"] * self.stutter["rho_bulk_in"]
        #                 )
        #             else:
        #                 error = log(
        #                     self.stutter["down_bulk_in"]
        #                     * self.stutter["rho_bulk_in"]
        #                     * (1 - self.stutter["rho_bulk_in"])
        #                     ** ((allele - read) / self.region.get("motif_len") - 1)
        #                 )

        # else:  # out-frame
        #     if prot == "sc":
        #         if read == allele:
        #             error = log(
        #                 1 - self.stutter["up_sc_out"] - self.stutter["down_sc_out"]
        #             )

        #         elif read > allele:
        #             if self.stutter["rho_sc_out"] == 1:
        #                 error = log(
        #                     self.stutter["up_sc_out"] * self.stutter["rho_sc_out"]
        #                 )
        #             else:
        #                 error = log(
        #                     self.stutter["up_sc_out"]
        #                     * self.stutter["rho_sc_out"]
        #                     * (1 - self.stutter["rho_sc_out"])
        #                     ** (
        #                         read
        #                         - allele
        #                         - ((read - allele) // self.region.get("motif_len"))
        #                         - 1
        #                     )
        #                 )

        #         else:
        #             if self.stutter["rho_sc_out"] == 1:
        #                 error = log(
        #                     self.stutter["down_sc_out"] * self.stutter["rho_sc_out"]
        #                 )
        #             else:
        #                 error = log(
        #                     self.stutter["down_sc_out"]
        #                     * self.stutter["rho_sc_out"]
        #                     * (1 - self.stutter["rho_sc_out"])
        #                     ** (
        #                         allele
        #                         - read
        #                         - ((allele - read) // self.region.get("motif_len"))
        #                         - 1
        #                     )
        #                 )

        #     elif prot == "bulk":
        #         if read == allele:
        #             error = log(
        #                 1 - self.stutter["up_bulk_out"] - self.stutter["down_bulk_out"]
        #             )

        #         elif read > allele:
        #             if self.stutter["rho_bulk_out"] == 1:
        #                 error = log(
        #                     self.stutter["up_bulk_out"] * self.stutter["rho_bulk_out"]
        #                 )
        #             else:
        #                 error = log(
        #                     self.stutter["up_bulk_out"]
        #                     * self.stutter["rho_bulk_out"]
        #                     * (1 - self.stutter["rho_bulk_out"])
        #                     ** (
        #                         read
        #                         - allele
        #                         - ((read - allele) // self.region.get("motif_len"))
        #                         - 1
        #                     )
        #                 )

        #         else:
        #             if self.stutter["rho_bulk_out"] == 1:
        #                 error = log(
        #                     self.stutter["down_bulk_out"] * self.stutter["rho_bulk_out"]
        #                 )
        #             else:
        #                 error = log(
        #                     self.stutter["down_bulk_out"]
        #                     * self.stutter["rho_bulk_out"]
        #                     * (1 - self.stutter["rho_bulk_out"])
        #                     ** (
        #                         allele
        #                         - read
        #                         - ((allele - read) // self.region.get("motif_len"))
        #                         - 1
        #                     )
        #                 )

    def gt_germ_bb(self, gt_germ_alleles, idx_ind):
        r"""Germline genotype priors."""
        j_allele, k_allele = gt_germ_alleles
        # het
        if self.freq.get(j_allele) == 0 or self.freq.get(k_allele) == 0:
            gt_germ_het = float("-inf")
        else:
            gt_germ_het = (
                log(2) + log(self.freq.get(j_allele)) + log(self.freq.get(k_allele))
            )
            # assume 0 is bulk
            gt_germ_het += self.bb_likelihoods[idx_ind][0][1]
            # assume rest is sc
            for idx_sample in range(1, len(self.bb_likelihoods[idx_ind])):
                gt_germ_het += logsumexp(
                    self.mu[idx_ind].get((j_allele, j_allele))
                    + self.mu[idx_ind].get((k_allele, k_allele))
                    + self.bb_likelihoods[idx_ind][idx_sample][1]
                )
        # refhom
        if self.freq.get(j_allele) == 0:
            gt_germ_refhom = float("-inf")
        else:
            gt_germ_refhom = 2 * log(self.freq.get(j_allele))
            # assume 0 is bulk
            gt_germ_refhom += self.prod_likelihoods[idx_ind][0][0]
            # assume rest is sc
            for idx_sample in range(1, len(self.prod_likelihoods[idx_ind])):
                cell_gts = []
                # [j / non_j]
                cell_gts.append(
                    log(2)
                    + self.mu[idx_ind].get((j_allele, j_allele))
                    + self.mu[idx_ind].get((j_allele, k_allele))
                    + self.prod_likelihoods[idx_ind][idx_sample][1]
                )
                # [j, j]
                cell_gts.append(
                    self.mu[idx_ind].get((j_allele, j_allele)) * 2
                    + self.prod_likelihoods[idx_ind][idx_sample][0]
                )
                gt_germ_refhom += logsumexp(cell_gts)
        # breakpoint()
        return gt_germ_refhom, gt_germ_het

    def gt_phase_likelihoods_bb_mm(
        self,
        genotype,
        reads_data_seq,
        reads_qualities,
        pos,
        p1=-0.6931471805599453,
        af=0.5,
    ):
        """phase genotype likelihoods for (un)phasable reads"""
        j_allele, k_allele = genotype
        # hom
        if j_allele == k_allele:
            r = [
                log(
                    align._emiss(
                        tools.desegmentation(reads_data_seq[idx])[pos - 1],
                        tools.desegmentation(j_allele)[pos - 1],
                        tools.desegmentation(reads_qualities[idx])[pos - 1],
                    )
                )
                for idx in range(len(reads_data_seq))
            ]
        else:  # het
            r = []
            p1 = p1 if p1 < -1e-12 else -1e-12
            p1 = p1 if p1 > -1e12 else -1e12
            p2 = logsumexp([p1, 0], b=[-1, 1])
            for idx, _ in enumerate(reads_data_seq):
                r.append(
                    logsumexp(
                        [
                            p1
                            + log(
                                af
                                * align._emiss(
                                    tools.desegmentation(reads_data_seq[idx])[pos - 1],
                                    tools.desegmentation(j_allele)[pos - 1],
                                    tools.desegmentation(reads_qualities[idx])[pos - 1],
                                )
                            ),
                            p1
                            + log(
                                (1 - af)
                                * align._emiss(
                                    tools.desegmentation(reads_data_seq[idx])[pos - 1],
                                    tools.desegmentation(k_allele)[pos - 1],
                                    tools.desegmentation(reads_qualities[idx])[pos - 1],
                                )
                            ),
                            p2
                            + log(
                                af
                                * align._emiss(
                                    tools.desegmentation(reads_data_seq[idx])[pos - 1],
                                    tools.desegmentation(k_allele)[pos - 1],
                                    tools.desegmentation(reads_qualities[idx])[pos - 1],
                                )
                            ),
                            p2
                            + log(
                                (1 - af)
                                * align._emiss(
                                    tools.desegmentation(reads_data_seq[idx])[pos - 1],
                                    tools.desegmentation(j_allele)[pos - 1],
                                    tools.desegmentation(reads_qualities[idx])[pos - 1],
                                )
                            ),
                        ]
                    )
                )
        return sum(r)

    def gt_phase_likelihoods_bb_len(
        self,
        genotype,
        reads_data,
        prot=None,
        p1=-0.6931471805599453,
        af=0.5,
    ) -> float:
        """phase genotype likelihoods for (un)phasable reads"""
        j_allele, k_allele = genotype
        # hom
        if j_allele == k_allele:
            r = [
                self.read_allele_prob_bb(read, j_allele, prot, 1) for read in reads_data
            ]
        else:  # het
            r = []
            p1 = p1 if p1 < -1e-12 else -1e-12
            p1 = p1 if p1 > -1e12 else -1e12
            p2 = logsumexp([p1, 0], b=[-1, 1])
            for read in reads_data:
                r.append(
                    logsumexp(
                        [
                            p1 + self.read_allele_prob_bb(read, j_allele, prot, af),
                            p1 + self.read_allele_prob_bb(read, k_allele, prot, 1 - af),
                            p2 + self.read_allele_prob_bb(read, k_allele, prot, af),
                            p2 + self.read_allele_prob_bb(read, j_allele, prot, 1 - af),
                        ]
                    )
                )
        return sum(r)

    def gt_germ_bb_p(self, gt_germ_alleles, idx_ind):
        r"""Germline genotype priors."""
        j_allele, k_allele = gt_germ_alleles
        # het
        if self.freq.get(j_allele) == 0 or self.freq.get(k_allele) == 0:
            gt_germ_het = float("-inf")
        else:
            gt_germ_het = (
                log(2) + log(self.freq.get(j_allele)) + log(self.freq.get(k_allele))
            )
            # assume 0 is bulk
            gt_germ_het += self.bb_likelihoods_p[idx_ind][0][1]
            # assume rest is sc
            for idx_sample in range(1, len(self.bb_likelihoods_p[idx_ind])):
                gt_germ_het += logsumexp(
                    self.mu[idx_ind].get((j_allele, j_allele))
                    + self.mu[idx_ind].get((k_allele, k_allele))
                    + self.bb_likelihoods_p[idx_ind][idx_sample][1]
                )
        # refhom
        if self.freq.get(j_allele) == 0:
            gt_germ_refhom = float("-inf")
        else:
            gt_germ_refhom = 2 * log(self.freq.get(j_allele))
            # assume 0 is bulk
            gt_germ_refhom += self.prod_likelihoods_p[idx_ind][0][0]
            # assume rest is sc
            for idx_sample in range(1, len(self.prod_likelihoods_p[idx_ind])):
                cell_gts = []
                # [j / non_j]
                cell_gts.append(
                    log(2)
                    + self.mu[idx_ind].get((j_allele, j_allele))
                    + self.mu[idx_ind].get((j_allele, k_allele))
                    + self.prod_likelihoods_p[idx_ind][idx_sample][1]
                )
                # [j, j]
                cell_gts.append(
                    self.mu[idx_ind].get((j_allele, j_allele)) * 2
                    + self.prod_likelihoods_p[idx_ind][idx_sample][0]
                )
                gt_germ_refhom += logsumexp(cell_gts)
        # breakpoint()
        return gt_germ_refhom, gt_germ_het

    def train_refine(self, reads_data_seq, reads_data_qual):
        """refine EM"""
        # init
        self.allele_pool_refine = []
        for v in self.mu_mosaic:
            vs = set([v[0], v[1], v[-1]])
            self.allele_pool_refine.append(sorted([v for v in vs if v]))
        self.mu_refine = [
            dict(
                zip(
                    tools.cartesian_product(ind_allele_pool),
                    itertools.repeat(log(self.mut_rate)),
                )
            )
            for ind_allele_pool in self.allele_pool_refine
        ]
        for idx_m, m in enumerate(self.mu_refine):
            for k in m.keys():
                if k[0] == k[1]:
                    self.mu_refine[idx_m][k] = log(1 - self.mut_rate)
        self.genotypes_refine = [
            tools.cartesian_prod(pool) for pool in self.allele_pool_refine
        ]
        termination = False
        if not self.phasable:  # unphasble
            self.sample_likelihoods_refine = []
            for idx_ind, bam_ind in enumerate(reads_data_seq):
                ind_sample_likelihood = []
                # pooled bulk
                ind_sample_likelihood.append(
                    [
                        self.gt_likelihoods(
                            gt,
                            bam_ind[0],
                            reads_data_qual[idx_ind][0],
                            "bulk",
                        )
                        for gt in self.genotypes_refine[idx_ind]
                    ]
                )
                # single cells
                for idx_reads, reads in enumerate(bam_ind[1]):
                    ind_sample_likelihood.append(
                        [
                            self.gt_likelihoods(
                                gt,
                                reads,
                                reads_data_qual[idx_ind][1][idx_reads],
                                self.sc_prot_list[idx_ind][idx_reads],
                            )
                            for gt in self.genotypes_refine[idx_ind]
                        ]
                    )
                self.sample_likelihoods_refine.append(ind_sample_likelihood)

            self.sample_likelihoods_refine = [
                [tools.normalize_log(ll) for ll in ind]
                for ind in self.sample_likelihoods_refine
            ]
        for _ in range(10):
            # ---------------------- E-step ----------------------
            if self.phasable:
                # Calculation of genotype likelihoods for all reads
                # num_ind * num_sample * num_gt
                self.sample_likelihoods_refine = []
                for idx_ind, bam_ind in enumerate(reads_data_seq):
                    if self.phasable[idx_ind]:
                        ind_sample_likelihood = []
                        # pooled bulk
                        if self.num_spn[idx_ind][0]:
                            ind_sample_likelihood.append(
                                [
                                    self.gt_phase_likelihoods(
                                        idx_ind,
                                        gt,
                                        bam_ind[0],
                                        reads_data_qual[idx_ind][0],
                                        "bulk",
                                    )
                                    for gt in self.genotypes_refine[idx_ind]
                                ]
                            )
                        else:
                            ind_sample_likelihood.append(
                                [
                                    self.gt_likelihoods(
                                        gt,
                                        bam_ind[0],
                                        reads_data_qual[idx_ind][0],
                                        "bulk",
                                    )
                                    for gt in self.genotypes_refine[idx_ind]
                                ]
                            )
                        # single cells
                        for idx_reads, reads in enumerate(bam_ind[1]):
                            if (
                                self.num_spn[idx_ind][idx_reads + 1]
                                and self.af[idx_ind][idx_reads]
                            ):
                                ind_sample_likelihood.append(
                                    [
                                        self.gt_phase_likelihoods(
                                            idx_ind,
                                            gt,
                                            reads,
                                            reads_data_qual[idx_ind][1][idx_reads],
                                            self.sc_prot_list[idx_ind][idx_reads],
                                            self.af[idx_ind][idx_reads],
                                        )
                                        for gt in self.genotypes_refine[idx_ind]
                                    ]
                                )
                            else:
                                ind_sample_likelihood.append(
                                    [
                                        self.gt_likelihoods(
                                            gt,
                                            reads,
                                            reads_data_qual[idx_ind][1][idx_reads],
                                            self.sc_prot_list[idx_ind][idx_reads],
                                        )
                                        for gt in self.genotypes_refine[idx_ind]
                                    ]
                                )

                        self.sample_likelihoods_refine.append(ind_sample_likelihood)
                    else:
                        ind_sample_likelihood = []
                        # pooled bulk
                        ind_sample_likelihood.append(
                            [
                                self.gt_likelihoods(
                                    gt,
                                    bam_ind[0],
                                    reads_data_qual[idx_ind][0],
                                    "bulk",
                                )
                                for gt in self.genotypes_refine[idx_ind]
                            ]
                        )
                        # single cells
                        for idx_reads, reads in enumerate(bam_ind[1]):
                            ind_sample_likelihood.append(
                                [
                                    self.gt_likelihoods(
                                        gt,
                                        reads,
                                        reads_data_qual[idx_ind][1][idx_reads],
                                        self.sc_prot_list[idx_ind][idx_reads],
                                    )
                                    for gt in self.genotypes_refine[idx_ind]
                                ]
                            )
                        self.sample_likelihoods_refine.append(ind_sample_likelihood)

                self.sample_likelihoods_refine = [
                    [tools.normalize_log(ll) for ll in ind]
                    for ind in self.sample_likelihoods_refine
                ]
            # new_total_sample_LL = sum(
            #     list(tools.flatten(self.sample_likelihoods_refine))
            # )  # before or after normalize, use diff of params

            # Calculation of each individuals' germline priors
            # for each candidate genotypes
            # num_ind * num_gt
            if self.mode == "sp":
                self.germ_priors_unnorm_refine = [
                    [
                        self.gt_germ_priors_refine(gt, idx_ind)
                        for gt in self.genotypes_refine[idx_ind]
                    ]
                    for idx_ind in range(self.num_ind)
                ]
                # normalization
                self.germ_priors_refine = [
                    tools.normalize_log(ll) for ll in self.germ_priors_unnorm_refine
                ]
            else:
                # Calculate mu-based bulk sample likelihoods
                self.bulk_mu_likelihoods_refine = []
                for idx_ind, bam_ind in enumerate(reads_data_seq):
                    self.bulk_mu_likelihoods_refine.append(
                        [
                            self.gt_mu_likelihoods(
                                gt,
                                bam_ind[0],
                                reads_data_qual[idx_ind][0],
                                idx_ind,
                            )
                            for gt in self.genotypes_refine[idx_ind]
                        ]
                    )
                self.bulk_mu_likelihoods_refine = [
                    tools.normalize_log(ind) for ind in self.bulk_mu_likelihoods_refine
                ]
                # Calculation of each individuals' mu-based germline priors
                # for each candidate genotypes
                # num_ind * num_gt
                self.mu_germ_priors_unnorm_refine = [
                    [
                        self.gt_germ_priors_refine(gt, idx_ind, mu_based=True)
                        for gt in self.genotypes_refine[idx_ind]
                    ]
                    for idx_ind in range(self.num_ind)
                ]
                # normalization
                self.mu_germ_priors_refine = [
                    tools.normalize_log(ll) for ll in self.mu_germ_priors_unnorm_refine
                ]

            # region calculate phasing probabilities
            # if self.phasable:
            #     for idx_ind in range(self.num_ind):
            #         if not self.phasable[idx_ind]:
            #             continue
            #         for phase_gt in self.phase_probs[idx_ind]:
            #             j_allele, k_allele = phase_gt
            #             hap_prob_1 = self.phasing_prob(idx_ind, (j_allele, k_allele))
            #             hap_prob_2 = self.phasing_prob(idx_ind, (k_allele, j_allele))
            #             hap_prob_sum = tools.normalize_log(hap_prob_1 + hap_prob_2)
            #             self.phase_probs[idx_ind][phase_gt] = logsumexp(
            #                 hap_prob_sum[:3]
            #             ) - logsumexp(hap_prob_sum)
            # endregion calculate phasing probabilities

            # Calculation of each single cell sample's posteriors for given germline
            # num_ind * germ_gt * num_sc_sample * num_gt, all sc, idx_sc starts from 0
            self.sc_posteriors_refine = []
            for idx_ind in range(self.num_ind):
                germ_posteriors = []
                for gt_germ in self.genotypes_refine[idx_ind]:
                    # idx_sample starts from 1 cuz sc in `get_gt_likelihoods`
                    germ_smpl_posteriors = []
                    for idx_sample in range(1, self.num_sc_sample[idx_ind] + 1):
                        germ_smpl_posteriors.append(
                            tools.normalize_log(
                                [
                                    self.gt_sc_posteriors_refine(
                                        gt_sc, gt_germ, idx_ind, idx_sample
                                    )
                                    for gt_sc in self.genotypes_refine[idx_ind]
                                ]
                            )
                        )
                    germ_posteriors.append(germ_smpl_posteriors)
                self.sc_posteriors_refine.append(germ_posteriors)

            # Calc of sc sample's posteriors
            # ind * sample * gt
            self.sample_posteriors_refine = [
                [
                    [
                        self.gt_posteriors_refine(idx_ind, idx_sample, gt)
                        for gt in self.genotypes_refine[idx_ind]
                    ]
                    for idx_sample in range(self.num_sc_sample[idx_ind])
                ]
                for idx_ind in range(self.num_ind)
            ]
            # normalization
            self.sample_posteriors_refine = [
                [tools.normalize_log(ss) for ss in ind]
                for ind in self.sample_posteriors_refine
            ]

            # ---------------------- M-step ----------------------
            # mu
            # warnings.simplefilter("error")  # HACK:
            with warnings.catch_warnings(action="error"):
                for idx_ind in range(self.num_ind):
                    for j_allele in self.allele_pool_refine[idx_ind]:
                        for k_allele in self.allele_pool_refine[idx_ind]:
                            if j_allele == k_allele:
                                continue
                            # calc of mu_{i, j->k}
                            n_vals = []
                            for gt_germ in self.genotypes_refine[idx_ind]:
                                if j_allele not in gt_germ:
                                    continue
                                for gt_cell in self.genotypes_refine[idx_ind]:
                                    if k_allele not in gt_cell:
                                        continue
                                    if not (
                                        (
                                            gt_germ[0] == gt_germ[1]
                                            and j_allele in gt_cell
                                        )
                                        or (
                                            (
                                                (
                                                    gt_germ[0]
                                                    if gt_germ[0] != j_allele
                                                    else gt_germ[1]
                                                )
                                                == (
                                                    gt_cell[0]
                                                    if gt_cell[0] != k_allele
                                                    else gt_cell[1]
                                                )
                                            )
                                            and (
                                                (
                                                    gt_germ[0]
                                                    if gt_germ[0] != j_allele
                                                    else gt_germ[1]
                                                )
                                                not in (j_allele, k_allele)
                                            )
                                        )
                                    ):
                                        continue  # elimate ADO and multi-mut and jk->jk, remain jj->jk, jm->km
                                    val = self.get_gt_germ_priors_refine(
                                        idx_ind,
                                        gt_germ,
                                        mu_based=(False if self.mode == "sp" else True),
                                    )
                                    val += logsumexp(
                                        [
                                            self.get_sc_posteriors_refine(
                                                idx_ind, idx_sample, gt_germ, gt_cell
                                            )
                                            for idx_sample in range(
                                                0, self.num_sc_sample[idx_ind]
                                            )
                                        ]
                                    )
                                    n_vals.append(val)

                            d_vals = []
                            for gt_germ in self.genotypes_refine[idx_ind]:
                                if j_allele not in gt_germ:
                                    continue
                                val = self.get_gt_germ_priors_refine(
                                    idx_ind,
                                    gt_germ,
                                    mu_based=(False if self.mode == "sp" else True),
                                ) + log(self.num_sc_sample[idx_ind])
                                if gt_germ[0] == gt_germ[1]:
                                    val += log(2)
                                d_vals.append(val)
                            try:
                                self.mu_refine[idx_ind][(j_allele, k_allele)] = (
                                    logsumexp(n_vals) - logsumexp(d_vals)
                                )
                            except:
                                self.mu_refine[idx_ind][(j_allele, k_allele)] = -100
                # j->j, should have non_j_allele.
                for idx_ind in range(self.num_ind):
                    for j_allele in self.allele_pool_refine[idx_ind]:
                        try:
                            l_factor = logsumexp(
                                [
                                    self.mu_refine[idx_ind][(j_allele, k_allele)]
                                    for k_allele in self.allele_pool_refine[idx_ind]
                                    if j_allele != k_allele
                                ]
                            )
                            self.mu_refine[idx_ind][(j_allele, j_allele)] = logsumexp(
                                [0, l_factor], b=[1, -1]
                            )
                        except:
                            self.mu_refine[idx_ind][(j_allele, j_allele)] = 0
                self.mu_refine = [
                    {key: value if value >= -100 else -100 for key, value in d.items()}
                    for d in self.mu_refine
                ]

            # determination of convergence based on params
            mu_refine = [
                dict((k, exp(v)) for k, v in ind.items()) for ind in self.mu_refine
            ]
            new_total_params = sum([abs(v) for ind in mu_refine for v in ind.values()])
            if self.threshold is None:
                self.threshold = abs(new_total_params * 0.0001)
            diff = new_total_params - self.total_params_refine
            self.total_params_refine = new_total_params
            if not self.args.quiet:
                tools.logger.info("%s %s", new_total_params, diff)
                # print([dict((k, exp(v)) for k, v in ind.items()) for ind in self.mu_refine])
            # break
            if termination:
                break
            if diff < self.threshold:
                break
                termination = True

        self.mu_refine = [
            dict((k, exp(v)) for k, v in ind.items()) for ind in self.mu_refine
        ]
        self.refined = True

    def get_gt_likelihoods_refine(self, idx_ind, idx_sample, genotype):
        idx_gt = self.genotypes_refine[idx_ind].index(genotype)
        return self.sample_likelihoods_refine[idx_ind][idx_sample][idx_gt]

    def get_mu_likelihoods_refine(self, idx_ind, genotype):
        idx_gt = self.genotypes_refine[idx_ind].index(genotype)
        return self.bulk_mu_likelihoods_refine[idx_ind][idx_gt]

    def gt_germ_priors_refine(self, gt_germ, idx_ind, mu_based=False):
        j_allele, k_allele = gt_germ
        if j_allele != k_allele:  # het
            gt_germ_priors = log(2)
            # assume 0 is bulk
            if mu_based:
                gt_germ_priors += self.get_mu_likelihoods_refine(
                    idx_ind, gt_germ
                )  # se mode
            else:
                # sp mode
                gt_germ_priors += self.get_gt_likelihoods_refine(idx_ind, 0, gt_germ)
            # assume rest is sc
            for idx_sample in range(1, len(self.sample_likelihoods_refine[idx_ind])):
                cell_gts = []
                for gt_sc in self.genotypes_refine[idx_ind]:
                    # ADO
                    if gt_sc == (j_allele, j_allele) or gt_sc == (k_allele, k_allele):
                        continue
                    if j_allele in gt_sc and k_allele not in gt_sc:  # [j / non_k] gt
                        non_k_allele = gt_sc[1] if gt_sc[1] != j_allele else gt_sc[0]
                        cell_gts.append(
                            self.mu_refine[idx_ind].get((k_allele, non_k_allele))
                            + self.mu_refine[idx_ind].get((j_allele, j_allele))
                            + self.get_gt_likelihoods_refine(idx_ind, idx_sample, gt_sc)
                        )
                    elif k_allele in gt_sc and j_allele not in gt_sc:  # [non_j / k] gt
                        non_j_allele = gt_sc[0] if gt_sc[0] != k_allele else gt_sc[1]
                        cell_gts.append(
                            self.mu_refine[idx_ind].get((j_allele, non_j_allele))
                            + self.mu_refine[idx_ind].get((k_allele, k_allele))
                            + self.get_gt_likelihoods_refine(idx_ind, idx_sample, gt_sc)
                        )
                # [j, k]
                cell_gts.append(
                    self.mu_refine[idx_ind].get((j_allele, j_allele))
                    + self.mu_refine[idx_ind].get((k_allele, k_allele))
                    + self.get_gt_likelihoods_refine(idx_ind, idx_sample, gt_germ)
                )
                gt_germ_priors += logsumexp(cell_gts)
        else:  # hom
            gt_germ_priors = 0
            # assume 0 is bulk
            if mu_based:
                gt_germ_priors += self.get_mu_likelihoods_refine(
                    idx_ind, gt_germ
                )  # se mode
            else:
                # sp mode
                gt_germ_priors += self.get_gt_likelihoods_refine(idx_ind, 0, gt_germ)
            # assume rest is sc
            for idx_sample in range(1, len(self.sample_likelihoods_refine[idx_ind])):
                # print(idx_sample)
                cell_gts = []
                # [j / non_j]
                for gt_sc in self.genotypes_refine[idx_ind]:
                    if j_allele in gt_sc and gt_sc[0] != gt_sc[1]:  # [j / non_k] gt
                        non_j_allele = gt_sc[1] if gt_sc[1] != j_allele else gt_sc[0]
                        cell_gts.append(
                            log(2)
                            + self.mu_refine[idx_ind].get((j_allele, j_allele))
                            + self.mu_refine[idx_ind].get((j_allele, non_j_allele))
                            + self.get_gt_likelihoods_refine(idx_ind, idx_sample, gt_sc)
                        )
                # [j, j]
                cell_gts.append(
                    # log(2 * exp(self.mu_refine[idx_ind].get((j_allele, j_allele))) - 1)
                    self.mu_refine[idx_ind].get((j_allele, j_allele)) * 2
                    + self.get_gt_likelihoods_refine(idx_ind, idx_sample, gt_germ)
                )
                gt_germ_priors += logsumexp(cell_gts)
        return gt_germ_priors

    def get_gt_germ_priors_refine(self, idx_ind, genotype, mu_based=False, norm=True):
        idx_gt = self.genotypes_refine[idx_ind].index(genotype)
        if norm:
            if mu_based or self.mode == "se":  # for mu bulk
                priors = self.mu_germ_priors_refine[idx_ind][idx_gt]
            else:
                priors = self.germ_priors_refine[idx_ind][idx_gt]
        else:  # for LR test
            if mu_based or self.mode == "se":  # for mu bulk
                priors = self.mu_germ_priors_unnorm[idx_ind][idx_gt]
            else:
                priors = self.germ_priors_unnorm[idx_ind][idx_gt]
        return priors

    def gt_sc_posteriors_refine(self, gt_sc, gt_germ, idx_ind, idx_sample):
        gt_sc_posteriors = self.get_gt_likelihoods_refine(idx_ind, idx_sample, gt_sc)
        if gt_sc == gt_germ:
            return (
                gt_sc_posteriors
                + (
                    log(2)
                    if ((not self.phasable) or (not self.phasable[idx_ind]))
                    else 0
                )
                + self.mu_refine[idx_ind][(gt_germ[0], gt_sc[0])]
                + self.mu_refine[idx_ind][(gt_germ[1], gt_sc[1])]
            )
        elif gt_sc == (gt_germ[1], gt_germ[0]):
            return (
                gt_sc_posteriors
                + (
                    log(2)
                    if ((not self.phasable) or (not self.phasable[idx_ind]))
                    else 0
                )
                + self.mu_refine[idx_ind][(gt_germ[0], gt_sc[1])]
                + self.mu_refine[idx_ind][(gt_germ[0], gt_sc[1])]
            )
        elif (gt_germ[0] == gt_germ[1]) and (gt_germ[0] in gt_sc):  # jj->jk
            mut_allele = gt_sc[0] if gt_sc[0] != gt_germ[0] else gt_sc[1]
            return (
                gt_sc_posteriors
                + (
                    log(2)
                    if ((not self.phasable) or (not self.phasable[idx_ind]))
                    else 0
                )
                + self.mu_refine[idx_ind][(gt_germ[0], gt_germ[0])]
                + self.mu_refine[idx_ind][(gt_germ[0], mut_allele)]
            )
        elif len(tuple(set(gt_sc) & set(gt_germ))) == 1:  # jm->km
            m_allele = tuple(set(gt_sc) & set(gt_germ))[0]
            j_allele = gt_germ[0] if gt_germ[0] != m_allele else gt_germ[1]
            k_allele = gt_sc[0] if gt_sc[0] != m_allele else gt_sc[1]
            return (
                gt_sc_posteriors
                + self.mu_refine[idx_ind][(j_allele, k_allele)]
                + self.mu_refine[idx_ind][(m_allele, m_allele)]
            )
        else:
            return float("-inf")

    def gt_posteriors_refine(self, idx_ind, idx_sample, gt_sc):
        gt_posteriors = []
        for gt_germ in self.genotypes_refine[idx_ind]:
            gt_posteriors.append(
                self.get_sc_posteriors_refine(idx_ind, idx_sample, gt_germ, gt_sc)
                + self.get_gt_germ_priors_refine(idx_ind, gt_germ)
            )
        return logsumexp(gt_posteriors)

    def get_sc_posteriors_refine(self, idx_ind, idx_sample, gt_germ, gt_sc):
        idx_gt_germ = self.genotypes_refine[idx_ind].index(gt_germ)
        idx_gt_sc = self.genotypes_refine[idx_ind].index(gt_sc)
        return self.sc_posteriors_refine[idx_ind][idx_gt_germ][idx_sample][idx_gt_sc]

    def gt_sc_posteriors_final_refine(
        self,
        idx_ind,
        idx_sample,
        gt_germ,
        gt_mut,
        ref_allele,
        germ_allele,
        mosaic_allele,
    ):
        """Single cell sample's joint genotype posteriors."""
        # idx_sample 0 is bulk, starting from 1
        gt_sc_posteriors_germ = (
            self.get_gt_germ_priors_refine(idx_ind, gt_germ)
            + self.get_gt_likelihoods_refine(idx_ind, idx_sample, gt_germ)
            + log(
                self.mu_refine[idx_ind][(germ_allele, germ_allele)]
                * self.mu_refine[idx_ind][(ref_allele, ref_allele)]
            )
        )
        gt_sc_posteriors_mut = (
            self.get_gt_germ_priors_refine(idx_ind, gt_germ)
            + self.get_gt_likelihoods_refine(idx_ind, idx_sample, gt_mut)
            + log(
                self.mu_refine[idx_ind][(germ_allele, mosaic_allele)]
                * self.mu_refine[idx_ind][(ref_allele, ref_allele)]
            )
        )
        # print(idx_sample, gt_sc_posteriors_germ, gt_sc_posteriors_mut, sep="\n")
        return tools.normalize_log([gt_sc_posteriors_germ, gt_sc_posteriors_mut])
