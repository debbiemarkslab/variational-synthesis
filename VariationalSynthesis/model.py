import argparse
import configparser
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import torch
from torch.distributions import OneHotCategorical
from torch import optim
from torch.utils.data import DataLoader

from pyro.contrib.mue.dataloaders import BiosequenceDataset
from VariationalSynthesis import bio_utils as bu


class SynthesisModel:

    def __init__(self, K, C, Ls, assembly='deterministic', alph_unit='codon',
                 alph_constraint='arbitrary', alphabet_size=None,
                 enzyme='mutazymeII', lr=0.01, grad_steps=5, tau_max=10,
                 epsilon=1e-300, pin_memory=False, cuda=False):

        self.K = K
        self.C = C
        self.Ls = Ls
        self.L = sum(Ls)
        self.D = len(bu.alphabets['aa'])
        self.B = len(bu.alphabets['dna'])
        self.indx = [slice(sum(Ls[:j]), sum(Ls[:(j+1)]))
                     for j in range(len(Ls))]
        assert assembly in ['deterministic', 'combinatorial']
        self.assembly = assembly
        assert alph_unit in ['nuc', 'codon']
        self.alph_unit = alph_unit
        assert alph_constraint in ['enzymatic', 'finite', 'arbitrary']
        assert not (alph_constraint == 'arbitrary' and alph_unit == 'nuc'), (
                'Not yet implemented.')
        assert not (alph_unit == 'codon' and alph_constraint == 'enzymatic'), (
                'Not available.')
        self.alph_constraint = alph_constraint
        self.lr = lr
        self.grad_steps = grad_steps
        self.pin_memory = pin_memory
        self.cuda = cuda
        if cuda:
            self.gen_device = torch.device('cuda')
        else:
            self.gen_device = torch.device('cpu')
        self.transfer = bu.transfer.to(self.gen_device)
        self.mask = bu.mask.to(self.gen_device)

        if alph_constraint == 'enzymatic':
            self.A = self.B
            submat = bu.substitution_mat[enzyme].to(self.gen_device)
            self.Stauln = torch.cat([torch.log(torch.matrix_power(
                            submat, tau))[None, :, :]
                            for tau in range(1, tau_max+1)], axis=0)
            self.tau_max = tau_max
        else:
            self.A = alphabet_size
        if alph_unit == 'nuc':
            self._make_rho(self.A)

        self.epsilon = epsilon
        self.params = dict()

    def _make_rho(self, A):
        """Construct the constant rho for converting codon representations."""
        rho = torch.zeros((A**3, 3, A))
        for n in range(rho.shape[0]):
            for l in range(3):
                for a in range(A):
                    rho[n, l, a] = torch.tensor(
                            float(math.floor((n % (A**(l+1)))/(A**l)) == a))
        self.rho = rho

    def _make_alpha(self, x):
        """Construct log(u . T . x)"""
        if self.alph_unit == 'codon':
            if self.alph_constraint == 'arbitrary':
                return torch.einsum('cjd,ijd->icj', self.params['uln'], x)
            elif self.alph_constraint == 'finite':
                return torch.einsum('cja,ad,ijd->icj', self.params['xt'],
                                    self.params['vln'], x)
        if self.alph_unit == 'nuc':
            if self.alph_constraint == 'finite':
                vtilde = self.params['vtilde']
                vln = vtilde - torch.logsumexp(vtilde, axis=1, keepdim=True)
            elif self.alph_constraint == 'enzymatic':
                vln = self.Stauln[self.params['tau']-1]
            nuc_lp = torch.einsum('cjla,ab->cjlb', self.params['xt'], vln)
            aa_lp = bu.codon_to_aa_lp(nuc_lp, self.transfer, self.mask)
            return torch.einsum('cjd,ijd->icj', aa_lp, x)

    def _make_c_marg_mat(self):
        """Construct log(u . T)"""
        if self.alph_unit == 'codon':
            if self.alph_constraint == 'arbitrary':
                return self.params['uln']
            elif self.alph_constraint == 'finite':
                return torch.einsum('cja,ad->cjd', self.params['xt'],
                                    self.params['vln'])
        if self.alph_unit == 'nuc':
            if self.alph_constraint == 'finite':
                vtilde = self.params['vtilde']
                vln = vtilde - torch.logsumexp(vtilde, axis=1, keepdim=True)
            elif self.alph_constraint == 'enzymatic':
                vln = self.Stauln[self.params['tau']-1]
            nuc_lp = torch.einsum('cjla,ab->cjlb', self.params['xt'], vln)
            aa_lp = bu.codon_to_aa_lp(nuc_lp, self.transfer, self.mask)
            return aa_lp

    def get_seq_marg(self):
        """Get marginals at each sequence of synthesis model."""
        with torch.no_grad():
            per_well_marg = torch.exp(self._make_c_marg_mat())
            if self.assembly == 'deterministic':
                return torch.einsum('cjd,c->jd',
                                    per_well_marg, self.params['w'])
            elif self.assembly == 'combinatorial':
                margs = []
                for k in range(self.K):
                    margs.append(torch.einsum('cjd,c->jd',
                                              per_well_marg[:, self.indx[k]],
                                              self.params['w'][:, k]))
                return torch.cat(margs, 0)

    def get_cross_cov(self):
        """Get L2 norm of cross covariance of synthesis model."""
        with torch.no_grad():
            per_well_marg = torch.exp(self._make_c_marg_mat())
            diag_ind = (torch.eye(per_well_marg.shape[1])[:, :, None, None] *
                        torch.eye(per_well_marg.shape[2])[None, None, :, :]
                        ).to(self.gen_device)
            if self.assembly == 'deterministic':
                total_marg = torch.einsum('cjd,c->jd', per_well_marg,
                                          self.params['w'])
                indep_probs = torch.einsum('jd,ef->jedf', total_marg,
                                           total_marg)
                dep_probs = torch.einsum('c,cjd,cef->jedf', self.params['w'],
                                         per_well_marg, per_well_marg)
            elif self.assembly == 'combinatorial':
                indep_probs = torch.zeros_like(diag_ind)
                dep_probs = torch.zeros_like(diag_ind)
                margs = []
                for k in range(self.K):
                    k_marg_mat = per_well_marg[:, self.indx[k]]
                    total_marg = torch.einsum('cjd,c->jd',
                                              k_marg_mat,
                                              self.params['w'][:, k])
                    indep_probs[self.indx[k], self.indx[k]] = torch.einsum(
                            'jd,ef->jedf', total_marg, total_marg)
                    dep_probs[self.indx[k], self.indx[k]] = torch.einsum(
                            'c,cjd,cef->jedf',
                            self.params['w'][:, k],
                            k_marg_mat, k_marg_mat)
                    margs.append(total_marg)
                total_marg = torch.cat(margs, 0)
            dep_probs = dep_probs - dep_probs * diag_ind
            dep_probs = dep_probs + diag_ind * total_marg[:, None, :, None]
            return torch.sqrt(torch.square(dep_probs-indep_probs).sum((2, 3)))

    def get_samples(self, n):
        """Sample from the synthesis model."""
        with torch.no_grad():
            per_well_marg = torch.exp(self._make_c_marg_mat())
            if self.assembly == 'deterministic':
                well_dists = [torch.distributions.OneHotCategorical(ps)
                              for ps in per_well_marg]
                ws_dist = torch.distributions.Multinomial(n, self.params['w'])
                num_ws = ws_dist.sample().to(torch.long)
                return torch.cat([dist.sample((ni,))
                                  for dist, ni in zip(well_dists, num_ws)
                                  if ni > 0], 0)[torch.randperm(n)]
            elif self.assembly == 'combinatorial':
                seqs = []
                for k in range(self.K):
                    well_dists = [torch.distributions.OneHotCategorical(ps)
                                  for ps in per_well_marg[:, self.indx[k]]]
                    ws_dist = torch.distributions.Multinomial(
                        n, self.params['w'][:, k])
                    num_ws = ws_dist.sample().to(torch.long)
                    segs = torch.cat([
                                dist.sample((ni,))
                                for dist, ni in zip(well_dists, num_ws)
                                if ni > 0], 0)
                    # Randomize for independence across segments.
                    seqs.append(segs[torch.randperm(n)])
                return torch.cat(tuple(seqs), 1)

    def get_log_probs(self, x):
        """Get log probabilities for sequences x"""
        with torch.no_grad():
            alpha = self._make_alpha(x)
            if self.assembly == 'deterministic':
                rtilde = (torch.sum(alpha, axis=2) +
                          torch.log(self.params['w'] + self.epsilon)[None, :])
                rtilde_norm = torch.logsumexp(rtilde, axis=1)
            elif self.assembly == 'combinatorial':
                rtilde_norm = torch.zeros(x.shape[0])
                for k in range(self.K):
                    rtilde = (torch.sum(alpha[:, :, self.indx[k]], axis=2) +
                              torch.log(self.params['w']
                                        + self.epsilon)[None, :, k])
                    rtilde_norm += torch.logsumexp(rtilde, axis=1)
        return rtilde_norm

    def online_update(self, new, prev, step, decay=-0.6):
        """Smoothing update for stochastic optimization."""
        if step == 0:
            return new
        else:
            gamma = step**decay
            return prev + gamma*(new - prev)

    def E_step(self, x, step, decay=-0.6):
        """E step in EM algorithm."""
        with torch.no_grad():
            alpha = self._make_alpha(x)
            if self.assembly == 'deterministic':
                rtilde = (torch.sum(alpha, axis=2) +
                          torch.log(self.params['w'] + self.epsilon)[None, :])
                rtilde_norm = torch.logsumexp(rtilde, axis=1, keepdim=True)
                logp = torch.sum(rtilde_norm)
                r = torch.exp(rtilde - rtilde_norm)
                rmn = r.mean(axis=0)
                rxmn = torch.einsum('ic,ijd->cjd', r, x) / x.shape[0]
            elif self.assembly == 'combinatorial':
                logp = torch.tensor(0.)
                r = torch.zeros([x.shape[0], self.C, self.K])
                rxmn = torch.zeros([self.C, self.K, self.L, self.D])
                for k in range(self.K):
                    rtilde = (torch.sum(alpha[:, :, self.indx[k]], axis=2) +
                              torch.log(self.params['w']
                                        + self.epsilon)[None, :, k])
                    rtilde_norm = torch.logsumexp(rtilde, axis=1, keepdim=True)
                    logp += torch.sum(rtilde_norm)
                    r[:, :, k] = torch.exp(rtilde - rtilde_norm)
                    rxmn[:, k, self.indx[k], :] = torch.einsum(
                            'ic,ijd->cjd', r[:, :, k],
                            x[:, self.indx[k], :]) / x.shape[0]
                rmn = r.mean(axis=0)
        self.params['rmn'] = self.online_update(rmn, self.params['rmn'],
                                                step, decay=decay)
        self.params['rxmn'] = self.online_update(rxmn, self.params['rxmn'],
                                                 step, decay=decay)
        return logp

    def M_step(self):
        """M step in EM algorithm."""
        # w parameter.
        self.params['w'] = self.params['rmn']
        # u parameter.
        dtype = self.params['w'].dtype
        if self.alph_constraint == 'arbitrary' and self.alph_unit == 'codon':
            if self.assembly == 'deterministic':
                utilde = torch.log(self.params['rxmn'] + self.epsilon)
                self.params['uln'] = (
                        utilde - torch.logsumexp(utilde, axis=2, keepdim=True))
            elif self.assembly == 'combinatorial':
                for k in range(self.K):
                    utilde = torch.log(self.params['rxmn'][:, k, self.indx[k]]
                                       + self.epsilon)
                    self.params['uln'][:, self.indx[k], :] = (
                        utilde - torch.logsumexp(utilde, axis=2, keepdim=True))
        elif self.alph_constraint == 'finite' and self.alph_unit == 'codon':
            if self.assembly == 'deterministic':
                amax = torch.argmax(
                    torch.einsum('ad,cjd->cja', self.params['vln'],
                                 self.params['rxmn']), axis=2, keepdim=True)
                self.params['xt'] = (
                    amax == torch.arange(self.A)[None, None, :]).to(dtype)
                vtilde = torch.log(torch.einsum(
                    'cja,cjd->ad', self.params['xt'], self.params['rxmn'])
                    + self.epsilon)
                self.params['vln'] = (
                    vtilde - torch.logsumexp(vtilde, axis=1, keepdim=True))
            elif self.assembly == 'combinatorial':
                vtilde = torch.zeros((self.A, self.D))
                for k in range(self.K):
                    amax = torch.argmax(
                        torch.einsum('ad,cjd->cja', self.params['vln'],
                                     self.params['rxmn'][:, k, self.indx[k]]),
                        axis=2, keepdim=True)
                    self.params['xt'][:, self.indx[k], :] = (
                        amax == torch.arange(self.A)[None, None, :]).to(dtype)
                    vtilde += torch.einsum(
                            'cja,cjd->ad',
                            self.params['xt'][:, self.indx[k]],
                            self.params['rxmn'][:, k, self.indx[k]])
                vtilde = torch.log(vtilde + self.epsilon)
                self.params['vln'] = (
                    vtilde - torch.logsumexp(vtilde, axis=1, keepdim=True))
        elif (self.alph_constraint in ['finite', 'enzymatic']
              and self.alph_unit == 'nuc'):
            with torch.no_grad():
                if self.alph_constraint == 'finite':
                    vtilde = self.params['vtilde']
                    vln = vtilde - torch.logsumexp(vtilde, axis=1,
                                                   keepdim=True)
                elif self.alph_constraint == 'enzymatic':
                    vln = self.Stauln[self.params['tau']-1]
                aa_lp = bu.codon_to_aa_lp(torch.einsum('nla,ab->nlb',
                                                       self.rho, vln),
                                          self.transfer, self.mask)
                if self.assembly == 'deterministic':
                    rxlp = torch.einsum('nd,cjd->cjn', aa_lp,
                                        self.params['rxmn'])
                    amax = torch.argmax(rxlp, axis=2, keepdim=True)
                    amax_oh = (amax == torch.arange(self.A**3)[None, None, :]
                               ).to(dtype)
                    self.params['xt'] = torch.einsum('nla,ijn->ijla', self.rho,
                                                     amax_oh)
                elif self.assembly == 'combinatorial':
                    for k in range(self.K):
                        rxlp = torch.einsum(
                                'nd,cjd->cjn', aa_lp,
                                self.params['rxmn'][:, k, self.indx[k]])
                        amax = torch.argmax(rxlp, axis=2, keepdim=True)
                        amax_oh = (amax ==
                                   torch.arange(self.A**3)[None, None, :]
                                   ).to(dtype)
                        self.params['xt'][:, self.indx[k], :] = torch.einsum(
                                'nla,ijn->ijla', self.rho, amax_oh)
            if self.alph_constraint == 'finite':
                for gstep in range(self.grad_steps):
                    self.optimizer.zero_grad()
                    vtilde = self.params['vtilde']
                    vln = vtilde - torch.logsumexp(vtilde, axis=1,
                                                   keepdim=True)
                    nuc_lp = torch.einsum('cjla,ab->cjlb', self.params['xt'],
                                          vln)
                    aa_lp = bu.codon_to_aa_lp(nuc_lp, self.transfer, self.mask)
                    if self.assembly == 'deterministic':
                        loss = -torch.sum(aa_lp * self.params['rxmn'])
                    elif self.assembly == 'combinatorial':
                        loss = torch.tensor(0.)
                        for k in range(self.K):
                            loss -= torch.sum(
                                aa_lp[:, self.indx[k]] *
                                self.params['rxmn'][:, k, self.indx[k]])
                    loss.backward()
                    self.optimizer.step()
            elif self.alph_constraint == 'enzymatic':
                t_Elp = torch.zeros(self.tau_max)
                # Iterate over t to avoid memory overflow.
                for t in range(self.tau_max):
                    nuc_lp = torch.einsum('cjla,ab->cjlb', self.params['xt'],
                                          self.Stauln[t])
                    aa_lp = bu.codon_to_aa_lp(nuc_lp, self.transfer, self.mask)
                    if self.assembly == 'deterministic':
                        t_Elp[t] = torch.einsum('cjd,cjd->', aa_lp,
                                                self.params['rxmn'])
                    elif self.assembly == 'combinatorial':
                        for k in range(self.K):
                            t_Elp[t] += torch.einsum(
                                'cjd,cjd->', aa_lp[:, self.indx[k]],
                                self.params['rxmn'][:, k, self.indx[k]])
                self.params['tau'] = torch.argmax(t_Elp) + 1

    def initialize_EM(self, x):
        """Initialize EM parameters."""
        N = x.shape[0]
        if self.alph_constraint == 'arbitrary' and self.alph_unit == 'codon':
            utilde = torch.randn((self.C, self.L, self.D))
            self.params['uln'] = utilde - torch.logsumexp(
                    utilde, axis=2, keepdim=True)
        elif self.alph_constraint == 'finite' and self.alph_unit == 'codon':
            self.params['xt'] = OneHotCategorical(
                probs=(1/self.A)*torch.ones(self.A)).sample((self.C, self.L))
            vtilde = torch.randn((self.A, self.D))
            self.params['vln'] = (
                vtilde - torch.logsumexp(vtilde, axis=1, keepdim=True))
        elif self.alph_constraint == 'finite' and self.alph_unit == 'nuc':
            self.params['xt'] = OneHotCategorical(
                probs=(1/self.A)*torch.ones(self.A)
                ).sample((self.C, self.L, 3))
            self.params['vtilde'] = torch.randn((self.A, self.B),
                                                requires_grad=True)
            self.optimizer = optim.Adam([self.params['vtilde']], lr=self.lr)
        elif self.alph_constraint == 'enzymatic' and self.alph_unit == 'nuc':
            # Get device.
            device = torch.tensor(1.).device
            # Initialize cluster means with subsample of data
            xsub = x[torch.randperm(N)[:self.C]].clone().to(device)
            # Probability of each codon.
            collapse_codon = torch.sum(self.transfer, axis=(0, 1))
            codon_to_aa_prob = collapse_codon / torch.sum(collapse_codon,
                                                          axis=0)
            xtcodon_prob = torch.einsum('cjd,nd->cjn', xsub,
                                        codon_to_aa_prob)
            # For padded zeros use a uniform distribution.
            xtcodon_prob = xtcodon_prob + (
                torch.sum(xtcodon_prob, axis=-1, keepdim=True) < 0.0001
                ) / xtcodon_prob.shape[-1]
            # Sample codons.
            xtcodon = OneHotCategorical(probs=xtcodon_prob).sample()
            self.params['xt'] = torch.einsum('cjn,land->cjla', xtcodon,
                                             self.transfer)
            self.params['tau'] = 5

        # Initialize mixture component weights.
        if self.assembly == 'deterministic':
            self.params['w'] = (1/self.C) * torch.ones(self.C)
        elif self.assembly == 'combinatorial':
            self.params['w'] = (1/self.C) * torch.ones((self.C, self.K))

        # Initialize summary statistics.
        self.params['rmn'] = None
        self.params['rxmn'] = None

    def train(self, dataset, epochs, batch_size=None, decay=-0.6,
              polyak=True, shuffle=True, initialize=True):

        if batch_size is None:
            batch_size = dataset.data_size
        dataload = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              pin_memory=self.pin_memory,
                              generator=torch.Generator(
                                            device=self.gen_device))

        if initialize:
            self.initialize_EM(dataset.seq_data)
        logp = []
        n = 0
        tot_steps = int(epochs * math.ceil(dataset.data_size / batch_size))
        polyak_thresh = int(math.ceil(tot_steps / 2))
        for epoch in range(epochs):
            for seq_data, L_data in dataload:
                if self.cuda:
                    seq_data = seq_data.cuda()

                # E step.
                logp_batch = self.E_step(seq_data, n, decay=decay).cpu()
                n += 1

                # Batch estimate of log probability.
                logp.append((dataset.data_size/seq_data.shape[0]) * logp_batch)

                # Polyak-Ruppert averaging.
                if polyak:
                    if n == polyak_thresh:
                        rmn, rxmn = self.params['rmn'], self.params['rxmn']
                    elif n > polyak_thresh:
                        rmn += self.params['rmn']
                        rxmn += self.params['rxmn']
                        if n == tot_steps:
                            self.params['rmn'] = rmn/(n - polyak_thresh + 1)
                            self.params['rxmn'] = rxmn/(n - polyak_thresh + 1)

                # M step.
                self.M_step()

        return torch.tensor(logp, device=torch.device('cpu'))

    def evaluate(self, dataset, batch_size=None):
        """Compute total log probability and per residue logp."""
        if batch_size is None:
            batch_size = dataset.data_size
        dataload = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              pin_memory=self.pin_memory,
                              generator=torch.Generator(
                                            device=self.gen_device))
        logps = []
        logpres = []
        for seq_data, L_data in dataload:
            if self.cuda:
                seq_data = seq_data.cuda()
            batch_logps = self.get_log_probs(seq_data).cpu()
            logps.append(batch_logps)
            logpres.append(batch_logps/L_data)
        return torch.cat(logps), torch.cat(logpres)


def generate_example_data(small_test):
    """Generate mini example dataset."""
    if small_test:
        mult_dat = 1
    else:
        mult_dat = 10

    seqs = ['CACCA']*mult_dat + ['CAAC']*mult_dat + ['CACCC']*mult_dat
    dataset = BiosequenceDataset(seqs, 'list', 'amino-acid',
                                 include_stop=True)

    return dataset


def main(config):

    # Configure.
    test = config['general']['test'] == 'True'
    small = config['general']['small'] == 'True'
    rng_seed = int(config['general']['rng_seed'])
    file = config['general']['target_samples']
    out_folder = config['general']['out_folder']
    no_save = config['general']['no_save'] == 'True'
    cuda = config['general']['cuda'] == 'True'
    if cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        pin_memory = True
    else:
        torch.set_default_dtype(torch.float64)
        pin_memory = False
    cpu_data = config['general']['cpu_data'] == 'True'
    if cpu_data or not args.cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    include_stop = config['general']['include_stop'] == 'True'

    noligos = int(config['model']['noligos'])
    npools = int(config['model']['npools'])
    assembly = config['model']['assembly']
    unit = config['model']['unit']
    constraint = config['model']['constraint']
    alph_size = int(config['model']['alph_size'])
    enzyme = config['model']['enzyme']

    lr = float(config['train']['lr'])
    grad_steps = int(config['train']['grad_steps'])
    tau_max = int(config['train']['tau_max'])
    epochs = int(config['train']['epochs'])
    batch_size = config['train']['batch_size']
    if batch_size == 'None':
        batch_size = None
    else:
        batch_size = int(batch_size)
    polyak = config['train']['polyak'] == 'True'

    torch.set_default_dtype(torch.float64)

    # Load dataset
    if test:
        dataset = generate_example_data(small)
    else:
        alph = ''.join(bu.alphabets['aa'])
        if include_stop:
            alph = alph[:-1]  # Stop will be automatically added by loader.
        dataset = BiosequenceDataset(file, 'fasta', alph,
                                     include_stop=include_stop, device=device)

    # Seed.
    torch.manual_seed(rng_seed)

    # Construct model.
    K = npools
    C = noligos
    L = dataset.max_length
    # split evenly for now.
    Ls = [len(elem) for elem in torch.split(torch.arange(L),
                                            torch.ceil(torch.tensor(L/K)))]
    model = SynthesisModel(
               K, C, Ls, assembly=assembly, alph_unit=unit,
               alph_constraint=constraint,
               alphabet_size=alph_size, enzyme=enzyme,
               lr=lr, grad_steps=grad_steps,
               tau_max=tau_max, pin_memory=pin_memory, cuda=cuda)

    # Fit synthesis model.
    logp_trains = model.train(dataset, epochs, batch_size=batch_size,
                              polyak=polyak)

    # Evaluation.
    logps, logpres = model.evaluate(dataset, batch_size)

    # Plot.
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_folder = os.path.join(out_folder, 'logs', time_stamp)
    os.mkdir(out_folder)
    if not no_save:
        # Training curve.
        plt.figure(figsize=(8, 6))
        plt.plot(logp_trains)
        plt.xlabel('iteration', fontsize=18)
        plt.ylabel(r'$\log p$', fontsize=18)
        plt.savefig(os.path.join(out_folder, 'logp.pdf'))

        # Save parameters.
        params_file = os.path.join(out_folder, 'params.pkl')
        with open(params_file, 'wb') as pw:
            pickle.dump(model, pw)
            pickle.dump(logps, pw)
            pickle.dump(logpres, pw)

        # Evaluation results.
        logp_trains_file = os.path.join(out_folder, 'logp_trains.npy')
        np.save(logp_trains_file, logp_trains)
        config['results']['logp_trains'] = logp_trains_file
        config['results']['logp_per_seq'] = str(
                    (logps.mean()).numpy())
        config['results']['logp_per_res'] = str(logpres.mean().numpy())
        config['results']['perpl_per_res'] = str(
                torch.exp(-logpres.mean()).numpy())
        config['results']['params_file'] = params_file
        with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
            config.write(cw)

    return config, logp_trains


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Synthesis model.")
    parser.add_argument('configPath')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.configPath)

    main(config)
