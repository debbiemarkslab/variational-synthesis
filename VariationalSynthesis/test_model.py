import numpy as np
import pytest
import torch

from pyro.contrib.mue.dataloaders import BiosequenceDataset, alphabets

from StochSynthSample.synthesis.model import SynthesisModel, bu
from StochSynthSample.synthesis.moment_comparison import estimate_cross_cov

import pdb


def test_alphabet():
    assert ''.join(alphabets['amino-acid']) == ''.join(bu.alphabets['aa'])[:20]


@pytest.mark.parametrize('K', [2, 3])
@pytest.mark.parametrize('C', [2, 5])
@pytest.mark.parametrize('L', [6, 8])
@pytest.mark.parametrize('assembly', ['combinatorial', 'deterministic'])
@pytest.mark.parametrize('alph_unit', ['nuc', 'codon'])
@pytest.mark.parametrize('alph_constraint',
                         ['enzymatic', 'finite', 'arbitrary'])
def test_SynthesisModel(K, C, L, assembly, alph_unit, alph_constraint):

    # Unavailable option combinations.
    if (alph_unit == 'codon' and alph_constraint == 'enzymatic') or (
            alph_unit == 'nuc' and alph_constraint == 'arbitrary'):
        return

    torch.set_default_dtype(torch.float64)

    # Setup data.
    N = 10
    seqs = [''.join(np.random.choice(bu.alphabets['aa'][:-1],
                                     np.random.randint(1, L)))
            for i in range(N)]
    # Duplicate to check batching.
    seqs = seqs + seqs
    dataset = BiosequenceDataset(seqs, include_stop=True, max_length=L)

    # Initialize model.
    alphabet_size = 3
    Ls = [len(elem) for elem in torch.split(torch.arange(L),
                                            torch.floor(torch.tensor(L/K)))]
    if len(Ls) > K:
        Ls[-2] += Ls[-1]
        Ls.pop()
    synthesismodel = SynthesisModel(K, C, Ls, assembly, alph_unit,
                                    alph_constraint, alphabet_size,
                                    lr=0.0001, grad_steps=2)

    # Run EM.
    n_steps = 5
    Elogp = synthesismodel.train(dataset, n_steps, batch_size=N, decay=0.,
                                 shuffle=False)

    # Check monotonic increase.
    assert torch.all(torch.diff(Elogp) >= -1e-8)

    # Check evaluation.
    logps_tst, logpres_tst = synthesismodel.evaluate(dataset, batch_size=3)
    logps_chk = synthesismodel.train(dataset, 1, batch_size=len(seqs),
                                     shuffle=False, initialize=False)
    assert torch.allclose(logps_tst.sum(), logps_chk[-1])

    # Check marginal functions.
    x = synthesismodel.get_samples(10)
    chk_alpha = synthesismodel._make_alpha(x)
    tst_alpha = torch.einsum('cjd,ijd->icj', synthesismodel._make_c_marg_mat(),
                             x)
    assert torch.allclose(tst_alpha, chk_alpha)

    # Check sampling function.
    samps = synthesismodel.get_samples(10000)
    tst_mean = samps.mean(0)
    chk_mean = synthesismodel.get_seq_marg()
    assert torch.allclose(tst_mean, chk_mean, 0.1, 0.1)

    if assembly == 'combinatorial':
        first_pos = torch.sum(samps[:, 0, :10], -1)
        last_pos = torch.sum(samps[:, -1, :10], -1)
        corr = ((first_pos * last_pos).mean()
                - first_pos.mean() * last_pos.mean())
        assert torch.allclose(corr, torch.tensor(0.0), 0.1, 0.1)

    # Check covariance.
    tst_cov = synthesismodel.get_cross_cov()
    chk_cov = estimate_cross_cov(samps)
    assert torch.allclose(tst_cov, chk_cov, 0.1, 0.1)
