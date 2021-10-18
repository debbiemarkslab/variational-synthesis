import numpy as np
from scipy.special import logsumexp
import torch
import pdb

# --- Fixed constants. ---
# Amino acid and dna alphabets. aa in same order as pyro dataloader.
alphabets = {'aa': list('RHKDESTNQCGPAVILMFYW*'),
             'dna': list('ATGC')}

# E. coli codon frequencies (* I couldn't find a global frequency table. *).
# From https://www.genscript.com/tools/codon-frequency-table.
codon_freqs = {'R': {'codon': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
                     'freq': [0.36, 0.36, 0.07, 0.11, 0.07, 0.04]},
               'H': {'codon': ['CAT', 'CAC'],
                     'freq': [0.57, 0.43]},
               'K': {'codon': ['AAA', 'AAG'],
                     'freq': [0.74, 0.26]},
               'D': {'codon': ['GAT', 'GAC'],
                     'freq': [0.63, 0.37]},
               'E': {'codon': ['GAA', 'GAG'],
                     'freq': [0.68, 0.32]},
               'S': {'codon': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
                     'freq': [0.17, 0.15, 0.14, 0.14, 0.16, 0.25]},
               'T': {'codon': ['ACT', 'ACC', 'ACA', 'ACG'],
                     'freq': [0.19, 0.40, 0.17, 0.25]},
               'N': {'codon': ['AAT', 'AAC'],
                     'freq': [0.49, 0.51]},
               'Q': {'codon': ['CAA', 'CAG'],
                     'freq': [0.34, 0.66]},
               'C': {'codon': ['TGT', 'TGC'],
                     'freq': [0.46, 0.54]},
               'G': {'codon': ['GGT', 'GGC', 'GGA', 'GGG'],
                     'freq': [0.35, 0.37, 0.13, 0.15]},
               'P': {'codon': ['CCT', 'CCC', 'CCA', 'CCG'],
                     'freq': [0.18, 0.13, 0.20, 0.49]},
               'A': {'codon': ['GCT', 'GCC', 'GCA', 'GCG'],
                     'freq': [0.18, 0.26, 0.23, 0.33]},
               'V': {'codon': ['GTT', 'GTC', 'GTA', 'GTG'],
                     'freq': [0.28, 0.20, 0.17, 0.35]},
               'I': {'codon': ['ATT', 'ATC', 'ATA'],
                     'freq': [0.49, 0.39, 0.11]},
               'L': {'codon': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
                     'freq': [0.14, 0.13, 0.12, 0.10, 0.04, 0.47]},
               'M': {'codon': ['ATG'],
                     'freq': [1.0]},
               'F': {'codon': ['TTT', 'TTC'],
                     'freq': [0.58, 0.42]},
               'Y': {'codon': ['TAT', 'TAC'],
                     'freq': [0.59, 0.41]},
               'W': {'codon': ['TGG'],
                     'freq': [1.0]},
               '*': {'codon': ['TAA', 'TAG', 'TGA'],
                     'freq': [0.61, 0.09, 0.30]}}


# --- Fixed constants. ---
# Substitution matrices.
# GeneMorph II Random Mutagenesis Kit, Instruction Manual, Catalog #200550,
# Revision C.0 https://www.chem-agilent.com/pdf/strata/200550.pdf
# Rate of all mutations: 10 mutations / 1000 kilobases = 0.01
# Fraction of mutations that are substitutions of A-T: 0.507
# Probability a particular site is mutated and it is an A or T sub.: 0.01 * 0.507
# Fraction of sites that are A or T: 0.5 (assumed).
# Probability a particular site is mutated given that it is an A or T: 0.00507/0.5
# = 0.01014 \approx 0.0101
# For G and C: 0.01 * 0.438 / 0.5 = 0.00876 \approx 0.0088
# Taq polymerase https://www.chem-agilent.com/pdf/strata/200550.pdf
# A: (4.9/1000) * (0.759/0.5) = 0.0074382 \approx 0.00744
# T: (4.9/1000) * (0.759/0.5) = 0.0074382 \approx 0.00744
# G: (4.9/1000) * (0.196/0.5) = 0.0019208 \approx 0.00192
# C: (4.9/1000) * (0.196/0.5) = 0.0019208 \approx 0.00192
substitution_mat = {'mutazymeII': torch.from_numpy(np.array(
  [[1.-0.0101, 0.0101*0.285/0.507, 0.0101*0.175/0.507, 0.0101*0.047/0.507],
   [0.0101*0.285/0.507, 1.-0.0101, 0.0101*0.047/0.507, 0.0101*0.175/0.507],
   [0.0088*0.255/0.438, 0.0088*0.141/0.438, 1.-0.0088, 0.0088*0.041/0.438],
   [0.0088*0.141/0.438, 0.0088*0.255/0.438, 0.0088*0.041/0.438, 1.-0.0088]])),
                    'taq': torch.from_numpy(np.array(
  [[1.-0.00744, 0.00744*0.409/0.759, 0.00744*0.276/0.759, 0.00744*0.073/0.759],
   [0.00744*0.409/0.759, 1.-0.00744, 0.00744*0.073/0.759, 0.00744*0.276/0.759],
   [0.00192*0.136/0.196, 0.00192*0.045/0.196, 1.-0.00192, 0.00192*0.014/0.196],
   [0.00192*0.045/0.196, 0.00192*0.136/0.196, 0.00192*0.014/0.196, 1.-0.00192]]
  ))}
# --- ---

# --- Codon functions ---
# Codon transfer matrix and max.
len_codon = 3
codon_list = [codon for aa in alphabets['aa']
              for codon in codon_freqs[aa]['codon']]
transfer = np.zeros((len_codon, len(alphabets['dna']), len(codon_list),
                     len(alphabets['aa'])))
eps = 1e-300
mask = np.zeros((len(codon_list), len(alphabets['aa']))) + np.log(eps)
ci = -1
for ai, aa in enumerate(alphabets['aa']):
    for codon in codon_freqs[aa]['codon']:
        ci += 1
        for ni, n in enumerate(list(codon)):
            transfer[ni, alphabets['dna'].index(n), ci, ai] = 1
        mask[ci, ai] = 0.
transfer = torch.from_numpy(transfer)
mask = torch.from_numpy(mask)


def dna_to_aa(seq):
    """Convert a one-hot encoded dna sequence into an aa sequence."""
    # Reshape by codons.
    seq_by_codon = torch.reshape(seq, [seq.shape[0],
                                       int(seq.shape[1]/len_codon),
                                       len_codon, len(alphabets['dna'])])
    # Identify codons.
    codons = torch.relu(torch.einsum('aijk,jklm->ailm', seq_by_codon, transfer)
                        - 2.)
    # Convert to amino acids.
    aa_seq = torch.sum(codons, axis=2)
    return aa_seq


def dna_to_aa_lp(seq_lp):
    """Convert a set of independent dna log prob. to aa log prob."""
    # Reshape by codons.
    seq_by_codon = torch.reshape(seq_lp, [seq_lp.shape[0],
                                          int(seq_lp.shape[1]/len_codon),
                                          len_codon, len(alphabets['dna'])])

    return codon_to_aa_lp(seq_by_codon)


def codon_to_aa_lp(seq_by_codon_lp, ctransfer=None, cmask=None):
    """Convert a set of independent per codon base log prob. to aa log prob."""
    if ctransfer is None:
        ctransfer = transfer
    if cmask is None:
        cmask = mask
    # Compute probability of drawing each codon for each amino acid.
    codon_lp = (torch.einsum('...ijk,jklm->...ilm', seq_by_codon_lp, ctransfer)
                + cmask[..., :, :])

    # Sum over probability of each codon for each amino acid.
    return torch.logsumexp(codon_lp, axis=-2)

# --- ---


# --- Error prone polymerase ---
def polymerase_mutation(seq, mut_enzyme, mut_rounds):
    """Sequence probability distribution produced by polymerase."""
    # Replace with log space matrix power?
    submat = torch.matrix_power(substitution_mat[mut_enzyme], mut_rounds)
    return torch.log(torch.einsum('aij,jk->aik', seq, submat))
