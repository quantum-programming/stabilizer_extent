import os
from itertools import product

import numpy as np
import scipy.sparse
from numba import njit
from qiskit.quantum_info import Pauli
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm

from exputils.math.popcount import popcount
from exputils.math.rref import enumerate_RREF
from exputils.stabilizer_group import total_stabilizer_group_size


@njit
def _make_Amat_sub(
    n: int, k: int, R: int, t: int, phases: np.ndarray, coeff: float
) -> np.ndarray:
    rows = np.zeros((1 << n, 1 << (k * (k + 1) // 2) + k), dtype=np.complex128)
    for x in range(1 << k):
        row_idx = t
        for idx in range(k):
            if x & (1 << idx):
                row_idx ^= R[idx]
        rows[row_idx] = phases[x]
    return rows * coeff


def make_Amat(n: int) -> np.ndarray:
    """make Amat for n qubits

    Reference: https://arxiv.org/abs/2008.05234

    All stabilizer states can be represented as follows:
    * \\ket{t} (if k=0)
    * 1/2^{k/2} \\sum_{x=0}^{2^k-1} (-1)^{x^T Q x} i^{c^T x} \\ket{Rx+t} (if k>0)

    It is enough to consider the case satisfying the following conditions:
    - $Q$ is top-left $k$ times $k$ F_2 matrix
    - $R$ is $k$ times $n-k$ F_2 rref matrix
    - $t$ belongs to the complement of the row space of $R$
    """
    ans = []
    for k in range(n + 1):
        if k == 0:
            rows = np.zeros((1 << n, 1 << n), dtype=np.complex128)
            for t in range(1 << n):
                rows[t][t] = 1.0
            ans.append(rows)
        else:
            coeff = 1 / np.sqrt(1 << k)
            Q_idxs = [-1 for _ in range(k * k)]
            Q_idx = 0
            for i in range(k):
                for j in range(i, k):
                    Q_idxs[i * k + j] = Q_idx
                    Q_idx += 1

            phases = np.ones((1 << k, 1 << (k * (k + 1) // 2 + k)), dtype=np.complex128)
            complexes = np.ones((1 << k, 1 << k), dtype=np.complex128)
            for c in range(1 << k):
                for x in range(1 << k):
                    complexes[c][x] *= 1j ** popcount(c & x)
            phase_idx = 0
            for Q in tqdm(range(1 << (k * (k + 1) // 2))):
                for x in range(1 << k):
                    for i in range(k):
                        for j in range(i, k):
                            if (
                                (x & (1 << i))
                                and (x & (1 << j))
                                and (Q & (1 << Q_idxs[i * k + j]))
                            ):
                                phases[x][phase_idx] *= -1
                for c in range(1 << k):
                    phases[:, phase_idx + c] = phases[:, phase_idx] * complexes[c]
                phase_idx += 1 << k

            for R, t_mask in enumerate_RREF(n, k):
                t = 0
                while True:
                    ans.append(_make_Amat_sub(n, k, R, t, phases, coeff))
                    t = (t + ~t_mask + 1) & t_mask
                    if t == 0:
                        break
    return np.block(ans)


def save_Amat():
    assert os.getcwd().endswith("stabilizer_extent")
    for n in range(1, 5 + 1):
        print(f"n={n}")
        Amat = make_Amat(n)
        Amat_csc = csc_matrix(Amat)
        Amat_csc.eliminate_zeros()
        assert np.isclose(np.abs(Amat_csc.data).min(), 2 ** (-n / 2))
        scipy.sparse.save_npz(f"data/Amat/Amat{n}.npz", Amat_csc)
        if n <= 3:
            # Check if Amat is correct
            # A stabilizer state stabilizes 2^n Pauli matrices
            pauli_matrices = []
            for idx in product("IXYZ", repeat=n):
                label = "".join(idx)
                pauli_matrices.append(Pauli(label).to_matrix())
            col_set = set()
            for col in tqdm(Amat.T, desc=f"checking n={n}"):
                cnt = 0
                for pauli in pauli_matrices:
                    lhs = pauli @ col
                    if np.allclose(lhs, col) or np.allclose(-lhs, col):
                        cnt += 1
                assert cnt == (1 << n)
                col_set.add(tuple(col))
            assert len(col_set) == total_stabilizer_group_size(n)
            print(f"n={n} is OK")


if __name__ == "__main__":
    save_Amat()
