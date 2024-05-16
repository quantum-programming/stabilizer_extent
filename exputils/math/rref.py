import numpy as np

from numba import njit
from itertools import combinations

from exputils.math.gauss_jordan import gauss_jordan_get_only_rank
from exputils.math.q_binom import q_binomial


@njit(cache=True)
def make_mat(
    bit: int,
    Is: np.ndarray,
    Js: np.ndarray,
    not_col_idxs: np.ndarray,
    default_matrix: np.ndarray,
) -> np.ndarray:
    mat = default_matrix.copy()
    for idx in range(len(Is)):
        if bit & (1 << idx):
            mat[Is[idx]] += 1 << not_col_idxs[Js[idx]]
    return mat


def enumerate_RREF(n: int, k: int, is_fast_mode: bool = False):
    """enumerate all k times n RREF matrixes (row full rank) over F_2.

    Reference:
        https://mathlandscape.com/rref-matrix

    Explanation:
        Let set n=5 k=2 and consider the following rref matrix:
           #   #   : col_idxs
        [0 1 1 0 1]
        [0 0 0 1 1]

        Then, the basis of the complementary space is:
         #   #   # : not_col_idxs
        [1 0 0 0 0]
        [0 0 1 0 0]
        [0 0 0 0 1]
    """
    assert 1 <= n and 1 <= k <= n
    cnt = 0
    for _col_idxs_tuple in combinations(range(n), k):
        col_idxs = list(_col_idxs_tuple)  # which columns are (0, ..., 1, ..., 0)^T
        not_col_idxs = []
        default_matrix = np.zeros(k, dtype=np.int32)
        complement = 0
        for col_idx in range(n):
            if col_idx in _col_idxs_tuple:
                default_matrix[col_idx - len(not_col_idxs)] += 1 << col_idx
            else:
                complement += 1 << col_idx
                not_col_idxs.append(col_idx)
        Is = []
        Js = []
        for i in range(k):
            for j in range(n - k):
                if col_idxs[i] < not_col_idxs[j]:
                    Is.append(i)
                    Js.append(j)
        sz = len(Is)

        Is = np.array(Is, dtype=np.int32)
        Js = np.array(Js, dtype=np.int32)
        not_col_idxs = np.array(not_col_idxs, dtype=np.int32)

        if is_fast_mode:
            yield (Is, Js, not_col_idxs, default_matrix, complement)
        else:
            for bit in range(1 << sz):
                yield (make_mat(bit, Is, Js, not_col_idxs, default_matrix), complement)
        cnt += 1 << sz

    assert cnt == q_binomial(n, k)


def test_rref():
    for n, k in [(1, 1), (2, 1), (2, 2), (3, 2), (4, 4)]:
        print(f"n={n}, k={k}")
        for mat, t_mask in enumerate_RREF(n, k):
            complement = []
            for idx in range(n):
                if t_mask & (1 << idx):
                    complement.append(1 << idx)
            assert len(complement) == n - k
            complement = np.array(complement, dtype=np.int32)
            print(f"{mat=}, {complement=}")
            assert gauss_jordan_get_only_rank(mat.copy(), n) == k
            if n - k > 0:
                assert gauss_jordan_get_only_rank(complement.copy(), n) == n - k
            assert gauss_jordan_get_only_rank(np.hstack([mat, complement]), n) == n
            print(
                *[bin(row)[2:].zfill(n)[::-1] for row in mat], sep="\n", end="\n---\n"
            )
            print(
                *[bin(row)[2:].zfill(n)[::-1] for row in complement],
                sep="\n",
                end="\n=====\n",
            )


if __name__ == "__main__":
    test_rref()
