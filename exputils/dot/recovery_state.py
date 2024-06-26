from typing import List

import numpy as np

from exputils.Amat.get import get_Amat_sparse
from exputils.Amat.recover import make_state_from_kQctR
from exputils.math.rref import enumerate_RREF, make_mat
from scipy.sparse import csc_matrix

from exputils.stabilizer_group import total_stabilizer_group_size


def make_reordering(k: int) -> List[int]:
    # bfs order | Amat order
    #  c   Q    |   c   Q
    #  8  765   |   0 345
    #  4   32   |   1  67
    #  1    0   |   2   8
    # reordering = [8, 4, 1, 7, 6, 5, 3, 2, 0]
    c_idxs = []
    q_idxs = []
    reorder_idx = k + k * (k + 1) // 2
    for i in range(k)[::-1]:
        reorder_idx -= 1
        c_idxs.append(reorder_idx)
        for _ in range(i + 1):
            reorder_idx -= 1
            q_idxs.append(reorder_idx)
    reordering = c_idxs + q_idxs
    if k == 3:
        assert reordering == [8, 4, 1, 7, 6, 5, 3, 2, 0]
    return reordering


def recovery_states_from_idxs(n: int, idxs: List[int]) -> csc_matrix:
    """recover states from the result of 'calc_dot' function

    Args:
        idxs (List[int]): the result of 'calc_dot' function

    Returns:
        List[np.ndarray]: the recovered states
    """
    assert len(idxs) > 0 and type(idxs) == list and type(idxs[0]) == int
    idxs.sort(reverse=True)
    idx = idxs[-1]
    csc_indices = []
    csc_data = []
    csc_indptr = [0]
    r = 0
    for k in range(n + 1):
        if k == 0:
            r = 1 << n
            while idx < r:
                csc_indices.append(idx)
                csc_data.append(1.0)
                csc_indptr.append(len(csc_indices))
                idxs.pop()
                if len(idxs) == 0:
                    return csc_matrix(
                        (csc_data, csc_indices, csc_indptr),
                        shape=(1 << n, len(csc_indptr) - 1),
                        dtype=np.complex128,
                    )
                idx = idxs[-1]
        else:
            reordering = make_reordering(k)
            for Is, Js, not_col_idxs, default_matrix, t_mask in enumerate_RREF(
                n, k, is_fast_mode=True
            ):
                for bit in range(1 << len(Is)):
                    if idx >= r + (1 << (n + k * (k + 1) // 2)):
                        r += 1 << (n + k * (k + 1) // 2)
                        continue
                    R = make_mat(bit, Is, Js, not_col_idxs, default_matrix)
                    t = 0
                    while True:
                        r += 1 << (k + k * (k + 1) // 2)
                        while idx < r:
                            Qc_orig = idx - (r - (1 << (k * (k + 1) // 2 + k)))
                            Qc = 0
                            for i in range(k * (k + 1) // 2 + k):
                                Qc |= ((Qc_orig >> reordering[i]) & 1) << i
                            Q = Qc >> k
                            c = Qc & ((1 << k) - 1)
                            temp = make_state_from_kQctR(n, k, Q, c, t, R)
                            csc_indices.extend(temp[0])
                            csc_data.extend(temp[1])
                            csc_indptr.append(len(csc_indices))
                            idxs.pop()
                            if len(idxs) == 0:
                                return csc_matrix(
                                    (csc_data, csc_indices, csc_indptr),
                                    shape=(1 << n, len(csc_indptr) - 1),
                                    dtype=np.complex128,
                                )
                            idx = idxs[-1]
                        t = (t + ~t_mask + 1) & t_mask
                        if t == 0:
                            break
    assert False, "This line should not be reached"


def test_recovery_states_from_idxs():
    # import random
    # n = 10
    # t_s_g_s = total_stabilizer_group_size(n)
    # random_indices = []
    # for _ in range(10):
    #     log2 = int(np.log2(float(t_s_g_s)))
    #     idx = 0
    #     for i in range(log2):
    #         idx |= random.randint(0, 1) << i
    #     if idx >= t_s_g_s:
    #         idx = t_s_g_s - 1
    #     random_indices.append(idx)
    # random_indices = list(set(random_indices))
    # recovery_states_from_idxs(n, random_indices)

    for n in [1, 2, 3]:
        Amat_actual = get_Amat_sparse(n)
        Amat_recovered = recovery_states_from_idxs(n, list(range(Amat_actual.shape[1])))
        for _ in range(5):
            random_vec = np.random.rand(Amat_actual.shape[0]) + 1j * np.random.rand(
                Amat_actual.shape[0]
            )
            dots_actual = np.sort(np.abs(random_vec.T @ Amat_actual))
            dots_recovered = np.sort(np.abs(random_vec.T @ Amat_recovered))
            assert np.allclose(dots_actual, dots_recovered)
        print(f"test_recovery_states_from_idxs({n}) passed.")


if __name__ == "__main__":
    test_recovery_states_from_idxs()
