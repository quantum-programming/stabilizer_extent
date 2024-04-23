from typing import List

import numpy as np
from numba import njit


# Reference: https://drken1215.hatenablog.com/entry/2019/03/20/202800
@njit(cache=True)
def gauss_jordan_get_only_rank(A: np.ndarray, m: int) -> int:
    """By using Gauss-Jordan elimination, get the rank of A"""
    rank = 0
    for col in range(m):
        pivot = -1
        for row in range(rank, A.shape[0]):
            if (A[row] >> col) & 1:
                pivot = row
                break
        if pivot == -1:
            continue
        A[pivot], A[rank] = A[rank], A[pivot]
        for row in range(A.shape[0]):
            if row != rank and (A[row] >> col) & 1:
                A[row] ^= A[rank]
        rank += 1
    return rank


# Reference: https://drken1215.hatenablog.com/entry/2019/03/20/202800
@njit(cache=True)
def gauss_jordan(A: np.ndarray, m: int) -> List[int]:
    """Solve Ax=b by using Gauss-Jordan elimination

    Args:
        A (np.ndarray): the representation of (A|b) in bit
        m (int): the number of A's columns

    Returns:
        List[int]: the list of x
    """
    rank = 0
    for col in range(m):
        pivot = -1
        for row in range(rank, A.shape[0]):
            if (A[row] >> col) & 1:
                pivot = row
                break
        if pivot == -1:
            continue
        A[pivot], A[rank] = A[rank], A[pivot]
        for row in range(A.shape[0]):
            if row != rank and (A[row] >> col) & 1:
                A[row] ^= A[rank]
        rank += 1
    for row in range(rank, A.shape[0]):
        assert A[row] & (1 << m) == 0
    ret = [0 for _ in range(m)]
    col = 0
    for row in range(rank):
        while (A[row] >> col) & 1 == 0:
            col += 1
        ret[col] = (A[row] >> m) & 1
    return ret
