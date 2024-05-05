import sys

import numpy as np
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm
from typing import Tuple
from exputils.stabilizer_group import total_stabilizer_group_size


def _calculate_by_cvx(n: int, Amat: np.ndarray, psi: np.ndarray, verbose=False):
    """Solve the following optimization problem using cvxopt:
    minimize ||x||_1
        s.t. Amat @ x = psi
    See documents in doc folder for more details.
    """
    import cvxopt as cvx

    assert Amat.shape[0] == psi.size == 2**n
    assert Amat.shape[1] <= total_stabilizer_group_size(n)
    sz = Amat.shape[1]

    # https://cvxopt.org/userguide/coneprog.html#cvxopt.solvers.socp
    c = cvx.matrix(np.hstack([np.ones(sz), np.zeros(sz), np.zeros(sz)]))
    A = cvx.matrix(
        np.vstack(
            [
                np.hstack([np.zeros((2**n, sz)), Amat.real, -Amat.imag]),
                np.hstack([np.zeros((2**n, sz)), Amat.imag, Amat.real]),
            ]
        )
    )
    b = cvx.matrix(np.hstack([psi.real, psi.imag]))
    G = []
    h = []
    for i in range(sz):
        mat = np.zeros((3, 3 * sz))
        mat[0, i] = 1
        mat[1, sz + i] = 1
        mat[2, 2 * sz + i] = 1
        G.append(cvx.matrix(-mat))
        h.append(cvx.matrix(np.zeros(3)))
    cvx.solvers.options["show_progress"] = verbose
    solution = cvx.solvers.socp(c, A=A, b=b, Gq=G, hq=h)
    l1_norm = solution["primal objective"]
    stabilizer_extent = l1_norm**2
    _x = np.array(solution["x"]).flatten()
    _y = np.array(solution["y"]).flatten()
    t = _x[:sz]
    x = _x[sz : 2 * sz] + 1j * _x[2 * sz :]
    y = _y[: 2**n] + 1j * _y[2**n :]
    assert l1_norm >= 0
    assert np.allclose(Amat @ x, psi)
    assert np.allclose(np.abs(x), t, atol=1e-5)
    assert 1 - 1e-5 <= np.max(np.abs(y.conj() @ Amat)) <= 1 + 1e-5
    assert np.isclose(-(np.conj(y) @ psi).real, l1_norm, atol=1e-5)
    return stabilizer_extent, x, y


def _calculate_by_mosek(n: int, Amat: csc_matrix, psi: np.ndarray, verbose=False):
    """Solve the following optimization problem using cvxopt:
    minimize ||x||_1
        s.t. Amat @ x = psi
    See documents in doc folder for more details.
    """
    import mosek

    assert Amat.shape[0] == psi.size == 2**n
    assert 2**n <= Amat.shape[1] <= total_stabilizer_group_size(n)
    sz = Amat.shape[1]

    with mosek.Task() as task:
        # https://docs.mosek.com/latest/pythonapi/tutorial-cqo-shared.html#example-cqo1
        # Define a stream printer to grab output from MOSEK
        def stream_printer(text):
            sys.stdout.write(text)
            sys.stdout.flush()

        if verbose:
            task.set_Stream(mosek.streamtype.log, stream_printer)

        # c: https://docs.mosek.com/latest/pythonapi/tutorial-lo-shared.html#example-lo1
        task.appendvars(3 * sz)
        inf = 0.0  # Since the actual value of Infinity is ignores, we define it solely for symbolic purposes
        for j in range(3 * sz):
            task.putvarbound(j, mosek.boundkey.fr, -inf, +inf)
        task.appendcons(2 * len(psi))
        for i in range(len(psi)):
            task.putconbound(i, mosek.boundkey.fx, psi[i].real, psi[i].real)
        for i in range(len(psi)):
            task.putconbound(len(psi) + i, mosek.boundkey.fx, psi[i].imag, psi[i].imag)
        task.putclist(range(sz), [1] * sz)

        # A: https://docs.mosek.com/latest/pythonapi/optimizer-task.html#mosek.task.putacollist
        sub = np.vstack(
            [
                np.arange(1 * sz, 2 * sz, dtype=np.int32),
                np.arange(2 * sz, 3 * sz, dtype=np.int32),
            ]
        ).T.flatten()
        ptrb = np.vstack(
            [2 * Amat.indptr[:-1], Amat.indptr[:-1] + Amat.indptr[1:]]
        ).T.flatten()
        ptre = np.append(ptrb[1:], 2 * Amat.indptr[-1])
        assert np.all(sub >= 0) and np.all(ptrb >= 0) and np.all(ptre >= 0)
        asub = []
        aval = []
        for j, col in tqdm(enumerate(Amat.T), disable=not verbose, total=sz):
            is_re = col.data.real != 0
            idx_re = col.indices[is_re]
            idx_im = col.indices[~is_re]
            data_re = col.data[is_re].real
            data_im = col.data[~is_re].imag
            asub.extend([idx_re, idx_im + (1 << n)])
            aval.extend([+data_re, +data_im])
            asub.extend([idx_im, idx_re + (1 << n)])
            aval.extend([-data_im, +data_re])
        task.putacollist(sub, ptrb, ptre, np.hstack(asub), np.hstack(aval))

        # Input the affine conic constraints
        task.appendafes(3 * sz)
        task.putafefentrylist(range(3 * sz), range(3 * sz), [1.0] * (3 * sz))
        for i in range(sz):
            quad_cone = task.appendquadraticconedomain(3)
            task.appendacc(quad_cone, [i, i + sz, i + 2 * sz], None)

        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)

        # Optimize the task
        task.optimize()
        # task.writedata("cqo1.ptf")
        # Print a summary containing information
        # about the solution for debugging purposes
        task.solutionsummary(mosek.streamtype.msg)
        sol_sta = task.getsolsta(mosek.soltype.itr)

        # Output a solution
        if sol_sta != mosek.solsta.optimal:
            raise ValueError(f"Solution status: {sol_sta}")

        l1_norm = task.getprimalobj(mosek.soltype.itr)
        stabilizer_extent = l1_norm**2
        _x = np.array(task.getxx(mosek.soltype.itr))
        _y = np.array(task.gety(mosek.soltype.itr))
        t = _x[:sz]
        x = _x[sz : 2 * sz] + 1j * _x[2 * sz :]
        y = _y[: 2**n] + 1j * _y[2**n :]

    assert l1_norm >= 0
    assert np.allclose(Amat @ x, psi)
    assert np.allclose(np.abs(x), t, atol=1e-5)
    assert 1 - 1e-5 <= np.max(np.abs(y.conj() @ Amat)) <= 1 + 1e-5
    assert np.isclose((np.conj(y) @ psi).real, l1_norm, atol=1e-5)

    return stabilizer_extent, x, y


def _calculate_by_gurobi(n: int, Amat: csc_matrix, psi: np.ndarray, verbose=False):
    """Solve the following optimization problem using cvxopt:
    minimize ||x||_1
        s.t. Amat @ x = psi
    See documents in doc folder for more details.
    """
    import gurobipy as gp
    from gurobipy import GRB

    assert Amat.shape[0] == psi.size == 2**n
    assert Amat.shape[1] <= total_stabilizer_group_size(n)
    sz = Amat.shape[1]
    m = gp.Model("StabilizerExtent")
    # m.Params.QCPDual = 1
    # m.Params.OptimalityTol = 1e-9
    if not verbose:
        m.Params.LogToConsole = 0

    xReal = m.addMVar(shape=sz, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xReal")
    xImag = m.addMVar(shape=sz, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xImag")
    aux = m.addMVar(shape=sz, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="aux")
    t = m.addMVar(shape=sz, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")
    m.setObjective(np.ones(sz) @ t, GRB.MINIMIZE)
    A_real = Amat.real
    A_imag = Amat.imag
    real_constr = m.addConstr(A_real @ xReal - A_imag @ xImag == psi.real, name="real")
    imag_constr = m.addConstr(A_real @ xImag + A_imag @ xReal == psi.imag, name="imag")
    for i in range(sz):
        m.addConstr(aux[i] == gp.norm([xReal[i], xImag[i]], 2), name=f"aux{i}")
        m.addConstr(aux[i] <= t[i], name=f"norm{i}")
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise ValueError(f"Optimization status: {m.status}")
    l1_norm = m.ObjVal
    stabilizer_extent = l1_norm**2
    t = t.X
    x = np.array(xReal.X + 1j * xImag.X)
    assert np.allclose(Amat @ x, psi)
    assert np.allclose(np.abs(x), t, atol=1e-5)
    # ? I don't know why, but gurobi fails to return the correct dual variable
    # y = -np.array(real_constr.Pi) - 1j * np.array(imag_constr.Pi)
    # assert np.all(np.abs(y.conj() @ Amat) <= 1), y.conj() @ Amat
    # assert np.isclose(-(np.conj(y) @ psi).real, l1_norm, atol=1e-5), (
    #     -(np.conj(y) @ psi),
    #     l1_norm,
    # )
    return stabilizer_extent, x, np.nan


def calculate_extent_custom(
    n: int, Amat: csc_matrix, psi: np.ndarray, method: str = "mosek", verbose=False
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve the following optimization problem:
    minimize ||x||_1
        s.t. Amat @ x = psi
    This is the stabilizer extent of the psi state.
    """
    assert method in ["cvx", "mosek", "gurobi"]
    if method == "cvx":
        return _calculate_by_cvx(n, Amat.toarray(), psi, verbose)
    elif method == "mosek":
        return _calculate_by_mosek(n, Amat, psi, verbose)
    else:
        return _calculate_by_gurobi(n, Amat, psi, verbose)
