import time

from exputils.extent.cg import calculate_extent_CG
from exputils.state.random_ket import make_random_quantum_state


def CG_for_data(n: int):
    seed = 0
    kind = "pure"
    print(f"{n=}")
    psi = make_random_quantum_state(kind, n, seed)
    t0 = time.perf_counter()
    stabilizer_extent, extends, max_values, _ = calculate_extent_CG(
        n, psi, verbose=False
    )
    t1 = time.perf_counter()
    print(f"{stabilizer_extent=} {t1-t0=}")

    with open(f"../data/CG/{kind}_{n}_data.txt", mode="w") as f:
        for i in range(len(extends)):
            print(f"{extends[i]} {max_values[i]}", file=f)

    with open(f"../data/CG/{kind}_{n}_time.txt", mode="w") as f:
        print(t1 - t0, file=f)


if __name__ == "__main__":
    n = 9
    CG_for_data(n)

# cd notebooks
# nohup python column_generation.py &
