import numpy as np


def total_stabilizer_group_size(n: int) -> int:
    ret = 2**n
    for k in range(n):
        ret *= (2 ** (n - k)) + 1
    return ret


def main():
    for n in range(1, 10 + 1):
        print(f"{n:>2}: {total_stabilizer_group_size(n)}")


if __name__ == "__main__":
    main()
