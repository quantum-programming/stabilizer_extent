# stabilizer extent

Compute stabilizer extent.

## How to run the code

We require three setups to run the code.

### 1. Install MOSEK

We use MOSEK to solve SOCP.
You can download the MOSEK software from the [MOSEK website](https://www.mosek.com/).
Please get a license file according to the instructions on the website.

### 2. Install the required python libraries

It is recommended that you prepare a new virtual Python environment and run
`pip install -r requirement.txt`
to ensure that libraries are compatible.

Or, if you could not build the virtual environment successfully,
you can use the following command to install the required libraries:

```bash
pip install -e .
pip install tqdm
pip install scipy
pip install numpy
pip install numba
pip install mosek
pip install qutip
pip install cvxopt
pip install qiskit
pip install seaborn
pip install matplotlib
pip install openfermion
```

### 3. (optional) compile C++ code

We highly recommended to compile the C++ code to accelerate the computation.
The only file you have to compile is `exputils/cpp/calc_dot.cpp`.

You can compile the C++ code by running the following command:

```bash
pwd # check the current directory is path/to/stabilizer_extent
g++ exputils/cpp/calc_dot.cpp -o exputils/cpp/calc_dot.exe -std=c++17 -O2 -mtune=native -march=native -fopenmp -lz
```
