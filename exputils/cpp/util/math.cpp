#pragma once
#include "base.cpp"

vvi generate_combinations(int n, int k) {
  // https://stackoverflow.com/questions/9430568/generating-combinations-in-c
  vb v(n);
  fill(v.begin(), v.begin() + k, true);
  vvi combinations;
  do {
    vi combination;
    for (int i = 0; i < n; i++)
      if (v[i]) combination.push_back(i);
    combinations.push_back(combination);
  } while (prev_permutation(v.begin(), v.end()));
  __int128_t nCk = 1;
  for (__int128_t i = 1; i <= k; i++) {
    nCk *= n - k + i;
    nCk /= i;
  }
  assert(__int128_t(combinations.size()) == nCk);
  return combinations;
}

__int128_t q_factorial(int k) {
  // https://mathworld.wolfram.com/q-Factorial.html
  // q_factorial where q=2
  // [k]_q! = \prod_{i=1}^k (1 + q + q^2 + ... + q^{i-1})
  assert(k >= 0);
  __int128_t ret = 1;
  for (int i = 1; i <= k; ++i) ret *= (1ll << i) - 1;
  vi small_k_results = {1, 1, 3, 21, 315, 9765};
  if (k <= 5) assert(ret == small_k_results[k]);
  return ret;
}

__int128_t q_binomial(int n, int k) {
  // https://mathworld.wolfram.com/q-BinomialCoefficient.html
  // q_binomial where q=2
  // [n k]_q = \frac{[n]_q!}{[k]_q! [n-k]_q!}
  if (n - 1 == k) return (1ll << n) - 1;
  __int128_t ret1 = q_factorial(n) / (q_factorial(k) * q_factorial(n - k));
  __int128_t ret2 = 1;
  for (int i = 0; i < k; i++) {
    ret2 *= 1 - (1ll << (n - i));
    ret2 /= 1 - (1ll << (i + 1));
  }
  assert(ret1 == ret2);
  return ret1;
}

constexpr __int128_t total_stabilizer_group_size(int n) {
  // https://arxiv.org/pdf/1711.07848.pdf
  // The number of n qubit pure stabilizer states
  // |S_n| = 2^n \prod_{k=1}^{n} (2^k + 1)
  __int128_t ret = 1ll << n;
  for (int k = 0; k < n; ++k) {
    ret *= (1ll << (n - k)) + 1;
  }
  return ret;
}
