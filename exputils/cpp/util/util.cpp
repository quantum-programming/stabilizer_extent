#pragma once
#include "base.cpp"

template <typename VAL>
vec<VAL> arange_psi_by_t(int k, int t, const vi& row_idxs,
                         const vec<VAL>& psi_1Over2k) {
  vec<VAL> Ps(1 << k);
  for (int x = 0; x < (1 << k); x++) Ps[x] = psi_1Over2k[row_idxs[x] ^ t];
  return Ps;
}

COMPLEX rotate_complex(const COMPLEX x) {
  // only for 'check_branch_cut'
  if (x.imag() >= 0)
    return (x.real() >= 0) ? +x : x * COMPLEX(0, -1);
  else
    return (x.real() <= 0) ? -x : x * COMPLEX(0, +1);
}

template <typename InputIterator>
double calc_threshold_1(const int k, const InputIterator Ps_begin) {
  // 1. The first condition:
  // MAX <= sum_{x} |(-1)^{x^T Q x} i^{c^T x} P_x| = sum_{x} |P_x|
  return std::accumulate(Ps_begin, Ps_begin + (1 << k), 0.0,
                         [](double a, auto b) { return a + std::abs(b); });
}

// only for !is_real
template <typename InputIterator>
double calc_threshold_2(const int k, const InputIterator Ps_begin) {
  // 2. The second condition:
  // MAX <= sum_{x} |i^{c_x} P_x| where c_x \in {0,1,2,3}
  // we can compute this threshold by sorting the complex numbers by their argument
  vc Ys(1 << k);
  double sum_ys_real = 0.0, sum_ys_imag = 0.0;
  // rotate complex numbers to the first quadrant
  for (size_t i = 0; i < Ys.size(); i++) {
    Ys[i] = rotate_complex(*(Ps_begin + i));
    sum_ys_real += Ys[i].real();
    sum_ys_imag += Ys[i].imag();
  }
  // sort by argument
  std::sort(ALL(Ys), [](const auto a, const auto b) {
    return a.imag() * b.real() < a.real() * b.imag();
  });
  // by iterating the sorted complex numbers, we can compute the threshold
  double max_abs2 = sum_ys_real * sum_ys_real + sum_ys_imag * sum_ys_imag;
  for (size_t i = 0; i < Ys.size(); ++i) {
    sum_ys_real += -Ys[i].real() - Ys[i].imag();
    sum_ys_imag += -Ys[i].imag() + Ys[i].real();
    max_abs2 =
        std::max(max_abs2, sum_ys_real * sum_ys_real + sum_ys_imag * sum_ys_imag);
  }
  return max_abs2;
}
