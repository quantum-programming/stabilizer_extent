#include "include/include.hpp"
#include "rref.cpp"

COMPLEX xQc_to_coeff(int k, int x, __int128_t Q, int c) {
  int cx = 0, xQx = 0, Q_idx = 0;
  for (int i = 0; i < k; i++) {
    cx ^= ((x >> i) & 1) * ((c >> i) & 1);
    for (int j = i; j < k; ++j) {
      if (((Q >> Q_idx) & 1) & ((x >> i) & 1) & ((x >> j) & 1)) xQx ^= 1;
      Q_idx += 1;
    }
  }
  return std::pow(-1, xQx) * std::pow(COMPLEX(0, 1), cx);
}

double calc_abs_from_bQc(int k, const vc& b, __int128_t Q, int c) {
  double coeff = 1.0 / std::pow(2, k / 2.0);
  COMPLEX ret(0.0, 0.0);
  for (int x = 0; x < (1 << k); x++) ret += coeff * xQc_to_coeff(k, x, Q, c) * b[x];
  return std::abs(ret);
}

__int128_t randRangePow2(int m) {
  assert(m < 128);  // 128bit
  __int128_t ret = 0;
  for (int i = 0; i < m; i++) ret |= __int128_t(myrand.randBool()) << i;
  return ret;
}

double fast(int k, const vc& b) {
  vec<std::pair<int, vi>> Q_non_diag;
  int Q_idx = 0;
  for (int i = 0; i < k; i++)
    for (int j = i; j < k; j++) {
      if (i != j) {
        vi xs;
        for (int x = 0; x < (1 << k); x++)
          if (((x >> i) & 1) & ((x >> j) & 1)) xs.push_back(x);
        Q_non_diag.emplace_back(Q_idx, xs);
      }
      Q_idx += 1;
    }

  double coeff = 1.0 / std::pow(2, k / 2.0);
  double max_val = 0;
  vc coeffs(1 << k);
  vi Ps(Q_non_diag.size());
  std::iota(ALL(Ps), 0);
  Timer timer_local;
  timer_local.start();
  while (timer_local.ms() < 3 * 1000) {
    __int128_t Q = randRangePow2(k * (k + 1) / 2);
    __int128_t c = randRangePow2(k);
    for (int x = 0; x < (1 << k); x++) coeffs[x] = coeff * xQc_to_coeff(k, x, Q, c);
    COMPLEX now_sum = std::inner_product(ALL(coeffs), b.begin(), COMPLEX(0, 0));
    double now_abs = std::abs(now_sum);
    while (true) {
      bool is_updated = false;
      myrand.shuffle(Ps);
      for (int p : Ps) {
        const auto& [Q_idx, xs] = Q_non_diag[p];
        COMPLEX cur_sum = now_sum;
        for (int x : xs) cur_sum -= COMPLEX(2, 0) * b[x] * coeffs[x];
        double cur_abs = std::abs(cur_sum);
        if (cur_abs > now_abs) {
          now_sum = cur_sum;
          now_abs = cur_abs;
          Q ^= __int128_t(1) << Q_idx;
          for (int x : xs) coeffs[x] *= -1;
          is_updated = true;
          break;
        }
      }
      if (!is_updated) break;
    }
    // assert(std::abs(now_abs - calc_abs_from_bQc(k, b, Q, c)) < 1e-9);
    max_val = std::max(max_val, now_abs);
  }
  return max_val;
}

int main() {
  int n;
  vc psi;

  try {
    cnpy::NpyArray psi_npy = cnpy::npz_load("temp_in.npz")["psi"];
    for (n = 0; n <= 15; n++)
      if (int(psi_npy.shape[0]) == (1 << n)) break;
    if (int(psi_npy.shape[0]) != (1 << n))
      throw std::runtime_error("Invalid input shape");
    psi.resize(1 << n);
    for (int i = 0; i < 1 << n; i++) psi[i] = psi_npy.data<COMPLEX>()[i];
  } catch (const std::exception& e) {
    debug("Error: ", e.what());
    n = 14;
    psi.resize(1 << n);
    std::mt19937 mt(1);
    for (int i = 0; i < 1 << n; i++)
      psi[i] = COMPLEX(std::uniform_real_distribution<double>(-0.5, 0.5)(mt),
                       std::uniform_real_distribution<double>(-0.5, 0.5)(mt));
  }

  const int MAX_THREAD_NUM = omp_get_max_threads();
  double final_ans = 0;

#pragma omp parallel for num_threads(MAX_THREAD_NUM)
  for (int i = 0; i < MAX_THREAD_NUM; i++) {
    double ans = fast(n, psi);
#pragma omp critical
    final_ans = std::max(final_ans, ans);
  }
  debug(final_ans);

  RREF_generator rref_gen(n, n - 1);
  debug(rref_gen.q_binom);

  vec<std::pair<double, vc>> t2_psi_pairs;
#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
  for (int rref_idx = 0; rref_idx < rref_gen.q_binom; rref_idx++) {
    const auto& [row_idxs, t_mask] = rref_gen.get(rref_idx, false);
    assert(__builtin_popcount(t_mask) == 1);
    for (int t : {0, t_mask}) {
      vc psi2 = arange_psi_by_t(n - 1, t, row_idxs, psi);
      double t2 = calc_threshold_2(n - 1, psi2.begin());
      // This t2 is not normalized and not root
#pragma omp critical
      t2_psi_pairs.emplace_back(t2, psi2);
    }
    if (rref_idx % 1000 == 0) {
      std::cerr << "progress: " << rref_idx << "/" << rref_gen.q_binom << std::endl;
    }
  }
  std::cerr << "RREF(n,n-1) enumeration done" << std::endl;

  std::sort(ALL(t2_psi_pairs),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  if (int(t2_psi_pairs.size()) > 10) t2_psi_pairs.resize(10);

  for (const auto& [t2, psi2] : t2_psi_pairs) {
    double ans = fast(n - 1, psi2);
    debug(ans);
    final_ans = std::max(final_ans, ans);
  }

  std::cout << final_ans << std::endl;

  return 0;
}