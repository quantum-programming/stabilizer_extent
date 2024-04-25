#include "include/include.hpp"
#include "rref.cpp"

constexpr INT calc_kkk12(int k) { return INT(1) << (k + k * (k + 1) / 2); }

INT kkk12s[11] = {calc_kkk12(0), calc_kkk12(1), calc_kkk12(2), calc_kkk12(3),
                  calc_kkk12(4), calc_kkk12(5), calc_kkk12(6), calc_kkk12(7),
                  calc_kkk12(8), calc_kkk12(9), calc_kkk12(10)};

struct dotCalculator {
  dotCalculator() {}

  auto calc_dot(int n, const vc& psi, bool is_dual_mode) {
    timer1.start();
    vc psi_conj = psi;
    for (auto& x : psi_conj) x = std::conj(x);
    calc_dot_main(n, psi_conj, is_dual_mode);
    timer1.stop();
    std::cerr << " calculation time : " << timer1.report() << std::endl;
    std::cerr << "branch cut / total: " << branch_cut << "/"
              << total_stabilizer_group_size(n) << std::endl;

    sort(values.rbegin(), values.rend());
    int result_sz = std::min(values.size(), MAX_VALUES_SIZE);
    // *** idx ***
    vec<INT> result;
    result.reserve(result_sz);
    for (int i = 0; i < result_sz; i++) result.push_back(values[i].second);
    return result;
    // *** value ***
    // vec<double> result;
    // result.reserve(result_sz);
    // for (int i = 0; i < result_sz; i++) result.push_back(values[i].first);
    // return result;
  }

 private:
  Timer timer1;
  AmatForSmallN Amats;
  vec<std::pair<double, INT>> values;
  INT branch_cut = 0;
  double threshold = 0.0;
  static constexpr size_t MAX_VALUES_SIZE = 5000;
  static constexpr int LARGE_K = 7;

  vc arange_psi_by_t(int k, int t, const vi& row_idxs, const vc& psi_normalized) {
    vc psi2(1 << k);
    for (int x = 0; x < (1 << k); x++) psi2[x] = psi_normalized[row_idxs[x] ^ t];
    return psi2;
  }

  template <bool is_final = false>
  void truncate_values() {
    size_t SZ = MAX_VALUES_SIZE;
    if constexpr (is_final)
      if (values.size() < SZ) return;
    std::nth_element(values.begin(), values.begin() + SZ, values.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    threshold = values[SZ].first;
    values.resize(SZ);
  }

  void add_to_values(vec<std::pair<double, INT>>& values_local) {
#pragma omp critical
    {
      values.insert(values.end(), std::make_move_iterator(values_local.begin()),
                    std::make_move_iterator(values_local.end()));
      if (values.size() > MAX_VALUES_SIZE) truncate_values();
    }
  }

  COMPLEX rotate_complex(const COMPLEX x) {
    if (x.imag() >= 0)
      return (x.real() >= 0) ? +x : x * COMPLEX(0, -1);
    else
      return (x.real() <= 0) ? -x : x * COMPLEX(0, +1);
  }

  bool check_branch_cut(const int k, const vec<COMPLEX>::const_iterator psi_begin) {
    double absSum =
        std::accumulate(psi_begin, psi_begin + (1 << k), 0.0,
                        [](double a, COMPLEX b) { return a + std::abs(b); });
    if (absSum >= threshold) return false;
    if (absSum < 1.2 * threshold) {
      vc Ys(1 << k);
      double sum_ys_real = 0.0, sum_ys_imag = 0.0;
      for (size_t i = 0; i < Ys.size(); i++) {
        Ys[i] = rotate_complex(*(psi_begin + i));
        sum_ys_real += Ys[i].real();
        sum_ys_imag += Ys[i].imag();
      }
      std::sort(ALL(Ys), [](const auto a, const auto b) {
        return a.imag() * b.real() < a.real() * b.imag();
      });
      double max_abs2 = sum_ys_real * sum_ys_real + sum_ys_imag * sum_ys_imag;
      for (size_t i = 0; i < Ys.size(); ++i) {
        sum_ys_real += -Ys[i].real() - Ys[i].imag();
        sum_ys_imag += -Ys[i].imag() + Ys[i].real();
        max_abs2 =
            std::max(max_abs2, sum_ys_real * sum_ys_real + sum_ys_imag * sum_ys_imag);
      }
      if (max_abs2 >= threshold * threshold) return false;
    }
#pragma omp atomic
    branch_cut += kkk12s[k];
    return true;
  }

  void dfs_sub(const int k, const vc& psi, const int c_0, const int q_00, INT& ret_idx,
               vec<std::tuple<int, vc, INT>>& stk1) {
    vc next(1 << (k - 1));
    next[0] = psi[0] + psi[1] * COMPLEX(q_00 ? -1 : 1) * (c_0 ? COMPLEX(0, 1) : 1);
    COMPLEX coeff = c_0 ? COMPLEX(0, 1) : 1.0;
    // non-recursive dfs
    vec<std::tuple<int, int, INT>> stk2;
    stk2.emplace_back(1, 1, ret_idx + (kkk12s[k] >> 3));
    stk2.emplace_back(0, 1, ret_idx);
    while (!stk2.empty()) {
      auto [q_0, i, ret_idx_local] = stk2.back();
      stk2.pop_back();
      for (int x1 = 1 << (i - 1); x1 < 1 << i; x1++) {
        if (__builtin_parity(q_00 ^ (q_0 & x1)))
          next[x1] = psi[x1 << 1] - coeff * psi[(x1 << 1) ^ 1];
        else
          next[x1] = psi[x1 << 1] + coeff * psi[(x1 << 1) ^ 1];
      }
      if (i < k - 1) {
        stk2.emplace_back(q_0 ^ (1 << i), i + 1,
                          ret_idx_local + (kkk12s[k] >> (3 + i)));
        stk2.emplace_back(q_0, i + 1, ret_idx_local);
      } else {
        stk1.emplace_back(k - 1, next, ret_idx_local);
      }
    }
    ret_idx += kkk12s[k] >> 2;
  }

  vec<std::pair<double, INT>> calc_dot_sub(const int k_orig, const vc& psi_orig,
                                           const INT ret_idx_orig) {
    assert(1 <= k_orig && int(psi_orig.size()) == (1 << k_orig));
    // psi_list[(1<<k)+i] = psi_list[(1<<k)^i] = k-th psi[i] (0<=i<(1<<k))
    vc psi_list(1 << (k_orig + 1), 0);
    for (int i = 0; i < (1 << k_orig); i++) psi_list[(1 << k_orig) ^ i] = psi_orig[i];
    vec<INT> stk1;
    vec<std::tuple<int, int, INT, bool, bool>> stk2;
    vec<std::pair<double, INT>> values_local;
    vi ks;  // +:stk1 -:stk2
    stk1.emplace_back(ret_idx_orig);
    ks.push_back(k_orig);
    while (!ks.empty()) {
      int k = ks.back();
      ks.pop_back();
      if (k > 0) {
        INT ret_idx = stk1.back();
        stk1.pop_back();
        if (k == 1) {
          for (int i = 0; i < kkk12s[1]; i++) {
            double val = std::abs(Amats.Amat1[i][0] * psi_list[2 + 0] +
                                  Amats.Amat1[i][1] * psi_list[2 + 1]);
            if (val > threshold) values_local.emplace_back(val, ret_idx);
            ret_idx++;
          }
        } else if (k == 2) {
          for (int i = 0; i < kkk12s[2]; i++) {
            double val = std::abs(Amats.Amat2[i][0] * psi_list[4 + 0] +
                                  Amats.Amat2[i][1] * psi_list[4 + 1] +
                                  Amats.Amat2[i][2] * psi_list[4 + 2] +
                                  Amats.Amat2[i][3] * psi_list[4 + 3]);
            if (val > threshold) values_local.emplace_back(val, ret_idx);
            ret_idx++;
          }
        } else {
          if (check_branch_cut(k, psi_list.begin() + (1 << k))) continue;
          for (int c_0 = 0; c_0 <= 1; c_0++)
            for (int q_00 = 0; q_00 <= 1; q_00++) {
              stk2.emplace_back(1, 1, ret_idx + (kkk12s[k] >> 3), c_0, q_00);
              stk2.emplace_back(0, 1, ret_idx, c_0, q_00);
              ks.push_back(-k);
              ks.push_back(-k);
              ret_idx += kkk12s[k] >> 2;
            }
        }
      } else {
        k = -k;
        auto [q_0, i, ret_idx_local, c_0, q_00] = stk2.back();
        stk2.pop_back();
        if (q_0 == 0 && i == 1) {
          psi_list[(1 << (k - 1)) ^ 0] =
              psi_list[(1 << k) ^ 0] + psi_list[(1 << k) ^ 1] * COMPLEX(q_00 ? -1 : 1) *
                                           (c_0 ? COMPLEX(0, 1) : 1);
        }
        COMPLEX coeff = c_0 ? COMPLEX(0, 1) : 1.0;
        for (int x1 = 1 << (i - 1); x1 < 1 << i; x1++) {
          int idx = (1 << (k - 1)) ^ x1;
          if (__builtin_parity(q_00 ^ (q_0 & x1)))
            psi_list[idx] = psi_list[idx << 1] - coeff * psi_list[(idx << 1) ^ 1];
          else
            psi_list[idx] = psi_list[idx << 1] + coeff * psi_list[(idx << 1) ^ 1];
        }
        if (i < k - 1) {
          stk2.emplace_back(q_0 ^ (1 << i), i + 1,
                            ret_idx_local + (kkk12s[k] >> (3 + i)), c_0, q_00);
          stk2.emplace_back(q_0, i + 1, ret_idx_local, c_0, q_00);
          ks.push_back(-k);
          ks.push_back(-k);
        } else {
          stk1.push_back(ret_idx_local);
          ks.push_back(k - 1);
        }
      }
    }
    return values_local;
  }

  void calc_dot_sub_large_k(const int k, const vc& psi, INT ret_idx) {
    assert(LARGE_K <= k && int(psi.size()) == (1 << k));
    if (check_branch_cut(k, psi.begin())) return;
    vec<std::tuple<int, vc, INT>> stk1;
    stk1.reserve(1 << (k + 1));
    for (int c_0 = 0; c_0 <= 1; c_0++)
      for (int q_00 = 0; q_00 <= 1; q_00++) dfs_sub(k, psi, c_0, q_00, ret_idx, stk1);
    assert(int(stk1.size()) == (1 << (k + 1)));

#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
    for (int i = 0; i < int(stk1.size()); i++) {
      auto [k, psi, ret_idx] = stk1[i];
      auto res = calc_dot_sub(k, psi, ret_idx);
      if (!res.empty()) add_to_values(res);
      std::get<1>(stk1[i]).clear();
    }
  }

  void calc_dot_main(int n, const vc& psi, const bool is_dual_mode) {
    assert(int(psi.size()) == (1 << n));
    if (is_dual_mode) threshold = 1.00;
    for (int i = 0; i < (1 << n); i++)
      if (std::abs(psi[i]) > threshold) values.emplace_back(std::abs(psi[i]), i);
    std::sort(ALL(values),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    INT t_s_g_s = total_stabilizer_group_size(n);
    for (int k = 1; k <= n; k++) {
      INT ret_idx = (1ll << n);
      for (int _k = 1; _k < k; _k++)
        ret_idx += q_binomial(n, _k) * (1ll << (n + _k * (_k + 1) / 2));
      RREF_generator rref_gen(n, k);
      vc psi_normalized = psi;
      double sqrt2k = 1 / std::pow(std::sqrt(2), k);
      for (auto& x : psi_normalized) x *= sqrt2k;
      if (k < LARGE_K) {
        vec<INT> ret_idxs = {ret_idx};
        for (int rref_idx = 0; rref_idx < rref_gen.q_binom; rref_idx++) {
          ret_idx += (1ll << (n + k * (k + 1) / 2));
          ret_idxs.push_back(ret_idx);
        }
#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
        for (int rref_idx = 0; rref_idx < rref_gen.q_binom; rref_idx++) {
          const auto& [row_idxs, t_mask] = rref_gen.get(rref_idx, false);
          INT ret_idx_local = ret_idxs[rref_idx];
          vec<std::pair<double, INT>> values_local;
          int t = 0;
          while (true) {
            vc psi2 = arange_psi_by_t(k, t, row_idxs, psi_normalized);
            auto res = calc_dot_sub(k, psi2, ret_idx_local);
            if (!res.empty()) {
              values_local.insert(values_local.end(),
                                  std::make_move_iterator(res.begin()),
                                  std::make_move_iterator(res.end()));
              if (values_local.size() > MAX_VALUES_SIZE) {
                add_to_values(values_local);
                values_local.clear();
              }
            }
            ret_idx_local += kkk12s[k];
            t = (t + ~t_mask + 1) & t_mask;
            if (t == 0) break;
          }
          assert(ret_idx_local == ret_idxs[rref_idx + 1]);
          if (!values_local.empty()) add_to_values(values_local);
        }
      } else {
        for (int rref_idx = 0; rref_idx < rref_gen.q_binom; rref_idx++) {
          const auto& [row_idxs, t_mask] = rref_gen.get(rref_idx, false);
          int t = 0;
          while (true) {
            vc psi2 = arange_psi_by_t(k, t, row_idxs, psi_normalized);
            calc_dot_sub_large_k(k, psi2, ret_idx);
            ret_idx += kkk12s[k];
            t = (t + ~t_mask + 1) & t_mask;
            if (t == 0) break;
          }
        }
      }
      double min = std::nan(""), max = std::nan("");
      if (!values.empty()) {
        min = (*min_element(ALL(values))).first;
        max = (*max_element(ALL(values))).first;
      }
      std::cerr << "[k|progress|range]: " << std::setw(std::to_string(n).size()) << k
                << " | " << std::scientific << std::setprecision(5) << double(ret_idx)
                << "/" << double(t_s_g_s) << " | [" << std::fixed
                << std::setprecision(5) << min << ", " << max << "] | "
                << timer1.report() << std::endl;
    }
    truncate_values<true>();
    return;
  }
};

int main() {
  int n;
  vc psi;
  bool is_dual_mode;

  try {
    cnpy::NpyArray psi_npy = cnpy::npz_load("temp_in.npz")["psi"];
    for (n = 0; n < 15; n++)
      if (int(psi_npy.shape[0]) == (1 << n)) break;
    if (int(psi_npy.shape[0]) != (1 << n))
      throw std::runtime_error("Invalid input shape");
    psi.resize(1 << n);
    for (int i = 0; i < 1 << n; i++) psi[i] = psi_npy.data<COMPLEX>()[i];
    is_dual_mode = cnpy::npz_load("temp_in.npz")["is_dual_mode"].data<bool>()[0];
  } catch (const std::exception& e) {
    n = 3;
    psi.resize(1 << n);
    std::mt19937 mt(1);
    for (int i = 0; i < 1 << n; i++)
      psi[i] = COMPLEX(std::uniform_real_distribution<double>(-0.5, 0.5)(mt),
                       std::uniform_real_distribution<double>(-0.5, 0.5)(mt));
    is_dual_mode = true;
  }

  dotCalculator calculator;
  auto res = calculator.calc_dot(n, psi, is_dual_mode);

  if (res.empty()) res.push_back(-1);

  // cnpy does not correspond to __int128_t, so we use long long instead.
  vec<long long> res_ll_1(res.size());
  vec<long long> res_ll_2(res.size());
  for (size_t i = 0; i < res.size(); i++) {
    res_ll_1[i] = res[i] >> 62;
    res_ll_2[i] = res[i] & ((1ll << 62) - 1);
  }
  cnpy::npz_save("temp_out.npz", "res1", res_ll_1.data(), {res.size()}, "w");
  cnpy::npz_save("temp_out.npz", "res2", res_ll_2.data(), {res.size()}, "a");

  return 0;
}
