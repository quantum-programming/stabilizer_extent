#include "include/include.hpp"
#include "rref.cpp"

// Please note that all stabilizer states can be represented by the form of
// |phi> = |x>
//    or = 1/2^{k/2} \sum_{x=0}^{2^k-1} -1^{x^T Q x} i^{c^T x} |Rx+t>.

// kkk12 represents the number of stabilizer states for each k with fixed R and t,
// which means the number of combinations of Q and c.
constexpr ll calc_kkk12(int k) { return 1ll << (k + k * (k + 1) / 2); }
static const ll kkk12s[11] = {calc_kkk12(0), calc_kkk12(1), calc_kkk12(2),
                              calc_kkk12(3), calc_kkk12(4), calc_kkk12(5),
                              calc_kkk12(6), calc_kkk12(7), calc_kkk12(8),
                              calc_kkk12(9), calc_kkk12(10)};

template <typename INT, typename VAL, bool is_real_mode = false,
          bool is_dual_mode = false>
struct dotCalculator {
  // compute top MAX_VALUES_SIZE values of |<psi|phi>|
  dotCalculator(size_t MAX_VALUES_SIZE) : MAX_VALUES_SIZE(MAX_VALUES_SIZE) {}

  auto calc_dot(int n, const vc& psi) {
    // Let b_i := <i|psi>, then
    // <phi|psi>
    // = 1/2^{k/2} (\sum_{x} (-1)^{x^T Q x} i^{c^T x} <Rx+t|) |psi>
    // = \sum_{x} (-1)^{x^T Q x} i^{c^T x} (1/2^{k/2} b_{Rx+t}^\dagger)
    // Thus, we use b_i^\dagger as the input
    // in order to avoid the conjugate operation in the inner loop.
    timer1.start();
    if constexpr (is_real_mode) {
      vec<double> psi_real(1 << n);
      for (int i = 0; i < 1 << n; i++) psi_real[i] = psi[i].real();
      for (int i = 0; i < 1 << n; i++) assert(std::abs(psi[i].imag()) < 1e-10);
      calc_dot_main(n, psi_real);
    } else {
      vc psi_conj = psi;
      for (auto& x : psi_conj) x = std::conj(x);
      calc_dot_main(n, psi_conj);
    }
    timer1.stop();
    std::cerr << " calculation time : " << timer1.report() << std::endl;
    sort(values.rbegin(), values.rend());
    int result_sz = std::min(values.size(), MAX_VALUES_SIZE);
    vec<INT> result(result_sz);
    for (int i = 0; i < result_sz; i++) result[i] = values[i].second;
    return result;
  }

 private:
  Timer timer1;
  vec<std::pair<double, INT>> values;
  double threshold = 0.0;
  size_t MAX_VALUES_SIZE;
  static constexpr int LARGE_K = 7;

  template <bool is_final = false>
  void truncate_values() {
    size_t SZ = MAX_VALUES_SIZE;
    if (is_final)
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

  template <typename InputIterator>
  bool check_branch_cut(const int k, const InputIterator Ps_begin) {
    // check if the branch cut is possible for k-th Ps ([Ps_begin, Ps_begin+(1<<k)))
    // max_{Q,c} |sum_{x} (-1)^{x^T Q x} i^{c^T x} P_x| < min(t1, sqrt(t2))
    double t1 = calc_threshold_1(k, Ps_begin);
    if (t1 < threshold) return true;
    if (!is_real_mode && t1 < 1.1 * threshold)
      if (calc_threshold_2(k, Ps_begin) < threshold * threshold) return true;
    return false;
  }

  vec<std::tuple<double, int, vec<VAL>, INT>> dfs_sub(const int k, const vec<VAL>& Ps,
                                                      const int ret_size,
                                                      INT& ret_idx) {
    vec<std::tuple<double, int, vec<VAL>, INT>> stk1;
    vec<std::tuple<int, int, INT>> stk2;
    vec<VAL> next(1 << (k - 1));
    for (bool c_0 : {false, true})
      for (bool q_00 : {false, true}) {
        if (is_real_mode && c_0) {
          ret_idx += kkk12s[k] >> 2;
          continue;
        }
        if constexpr (is_real_mode)
          next[0] = Ps[0] + Ps[1] * (q_00 ? -1 : 1);
        else
          next[0] = Ps[0] + Ps[1] * COMPLEX(q_00 ? -1 : 1) * (c_0 ? COMPLEX(0, 1) : 1);
        VAL coeff = 1.0;
        if constexpr (!is_real_mode) coeff = c_0 ? COMPLEX(0, 1) : 1.0;
        // non-recursive dfs
        stk2.emplace_back(1, 1, ret_idx + (kkk12s[k] >> 3));
        stk2.emplace_back(0, 1, ret_idx);
        while (!stk2.empty()) {
          auto [q_0, i, ret_idx_local] = stk2.back();
          stk2.pop_back();
          for (int x1 = 1 << (i - 1); x1 < 1 << i; x1++) {
            if (q_00 ^ __builtin_parity(q_0 & x1))
              next[x1] = Ps[x1 << 1] - coeff * Ps[(x1 << 1) ^ 1];
            else
              next[x1] = Ps[x1 << 1] + coeff * Ps[(x1 << 1) ^ 1];
          }
          if (i < k - 1) {
            stk2.emplace_back(q_0 ^ (1 << i), i + 1,
                              ret_idx_local + (kkk12s[k] >> (3 + i)));
            stk2.emplace_back(q_0, i + 1, ret_idx_local);
          } else {
            double t1 = std::accumulate(
                ALL(next), 0.0, [](double a, VAL b) { return a + std::abs(b); });
            if (t1 < threshold) continue;
            double t2 = ret_size == -1 ? std::numeric_limits<double>::infinity()
                                       : calc_threshold_2(k - 1, next.begin());
            if (t2 < threshold * threshold) continue;
            stk1.emplace_back(std::sqrt(t2), k - 1, next, ret_idx_local);
            if (ret_size != -1 && int(stk1.size()) > 2 * ret_size) {
              std::sort(ALL(stk1), [](const auto& a, const auto& b) {
                return std::get<0>(a) > std::get<0>(b);
              });
              stk1.resize(ret_size);
            }
          }
        }
        ret_idx += kkk12s[k] >> 2;
      }
    if (ret_size != -1 && int(stk1.size()) > ret_size) {
      std::sort(ALL(stk1), [](const auto& a, const auto& b) {
        return std::get<0>(a) > std::get<0>(b);
      });
      stk1.resize(ret_size);
    }
    return stk1;
  }

  vec<std::pair<double, INT>> calc_dot_sub(const int k_orig, const vec<VAL>& Ps_orig,
                                           const INT ret_idx_orig) {
    assert(1 <= k_orig && int(Ps_orig.size()) == (1 << k_orig));
    // Ps_list[(1<<k)+i] = Ps_list[(1<<k)^i] = k-th Ps[i] (0<=i<(1<<k))
    vec<VAL> Ps_list(1 << (k_orig + 1), 0);
    for (int i = 0; i < (1 << k_orig); i++) Ps_list[(1 << k_orig) ^ i] = Ps_orig[i];

    // we use non-recursive dfs. The each step of dfs is divided into two parts:
    // 1. set the value of c[k] and Q[k,k] (stk1)
    // 2. set the value of Q[k,i] (k<=i)   (stk2)
    vec<INT> stk1;
    vec<std::tuple<int, int_fast8_t, INT, bool, bool>> stk2;

    // in order to reduce critical section, we save values to local variable temporarily
    vec<std::pair<double, INT>> values_local;
    // +:stk1 -:stk2
    vi ks;

    stk1.emplace_back(ret_idx_orig);
    ks.push_back(k_orig);
    while (!ks.empty()) {
      int k = ks.back();
      ks.pop_back();
      if (k > 0) {
        // 1. set the value of c[k] and Q[k,k]
        INT ret_idx = stk1.back();
        stk1.pop_back();
        if (k == 1) {
          // 1+1, 1-1
          double val = std::abs(Ps_list[2 + 0] + Ps_list[2 + 1]);
          if (val > threshold) values_local.emplace_back(val, ret_idx);
          ret_idx++;
          val = std::abs(Ps_list[2 + 0] - Ps_list[2 + 1]);
          if (val > threshold) values_local.emplace_back(val, ret_idx);
          ret_idx++;
          // 1+1i, 1-1i
          if constexpr (is_real_mode) {
            ret_idx += 2;
          } else {
            val = std::abs(Ps_list[2 + 0] + COMPLEX(0, 1) * Ps_list[2 + 1]);
            if (val > threshold) values_local.emplace_back(val, ret_idx);
            ret_idx++;
            val = std::abs(Ps_list[2 + 0] - COMPLEX(0, 1) * Ps_list[2 + 1]);
            if (val > threshold) values_local.emplace_back(val, ret_idx);
            ret_idx++;
          }
        } else {
          if (check_branch_cut(k, Ps_list.begin() + (1 << k))) continue;
          for (bool c_0 : {false, true})
            for (bool q_00 : {false, true}) {
              if (is_real_mode && c_0) {
                ret_idx += kkk12s[k] >> 2;
                continue;
              }
              stk2.emplace_back(1, 1, ret_idx + (kkk12s[k] >> 3), c_0, q_00);
              stk2.emplace_back(0, 1, ret_idx, c_0, q_00);
              ks.push_back(-k);
              ks.push_back(-k);
              ret_idx += kkk12s[k] >> 2;
            }
        }
      } else {
        // 2. set the value of Q[k,i] (k<=i)
        k = -k;
        auto [q_0, i, ret_idx_local, c_0, q_00] = stk2.back();
        stk2.pop_back();
        if (q_0 == 0 && i == 1) {
          if constexpr (is_real_mode)
            Ps_list[(1 << (k - 1)) ^ 0] =
                Ps_list[(1 << k) ^ 0] + Ps_list[(1 << k) ^ 1] * (q_00 ? -1 : 1);
          else
            Ps_list[(1 << (k - 1)) ^ 0] =
                Ps_list[(1 << k) ^ 0] + Ps_list[(1 << k) ^ 1] * COMPLEX(q_00 ? -1 : 1) *
                                            (c_0 ? COMPLEX(0, 1) : 1);
        }
        VAL coeff = 1.0;
        if constexpr (!is_real_mode) coeff = c_0 ? COMPLEX(0, 1) : 1.0;
        for (int x1 = 1 << (i - 1); x1 < 1 << i; x1++) {
          int idx = (1 << (k - 1)) ^ x1;
          if (q_00 ^ __builtin_parity(q_0 & x1))
            Ps_list[idx] = Ps_list[idx << 1] - coeff * Ps_list[(idx << 1) ^ 1];
          else
            Ps_list[idx] = Ps_list[idx << 1] + coeff * Ps_list[(idx << 1) ^ 1];
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

  void calc_dot_sub_large_k(const int k, const vec<VAL>& Ps, INT ret_idx) {
    // If k is large, the size of rref (which means R and t) becomes too small to
    // parallelize. Thus, we parallelize by the first step of the non-recursive dfs.
    assert(LARGE_K <= k && int(Ps.size()) == (1 << k));
    if (check_branch_cut(k, Ps.begin())) return;
    auto stk1 = dfs_sub(k, Ps, -1, ret_idx);

#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
    for (int i = 0; i < int(stk1.size()); i++) {
      auto [_, k, PsLocal, ret_idx] = stk1[i];
      auto res = calc_dot_sub(k, PsLocal, ret_idx);
      if (!res.empty()) add_to_values(res);
      std::get<2>(stk1[i]).clear();
    }
  }

  void calc_dot_main(int n, const vec<VAL>& psi) {
    assert(int(psi.size()) == (1 << n));
    if constexpr (is_dual_mode) threshold = 0.97;  // little bit smaller than 1.0

    // Total number of stabilizer states
    INT t_s_g_s = total_stabilizer_group_size(n);

    // For the case k=0
    for (int i = 0; i < (1 << n); i++)
      if (std::abs(psi[i]) > threshold) values.emplace_back(std::abs(psi[i]), i);
    std::sort(ALL(values),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (int k = 1; k <= n; k++) {
      // rref means the reduced row echelon form of the matrix R.
      // R = [row_idxs[0]//row_idxs[1]//...//row_idxs[k-1]].

      // t is a element from the complement of R.
      // The complement of R can be expressed by basis vectors,
      // where each basis vector has only one element of 1 and the others are 0.
      // t_mask is a sum of the basis vectors.
      INT ret_idx = (INT(1) << n);
      for (int _k = 1; _k < k; _k++)
        ret_idx += q_binomial(n, _k) * (INT(1) << (n + _k * (_k + 1) / 2));
      RREF_generator rref_gen(n, k);
      double sqrt2k = 1 / std::pow(std::sqrt(2), k);
      vec<VAL> psi_1Over2k = psi;
      for (auto& x : psi_1Over2k) x *= sqrt2k;
      if (k < LARGE_K) {
        vec<INT> ret_idxs = {ret_idx};
        for (int rref_idx = 0; rref_idx < rref_gen.q_binom; rref_idx++) {
          ret_idx += (INT(1) << (n + k * (k + 1) / 2));
          ret_idxs.push_back(ret_idx);
        }
#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
        for (int rref_idx = 0; rref_idx < rref_gen.q_binom; rref_idx++) {
          const auto& [row_idxs, t_mask] = rref_gen.get(rref_idx, false);
          INT ret_idx_local = ret_idxs[rref_idx];
          vec<std::pair<double, INT>> values_local;
          int t = 0;
          while (true) {
            vec<VAL> Ps = arange_psi_by_t(k, t, row_idxs, psi_1Over2k);
            auto res = calc_dot_sub(k, Ps, ret_idx_local);
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
            // Iterate the subset of t_mask. Refer to the following URL for details.
            // https://stackoverflow.com/questions/7277554/what-is-a-good-way-to-iterate-a-number-through-all-the-possible-values-of-a-mask#comment9091440_7277818
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
            vec<VAL> Ps = arange_psi_by_t(k, t, row_idxs, psi_1Over2k);
            calc_dot_sub_large_k(k, Ps, ret_idx);
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
  bool is_real_mode;
  size_t MAX_VALUES_SIZE;

  try {
    cnpy::NpyArray psi_npy = cnpy::npz_load("temp_in.npz")["psi"];
    for (n = 0; n <= 10; n++)
      if (int(psi_npy.shape[0]) == (1 << n)) break;
    if (int(psi_npy.shape[0]) != (1 << n))
      throw std::runtime_error("Invalid input shape");
    psi.resize(1 << n);
    for (int i = 0; i < 1 << n; i++) psi[i] = psi_npy.data<COMPLEX>()[i];
    is_dual_mode = cnpy::npz_load("temp_in.npz")["is_dual_mode"].data<bool>()[0];
    MAX_VALUES_SIZE = cnpy::npz_load("temp_in.npz")["K"].data<int>()[0];
  } catch (const std::exception& e) {
    n = 6;
    psi.resize(1 << n);
    std::mt19937 mt(1);
    for (int i = 0; i < 1 << n; i++)
      psi[i] = COMPLEX(std::uniform_real_distribution<double>(-0.5, 0.5)(mt),
                       std::uniform_real_distribution<double>(-0.5, 0.5)(mt));
    is_dual_mode = true;
    MAX_VALUES_SIZE = 10000;
  }
  is_real_mode = true;
  for (int i = 0; i < 1 << n; i++)
    if (std::abs(psi[i].imag()) > 1e-10) is_real_mode = false;

  // cnpy does not correspond to __int128_t, so we use long long instead.
  vec<ll> res_ll_1, res_ll_2;

  // If n>=10, use __int128_t instead of long long for the data type INT.
  // Ff the result is empty, cnpy fails to save the file.
  // We add a dummy value (0) in this case.
  if (n <= 9) {
    vec<ll> res;
    if (is_real_mode) {
      if (is_dual_mode) {
        dotCalculator<ll, double, true, true> calculator(MAX_VALUES_SIZE);
        res = calculator.calc_dot(n, psi);
      } else {
        dotCalculator<ll, double, true, false> calculator(MAX_VALUES_SIZE);
        res = calculator.calc_dot(n, psi);
      }
    } else {
      if (is_dual_mode) {
        dotCalculator<ll, COMPLEX, false, true> calculator(MAX_VALUES_SIZE);
        res = calculator.calc_dot(n, psi);
      } else {
        dotCalculator<ll, COMPLEX, false, false> calculator(MAX_VALUES_SIZE);
        res = calculator.calc_dot(n, psi);
      }
    }
    if (res.empty()) res.push_back(0);
    res_ll_1.resize(res.size());
    res_ll_2 = res;
  } else if (n == 10) {
    vec<__int128_t> res;
    if (is_real_mode) {
      if (is_dual_mode) {
        dotCalculator<__int128_t, double, true, true> calculator(MAX_VALUES_SIZE);
        res = calculator.calc_dot(n, psi);
      } else {
        dotCalculator<__int128_t, double, true, false> calculator(MAX_VALUES_SIZE);
        res = calculator.calc_dot(n, psi);
      }
    } else {
      if (is_dual_mode) {
        dotCalculator<__int128_t, COMPLEX, false, true> calculator(MAX_VALUES_SIZE);
        res = calculator.calc_dot(n, psi);
      } else {
        dotCalculator<__int128_t, COMPLEX, false, false> calculator(MAX_VALUES_SIZE);
        res = calculator.calc_dot(n, psi);
      }
    }
    if (res.empty()) res.push_back(0);
    res_ll_1.resize(res.size());
    res_ll_2.resize(res.size());
    for (size_t i = 0; i < res.size(); i++) {
      res_ll_1[i] = res[i] >> 62;
      res_ll_2[i] = res[i] & ((1ll << 62) - 1);
    }
  } else {
    std::cerr << "n should be less than or equal to 10." << std::endl;
    std::cerr << "Otherwise, we cannot guarantee the overflow does not occur."
              << std::endl;
    return 1;
  }

  size_t res_size = res_ll_1.size();
  cnpy::npz_save("temp_out.npz", "res1", res_ll_1.data(), {res_size}, "w");
  cnpy::npz_save("temp_out.npz", "res2", res_ll_2.data(), {res_size}, "a");

  return 0;
}
