#include "include/include.hpp"
#include "rref.cpp"

// {value, k, Q, c, row_idxs, t}
using Info = std::tuple<double, int, __int128_t, int, vi, int>;

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

template <typename... T>
void truncate_goods(vec<std::tuple<T...>>& goods, size_t max_size,
                    bool do_remove_duplicates) {
  if (do_remove_duplicates) {
    std::sort(ALL(goods), std::greater<>());
    goods.erase(std::unique(ALL(goods),
                            [&](const auto& a, const auto& b) {
                              return std::get<1>(a) == std::get<1>(b) &&
                                     std::get<2>(a) == std::get<2>(b);
                            }),
                goods.end());
  } else {
    std::sort(ALL(goods), [](const auto& a, const auto& b) {
      return std::get<0>(a) > std::get<0>(b);
    });
  }
  if (goods.size() > max_size) goods.resize(max_size);
}

std::pair<vi, vc> make_col_from_QcRt(int k, __int128_t Q, int c, const vi& row_idxs,
                                     int t) {
  vi indices;
  vc values;
  double coeff_2k = 1.0 / std::pow(2, k / 2.0);
  assert(!row_idxs.empty() || t == 0);
  for (int x = 0; x < (1 << k); x++) {
    values.push_back(xQc_to_coeff(k, x, Q, c) * coeff_2k);
    indices.push_back((row_idxs.empty() ? x : row_idxs[x]) ^ t);
  }
  return {indices, values};
}

vec<Info> hill_climbing(int k, const vc& b, const bool is_dual_mode) {
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
  vc coeffs(1 << k);
  vi Ps(Q_non_diag.size());
  std::iota(ALL(Ps), 0);
  Timer timer_local;
  timer_local.start();
  vec<std::tuple<double, __int128_t, int>> good_Qc;
  // This function is based on hill climbing algorithm.
  while (timer_local.ms() < 30000) {
    // 1. Randomly choose Q and c
    __int128_t Q = randRangePow2(k * (k + 1) / 2);
    int c = randRangePow2(k);
    for (int x = 0; x < (1 << k); x++) coeffs[x] = coeff * xQc_to_coeff(k, x, Q, c);
    COMPLEX now_sum = std::inner_product(ALL(coeffs), b.begin(), COMPLEX(0, 0));
    double now_abs = std::abs(now_sum);
    // 2. Hill climbing
    //    If there is a better Q and c with one bit flip, update it.
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
    if (!is_dual_mode || now_abs > 1.0) good_Qc.emplace_back(now_abs, Q, c);
    if (good_Qc.size() > 2000) truncate_goods(good_Qc, 1000, true);
  }
  truncate_goods(good_Qc, 1000, true);
  vec<Info> ret;
  for (const auto& [val, Q, c] : good_Qc) ret.emplace_back(val, k, Q, c, vi(), 0);
  return ret;
}

std::tuple<vi, vi, vc> get_rough_topK_Amat(int n, const vc& _psi,
                                           const bool is_dual_mode) {
  // Roughly estimate the following problem
  // max_{phi \in S_n} |<phi|psi>|^2
  // We only consider the case of k=n and k=n-1.

  vc psi = _psi;
  for (int i = 0; i < (1 << n); i++) psi[i] = std::conj(psi[i]);

  const int MAX_THREAD_NUM = omp_get_max_threads();
  vec<Info> good_kQcRt;

  // 1. Search for the case k=n
#pragma omp parallel for num_threads(MAX_THREAD_NUM)
  for (int i = 0; i < MAX_THREAD_NUM; i++) {
    vec<Info> ans = hill_climbing(n, psi, is_dual_mode);
#pragma omp critical
    good_kQcRt.insert(good_kQcRt.end(), std::make_move_iterator(ans.begin()),
                      std::make_move_iterator(ans.end()));
  }
  std::sort(ALL(good_kQcRt), std::greater<>());
  good_kQcRt.erase(std::unique(ALL(good_kQcRt),
                               [](const Info& a, const Info& b) {
                                 return std::get<2>(a) == std::get<2>(b) &&
                                        std::get<3>(a) == std::get<3>(b);
                               }),
                   good_kQcRt.end());
  if (!good_kQcRt.empty())
    std::cerr << "Case k=n done | current max: " << std::get<0>(good_kQcRt.front())
              << std::endl;

  // 2. Search for the case k=n-1
  RREF_generator rref_gen(n, n - 1);
  vec<std::tuple<double, vi, int>> good_Rt;
#pragma omp parallel for schedule(dynamic) num_threads(MAX_THREAD_NUM)
  for (int rref_idx = 0; rref_idx < rref_gen.q_binom; rref_idx++) {
    const auto& [row_idxs, t_mask] = rref_gen.get(rref_idx, false);
    assert(__builtin_popcount(t_mask) == 1);
    for (int t : {0, t_mask}) {
      vc psi2 = arange_psi_by_t(n - 1, t, row_idxs, psi);
      double t1 = calc_threshold_1(n - 1, psi2.begin());  // not normalized
#pragma omp critical
      {
        good_Rt.emplace_back(t1, row_idxs, t);
        if (good_Rt.size() > 20) truncate_goods(good_Rt, 10, false);
      }
    }
  }
  truncate_goods(good_Rt, 10, false);
  std::cerr << "progress: " << rref_gen.q_binom << "/" << rref_gen.q_binom << std::endl;

#pragma omp parallel for num_threads(MAX_THREAD_NUM)
  for (const auto& [_, row_idxs, t] : good_Rt) {
    vc psi2 = arange_psi_by_t(n - 1, t, row_idxs, psi);
    vec<Info> ans = hill_climbing(n - 1, psi2, is_dual_mode);
    for (auto& info : ans) {
      std::get<4>(info) = row_idxs;
      std::get<5>(info) = t;
    }
    if (ans.empty()) continue;
#pragma omp critical
    {
      std::cerr << "Case k=n-1 | current max:" << std::get<0>(ans.front()) << std::endl;
      good_kQcRt.insert(good_kQcRt.end(), std::make_move_iterator(ans.begin()),
                        std::make_move_iterator(ans.end()));
    }
  }

  // 3. from kQcRt to A matrix
  //   indptr, indices, data
  if (good_kQcRt.empty()) {
    std::cerr << "No good kQcRt found" << std::endl;
    return {vi{-1}, vi{-1}, vc{-1}};
  }

  std::tuple<vi, vi, vc> ret = {vi(), vi(), vc()};
  std::get<0>(ret).push_back(0);
  for (const auto& [_, k, Q, c, row_idxs, t] : good_kQcRt) {
    auto [indices, values] = make_col_from_QcRt(k, Q, c, row_idxs, t);
    std::get<0>(ret).push_back(std::get<0>(ret).back() + (1 << k));
    std::get<1>(ret).insert(std::get<1>(ret).end(), ALL(indices));
    std::get<2>(ret).insert(std::get<2>(ret).end(), ALL(values));
  }
  std::cerr << "A matrix construction done" << std::endl;

  return ret;
}

int main() {
  int n;
  vc psi;
  bool is_dual_mode;

  try {
    cnpy::NpyArray psi_npy = cnpy::npz_load("temp_in.npz")["psi"];
    for (n = 5; n <= 15; n++)
      if (int(psi_npy.shape[0]) == (1 << n)) break;
    if (int(psi_npy.shape[0]) != (1 << n))
      throw std::runtime_error("Invalid input shape");
    psi.resize(1 << n);
    for (int i = 0; i < 1 << n; i++) psi[i] = psi_npy.data<COMPLEX>()[i];
    is_dual_mode = cnpy::npz_load("temp_in.npz")["is_dual_mode"].data<bool>()[0];
  } catch (const std::exception& e) {
    n = 5;
    psi.resize(1 << n);
    std::mt19937 mt(1);
    for (int i = 0; i < 1 << n; i++)
      psi[i] = COMPLEX(std::uniform_real_distribution<double>(-0.5, 0.5)(mt),
                       std::uniform_real_distribution<double>(-0.5, 0.5)(mt));
    is_dual_mode = false;
  }

  auto rough_topK_Amat = get_rough_topK_Amat(n, psi, is_dual_mode);

  auto [indptr, indices, data] = rough_topK_Amat;
  cnpy::npz_save("temp_out.npz", "indptr", indptr.data(), {indptr.size()}, "w");
  cnpy::npz_save("temp_out.npz", "indices", indices.data(), {indices.size()}, "a");
  cnpy::npz_save("temp_out.npz", "data", data.data(), {data.size()}, "a");

  return 0;
}