#include "include/include.hpp"

struct RREF_generator {
  int n, k;  // k times n
  int q_binom;

  RREF_generator(int n, int k) : n(n), k(k) {
    assert(1 <= n && 1 <= k && k <= n);
    q_binom = q_binomial(n, k);  // within the range of int
    combinations = generate_combinations(n, k);
    info.reserve(combinations.size());
    for (int i = 0; i < int(combinations.size()); i++)
      info.push_back(update_default_matrix(i));
    prefix_sum.assign(combinations.size() + 1, 0);
    for (int i = 0; i < int(combinations.size()); i++)
      prefix_sum[i + 1] = prefix_sum[i] + (1 << std::get<2>(info[i]).size());
    assert(prefix_sum.back() == q_binom);
  }

  std::pair<vi, int> get(int idx, bool return_mat) {
    assert(0 <= idx && idx < q_binom);
    int info_idx = std::upper_bound(ALL(prefix_sum), idx) - prefix_sum.begin() - 1;
    assert(prefix_sum[info_idx] <= idx && idx < prefix_sum[info_idx + 1]);
    idx -= prefix_sum[info_idx];
    assert(0 <= info_idx && info_idx < int(info.size()));
    const auto& [default_matrix, complement, Is, Js, not_col_idxs] = info[info_idx];
    assert(0 <= idx && idx < (1 << Is.size()));
    vi mat = default_matrix;
    for (int i = 0; i < int(Is.size()); i++)
      if (idx & (1 << i)) mat[Is[i]] += 1 << not_col_idxs[Js[i]];
    if (return_mat) return {mat, complement};
    vi row_idxs(1 << k);
    for (int x = 0; x < (1 << k); x++) {
      int row_idx = 0;
      for (int i = 0; i < k; i++)
        if (x & (1 << i)) row_idx ^= mat[i];
      row_idxs[x] = row_idx;
    }
    return {row_idxs, complement};
  }

 private:
  using Info = std::tuple<vi, int, vi, vi, vi>;
  vvi combinations;  // all combinations of k elements from n
  vec<Info> info;    // {default_matrix, complement, Is, Js, not_col_idxs}
  vi prefix_sum;     // prefix sum of 1<<Is.size()

  Info update_default_matrix(int combination_idx) {
    const vi& col_idxs = combinations[combination_idx];
    int complement = 0;
    vi not_col_idxs;
    vi default_matrix(k, 0);
    for (int col_idx = 0; col_idx < n; col_idx++) {
      if (find(ALL(col_idxs), col_idx) != col_idxs.end()) {
        default_matrix[col_idx - not_col_idxs.size()] += 1 << col_idx;
      } else {
        complement += 1 << col_idx;
        not_col_idxs.push_back(col_idx);
      }
    }
    vi Is, Js;
    for (int i = 0; i < k; i++)
      for (int j = 0; j < n - k; j++)
        if (col_idxs[i] < not_col_idxs[j]) {
          Is.push_back(i);
          Js.push_back(j);
        }
    return {default_matrix, complement, Is, Js, not_col_idxs};
  }
};
