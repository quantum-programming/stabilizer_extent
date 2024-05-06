#pragma once
#include "base.cpp"

#define REP_01(i) for (int i = 0; i <= 1; i++)

struct AmatForSmallN {
  std::complex<double> Amat1[1 << (1 + 1 * (1 + 1) / 2)][1 << 1];
  std::complex<double> Amat2[1 << (2 + 2 * (2 + 1) / 2)][1 << 2];

  AmatForSmallN() { make_Amat_for_small_n(); }

 private:
  vc from_CQ_to_col(int n, const vi& Cs, const vvi& Qs) {
    vc col(1 << n);
    for (int x = 0; x < (1 << n); x++) {
      int c_cnt = 0, q_cnt = 0;
      for (int i = 0; i < n; i++) {
        if (x & (1 << i)) {
          c_cnt += Cs[i];
          q_cnt += Qs[i][i];
        }
        for (int j = i + 1; j < n; j++)
          if ((x & (1 << i)) && (x & (1 << j))) q_cnt += Qs[i][j];
      }
      col[x] = pow(-1.0, q_cnt) * pow(std::complex<double>(0, 1), c_cnt);
    }
    return col;
  }

  void make_Amat_for_small_n() {
    size_t idx = 0;
    REP_01(c_0) REP_01(q_00) {
      vc col = from_CQ_to_col(1, {c_0}, {{q_00}});
      for (int i = 0; i < (1 << 1); i++) Amat1[idx][i] = col[i];
      idx++;
    }
    idx = 0;
    REP_01(c_0) REP_01(q_00) REP_01(q_01) REP_01(c_1) REP_01(q_11) {
      vc col = from_CQ_to_col(2, {c_0, c_1}, {{q_00, q_01}, {0, q_11}});
      for (int i = 0; i < (1 << 2); i++) Amat2[idx][i] = col[i];
      idx++;
    }
  }
};