#include "../rref.cpp"

void test_rref() {
  std::vector<std::pair<int, int>> nk_pairs = {{1, 1}, {2, 1}, {2, 2}, {3, 2}, {4, 4}};
  for (auto [n, k] : nk_pairs) {
    std::cout << "n: " << n << ", k: " << k << std::endl;
    RREF_generator rref_gen(n, k);
    for (int rref_idx = 0; rref_idx < rref_gen.q_binom; rref_idx++) {
      auto [mat, t_mask] = rref_gen.get(rref_idx, true);
      vi complement;
      for (int idx = 0; idx < n; ++idx)
        if (t_mask & (1 << idx)) complement.push_back(1 << idx);
      assert(int(complement.size()) == n - k);
      std::cout << "mat: ";
      for (auto m : mat) std::cout << m << " ";
      std::cout << ", complement: ";
      for (auto c : complement) std::cout << c << " ";
      std::cout << std::endl;
      for (auto row : mat)
        std::cout << std::bitset<32>(row).to_string().substr(32 - n) << std::endl;
      std::cout << "---" << std::endl;
      for (auto row : complement)
        std::cout << std::bitset<32>(row).to_string().substr(32 - n) << std::endl;
      std::cout << "=====" << std::endl;
    }
  }
}

int main() {
  test_rref();
  return 0;
}