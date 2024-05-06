#pragma once
#include "base.cpp"

namespace viewer {
constexpr int INF = 1001001001;
constexpr long long INFll = 1001001001001001001;

template <class T>
std::string f(T i) {
  std::string S;
  if (i == INF) {
    S = "INF";
  } else if (i == -INF) {
    S = "-INF";
  } else if (i == INFll) {
    S = "INFll";
  } else if (i == -INFll) {
    S = "-INFll";
  } else {
    S = std::to_string(i);
  }
  return std::string(std::max(0, 3 - int(S.size())), ' ') + S;
}

template <class T>
auto v(T& x, std::string&& e) -> decltype(std::cerr << x) {
  return std::cerr << x << e;
}

void v(bool x, std::string&& e) { std::cerr << (x ? "T" : "F") << e; }

void v(int x, std::string&& e) { std::cerr << f(x) << e; }

void v(long long x, std::string&& e) { std::cerr << f(x) << e; }

void v(float x, std::string&& e) {
  std::cerr << std::fixed << std::setprecision(5) << x << e;
}

void v(double x, std::string&& e) {
  std::cerr << std::fixed << std::setprecision(10) << x << e;
}

void v(long double x, std::string&& e) {
  std::cerr << std::fixed << std::setprecision(15) << x << e;
}

template <class T, class U>
void v(const std::pair<T, U>& p, std::string&& e = "\n") {
  std::cerr << "(";
  v(p.first, ", ");
  v(p.second, ")" + e);
}

template <class T, class U>
void v(const std::tuple<T, U>& t, std::string&& e = "\n") {
  std::cerr << "(";
  v(std::get<0>(t), ", ");
  v(std::get<1>(t), ")" + e);
}

template <class T, class U, class V>
void v(const std::tuple<T, U, V>& t, std::string&& e = "\n") {
  std::cerr << "(";
  v(std::get<0>(t), ", ");
  v(std::get<1>(t), ", ");
  v(std::get<2>(t), ")" + e);
}

template <class T, class U, class V, class W>
void v(const std::tuple<T, U, V, W>& t, std::string&& e = "\n") {
  std::cerr << "(";
  v(std::get<0>(t), ", ");
  v(std::get<1>(t), ", ");
  v(std::get<2>(t), ", ");
  v(std::get<3>(t), ")" + e);
}
template <class T, class U, class V, class W, class X>
void v(const std::tuple<T, U, V, W, X>& t, std::string&& e = "\n") {
  std::cerr << "(";
  v(std::get<0>(t), ", ");
  v(std::get<1>(t), ", ");
  v(std::get<2>(t), ", ");
  v(std::get<3>(t), ", ");
  v(std::get<4>(t), ")" + e);
}

template <class T>
void v(const std::vector<T>& vx, std::string);

template <class T>
auto ve(int, const std::vector<T>& vx) -> decltype(std::cerr << vx[0]) {
  std::cerr << "{";
  for (const T& x : vx) v(x, ",");
  return std::cerr << "}\n";
}

template <class T>
auto ve(bool, const std::vector<T>& vx) {
  std::cerr << "{\n";
  for (const T& x : vx) std::cerr << "  ", v(x, ",");
  std::cerr << "}\n";
}

template <class T>
void v(const std::vector<T>& vx, std::string) {
  ve(0, vx);
}

template <class T>
void v(const std::deque<T>& q, std::string&& e) {
  v(std::vector<T>(q.begin(), q.end()), e);
}

template <class T, class C>
void v(const std::set<T, C>& S, std::string&& e) {
  v(std::vector<T>(S.begin(), S.end()), e);
}

template <class T, class C>
void v(const std::multiset<T, C>& S, std::string&& e) {
  v(std::vector<T>(S.begin(), S.end()), e);
}

template <class T>
void v(const std::unordered_set<T>& S, std::string&& e) {
  v(std::vector<T>(S.begin(), S.end()), e);
}

template <class T, class U, class V>
void v(const std::priority_queue<T, U, V>& p, std::string&& e) {
  std::priority_queue<T, U, V> q = p;
  std::vector<T> z;
  while (!q.empty()) {
    z.push_back(q.top());
    q.pop();
  }
  v(z, e);
}

template <class T, class U>
void v(const std::map<T, U>& m, std::string&& e) {
  std::cerr << "{" << (m.empty() ? "" : "\n");
  for (const auto& kv : m) {
    std::cerr << "  [";
    v(kv.first, "");
    std::cerr << "] : ";
    v(kv.second, "");
    std::cerr << "\n";
  }
  std::cerr << "}" + e;
}

template <class T>
void grid(T) {
  assert(false);
}

void grid(const std::vector<std::vector<bool>>& vvb) {
  std::cerr << "\n";
  for (const std::vector<bool>& vb : vvb) {
    for (const bool& b : vb) std::cerr << (b ? "." : "#");
    std::cerr << "\n";
  }
}

void _debug(int, std::string) {}

template <typename H, typename... T>
void _debug(int n, std::string S, const H& h, const T&... t) {
  if (n != -1 && S.size() >= 8 && std::string(S.end() - 6, S.end()) == "\"grid\"") {
    std::cerr << "\033[1;32m" << n << "\033[0m: \033[1;36m"
              << std::string(S.begin(), S.end() - 8) << "(.:true, #:false)"
              << "\033[0m = ";
    grid(h);
    return;
  } else if (n != -1) {
    std::cerr << "\033[1;32m" << n << "\033[0m: \033[1;36m" << S << "\033[0m = ";
  }
  v(h, sizeof...(t) ? "," : "\n");
  _debug(-1, "", t...);
}
}  // namespace viewer

#define debug(...) viewer::_debug(__LINE__, #__VA_ARGS__, __VA_ARGS__)
