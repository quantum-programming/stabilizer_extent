#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <complex>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// clang-format off
template <typename T> using vec = std::vector<T>;
template <typename T> using vvec = std::vector<vec<T>>;
template <typename T> using vvvec = std::vector<vvec<T>>;
using INT = long long;
using COMPLEX = std::complex<double>;
using vi = vec<int>; using vvi = vec<vi>; using vvvi = vec<vvi>;
using vb = vec<bool>; using vvb = vec<vb>; using vvvb = vec<vvb>;
using vc = vec<COMPLEX>; using vvc = vec<vc>; using vvvc = vec<vvc>;
#define ALL(x) begin(x), end(x)

struct Timer{
    void start(){_start=std::chrono::system_clock::now();}
    void stop(){_end=std::chrono::system_clock::now();sum+=std::chrono::duration_cast<std::chrono::nanoseconds>(_end-_start).count();}
    inline int ms()const{const std::chrono::system_clock::time_point now=std::chrono::system_clock::now();return static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(now-_start).count()/1000);}
    inline int ns()const{const std::chrono::system_clock::time_point now=std::chrono::system_clock::now();return static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(now-_start).count());}
    std::string report(){stop();start();return std::to_string(sum/1000000)+"[ms]";}
    void reset(){_start=std::chrono::system_clock::now();sum=0;}
    private: std::chrono::system_clock::time_point _start,_end;long long sum=0;
} timer;

struct Xor128{  // period 2^128 - 1
    uint32_t x,y,z,w;
    Xor128(uint32_t seed=0):x(123456789),y(362436069),z(521288629),w(88675123+seed){}
    uint32_t operator()(){uint32_t t=x^(x<<11);x=y;y=z;z=w;return w=(w^(w>>19))^(t^(t>>8));}
    uint32_t operator()(uint32_t l,uint32_t r){return((*this)()%(r-l))+l;}
    uint32_t operator()(uint32_t r){return(*this)()%r;}};
struct Rand {  // https://docs.python.org/ja/3/library/random.html
    Rand(int seed):gen(seed){};
    bool randBool(){return gen()&1;}
    template<class T>
    void shuffle(vec<T>&x){for(int i=x.size(),j;i>1;){j=gen(i);std::swap(x[j],x[--i]);}}
   private:
    Xor128 gen;
} myrand(0);
// clang-format on

// https://stackoverflow.com/questions/25114597/how-to-print-int128-in-g
std::ostream& operator<<(std::ostream& dest, __int128_t value) {
  std::ostream::sentry s(dest);
  if (s) {
    __uint128_t tmp = value < 0 ? -value : value;
    char buffer[128];
    char* d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[tmp % 10];
      tmp /= 10;
    } while (tmp != 0);
    if (value < 0) {
      --d;
      *d = '-';
    }
    int len = std::end(buffer) - d;
    if (dest.rdbuf()->sputn(d, len) != len) {
      dest.setstate(std::ios_base::badbit);
    }
  }
  return dest;
}