#pragma once

#include <algorithm>
#include <glass/neighbor.hpp>
#include <mutex>
#include <random>
#include <unordered_set>
#include <vector>

namespace glass {

using LockGuard = std::lock_guard<std::mutex>;

inline void GenRandom(std::mt19937& rng, int* addr, const int size, const int N) {
  for (int i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (int i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  int off = rng() % N;
  for (int i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

struct RandomGenerator {
  std::mt19937 mt;

  explicit RandomGenerator(int64_t seed = 1234) : mt((unsigned int)seed) {}

  /// random positive integer
  int rand_int() { return mt() & 0x7fffffff; }

  /// random int64_t
  int64_t rand_int64() { return int64_t(rand_int()) | int64_t(rand_int()) << 31; }

  /// generate random integer between 0 and max-1
  int rand_int(int max) { return mt() % max; }

  /// between 0 and 1
  float rand_float() { return mt() / float(mt.max()); }

  double rand_double() { return mt() / double(mt.max()); }
};

// template <typename IDType = uint32_t>
// float CalRecallById(const IDType* kResVal, const uint32_t kNum, const uint32_t kK, const IDType* kGtVal,
//                     const uint32_t kGtLine) {
//   float exact_num = 0;
//   for (auto n = 0; n < kNum; ++n) {
//     std::vector<bool> visted(kK, false);
//     for (uint32_t i = 0; i < kK; ++i) {
//       for (uint32_t j = 0; j < kK; ++j) {
//         if (!visted[j] && kResVal[n * kK + i] == kGtVal[n * kGtLine + j]) {
//           exact_num++;
//           visted[j] = true;
//           break;
//         }
//       }
//     }
//   }
//   return exact_num / (kNum * kK);
// }

float CalRecallById(const std::vector<glass::searcher::LPool<float>>& res, const uint32_t kNum, const uint32_t* kGtVal,
                    const uint32_t kGtLine) {
  float exact_num = 0;
  for (auto q = 0; q < res.size(); ++q) {
    auto& res_pool = res[q];
    std::vector<bool> visted(kNum, false);
    for (uint32_t i = 0; i < kNum; ++i) {
      for (uint32_t j = 0; j < kNum; ++j) {
        if (!visted[j] && res_pool.data_[i].id == kGtVal[q * kGtLine + j]) {
          exact_num++;
          visted[j] = true;
          break;
        }
      }
    }
  }
  return exact_num / (res.size() * kNum);
}

}  // namespace glass