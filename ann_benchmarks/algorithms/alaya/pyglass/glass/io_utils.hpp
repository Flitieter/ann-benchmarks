#pragma once

#include <fmt/core.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace glass {

inline void load_fvecs(char *filename, float *&data, unsigned &num,
                       unsigned &dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char *)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[num * dim * sizeof(float)];

  fmt::println("Read {}", filename);
  fmt::println("data number: {}, data dimension: {}", num, dim);

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char *)(data + i * dim), dim * 4);
  }
  in.close();
}

inline void load_ivecs(char *filename, unsigned *&data, unsigned &num,
                       unsigned &dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char *)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);

  //  fmt::println("Read {}", filename);
  //  fmt::println("data number: {}, data dimension: {}", num, dim);

  data = new unsigned[num * dim * sizeof(int)];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char *)(data + i * dim), dim * 4);
  }
  in.close();
}

template <typename IDType = uint32_t>
float CalRecallById(const IDType *kResVal, const uint32_t kNum, const uint32_t kK, const IDType *kGtVal,
                    const uint32_t kGtLine) {
  float exact_num = 0;
  for (auto n = 0; n < kNum; ++n) {
    std::vector<bool> visted(kK, false);
    for (uint32_t i = 0; i < kK; ++i) {
      for (uint32_t j = 0; j < kK; ++j) {
        if (!visted[j] && kResVal[n * kK + i] == kGtVal[n * kGtLine + j]) {
          exact_num++;
          visted[j] = true;
          break;
        }
      }
    }
  }
  return exact_num / (kNum * kK);
}

// inline double compute_recall(unsigned *answers, unsigned *results, unsigned query_num, unsigned k, unsigned dim) {
//   double sum_recall = 0.0;

//   // for (unsigned query_idx = 0; query_idx < query_num; query_idx++) {
//   //   std::unordered_set<int> res_set;
//   //   for (unsigned i = 0; i < i; i++) {
//   //     res_set.insert(results[query_idx * k + i]);
//   //   }

//   //   int true_pos = 0;
//   //   for (unsigned i = 0; i < query_num; i++) {
//   //     unsigned retuned_idx = answers[query_idx * k + i];
//   //     if (res_set.find(retuned_idx) != res_set.end()) {
//   //       true_pos += 1;
//   //     }
//   //   }

//   //   sum_recall += static_cast<double>(true_pos) / k;
//   // }

//   return sum_recall / query_num;
// }

}  // namespace glass