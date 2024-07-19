#include <fmt/chrono.h>
#include <fmt/core.h>

#include <filesystem>
#include <glass/graph.hpp>
#include <glass/hnsw/hnsw.hpp>
#include <glass/io_utils.hpp>
#include <glass/neighbor.hpp>
#include <glass/rerank.hpp>
#include <glass/searcher.hpp>
#include <glass/timer.hpp>
#include <set>
#include <string>

void rerank(std::vector<int> &tmp_id, const float *query, const float *data, glass::searcher::LPool<float> &res_pool,
            int k, int rerank, int dim) {
  for (size_t i = 0; i < k; ++i) {
    res_pool.insert_back(tmp_id[i], glass::L2SqrRef(query, data + tmp_id[i] * dim, dim));
  }
  res_pool.sorted();
  for (size_t i = k; i < rerank; ++i) {
    res_pool.emplace_insert(tmp_id[i], glass::L2SqrRef(query, data + tmp_id[i] * dim, dim));
  }
  // for (size_t i = 0; i < k; ++i) {
  //   printf("(%d, %f), ", res_pool.data_[i].id, res_pool.data_[i].distance);
  // }
  // printf("\n");
}

float cal_recall(const std::vector<std::vector<int>> &res_id, const int *gt_id, int gt_line, int query_num,
                 int data_num, int dim, int topk) {
  float exact_num = 0;
  for (auto q = 0; q < query_num; ++q) {
    auto res_id = gt_id + q * topk;
    std::vector<bool> visted(topk, false);
    for (uint32_t i = 0; i < topk; ++i) {
      for (uint32_t j = 0; j < topk; ++j) {
        if (!visted[j] && res_id[i] == gt_id[q * gt_line + j]) {
          exact_num++;
          visted[j] = true;
          break;
        }
      }
    }
  }
  return exact_num / (query_num * topk);
}

int main(int argc, char **argv) {
  std::filesystem::path dir_path(argv[1]);
  int M = atoi(argv[2]);
  int topk = atoi(argv[3]);
  int ef = atoi(argv[4]);
  int rerank_k = atoi(argv[5]);

  std::string dir_name = dir_path.filename().string();
  std::filesystem::path data_file = dir_path / fmt::format("{}_base.fvecs", dir_name);
  std::filesystem::path query_file = dir_path / fmt::format("{}_query.fvecs", dir_name);
  std::filesystem::path gt_file = dir_path / fmt::format("{}_gti.ivecs", dir_name);

  float *data_load = NULL;
  unsigned points_num, dim;
  glass::load_fvecs(data_file.string().c_str(), data_load, points_num, dim);

  float *query_load = NULL;
  unsigned query_num, query_dim;
  glass::load_fvecs(query_file.string().c_str(), query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned *answers = NULL;
  unsigned ans_num, gt_col;
  glass::load_ivecs(gt_file.string().c_str(), answers, ans_num, gt_col);
  assert(ans_num == query_num);

  std::filesystem::path index_file = dir_path / fmt::format("{}_M{}.hnsw", dir_name, M);

  glass::Graph<int> *graph = nullptr;

  if (std::filesystem::exists(index_file)) {
    glass::Graph<int> *load_graph = new glass::Graph<int>();
    load_graph->load(index_file.string());
    graph = load_graph;
  } else {
    auto index = std::unique_ptr<glass::Builder>((glass::Builder *)new glass::HNSW(dim, "L2", M));
    index->Build(data_load, points_num);
    graph = new glass::Graph<int>(index->GetGraph());
    graph->save(index_file.string());
  }

  // index->GetGraph().save("sift.hsnw");

  // std::cout << "Build HNSW Over" << std::endl;

  const std::string metric = "L2";
  int level = 3;
  auto searcher = std::unique_ptr<glass::SearcherBase>(glass::create_searcher(*graph, metric, level));
  searcher->SetEf(ef);

  searcher->SetData(data_load, points_num, dim);

  searcher->Optimize(1);

  std::cout << "create searcher" << std::endl;

  // std::vector<maxPQIFCS<float>> candidate_pool(query_num, maxPQIFCS<float>(topk));
  // std::vector<std::vector<int>> tmp_id(query_num, std::vector<int>(ef));

  // std::vector<std::vector<unsigned>> res_id(query_num, std::vector<int>(topk));

  Timer<std::chrono::milliseconds> timer;

  // int num = 100;

  timer.reset();
  // #pragma omp parallel for schedule(dynamic)

  // while (num--) {
  // query_num = 10;
  std::vector<glass::searcher::LPool<float>> res_pool(query_num, glass::searcher::LPool<float>(topk));
  for (size_t q = 0; q < query_num; ++q) {
    // printf("%d\n", q);
    // auto &ids = tmp_id[q];
    std::vector<int> ids(ef);
    // glass::searcher::LPool<float> res_pool(rerank_k);
    auto cur_query = query_load + q * query_dim;
    searcher->Search(cur_query, ef, ids.data());

    // for (int i = 0; i < ef; ++i) {
    //   printf("id: %d, ", ids[i]);
    // }
    // printf("\n");

    rerank(ids, cur_query, data_load, res_pool[q], topk, rerank_k, dim);
  }
  // }

  timer.end();

  // for (size_t q = 0; q < 10; ++q) {
  //   auto &ids = res_pool[q].data_;
  //   for (size_t i = 0; i < topk; ++i) {
  //     printf("%d, ", ids[i].id);
  //   }
  //   printf("\n");
  // }

  fmt::println("topk:{}, ef:{}, rerank_k:{}", topk, ef, rerank_k);
  float sec = timer.getElapsedTime().count() / 1000.0;

  fmt::println("Query Num: {}, Data Num: {}", query_num, points_num);
  fmt::println("Search Time: {}, in sec: {}", timer.getElapsedTime(), sec);
  fmt::println("QPS: {}", query_num / sec);

  float recall = glass::CalRecallById(res_pool, topk, answers, gt_col);
  printf("Recall: %f\n", recall);

  return 0;
}

//  fmt::println("Recall: {}", recall);
// for (int q = 0; q < 10; ++q) {
//   printf("res\n");
//   for (int i = 0; i < topk; ++i) {
//     printf("%d, ", res_pool[q].data_[i].id);
//   }
// }
// printf("\n");
// std::cout << "Time: " << timer.getElapsedTime() << std::endl;
//  fmt::println("Search Time: {}", timer.getElapsedTime());

// for (int i = 0; i < 10; ++i) {
//   printf("search res: \n");
//   for (int j = 0; j < topk; ++j) {
//     printf("%d, ", tmp_id[i][j]);
//   }
//   printf("\n");
//   std::set<int> S = {};
//   for (int j = 0; j < topk; ++j) {
//     printf("%d, ", answers[i * kk + j]);
//     S.insert(answers[i * kk + j]);
//   }
//   printf("\n");

//   int hit_cou = 0;
//   for (int j = 0; j < topk; ++j) {
//     if (S.find(tmp_id[i][j]) != S.end()) hit_cou++;
//   }
//   printf("hit count: %d\n", hit_cou);
//   //    for (int j=0; j<topk; ++j) {
//   //        printf("%d, ", bru_id[i][j]);
//   //    }
//   //    printf("\n");
// }
