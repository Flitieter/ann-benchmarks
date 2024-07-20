
#include <glass/graph.hpp>
#include <cassert>
#include <glass/mergraph/mergraph.hpp>
#include <glass/nsg/nsg.hpp>
#include <glass/io_utils.hpp>
#include <glass/neighbor.hpp>
#include <glass/rerank.hpp>
#include <glass/searcher.hpp>
#include <glass/timer.hpp>
#include <set>
#include <string>

int main(int argc, char **argv) {
  if (argc != 8) {
    //    fmt::println("{}  data_file query_file answer_file result_path", argv[0]);
    exit(-1);
  }

  char *data_file = argv[1];
  char *query_file = argv[2];
  char *ans_file = argv[3];

  int topk = atoi(argv[4]);
  int ef = atoi(argv[5]);
  int rerank_k = atoi(argv[6]);

  std::string graph_type(argv[7]);
    std::cout<<"topK: "<<topk<<",ef:  "<<ef<<",rerank: "<<rerank_k<<",type: "<<graph_type<<"\n";
  float *data_load = NULL;
  unsigned points_num, dim;
  glass::load_fvecs(data_file, data_load, points_num, dim);

  float *query_load = NULL;
  unsigned query_num, query_dim;
  glass::load_fvecs(query_file, query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned *answers = NULL;
  unsigned ans_num, kk;
  glass::load_ivecs(ans_file, answers, ans_num, kk);
  assert(ans_num == query_num);
  assert(graph_type=="HNSW"||graph_type=="NSG"||graph_type=="MERGE"||graph_type=="MERGE_CUT");
  auto index=(graph_type=="HNSW")?
          std::unique_ptr<glass::Builder>((glass::Builder *)new glass::HNSW(dim, "L2"))
          :(graph_type=="NSG")?std::unique_ptr<glass::Builder>((glass::Builder *)new glass::NSG(dim, "L2"))
          :(graph_type=="MERGE")?std::unique_ptr<glass::Builder>((glass::Builder *)new glass::MERGRAPH(dim, "L2",false))
          :std::unique_ptr<glass::Builder>((glass::Builder *)new glass::MERGRAPH(dim, "L2",true));

  index->Build(data_load, points_num);

  // index->GetGraph().save("sift.hsnw");

  // std::cout << "Build HNSW Over" << std::endl;

  const std::string metric = "L2";
  int level = 3;
  auto searcher = std::unique_ptr<glass::SearcherBase>(glass::create_searcher(index->GetGraph(), metric, level));
  searcher->SetEf(ef);

  searcher->SetData(data_load, points_num, dim);

  searcher->Optimize(96);

  std::cout << "create searcher" << std::endl;

  // std::vector<maxPQIFCS<float>> candidate_pool(query_num, maxPQIFCS<float>(topk));
  std::vector<std::vector<int>> tmp_id(query_num, std::vector<int>(ef));
  std::vector<glass::searcher::LPool<float>> res_pool(query_num, glass::searcher::LPool<float>(rerank_k));

  Timer<std::chrono::milliseconds> timer;

  timer.reset();
  // #pragma omp parallel for schedule(dynamic)
  for (size_t q = 0; q < query_num; ++q) {
//    printf("%d\n", q);
    auto &ids = tmp_id[q];
    auto cur_query = query_load + q * query_dim;
    searcher->Search(cur_query, ef, ids.data());

    for (size_t i = 0; i < rerank_k; ++i) {
      res_pool[q].insert_back(ids[i], glass::L2SqrRef(cur_query, data_load + ids[i] * dim, dim));
    }
    res_pool[q].sorted();
    for (size_t i = rerank_k; i < ef; ++i) {
      res_pool[q].emplace_insert(ids[i], glass::L2SqrRef(cur_query, data_load + ids[i] * dim, dim));
    }
  }
  timer.end();

//  for (int q = 0; q < 10; ++q) {
//    printf("res\n");
//    for (int i = 0; i < topk; ++i) {
//      printf("%d, ", res_pool[q].data_[i].id);
//    }
//  }
//  printf("\n");
   std::cout << "Time: " << timer.getElapsedTime() << std::endl;
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

  float recall = glass::CalRecallById(res_pool, topk, answers, kk);
  printf("Recall: %f\n", recall);

  //  fmt::println("Recall: {}", recall);

  return 0;
}