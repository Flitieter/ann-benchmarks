#include <glass/common.hpp>
#include <glass/graph.hpp>
#include <glass/hnsw/hnsw.hpp>
#include <glass/neighbor.hpp>
#include <glass/quant/quant.hpp>
#include <glass/searcher.hpp>

extern "C" {

void *searcher;
float *data;

void fit(float *vec_data, int data_num, int data_dim) {
  data = vec_data;
  auto hnsw = (glass::HNSW *)new glass::HNSW(data_dim, "L2");
  hnsw->Build(data, data_num);

  // const std::string metric = "L2";
  // int level = 3;

  searcher = (glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>>
                  *)(new glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>>(hnsw->GetGraph()));

  auto hnsw_s = (glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>> *)searcher;

  // hnsw_s->SetEf(ef);

  hnsw_s->SetData(data, data_num, data_dim);

  hnsw_s->Optimize(1);
}

void batch_query(float *query, int query_num, int dim, int k, int ef, int rerank_k, unsigned *res) {
  printf("init search\n");
  std::vector<std::vector<int>> tmp_id(query_num, std::vector<int>(ef));
  std::vector<glass::searcher::LPool<float>> res_pool(query_num, glass::searcher::LPool<float>(rerank_k));

  printf("before search\n");

  auto hnsw_s = (glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>> *)searcher;
  hnsw_s->SetEf(ef);
  for (size_t q = 0; q < query_num; ++q) {
    // printf("query id: %d\n", q);
    auto &ids = tmp_id[q];
    auto cur_query = query + q * dim;
    hnsw_s->Search(cur_query, ef, ids.data());

    for (size_t i = 0; i < k; ++i) {
      res[q * k + i] = ids[i];
    }

    // printf("a search");

    // printf("rerank: %d, k: %d, ef: %d\n", rerank_k, k, ef);
    // for (size_t i = 0; i < rerank_k; ++i) {
    //   res_pool[q].insert_back(ids[i], glass::L2SqrRef(cur_query, data + ids[i] * dim, dim));
    // }
    // res_pool[q].sorted();
    // for (size_t i = rerank_k; i < ef; ++i) {
    //   res_pool[q].emplace_insert(ids[i], glass::L2SqrRef(cur_query, data + ids[i] * dim, dim));
    // }

    // printf("res\n");
    // for (size_t i = 0; i < k; ++i) {
    //   res[q * k + i] = res_pool[q].data_[i].id;
    //   printf("%d, ", res[q * k + i]);
    // }
    // printf("\n");
  }
}

}  // extern C