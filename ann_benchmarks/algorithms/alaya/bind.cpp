// #include <omp.h>
// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

#include <glass/common.hpp>
#include <glass/graph.hpp>
#include <glass/hnsw/hnsw.hpp>
#include <glass/neighbor.hpp>
#include <glass/quant/quant.hpp>
#include <glass/searcher.hpp>
#include <glass/mergraph/mergraph.hpp>

class Array1DViewInt64 {
 public:
  void* data;

 public:
  Array1DViewInt64(void* data) : data(data) {}

  int64_t& operator[](int i) { return *(int64_t*)((char*)data + i * sizeof(int64_t)); }
};

class Array2DViewFloat {
 public:
  void* data;
  long shape[2];
  long strides[2];

 public:
  Array2DViewFloat() : data(nullptr) {}

  void init(void* data, long* shape, long* strides) {
    this->data = data;
    ::memcpy(&this->shape[0], shape, sizeof(this->shape));
    ::memcpy(&this->strides[0], strides, sizeof(this->strides));
  }

  float& operator()(int i, int j) { return *(float*)((char*)data + i * strides[0] + j * strides[1]); }
};

class Alaya {
 private:
  glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>>* searcher{nullptr};
//  glass::HNSW* hnsw{nullptr};
  glass::MERGRAPH* mergraph{nullptr};
  Array2DViewFloat data;

 public:
  Alaya() = default;
  Alaya(const Alaya&) = delete;
  Alaya& operator=(const Alaya&) = delete;
  const Alaya& operator=(Alaya&&) = delete;
  ~Alaya() {
    delete searcher;
//    delete hnsw;
    delete mergraph;
  }

  void fit(void* data, long* shape, long* strides, int M) {
      mergraph = new glass::MERGRAPH(shape[1], "L2", M);
    assert(strides[1] == sizeof(float));
    assert(strides[0] == shape[1] * strides[1]);
    mergraph->Build((float*)data, shape[0]);
    searcher = new glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>>(mergraph->GetGraph());
    searcher->SetData((float*)data, shape[0], shape[1]);
    // int num_threads = omp_get_num_threads();
    searcher->Optimize(1);
    this->data.init(data, shape, strides);
  }

  void query(void* query_data, long* query_shape, long* query_stirdes, int k, int ef, int rerank_k, void* res) {
    assert(query_shape[0] == 1);
    assert(query_shape[1] == data.shape[1]);
    assert(query_stirdes[1] == sizeof(float));
    assert(query_stirdes[0] == query_shape[1] * query_stirdes[1]);
    int tmp_id[ef];
    glass::searcher::LPool<float> res_pool(k);
    searcher->SetEf(ef);
    searcher->Search((float*)query_data, k, tmp_id);
    for (size_t i = 0; i < k; ++i) {
      res_pool.insert_back(tmp_id[i], glass::L2SqrRef((float*)query_data, &this->data(tmp_id[i], 0), query_shape[1]));
    }
    res_pool.sorted();
    for (size_t i = k; i < rerank_k; ++i) {
      res_pool.emplace_insert(tmp_id[i],
                              glass::L2SqrRef((float*)query_data, &this->data(tmp_id[i], 0), query_shape[1]));
    }
    auto res_view = Array1DViewInt64(res);
    for (size_t i = 0; i < k; ++i) {
      res_view[i] = res_pool.data_[i].id;
    }
  }
};

extern "C" {

void* init_alaya() { return new Alaya(); }

void del_alaya(void* self) { delete (Alaya*)self; }

void fit(void* self, void* data, long* shape, long* strides, int M) { ((Alaya*)self)->fit(data, shape, strides, M); }

void query(void* self, void* query_data, long* query_shape, long* query_stirdes, int k, int ef, int rerank_k, void* res) {
  ((Alaya*)self)->query(query_data, query_shape, query_stirdes, k, ef, rerank_k, res);
}

// void* searcher;
// // float *data;
// // std::vector<float> Data;
// void fit(float* vec_data, int data_num, int data_dim, int M) {
//   printf("fit, num:%d, dim:%d, M:%d\n", data_num, data_dim, M);
//   // data = vec_data;
//   auto hnsw = (glass::HNSW*)new glass::HNSW(data_dim, "L2", M);
//   hnsw->Build(vec_data, data_num);
//   printf("Build Hnsw over\n");
//   // Data.reserve(data_num * data_dim);
//   // for (int i = 0; i < data_num * data_dim; i++) Data.emplace_back(vec_data[i]);
//   printf("data push back over\n");

//   searcher =
//       (glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>>*)(new glass::Searcher<
//                                                                           glass::SQ8SymmetricQuantizer<
//                                                                               glass::Metric::L2>>(hnsw->GetGraph()));

//   auto hnsw_s = (glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>>*)searcher;

//   hnsw_s->SetData(vec_data, data_num, data_dim);

//   int num_threads = omp_get_num_threads();
//   printf("num_threads: %d\n", num_threads);
//   hnsw_s->Optimize(num_threads);
// }

// void query(float* query, float* data, int query_dim, int k, int ef, int rerank_k, unsigned* res) {
//   // printf("query\n");
//   // std::vector<int> tmp_id(ef);
//   // int *tmp_id = new int[ef];
//   int tmp_id[ef];
//   glass::searcher::LPool<float> res_pool(rerank_k);
//   // data = Data.data();
//   auto hnsw_s = (glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>>*)searcher;
//   hnsw_s->SetEf(ef);
//   // printf("set ef\n");

//   printf("query: ");
//   for (int i = 0; i < query_dim; i++) {
//     printf("i: %d, %f ", i, query[i]);
//   }
//   printf("\n");

//   // printf("data\n");
//   // for (int i = 0; i < query_dim; i++) {
//   //   printf("i: %d, %f ", i, data[i]);
//   // }
//   // printf("\n");

//   hnsw_s->Search(query, k, tmp_id);

//   printf("before rerank\n");
//   for (size_t i = 0; i < k; ++i) {
//     printf("%d ", tmp_id[i]);
//   }
//   printf("\n");

//   // printf("search over\n");
//   for (size_t i = 0; i < rerank_k; ++i) {
//     printf("i%d, data\n", i);
//     for (auto d = 0; d < query_dim; d++) {
//       printf("d: %d, %f ", d, data[tmp_id[i] * query_dim + d]);
//     }
//     printf("\n");
//     res_pool.insert_back(tmp_id[i], glass::L2SqrRef(query, data + tmp_id[i] * query_dim, query_dim));
//     printf("rerank dist: %f\n", glass::L2SqrRef(query, data + tmp_id[i] * query_dim, query_dim));
//   }
//   res_pool.sorted();
//   for (size_t i = rerank_k; i < ef; ++i) {
//     res_pool.emplace_insert(tmp_id[i], glass::L2SqrRef(query, data + tmp_id[i] * query_dim, query_dim));
//   }
//   printf("res: \n");
//   for (size_t i = 0; i < k; ++i) {
//     res[i] = res_pool.data_[i].id;
//     printf("%d ", res[i]);
//   }
//   printf("\n");
// }

// void batch_query(float* query, float* data, int query_num, int dim, int k, int ef, int rerank_k, unsigned* res) {
//   // printf("init search\n");
//   std::vector<std::vector<int>> tmp_id(query_num, std::vector<int>(ef));
//   std::vector<glass::searcher::LPool<float>> res_pool(query_num, glass::searcher::LPool<float>(rerank_k));
//   // data = Data.data();
//   // printf("before search\n");

//   auto hnsw_s = (glass::Searcher<glass::SQ8SymmetricQuantizer<glass::Metric::L2>>*)searcher;
// #pragma omp parallel for schedule(dynamic)
//   for (size_t q = 0; q < query_num; ++q) {
//     //     printf("query id: %d\n", q);
//     auto& ids = tmp_id[q];
//     auto cur_query = query + q * dim;
//     hnsw_s->Search(cur_query, ef, ids.data());

//     for (size_t i = 0; i < k; ++i) {
//       res[q * k + i] = ids[i];
//     }

//     //  printf("a search\n");

//     //     printf("rerank: %d, k: %d, ef: %d\n", rerank_k, k, ef);
//     for (size_t i = 0; i < rerank_k; ++i) {
//       res_pool[q].insert_back(ids[i], glass::L2SqrRef(cur_query, data + ids[i] * dim, dim));
//     }
//     res_pool[q].sorted();
//     for (size_t i = rerank_k; i < ef; ++i) {
//       res_pool[q].emplace_insert(ids[i], glass::L2SqrRef(cur_query, data + ids[i] * dim, dim));
//     }

//     for (size_t i = 0; i < k; ++i) {
//       res[q * k + i] = res_pool[q].data_[i].id;
//     }
//   }
// }

}  // extern C