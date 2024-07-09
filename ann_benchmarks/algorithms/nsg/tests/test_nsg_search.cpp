//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <omp.h>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}


void load_ivecs_data(const char* filename,
                 std::vector<std::vector<unsigned> >& results, unsigned &num, unsigned &dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "ivecs open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  //std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  results.resize(num);
  for (unsigned i = 0; i < num; i++) results[i].resize(dim);

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)results[i].data(), dim * 4);
  }
  in.close();
}

void eval_recall(std::vector<std::vector<unsigned> >& query_set, std::vector<std::vector<unsigned> > &acc_eval_set){
  float mean_acc=0;

  for(unsigned i=0; i<query_set.size(); i++){
    float acc = 0;
    auto &g = query_set[i];
    auto &v = acc_eval_set[i];
    int top_k = g.size();
    for(unsigned j = 0; j < top_k; j ++){
      for(unsigned k = 0; k < top_k; k++){
        if(g[j] == v[k]){
          acc++;
          break;
        }
      }
    }
    mean_acc += acc / top_k;
  }
  std::cout<<"recall : "<<mean_acc / query_set.size() <<std::endl;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path grouth_file"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
  index.Load(argv[3]);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);
  std::cout << "query num: " << query_num << "\n";

  auto s = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<unsigned> > res(query_num, std::vector<unsigned>(K));
  // res.resize(query_num);

  // omp_set_num_threads(32);
  
  #pragma omp parallel for
  for (unsigned i = 0; i < query_num; i++) {
    
    std::vector<unsigned> tmp(K);
    index.Search(query_load + i * dim, data_load, K, paras, tmp.data());
    
    // #pragma omp critical
    res[i] = tmp;
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "search time: " << diff.count() << "s \n";

  std::vector<std::vector<unsigned> > true_load;
  unsigned G_dim, G_num;
  load_ivecs_data(argv[7], true_load, G_num, G_dim);  
  
  eval_recall(res , true_load);

  std::cout<<"qps: "<< query_num * 1.0 / diff.count()<<"\n";
  save_result(argv[6], res);

  return 0;
}
//./tests/test_nndescent ~/dataset/sift/sift_base.fvecs sift_200nn.graph 200 200 10 10 100
//./tests/test_nsg_index ~/dataset/sift/sift_base.fvecs sift_200nn.graph 40 50 500 sift.nsg  
//./tests/test_nsg_search ~/dataset/sift/sift_base.fvecs  ~/dataset/sift/sift_query.fvecs  sift.nsg 40 10 result.nsg  ~/dataset/sift/sift_groundtruth.ivecs