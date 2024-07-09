#include <cstddef>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>
#include <string.h>
#include <string>

void load_data(char* &filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
    in.seekg(4,std::ios::cur);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
}

std::string get_graph_file(char* filename , std::string file_suffix){
    std::string save_nndescent_graph_file_name(filename);
    size_t last_postion = save_nndescent_graph_file_name.find_last_of('.');
    if(last_postion != std::string::npos){
        save_nndescent_graph_file_name = save_nndescent_graph_file_name.substr(0 , last_postion - 1);
    }

    save_nndescent_graph_file_name += file_suffix ;
    return save_nndescent_graph_file_name;
}

void load_ivecs_data(const char* filename,
                 std::vector<std::vector<unsigned> >& results, unsigned &num, unsigned &dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
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

int main(int argc , char** argv){


  if(argc != 8){std::cout<< argv[0] <<" data_file K L iter S R C"<<std::endl; exit(-1);}
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  std::string graph_filename = get_graph_file(argv[1], "_nndescent.graph");
  std::cout<<"load base data finish\n";
  unsigned K = (unsigned)atoi(argv[2]);
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned iter = (unsigned)atoi(argv[4]);
  unsigned S = (unsigned)atoi(argv[5]);
  unsigned R = (unsigned)atoi(argv[6]);
  unsigned C = (unsigned)atoi(argv[7]);

  //data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));

  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("iter", iter);
  paras.Set<unsigned>("S", S);
  paras.Set<unsigned>("R", R);

  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e-s;
  std::cout <<"Time cost: "<< diff.count() << "\n";

  index.Save(graph_filename.c_str());
  std::cout<<"graph file has save to "<<graph_filename<<"\n";  
  
  efanna2e::IndexNSG nsg_index(dim, points_num, efanna2e::L2, nullptr);

  auto nsg_s = std::chrono::high_resolution_clock::now();
  efanna2e::Parameters nsg_paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<std::string>("nn_graph_path", graph_filename);

  nsg_index.Build(points_num, data_load, paras);
  auto nsg_e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> nsg_diff = nsg_e - nsg_s;

  std::cout << "nsg build indexing time: " << nsg_diff.count() << "\n";
  std::string index_result_filename = get_graph_file(argv[1], ".nsg");
  nsg_index.Save(index_result_filename.c_str());   
   

  std::cout<<"nsg index has save to "<<index_result_filename<<std::endl;

  
}