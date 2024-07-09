#include <iostream>
#include <fstream>
#include <vector>

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

int main(int argc, char** argv) {
  std::vector<std::vector<unsigned> > true_load;
  unsigned dim, num;
  load_ivecs_data(argv[1], true_load, num, dim);
  for(size_t i = 0; i < num; i++) {
    for(size_t j = 0; j < dim; j++) {
      std::cout << true_load[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "result_num："<< num << std::endl << "result dimension：" << dim << std::endl;
  return 0;
}