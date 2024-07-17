#include <fstream>
#include <iostream>

template <typename T>
void read_file(char *data_file, T *r_data, size_t num, size_t dim) {
  std::cout << "num: " << num << ", dim: " << dim << std::endl;
  std::ifstream in(data_file, std::ios::binary);
  // r_data = new T[num * dim];
  in.read((char *)r_data, num * dim * sizeof(T));
  in.close();
}

template <typename T>
void write_file(char *output_file, T *r_data, size_t num, size_t dim) {
  std::cout << "Out file: " << output_file << std::endl;
  std::ofstream out(output_file, std::ios::binary);
  T *w_data = new T[num * (dim + 1)];
  unsigned out_dim = dim;
  std::cout << "num: " << num << ", dim: " << dim << std::endl;
  for (size_t i = 0; i < num; ++i) {
    out.write((char *)&out_dim, sizeof(out_dim));
    out.write((char *)(r_data + i * dim), dim * sizeof(T));
  }
  out.close();
}

int main(int argc, char **argv) {
  char *data_file = argv[1];
  size_t num = atoi(argv[2]);
  size_t dim = atoi(argv[3]);
  char *output_file = argv[4];
  std::string type = argv[5];

  if (type == "float") {
    std::cout << "float" << std::endl;
    float *r_data = new float[num * dim];
    // float *w_data = nullptr;
    read_file(data_file, r_data, num, dim);
    write_file(output_file, r_data, num, dim);
  } else {
    std::cout << "unsigned" << std::endl;
    int64_t *r_data = new int64_t[num * dim];
    read_file(data_file, r_data, num, dim);
    std::cout << "read over" << std::endl;
    unsigned *rr_data = new unsigned[num * dim];
    for (size_t i = 0; i < num; ++i) {
      for (size_t j = 0; j < dim; ++j) {
        rr_data[i * dim + j] = r_data[i * dim + j];
      }
    }
    std::cout << "before write" << std::endl;
    write_file(output_file, rr_data, num, dim);
  }

  return 0;
  // std::ifstream in(data_file, std::ios::binary);

  // float *w_data = new float[num * (dim + 1)];
  // float *r_data = new float[num * dim];

  // in.read((char *)(r_data), num * dim * 4);

  // unsigned out_dim = (unsigned)dim;
  // std::ofstream out(output_file, std::ios::binary);
  // for (size_t i = 0; i < num; ++i) {
  //   out.write((char *)&out_dim, sizeof(out_dim));
  //   out.write((char *)(r_data + i * dim), dim * 4);
  // }
}