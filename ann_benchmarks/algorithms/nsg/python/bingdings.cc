#include "efanna2e/index_graph.h"
#include "efanna2e/index_random.h"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>

#include <omp.h>

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals; // needed to bring in _a literal

inline void get_input_array_shapes(const py::buffer_info &buffer, size_t *rows,
                                   size_t *features) {
  if (buffer.ndim != 2 && buffer.ndim != 1) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "Input vector data wrong shape. Number of dimensions %d. Data "
             "must be a 1D or 2D array.",
             buffer.ndim);
  }
  if (buffer.ndim == 2) {
    *rows = buffer.shape[0];
    *features = buffer.shape[1];
  } else {
    *rows = 1;
    *features = buffer.shape[0];
  }
  
}

void set_num_threads(int num_threads) { omp_set_num_threads(num_threads); }

struct Index {
  std::unique_ptr<efanna2e::IndexGraph> index = nullptr;

  typedef std::vector<std::vector<unsigned > > CompactGraph;

  Index(int dim, efanna2e::Metric metric,
        int R = 32, int L = 200 , int n) {
    
    efanna2e::IndexRandom init_index(dim  , n);

    index = std::make_unique<efanna2e::IndexGraph>(dim, R, efanna2e::L2, (efanna2e::Index*)(&init_index));

  }

  CompactGraph build(py::object input , py::object paras) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);

    py::array_t<unsigned, py::array::c_style | py::array::forcecast> indices(paras);
    efanna2e

    float *vector_data = (float *)items.data(0);
    index->Build(vector_data, rows);
    return Graph(index->GetGraph());
  }
};
