#pragma once

#include <chrono>
#include <memory>

#include "glass/builder.hpp"
#include "glass/common.hpp"
#include "glass/graph.hpp"
#include "glass/hnsw/HNSWInitializer.hpp"
#include "glass/hnswlib/hnswalg.h"
#include "glass/hnswlib/hnswlib.h"
#include "glass/hnswlib/space_ip.h"
#include "glass/hnswlib/space_l2.h"

// #include "fmt/core.h"

namespace glass {

struct HNSW : public Builder {
  int nb, dim;
  int M, efConstruction;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space = nullptr;

  Graph<int> final_graph;

  HNSW(int dim, const std::string &metric, int R = 32, int L = 200) : dim(dim), M(R / 2), efConstruction(L) {
    auto m = metric_map[metric];
    if (m == Metric::L2) {
      space = std::make_unique<hnswlib::L2Space>(dim);
    } else if (m == Metric::IP) {
      space = std::make_unique<hnswlib::InnerProductSpace>(dim);
    } else {
      printf("Unsupported metric type\n");
    }
  }

  void Build(float *data, int N) override {
    // fmt::println("in build");
    nb = N;
    hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), N, M, efConstruction);
    std::atomic<int> cnt{0};
    // fmt::println("make space");
    auto st = std::chrono::high_resolution_clock::now();
    hnsw->addPoint(data, 0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < nb; ++i) {
      hnsw->addPoint(data + i * dim, i);
      int cur = cnt += 1;
      if (cur % 10000 == 0) {
        printf("HNSW building progress: [%d/%d]\n", cur, nb);
      }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    printf("HNSW building cost: %.2lfs\n", ela);
    final_graph.init(nb, 2 * M);
#pragma omp parallel for
    for (int i = 0; i < nb; ++i) {
      auto internal_id = hnsw->label_lookup_[i];
      int *edges = (int *)hnsw->get_linklist0(internal_id);
      for (int j = 1; j <= edges[0]; ++j) {
        int external_id = hnsw->getExternalLabel(edges[j]);
        final_graph.at(i, j - 1) = external_id;
        // final_graph.at_time(i, j - 1) = timestamps[external_id];
      }
      // NOT in use! sort the edges by timestamp, to find valid position of edges while in-filtering
      // std::sort(ext_ids.begin(), ext_ids.end(), [&](int a, int b) {
      //   return timestamps[a] < timestamps[b];
      // });
    }
    auto initializer = std::make_unique<HNSWInitializer>(nb, M);
    initializer->ep = hnsw->getExternalLabel(hnsw->enterpoint_node_);
#pragma omp parallel for
    for (int i = 0; i < nb; ++i) {
      auto internal_id = hnsw->label_lookup_[i];
      int level = hnsw->element_levels_[internal_id];
      initializer->levels[i] = level;
      if (level > 0) {
        initializer->lists[i].assign(level * M, -1);
        for (int j = 1; j <= level; ++j) {
          int *edges = (int *)hnsw->get_linklist(internal_id, j);
          for (int k = 1; k <= edges[0]; ++k) {
            initializer->at(j, i, k - 1) = hnsw->getExternalLabel(edges[k]);
          }
        }
      }
    }
    final_graph.initializer = std::move(initializer);
  }

  Graph<int> GetGraph() override { return final_graph; }
};
}  // namespace glass