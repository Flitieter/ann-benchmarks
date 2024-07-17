//
// Created by ann-benchmark on 7/16/24.
//
#include "glass/builder.hpp"
#include "glass/common.hpp"
#include "glass/graph.hpp"
#include <glass/hnsw/hnsw.hpp>
#include <glass/nsg/nsg.hpp>
#include <set>
namespace glass{
    struct MERGRAPH : public Builder{
        int nb, dim;
        int M, efConstruction;
        Graph<int> final_graph;
        glass::HNSW* hnsw{nullptr};
        glass::NSG* nsg{nullptr};
        MERGRAPH(int dim, const std::string &metric, int R = 32, int L = 200): dim(dim), M(R), efConstruction(L){
            hnsw = new glass::HNSW(dim, metric, M, efConstruction);
            nsg  = new glass::NSG(dim, metric, M, efConstruction);
        }

        void Build(float *data, int N) override {
            nb=N;
            hnsw->Build((float*)data, N);
            nsg->Build((float*)data, N);
            final_graph.init(nb, 2 * M);
            for(int i=0;i<nb;i++){
                std::set<int> Union={};
                for(int j=0;j<M;j++)Union.insert(hnsw->final_graph.at(i,j));
                for(int j=0;j<M;j++)Union.insert(nsg->final_graph.at(i,j));
                int cnt=0;
                for(auto x:Union){
                    final_graph.at(i,cnt)=x;
                    cnt++;
                }
            }
        }




        Graph<int> GetGraph() override { return final_graph; }
    };
}
