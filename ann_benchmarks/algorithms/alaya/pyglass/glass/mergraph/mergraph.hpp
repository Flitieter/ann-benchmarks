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
                    if(x==-1)continue;
                    final_graph.at(i,cnt)=x;
                    cnt++;
                }
                if(i<10){
                    std::cout<<"id: "<<i<<"deg: "<<Union.size()<<"\n";
                    for(int j=0;j<2*M;j++)std::cout<<final_graph.at(i,j)<<" ";
                    std::cout<<"\n";
                }
            }
            int hnsw_ep=hnsw->final_graph.initializer->ep;
            bool hnsw_ep_exist=false;
            for(auto x:nsg->final_graph.eps)if(x==hnsw_ep)hnsw_ep_exist=true;
            final_graph.eps=nsg->final_graph.eps;
            if(!hnsw_ep_exist)final_graph.eps.push_back(hnsw_ep);

            std::cout<<"eps: \n";
            for(auto ep:final_graph.eps){
                std::cout<<ep<<" ";
            }
            std::cout<<"\n";
        }




        Graph<int> GetGraph() override { return final_graph; }
    };
}
