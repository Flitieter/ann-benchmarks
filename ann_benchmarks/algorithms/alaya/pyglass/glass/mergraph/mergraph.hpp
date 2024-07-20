//
// Created by ann-benchmark on 7/16/24.
//
#include "glass/builder.hpp"
#include "glass/common.hpp"
#include "glass/graph.hpp"
#include <glass/hnsw/hnsw.hpp>
#include <glass/nsg/nsg.hpp>
#include <set>
#include <queue>
#include <glass/hnswlib/hnswalg.h>
namespace glass{
    struct MERGRAPH : public Builder{
        int nb, dim;
        int M, efConstruction;
        Graph<int> final_graph;
        glass::HNSW* hnsw{nullptr};
        glass::NSG* nsg{nullptr};
        bool Cut;
        MERGRAPH(int dim, const std::string &metric, bool cut,int R = 32, int L = 200): dim(dim), M(R), efConstruction(L),Cut(cut){
            hnsw = new glass::HNSW(dim, metric, M, efConstruction);
            nsg  = new glass::NSG(dim, metric, M, efConstruction);
        }

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int>>,
                        std::greater<>> &top_candidates,
                const size_t M,float *data) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<float, int>> queue_closest;
            std::vector<std::pair< float, int>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first,
                                      top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<float, int> curent_pair = queue_closest.top();
                float dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<float, int> second_pair : return_list) {
//                    float curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
//                                                  getDataByInternalId(curent_pair.second),
//                                                  dist_func_param_);
                    float curdist=glass::L2SqrRef(data+second_pair.second*dim, data+curent_pair.second * dim, dim);
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<float, int> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        void Build(float *data, int N) override {
            nb=N;
            hnsw->Build((float*)data, N);
            nsg->Build((float*)data, N);
            final_graph.init(nb, 2 * M);
            std::vector<std::set<int> > Union(nb);
            std::vector<std::priority_queue<std::pair<float,int>,std::vector<std::pair<float,int>>, std::greater<>> > PL(nb);
            if(Cut)std::cout<<"!!!Execute getNeighborsByHeuristic!!!\n";
            #pragma omp parallel for schedule(dynamic)
            for(int i=0;i<nb;i++){

                for(int j=0;j<M;j++)Union[i].insert(hnsw->final_graph.at(i,j));
                for(int j=0;j<M;j++)Union[i].insert(nsg->final_graph.at(i,j));
                int cnt=0;
                if(!Cut){
                    for(auto x:Union[i]){
                        if(x==-1)continue;
                        final_graph.at(i,cnt)=x;
                        cnt++;
                    }
                }
                else{

                    for(auto x:Union[i]){
                        if(x==-1)continue;
                        PL[i].emplace(glass::L2SqrRef(data+i*dim,data+x*dim,dim),x);
                    }
                    getNeighborsByHeuristic2(PL[i],M,data);
                    while(!PL[i].empty()){
                        final_graph.at(i,cnt)=PL[i].top().second;
                        PL[i].pop();
                        cnt++;
                    }
                }
//                if(i<10){
//                    std::cout<<"id: "<<i<<"deg: "<<Union.size()<<"\n";
//                    for(int j=0;j<2*M;j++)std::cout<<final_graph.at(i,j)<<" ";
//                    std::cout<<"\n";
//                }
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
