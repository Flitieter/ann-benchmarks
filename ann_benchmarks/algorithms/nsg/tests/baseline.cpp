#include <cstddef>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>
#include <memory>
#include <string.h>
#include <string>

const char* SAVE_KNN_GRAPH_FILANAME = "save_knn_graph.graph";

unsigned KNN_K; //is the 'K' of kNN graph.
unsigned KNN_L; //is the parameter controlling the graph quality, larger is more accurate but slower, no smaller than K.
unsigned KNN_iter; //is the parameter controlling the iteration times, iter usually < 30.
unsigned KNN_S; //is the parameter contollling the graph quality, larger is more accurate but slower.
unsigned KNN_R; //is the parameter controlling the graph quality, larger is more accurate but slower.
unsigned NSG_L; //controls the quality of the NSG, the larger the better.
unsigned NSG_R; //controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
unsigned NSG_C; //controls the maximum candidate pool size during NSG contruction.
unsigned SEARCH_L; //controls the quality of the search results, the larger the better but slower. The SEARCH_L cannot be samller than the SEARCH_K
unsigned SEARCH_K; //controls the number of result neighbors we want to query.
std::unique_ptr<efanna2e::IndexNSG> nsg_index = nullptr;
unsigned points_num, dim;
float* data = NULL;

void init_KNN_parameters(unsigned k , unsigned l , unsigned iter , unsigned s , unsigned r)
{
    KNN_K = k;
    KNN_L = l;
    KNN_iter = iter;
    KNN_S = s;
    KNN_R = r;
}

void init_NSG_parameters(unsigned l , unsigned r , unsigned c)
{
    NSG_L = l;
    NSG_R = r;
    NSG_C = c;
}


void fit(float *Data, int rows, int cols)
{
    points_num = rows ;
    dim = cols;
    data = new float[rows * 1ll * cols * sizeof(float)];
    memcpy(data, Data, rows * 1ll * cols * sizeof(float));


    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", KNN_K);
    paras.Set<unsigned>("L", KNN_L);
    paras.Set<unsigned>("iter", KNN_iter);
    paras.Set<unsigned>("S", KNN_S);
    paras.Set<unsigned>("R", KNN_R);

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(points_num, Data, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e-s;
    std::cout <<"Time cost: "<< diff.count() << "\n";   

    index.Save(SAVE_KNN_GRAPH_FILANAME);
    std::cout<<"Knn graph file has save to "<<SAVE_KNN_GRAPH_FILANAME<<"\n";  

    nsg_index = std::make_unique<efanna2e::IndexNSG>(dim, points_num, efanna2e::L2, nullptr);
    
    efanna2e::Parameters nsg_paras;
    paras.Set<unsigned>("L", NSG_L);
    paras.Set<unsigned>("R", NSG_R);
    paras.Set<unsigned>("C", NSG_C);
    paras.Set<std::string>("nn_graph_path", SAVE_KNN_GRAPH_FILANAME);

    auto nsg_s = std::chrono::high_resolution_clock::now();
    nsg_index->Build(points_num, Data, paras);
    auto nsg_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> nsg_diff = nsg_e - nsg_s;
    std::cout << "nsg build indexing time: " << nsg_diff.count() << "\n";

}


void query(float* query_data , unsigned l , unsigned k , unsigned* result){
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", l);
    paras.Set<unsigned>("P_search", l);

    // std::vector<unsigned> tmp(k);
    nsg_index->Search(query_data, data, k, paras, result);  
}


int main()
{
    return 0;
}