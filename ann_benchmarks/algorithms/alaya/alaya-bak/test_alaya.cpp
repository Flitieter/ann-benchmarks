#include <iostream>
#include <cstring>
#include <set>
#include "baseline.cpp"

void load_fvecs(char *filename, float *&data, unsigned &num,
                unsigned &dim){ // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
}

void load_ivecs(char *filename, unsigned *&data, unsigned &num,
                unsigned &dim) { // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
    // std::cout<<"data dimension: "<<dim<<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);

    data = new unsigned[num * dim * sizeof(int)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
}

double compute_recall(unsigned *answers, unsigned *results,
                      unsigned query_num, unsigned k, unsigned dim) {

    double sum_recall = 0.0;

    for (unsigned query_idx = 0; query_idx < query_num; query_idx++) {
        std::unordered_set<int> res_set;
        for (unsigned i = 0; i < i; i++) {
            res_set.insert(results[query_idx * k + i]);
        }

        int true_pos = 0;
        for (unsigned i = 0; i < query_num; i++) {
            unsigned retuned_idx = answers[query_idx * k + i];
            if (res_set.find(retuned_idx) != res_set.end()) {
                true_pos += 1;
            }
        }

        sum_recall += static_cast<double>(true_pos) / k;
    }

    return sum_recall / query_num;
}

void save_result(char *filename, double time, double recall,
                 unsigned *results, unsigned query_num, unsigned k) {
    std::ofstream out(filename, std::ios::out);

    out << time << std::endl
        << recall << std::endl;

    for (unsigned i = 0; i < query_num; i++) {
        for (unsigned j = 0; j < k; j++) {
            out << results[i * k + j];
            if (j != k - 1) {
                out << ",";
            } else {
                out << std::endl;
            }
        }
    }
    out.close();
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << argv[0]
                  << " data_file query_file answer_file result_path"
                  << std::endl;
        exit(-1);
    }
    char *data_file = argv[1];
    char *query_file = argv[2];
    char *ans_file = argv[3];
    char *result_path = argv[4];

    float *data_load = NULL;
    unsigned points_num, dim;
    load_fvecs(data_file, data_load, points_num, dim);

    float *query_load = NULL;
    unsigned query_num, query_dim;
    load_fvecs(query_file, query_load, query_num, query_dim);
    assert(dim == query_dim);

    unsigned *answers = NULL;
    unsigned ans_num, k;
    load_ivecs(ans_file, answers, ans_num, k);
    assert(ans_num == query_num);

    // fit
    fit(data_load, query_num, dim);

    auto s = std::chrono::high_resolution_clock::now();
    unsigned results[query_num * k];
    for (unsigned i = 0; i < query_num; i++) {
        // query
        unsigned *ans = results + i * k;
        batch_query(query_load, 1, dim, k, ans);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "search time: " << diff.count() << "\n";       // in sec

    double recall = compute_recall(answers, results, query_num, k, dim);

    save_result(result_path, diff.count(), recall, results, query_num, dim);

    return 0;
}