start=50
end=300
step=25 # 更改为更合理的步长

>result.txt

for i in $(seq $start $step $end)
do
 rerank=15
 if [ "$i" -gt 75 ]; then
 rerank=20
 fi
 if [ "$i" -gt 200 ]; then
  rerank=30
 fi

 # 执行 main 命令
 echo "---------------------------"
 echo "Running with HNSW"
 /home/ann-benchmark/yitao/ann-bench/ann-benchmarks/ann_benchmarks/algorithms/alaya/cmake-build-release/main /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_base.fvecs /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_query.fvecs /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_gti.ivecs 10 $i $rerank HNSW
done

for i in $(seq $start $step $end)
do
 rerank=15
 if [ "$i" -gt 75 ]; then
 rerank=20
 fi
 if [ "$i" -gt 200 ]; then
  rerank=30
 fi

 # 执行 main 命令
 echo "---------------------------"
 echo "Running with NSG"
 /home/ann-benchmark/yitao/ann-bench/ann-benchmarks/ann_benchmarks/algorithms/alaya/cmake-build-release/main /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_base.fvecs /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_query.fvecs /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_gti.ivecs 10 $i $rerank NSG
done

for i in $(seq $start $step $end)
do
 rerank=15
 if [ "$i" -gt 75 ]; then
 rerank=20
 fi
 if [ "$i" -gt 200 ]; then
  rerank=30
 fi

 # 执行 main 命令
 echo "---------------------------"
 echo "Running with MERGE"
 /home/ann-benchmark/yitao/ann-bench/ann-benchmarks/ann_benchmarks/algorithms/alaya/cmake-build-release/main /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_base.fvecs /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_query.fvecs /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_gti.ivecs 10 $i $rerank MERGE
done

for i in $(seq $start $step $end)
do
 rerank=15
 if [ "$i" -gt 75 ]; then
 rerank=20
 fi
 if [ "$i" -gt 200 ]; then
  rerank=30
 fi

 # 执行 main 命令
 echo "---------------------------"
 echo "Running with MERGE_CUT"
 /home/ann-benchmark/yitao/ann-bench/ann-benchmarks/ann_benchmarks/algorithms/alaya/cmake-build-release/main /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_base.fvecs /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_query.fvecs /data/dbpedia-openai-100k-euclidean/dbpedia-openai-100k_gti.ivecs 10 $i $rerank MERGE_CUT
done