# Benchmarking Image Search Runtimes

We test different methods of similarity searching to determine the runtime performance of different methods

* Cosine Similarity
* Euclidean Distance

We test on different sized vector lists
* 1,000
* 100,000
* 1,000,000

Additionally we show the relationship between projection dimension and space complexity

Tests are performed with torch 1.13 and cuda 1.17 on NVIDIA Tesla V100 32GB