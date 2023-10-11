# Backend: transformers

For [Interactive visualization of the results](https://raw.githubusercontent.com/h2oai/h2ogpt/blob/main/benchmarks/llm_gpu_benchmark_transformers.html), save the linked file as html on your machine and open it in a browser.


## Model: h2oai/h2ogpt-4096-llama2-7b-chat (transformers)
### Number of GPUs: 0
|   bits | gpus   |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-------|---------------------------:|--------------------------------:|:------------|
|     16 | CPU    |                    1215.52 |                         1.17546 |             |
|      8 | CPU    |                    1216.98 |                         1.17641 |             |
|      4 | CPU    |                    1217.17 |                         1.16575 |             |
### Number of GPUs: 1
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    31.8619 |                        41.9433  |             |
|     16 | 1 x NVIDIA GeForce RTX 4090 (24564 MiB)        |                    32.2947 |                        40.9252  |             |
|     16 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    37.1139 |                        32.4529  |             |
|     16 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                    47.0375 |                        29.8526  |             |
|     16 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    67.9752 |                        18.0571  |             |
|      8 | 1 x NVIDIA GeForce RTX 4090 (24564 MiB)        |                   114.622  |                         9.21246 |             |
|      8 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    94.1774 |                         8.95532 |             |
|      8 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                   181.246  |                         7.47991 |             |
|      8 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                   148.616  |                         6.61984 |             |
|      8 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                   185.146  |                         4.35807 |             |
|      4 | 1 x NVIDIA GeForce RTX 4090 (24564 MiB)        |                    39.544  |                        32.571   |             |
|      4 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    42.8067 |                        32.3408  |             |
|      4 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    53.3973 |                        23.3267  |             |
|      4 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                    61.5241 |                        22.8456  |             |
|      4 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    90.5194 |                        14.9456  |             |
### Number of GPUs: 2
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    32.1395 |                        40.3871  |             |
|     16 | 2 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    39.9269 |                        32.248   |             |
|     16 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                    47.4105 |                        28.8472  |             |
|     16 | 2 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    71.4808 |                        17.7518  |             |
|      8 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    94.9813 |                         9.03765 |             |
|      8 | 2 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                   178.2    |                         7.55443 |             |
|      8 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                   152.544  |                         6.43862 |             |
|      8 | 2 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                   186.884  |                         4.35012 |             |
|      4 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    43.235  |                        32.0566  |             |
|      4 | 2 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    57.0808 |                        22.6791  |             |
|      4 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                    64.6442 |                        21.972   |             |
|      4 | 2 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    94.5099 |                        14.6162  |             |
### Number of GPUs: 4
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    42.3398 |                        30.2181  |             |
|     16 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                    49.089  |                        27.7344  |             |
|      8 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                   180.534  |                         7.53804 |             |
|      8 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                   153.411  |                         6.46469 |             |
|      4 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    58.6287 |                        21.9123  |             |
|      4 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                    66.4926 |                        21.409   |             |
### Number of GPUs: 8
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    40.4986 |                        30.5489  |             |
|      8 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                   186.713  |                         7.23498 |             |
|      4 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    60.1828 |                        21.9172  |             |
## Model: h2oai/h2ogpt-4096-llama2-13b-chat (transformers)
### Number of GPUs: 1
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    52.4984 |                        26.2487  |             |
|     16 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    49.7972 |                        24.9301  |             |
|     16 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                    71.9114 |                        18.4362  |             |
|     16 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                   nan      |                       nan       | OOM         |
|     16 | 1 x NVIDIA GeForce RTX 4090 (24564 MiB)        |                   nan      |                       nan       | OOM         |
|      8 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                   168.967  |                         7.67522 |             |
|      8 | 1 x NVIDIA GeForce RTX 4090 (24564 MiB)        |                   185.442  |                         6.0205  |             |
|      8 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                   174.458  |                         5.69269 |             |
|      8 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                   193.993  |                         5.56359 |             |
|      8 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                   280.467  |                         3.75936 |             |
|      4 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    45.3051 |                        20.4771  |             |
|      4 | 1 x NVIDIA GeForce RTX 4090 (24564 MiB)        |                    68.0646 |                        16.1241  |             |
|      4 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                    81.1389 |                        15.6933  |             |
|      4 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    74.271  |                        15.0868  |             |
|      4 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    96.6189 |                         9.77255 |             |
### Number of GPUs: 2
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    51.6428 |                        26.1842  |             |
|     16 | 2 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    51.299  |                        24.8757  |             |
|     16 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                    72.8565 |                        18.2039  |             |
|     16 | 2 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    89.5996 |                        12.8295  |             |
|      8 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                   167.523  |                         7.82793 |             |
|      8 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                   195.929  |                         5.51238 |             |
|      8 | 2 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                   180.781  |                         5.43787 |             |
|      8 | 2 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                   280.831  |                         3.72157 |             |
|      4 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    47.1425 |                        19.9791  |             |
|      4 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                    84.5776 |                        15.1326  |             |
|      4 | 2 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    79.9461 |                        14.3455  |             |
|      4 | 2 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    98.4705 |                         9.68779 |             |
### Number of GPUs: 4
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    55.3779 |                        21.7073  |             |
|     16 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                    74.4377 |                        17.8537  |             |
|      8 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                   179.505  |                         5.45185 |             |
|      8 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                   199.799  |                         5.39725 |             |
|      4 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                    87.6579 |                        14.6779  |             |
|      4 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    78.9061 |                        14.6754  |             |
### Number of GPUs: 8
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    55.3965 |                        22.302   |             |
|      8 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                   185.328  |                         5.38647 |             |
|      4 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    83.0479 |                        13.969   |             |
## Model: h2oai/h2ogpt-4096-llama2-70b-chat (transformers)
### Number of GPUs: 1
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    nan     |                       nan       | OOM         |
|     16 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    nan     |                       nan       | OOM         |
|     16 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    nan     |                       nan       | OOM         |
|     16 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                    nan     |                       nan       | OOM         |
|      8 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    nan     |                       nan       | OOM         |
|      8 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    nan     |                       nan       | OOM         |
|      8 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                    nan     |                       nan       | OOM         |
|      4 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    122.132 |                        10.6495  |             |
|      4 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                    165.058 |                         6.94248 |             |
|      4 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    nan     |                       nan       | OOM         |
### Number of GPUs: 2
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                    nan     |                       nan       | OOM         |
|      8 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                    410.069 |                         2.25687 |             |
|      4 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    120.538 |                        10.5008  |             |
|      4 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                    171.744 |                         6.71342 |             |
|      4 | 2 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    nan     |                       nan       | OOM         |
### Number of GPUs: 4
|   bits | gpus                             |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:---------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 4 x NVIDIA RTX A6000 (46068 MiB) |                    267.056 |                         4.24242 |             |
|      8 | 4 x NVIDIA RTX A6000 (46068 MiB) |                    413.957 |                         2.22551 |             |
|      4 | 4 x NVIDIA RTX A6000 (46068 MiB) |                    175.491 |                         6.5798  |             |
# Backend: text-generation-inference

For [Interactive visualization of the results](https://raw.githubusercontent.com/h2oai/h2ogpt/blob/main/benchmarks/llm_gpu_benchmark_text-generation-inference.html), save the linked file as html on your machine and open it in a browser.


## Model: h2oai/h2ogpt-4096-llama2-7b-chat (text-generation-inference)
### Number of GPUs: 1
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    39.0155 |                         55.2139 |             |
|     16 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    29.129  |                         45.9535 |             |
|     16 | 1 x NVIDIA GeForce RTX 4090 (24564 MiB)        |                    24.3988 |                         44.5878 |             |
|     16 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    39.2697 |                         30.3068 |             |
|     16 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                    40.3622 |                         29.9724 |             |
### Number of GPUs: 2
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    7.63612 |                         71.7881 |             |
|     16 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                   41.0461  |                         30.3726 |             |
|     16 | 2 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                   41.0245  |                         29.36   |             |
### Number of GPUs: 4
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                    42.8377 |                         29.388  |             |
|     16 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    41.0995 |                         28.4403 |             |
### Number of GPUs: 8
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    42.8594 |                         27.8644 |             |
## Model: h2oai/h2ogpt-4096-llama2-13b-chat (text-generation-inference)
### Number of GPUs: 1
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 1 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    21.7823 |                         33.7132 |             |
|     16 | 1 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    51.8428 |                         19.083  |             |
|     16 | 1 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                   nan      |                        nan      | OOM         |
|     16 | 1 x NVIDIA RTX A6000 (46068 MiB)               |                   nan      |                        nan      | OOM         |
### Number of GPUs: 2
|   bits | gpus                                           |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:-----------------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 2 x NVIDIA RTX 6000 Ada Generation (49140 MiB) |                    10.8242 |                         57.8237 |             |
|     16 | 2 x NVIDIA GeForce RTX 3090 (24576 MiB)        |                    42.2111 |                         31.4247 |             |
|     16 | 2 x NVIDIA A100-SXM4-80GB (81920 MiB)          |                    53.3837 |                         22.223  |             |
|     16 | 2 x NVIDIA RTX A6000 (46068 MiB)               |                    64.782  |                         21.3549 |             |
### Number of GPUs: 4
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    52.7912 |                         21.3862 |             |
|     16 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                    66.5247 |                         20.777  |             |
### Number of GPUs: 8
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    56.3847 |                         20.3764 |             |
## Model: h2oai/h2ogpt-4096-llama2-70b-chat (text-generation-inference)
### Number of GPUs: 4
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 4 x NVIDIA A100-SXM4-80GB (81920 MiB) |                    131.453 |                         9.61851 |             |
|     16 | 4 x NVIDIA RTX A6000 (46068 MiB)      |                    nan     |                       nan       | OOM         |
### Number of GPUs: 8
|   bits | gpus                                  |   summarization time [sec] |   generation speed [tokens/sec] | exception   |
|-------:|:--------------------------------------|---------------------------:|--------------------------------:|:------------|
|     16 | 8 x NVIDIA A100-SXM4-80GB (81920 MiB) |                     133.53 |                         9.53011 |             |
