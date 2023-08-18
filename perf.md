## Backend: transformers
### Model: h2oai/h2ogpt-4096-llama2-7b-chat
### Number of GPUs: 1
|   bits | gpus                               |   summarization time [sec] |   generation speed [tokens/sec] |
|-------:|:-----------------------------------|---------------------------:|--------------------------------:|
|     16 | ['NVIDIA RTX 6000 Ada Generation'] |                    32.3246 |                        40.9962  |
|     16 | ['NVIDIA GeForce RTX 4090']        |                    37.7366 |                        37.6173  |
|     16 | ['NVIDIA GeForce RTX 3090']        |                    68.8296 |                        18.6757  |
|     16 | ['NVIDIA GeForce GTX 1080 Ti']     |                   OOM      |                       OOM       |
|      8 | ['NVIDIA GeForce RTX 4090']        |                   122.068  |                         8.75473 |
|      8 | ['NVIDIA RTX 6000 Ada Generation'] |                    99.3556 |                         8.75433 |
|      8 | ['NVIDIA GeForce RTX 3090']        |                   183.219  |                         4.27543 |
|      8 | ['NVIDIA GeForce GTX 1080 Ti']     |                   OOM      |                       OOM       |
|      4 | ['NVIDIA RTX 6000 Ada Generation'] |                    43.8436 |                        32.4415  |
|      4 | ['NVIDIA GeForce RTX 4090']        |                    42.1202 |                        30.8347  |
|      4 | ['NVIDIA GeForce GTX 1080 Ti']     |                    30.6793 |                        23.2769  |
|      4 | ['NVIDIA GeForce RTX 3090']        |                    89.302  |                        15.2251  |
### Number of GPUs: 2
|   bits | gpus                                                                 |   summarization time [sec] |   generation speed [tokens/sec] |
|-------:|:---------------------------------------------------------------------|---------------------------:|--------------------------------:|
|     16 | ['NVIDIA RTX 6000 Ada Generation', 'NVIDIA RTX 6000 Ada Generation'] |                    33.7165 |                        38.1794  |
|     16 | ['NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3090']               |                    72.2251 |                        17.8732  |
|     16 | ['NVIDIA GeForce GTX 1080 Ti', 'NVIDIA GeForce GTX 1080 Ti']         |                   OOM      |                       OOM       |
|      8 | ['NVIDIA RTX 6000 Ada Generation', 'NVIDIA RTX 6000 Ada Generation'] |                    97.6089 |                         8.69043 |
|      8 | ['NVIDIA GeForce GTX 1080 Ti', 'NVIDIA GeForce GTX 1080 Ti']         |                    39.9321 |                         6.33888 |
|      8 | ['NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3090']               |                   186.189  |                         4.20081 |
|      4 | ['NVIDIA RTX 6000 Ada Generation', 'NVIDIA RTX 6000 Ada Generation'] |                    45.2326 |                        30.3736  |
|      4 | ['NVIDIA GeForce GTX 1080 Ti', 'NVIDIA GeForce GTX 1080 Ti']         |                    30.5984 |                        23.7388  |
|      4 | ['NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3090']               |                    91.7171 |                        14.6729  |
