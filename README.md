# Arena-Hard-Auto

> 🚨 New feature: **Style Control** is now added to Arena Hard Auto! Check this [section](#style-control) to start using style control!

Arena-Hard-Auto-v0.1 ([See Paper](https://arxiv.org/abs/2406.11939)) is an automatic evaluation tool for instruction-tuned LLMs. It contains 500 challenging user queries sourced from Chatbot Arena. We prompt GPT-4-Turbo as judge to compare the models' responses against a baseline model (default: GPT-4-0314). Notably, Arena-Hard-Auto has the highest correlation and separability to Chatbot Arena among popular open-ended LLM benchmarks ([See Paper](https://arxiv.org/abs/2406.11939)). If you are curious to see how well your model might perform on Chatbot Arena, we recommend trying Arena-Hard-Auto.

Although both Arena-Hard-Auto and Chatbot Arena Category Hard ([See Blog](https://lmsys.org/blog/2024-05-17-category-hard/)) employ similar pipeline to select hard prompts, Arena-Hard-Auto employs automatic judge as a cheaper and faster approximator to human preference. Checkout [BenchBuilder](BenchBuilder) folder for code and resources on how we curate Arena-Hard-Auto. In the paper we also purposed metrics, such as model separability and agreement to human preference, for evaluating benchmarks' ability to rank models (See [Evaluate Benchmarks](#evaluate-benchmarks) for more information and code).

## Content
- [Style Control Leaderboard](#style-control-leaderboard)
- [Leaderboard](#leaderboard)
- [Install](#install-dependencies)
- [Evaluation](#evaluate)
- [Style Control: how to mitigate biases](#style-control)
- [Evaluate Benchmarks: how to evaluate benchmarks](#evaluate-benchmarks)
- [Citation](#citation)

## Style Control Leaderboard
Following the newly introduced Style Control on Chatbot Arena, we release Style Control on Arena Hard Auto! We employ the same Style Control methods as proposed in the [blogpost](https://lmsys.org/blog/2024-08-28-style-control/). Please refer to the blogpost for methodology and technical background. 

(Updated: 11/14)
```console
Llama-3.3-Nemotron-Super-49B-v1-Feedback-Edit-ITS | score: 88.7  | 95% CI: (-2.3, 2.2)  | average #tokens: 1217
claude-3-5-sonnet-20241022                        | score: 86.4  | 95% CI: (-1.3, 1.3)  | average #tokens: 691
claude-3-5-sonnet-20240620                        | score: 82.2  | 95% CI: (-1.9, 1.6)  | average #tokens: 567
o1-preview-2024-09-12                             | score: 81.7  | 95% CI: (-2.2, 2.1)  | average #tokens: 1193
o1-mini-2024-09-12                                | score: 79.3  | 95% CI: (-2.8, 2.3)  | average #tokens: 1399
gpt-4-turbo-2024-04-09                            | score: 74.3  | 95% CI: (-2.4, 2.4)  | average #tokens: 662
gpt-4-0125-preview                                | score: 73.6  | 95% CI: (-2.0, 2.0)  | average #tokens: 619
athene-v2-chat                                    | score: 72.1  | 95% CI: (-2.5, 2.5)  | average #tokens: 884
gpt-4o-2024-08-06                                 | score: 71.1  | 95% CI: (-2.5, 2.0)  | average #tokens: 594
llama-3.1-nemotron-70b-instruct                   | score: 71.0  | 95% CI: (-2.8, 3.1)  | average #tokens: 869
gpt-4o-2024-05-13                                 | score: 69.9  | 95% CI: (-2.5, 2.0)  | average #tokens: 696
athene-70b-0725                                   | score: 68.3  | 95% CI: (-2.6, 2.4)  | average #tokens: 683
llama-3.1-405b-instruct-fp8                       | score: 67.1  | 95% CI: (-2.2, 2.8)  | average #tokens: 658
yi-lightning                                      | score: 66.9  | 95% CI: (-3.3, 2.7)  | average #tokens: 875
claude-3-opus-20240229                            | score: 65.5  | 95% CI: (-2.3, 2.2)  | average #tokens: 541
yi-large-preview                                  | score: 65.1  | 95% CI: (-2.5, 2.5)  | average #tokens: 720
gpt-4o-mini-2024-07-18                            | score: 64.0  | 95% CI: (-3.5, 2.9)  | average #tokens: 668
qwen2.5-72b-instruct                              | score: 63.3  | 95% CI: (-2.5, 2.3)  | average #tokens: 821
mistral-large-2407                                | score: 63.1  | 95% CI: (-3.0, 2.6)  | average #tokens: 623
gemini-1.5-pro-api-0514                           | score: 62.7  | 95% CI: (-3.2, 3.0)  | average #tokens: 676
glm-4-0520                                        | score: 61.4  | 95% CI: (-2.6, 2.4)  | average #tokens: 636
yi-large                                          | score: 59.4  | 95% CI: (-2.8, 2.5)  | average #tokens: 626
deepseek-coder-v2                                 | score: 58.3  | 95% CI: (-2.8, 2.6)  | average #tokens: 578
glm-4-0116                                        | score: 54.2  | 95% CI: (-2.2, 2.2)  | average #tokens: 622
llama-3.1-70b-instruct                            | score: 51.8  | 95% CI: (-3.4, 2.1)  | average #tokens: 628
glm-4-air                                         | score: 50.6  | 95% CI: (-2.6, 2.4)  | average #tokens: 619
gpt-4-0314                                        | score: 50.0  | 95% CI:  (0.0, 0.0)  | average #tokens: 423                   
claude-3-sonnet-20240229                          | score: 49.9  | 95% CI: (-2.7, 2.4)  | average #tokens: 552                   
gpt-4-0613                                        | score: 49.7  | 95% CI: (-2.3, 2.5)  | average #tokens: 354
qwen2-72b-instruct                                | score: 49.6  | 95% CI: (-2.1, 2.2)  | average #tokens: 515
gemma-2-27b-it                                    | score: 47.5  | 95% CI: (-2.5, 2.7)  | average #tokens: 577
gemini-1.5-pro-api-0409-preview                   | score: 46.7  | 95% CI: (-2.6, 3.1)  | average #tokens: 478
mistral-large-2402                                | score: 45.6  | 95% CI: (-2.1, 2.3)  | average #tokens: 400
claude-3-haiku-20240307                           | score: 45.4  | 95% CI: (-2.5, 2.7)  | average #tokens: 505
llama-3-70b-instruct                              | score: 44.5  | 95% CI: (-2.4, 2.0)  | average #tokens: 591
mixtral-8x22b-instruct-v0.1                       | score: 44.2  | 95% CI: (-2.7, 3.1)  | average #tokens: 430
gemini-1.5-flash-api-0514                         | score: 39.9  | 95% CI: (-2.5, 2.1)  | average #tokens: 642
llama-3.1-nemotron-51b-instruct                   | score: 39.9  | 95% CI: (-2.9, 2.7)  | average #tokens: 747
qwen1.5-72b-chat                                  | score: 39.9  | 95% CI: (-2.1, 2.4)  | average #tokens: 474
mistral-next                                      | score: 39.6  | 95% CI: (-2.4, 2.7)  | average #tokens: 297
mistral-medium                                    | score: 39.1  | 95% CI: (-2.4, 2.8)  | average #tokens: 485
phi-3-medium-4k-instruct                          | score: 38.8  | 95% CI: (-2.5, 2.7)  | average #tokens: 517
command-r-plus                                    | score: 37.5  | 95% CI: (-2.4, 2.3)  | average #tokens: 541
claude-2.0                                        | score: 36.6  | 95% CI: (-3.0, 3.0)  | average #tokens: 295
claude-2.1                                        | score: 35.0  | 95% CI: (-2.6, 2.3)  | average #tokens: 290
gpt-3.5-turbo-0613                                | score: 34.9  | 95% CI: (-2.4, 2.9)  | average #tokens: 401
gpt-3.5-turbo-0125                                | score: 34.6  | 95% CI: (-2.3, 2.6)  | average #tokens: 329
phi-3-small-8k-instruct                           | score: 33.8  | 95% CI: (-2.4, 1.9)  | average #tokens: 568
gemma-2-9b-it                                     | score: 33.6  | 95% CI: (-2.3, 2.2)  | average #tokens: 541
gpt-3.5-turbo-1106                                | score: 32.9  | 95% CI: (-3.7, 2.4)  | average #tokens: 285
dbrx-instruct-preview                             | score: 32.0  | 95% CI: (-2.5, 2.4)  | average #tokens: 415
internlm2-20b-5-chat                              | score: 30.4  | 95% CI: (-2.2, 2.6)  | average #tokens: 576
mixtral-8x7b-instruct-v0.1                        | score: 29.8  | 95% CI: (-2.3, 2.2)  | average #tokens: 457
gpt-3.5-turbo-0314                                | score: 29.4  | 95% CI: (-2.5, 3.0)  | average #tokens: 334
starling-lm-7b-beta                               | score: 26.1  | 95% CI: (-2.6, 2.0)  | average #tokens: 530
snowflake-arctic-instruct                         | score: 25.8  | 95% CI: (-2.3, 2.1)  | average #tokens: 365
gemini-pro                                        | score: 24.9  | 95% CI: (-1.8, 2.5)  | average #tokens: 322
command-r                                         | score: 23.3  | 95% CI: (-1.9, 2.0)  | average #tokens: 432
snorkel-mistral-pairrm-dpo                        | score: 21.9  | 95% CI: (-1.6, 1.9)  | average #tokens: 564
yi-34b-chat                                       | score: 21.9  | 95% CI: (-1.5, 2.2)  | average #tokens: 611
internlm2-20b-chat                                | score: 21.3  | 95% CI: (-2.1, 1.8)  | average #tokens: 667
llama-3-8b-instruct                               | score: 19.8  | 95% CI: (-1.6, 1.9)  | average #tokens: 585
llama-3.1-8b-instruct                             | score: 18.3  | 95% CI: (-1.6, 1.6)  | average #tokens: 861
tulu-2-dpo-70b                                    | score: 18.0  | 95% CI: (-1.9, 2.4)  | average #tokens: 550
starling-lm-7b-alpha                              | score: 16.4  | 95% CI: (-1.4, 1.5)  | average #tokens: 483
phi-3-mini-128k-instruct                          | score: 16.1  | 95% CI: (-1.5, 1.6)  | average #tokens: 609
mistral-7b-instruct                               | score: 15.2  | 95% CI: (-1.6, 2.0)  | average #tokens: 541
llama-2-70b-chat                                  | score: 13.4  | 95% CI: (-1.5, 1.8)  | average #tokens: 595
vicuna-33b                                        | score: 11.8  | 95% CI: (-1.8, 1.3)  | average #tokens: 451
gemma-1.1-7b-it                                   | score: 11.5  | 95% CI: (-1.5, 1.3)  | average #tokens: 341
gemma-7b-it                                       | score:  7.1  | 95% CI: (-1.3, 1.2)  | average #tokens: 378
gemma-1.1-2b-it                                   | score:  3.5  | 95% CI: (-0.5, 0.8)  | average #tokens: 316
gemma-2b-it                                       | score:  2.9  | 95% CI: (-0.6, 0.7)  | average #tokens: 369
```

# Leaderboard
The following leaderboard has no style control.

(Updated: 11/14)
```console
Llama-3.3-Nemotron-Super-49B-v1-Feedback-Edit-ITS | score: 93.4  | 95% CI: (-1.1, 1.0)  | average #tokens: 1217
o1-mini-2024-09-12                                | score: 92.0  | 95% CI: (-1.2, 1.0)  | average #tokens: 1399
o1-preview-2024-09-12                             | score: 90.4  | 95% CI: (-1.1, 1.3)  | average #tokens: 1193
claude-3-5-sonnet-20241022                        | score: 85.2  | 95% CI: (-1.4, 1.6)  | average #tokens: 691
athene-v2-chat                                    | score: 85.0  | 95% CI: (-1.4, 1.7)  | average #tokens: 884
llama-3.1-nemotron-70b-instruct                   | score: 84.9  | 95% CI: (-1.7, 1.8)  | average #tokens: 869
gpt-4-turbo-2024-04-09                            | score: 82.6  | 95% CI: (-1.8, 1.5)  | average #tokens: 662
yi-lightning                                      | score: 81.5  | 95% CI: (-1.6, 1.6)  | average #tokens: 875
claude-3-5-sonnet-20240620                        | score: 79.3  | 95% CI: (-2.1, 2.0)  | average #tokens: 567
gpt-4o-2024-05-13                                 | score: 79.2  | 95% CI: (-1.9, 1.7)  | average #tokens: 696
gpt-4-0125-preview                                | score: 78.0  | 95% CI: (-2.1, 2.4)  | average #tokens: 619
qwen2.5-72b-instruct                              | score: 78.0  | 95% CI: (-1.8, 1.8)  | average #tokens: 821
gpt-4o-2024-08-06                                 | score: 77.9  | 95% CI: (-2.0, 2.1)  | average #tokens: 594
athene-70b                                        | score: 77.6  | 95% CI: (-2.7, 2.2)  | average #tokens: 684
gpt-4o-mini                                       | score: 74.9  | 95% CI: (-2.5, 1.9)  | average #tokens: 668
gemini-1.5-pro-api-preview                        | score: 72.0  | 95% CI: (-2.1, 2.5)  | average #tokens: 676
mistral-large-2407                                | score: 70.4  | 95% CI: (-1.6, 2.1)  | average #tokens: 623
llama-3.1-405b-instruct-fp8                       | score: 69.3  | 95% CI: (-2.4, 2.2)  | average #tokens: 658
glm-4-0520                                        | score: 63.8  | 95% CI: (-2.9, 2.8)  | average #tokens: 636
yi-large                                          | score: 63.7  | 95% CI: (-2.6, 2.4)  | average #tokens: 626
deepseek-coder-v2                                 | score: 62.3  | 95% CI: (-2.1, 1.8)  | average #tokens: 578
claude-3-opus-20240229                            | score: 60.4  | 95% CI: (-2.5, 2.5)  | average #tokens: 541
gemma-2-27b-it                                    | score: 57.5  | 95% CI: (-2.1, 2.4)  | average #tokens: 577
llama-3.1-70b-instruct                            | score: 55.7  | 95% CI: (-2.9, 2.7)  | average #tokens: 628
glm-4-0116                                        | score: 55.7  | 95% CI: (-2.4, 2.3)  | average #tokens: 622
glm-4-air                                         | score: 50.9  | 95% CI: (-2.9, 2.7)  | average #tokens: 619
gpt-4-0314                                        | score: 50.0  | 95% CI:  (0.0, 0.0)  | average #tokens: 423
gemini-1.5-flash-api-preview                      | score: 49.6  | 95% CI: (-2.2, 2.8)  | average #tokens: 642
qwen2-72b-instruct                                | score: 46.9  | 95% CI: (-2.5, 2.7)  | average #tokens: 515
claude-3-sonnet-20240229                          | score: 46.8  | 95% CI: (-2.3, 2.7)  | average #tokens: 552
llama-3-70b-instruct                              | score: 46.6  | 95% CI: (-2.3, 2.6)  | average #tokens: 591
claude-3-haiku-20240307                           | score: 41.5  | 95% CI: (-2.5, 2.5)  | average #tokens: 505
gpt-4-0613                                        | score: 37.9  | 95% CI: (-2.8, 2.4)  | average #tokens: 354
mistral-large-2402                                | score: 37.7  | 95% CI: (-2.1, 2.6)  | average #tokens: 400
mixtral-8x22b-instruct-v0.1                       | score: 36.4  | 95% CI: (-2.4, 2.6)  | average #tokens: 430
Qwen1.5-72B-Chat                                  | score: 36.1  | 95% CI: (-2.0, 2.7)  | average #tokens: 474
phi-3-medium-4k-instruct                          | score: 33.4  | 95% CI: (-2.6, 2.1)  | average #tokens: 517
command-r-plus                                    | score: 33.1  | 95% CI: (-2.8, 2.4)  | average #tokens: 541
mistral-medium                                    | score: 31.9  | 95% CI: (-1.9, 2.2)  | average #tokens: 485
internlm2.5-20b-chat                              | score: 31.2  | 95% CI: (-2.4, 2.8)  | average #tokens: 576
phi-3-small-8k-instruct                           | score: 29.8  | 95% CI: (-1.8, 1.9)  | average #tokens: 568
mistral-next                                      | score: 27.4  | 95% CI: (-2.4, 2.4)  | average #tokens: 297
gpt-3.5-turbo-0613                                | score: 24.8  | 95% CI: (-1.9, 2.3)  | average #tokens: 401
dbrx-instruct-preview                             | score: 24.6  | 95% CI: (-2.0, 2.6)  | average #tokens: 415
internlm2-20b-chat                                | score: 24.4  | 95% CI: (-2.0, 2.2)  | average #tokens: 667
claude-2.0                                        | score: 24.0  | 95% CI: (-1.8, 1.8)  | average #tokens: 295
Mixtral-8x7B-Instruct-v0.1                        | score: 23.4  | 95% CI: (-2.0, 1.9)  | average #tokens: 457
gpt-3.5-turbo-0125                                | score: 23.3  | 95% CI: (-2.2, 1.9)  | average #tokens: 329
Yi-34B-Chat                                       | score: 23.1  | 95% CI: (-1.6, 1.8)  | average #tokens: 611
Starling-LM-7B-beta                               | score: 23.0  | 95% CI: (-1.8, 1.8)  | average #tokens: 530
claude-2.1                                        | score: 22.8  | 95% CI: (-2.3, 1.8)  | average #tokens: 290
llama-3.1-8b-instruct                             | score: 21.3  | 95% CI: (-1.9, 2.2)  | average #tokens: 861
Snorkel-Mistral-PairRM-DPO                        | score: 20.7  | 95% CI: (-1.8, 2.2)  | average #tokens: 564
llama-3-8b-instruct                               | score: 20.6  | 95% CI: (-2.0, 1.9)  | average #tokens: 585                       
gpt-3.5-turbo-1106                                | score: 18.9  | 95% CI: (-1.8, 1.6)  | average #tokens: 285
gpt-3.5-turbo-0301                                | score: 18.1  | 95% CI: (-1.9, 2.1)  | average #tokens: 334
gemini-1.0-pro                                    | score: 17.8  | 95% CI: (-1.2, 2.2)  | average #tokens: 322
snowflake-arctic-instruct                         | score: 17.6  | 95% CI: (-1.8, 1.5)  | average #tokens: 365
command-r                                         | score: 17.0  | 95% CI: (-1.7, 1.8)  | average #tokens: 432
phi-3-mini-128k-instruct                          | score: 15.4  | 95% CI: (-1.4, 1.4)  | average #tokens: 609
tulu-2-dpo-70b                                    | score: 15.0  | 95% CI: (-1.6, 1.3)  | average #tokens: 550
Starling-LM-7B-alpha                              | score: 12.8  | 95% CI: (-1.6, 1.4)  | average #tokens: 483
mistral-7b-instruct                               | score: 12.6  | 95% CI: (-1.7, 1.4)  | average #tokens: 541
gemma-1.1-7b-it                                   | score: 12.1  | 95% CI: (-1.3, 1.3)  | average #tokens: 341
Llama-2-70b-chat-hf                               | score: 11.6  | 95% CI: (-1.5, 1.2)  | average #tokens: 595
vicuna-33b-v1.3                                   | score:  8.6  | 95% CI: (-1.1, 1.1)  | average #tokens: 451
gemma-7b-it                                       | score:  7.5  | 95% CI: (-1.2, 1.3)  | average #tokens: 378
Llama-2-7b-chat-hf                                | score:  4.6  | 95% CI: (-0.8, 0.8)  | average #tokens: 561
gemma-1.1-2b-it                                   | score:  3.4  | 95% CI: (-0.6, 0.8)  | average #tokens: 316
gemma-2b-it                                       | score:  3.0  | 95% CI: (-0.6, 0.6)  | average #tokens: 369
```

## Install Dependencies
```
git clone https://github.com/lm-sys/arena-hard.git
cd arena-hard
pip install -r requirements.txt

# Running a model
python run.py --answer_model [HF_MODEL_PATH]
```
>Assuming python >= 3.10
