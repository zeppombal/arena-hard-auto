# MT

I used tower-eval with this config: `/mnt/cephfs-nvme/jpombal/tower-results/tower_v4.yaml`.

# IFEval

I used Ricardo's command in a loop; e.g.,
```bash
N_GPUS=1
GPU_MEM=0.9

orgs=("Widn/" "Widn/")
models=("Tower-4-Anthill" "Tower-4-Sugarloaf")

for i in "${!orgs[@]}"; do
    org=${orgs[$i]}
    model=${models[$i]}
    MODEL_PATH=${org}${model}
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_PATH,gpu_memory_utilization=$GPU_MEM,tensor_parallel_size=$N_GPUS,max_model_len=4096 \
        --tasks leaderboard_ifeval --batch_size auto --output_path $SAVE_DIR/ifeval \
        --apply_chat_template --write_out --log_samples \
        --gen_kwargs max_gen_toks=2048,do_sample=False,temperature=0.0 \
        --output_path results_tower_v4_grpo
done
```


# M-ArenaHard
This is a bit more involved, I'm going to break it down in steps.

## 1. Setup my most recent repo

```bash
cp -r /mnt/cephfs-nvme/jpombal/eval-bias/arena-hard-auto .
cd arena-hard-auto
pip install -r requirements.txt
```

## 2. Get answers with a model

```bash
python run.py [MODEL]
```

MODEL can be any HF model. This will create a MODEL.jsonl file under `data/[bench]/model_answers`, for benches `arena-hard-v0.1, m-arena-hard-de, m-arena-hard-es, m-arena-hard-zh, m-arena-hard-ru`.

## 3. Run judgments

First, deploy a Llama-3.3-70B instruct on 4 GPUS with the command
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.3-70B-Instruct --tensor_parallel_size 4 --served-model-name Llama-3.3-70B-Instruct --generation-config . --api-key token-abc123 --port 8001
```

Then, add MODEL to `configs/judge_config.yaml` (remove models from there if necessary), run

```bash
python gen_judgment.py --bench_name [BENCH]
```

and show results

```bash
python show_result.py --bench-name arena-hard-v0.1
```

BENCH is one of `arena-hard-v0.1, m-arena-hard-de, m-arena-hard-es, m-arena-hard-zh, m-arena-hard-ru`.
