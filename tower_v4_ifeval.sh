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