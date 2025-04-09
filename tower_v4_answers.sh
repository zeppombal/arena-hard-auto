#!/bin/bash
orgs=("Widn/" "Widn/" "Widn/" "Unbabel/" "Unbabel/" "utter-project/" "utter-project/" "haoranxu/" "haoranxu/")
models=(Tower-3.0-anthill-241001 Tower-3.0-sugarloaf-241001 Tower-3.0-vesuvius-241001 TowerInstruct-13B-v0.1 TowerInstruct-7B-v0.2 EuroLLM-1.7B-Instruct EuroLLM-9B-Instruct ALMA-7B-R ALMA-13B-R)

# 8192: gemma-2-2b-it gemma-2-9b-it gemma-2-27b-it GemmaX2-28-2B-v0.1 GemmaX2-28-9B-v0.1 Tower-4-Anthill Tower-4-Anthill-SFT Tower-4-Sugarloaf Tower-4-Sugarloaf-SFT Tower-4-Vesuvius-SFT aya-expanse-8b aya-expanse-32b
# 4096: Tower-3.0-anthill-241001 Tower-3.0-sugarloaf-241001 Tower-3.0-vesuvius-241001 TowerInstruct-13B-v0.1 TowerInstruct-7B-v0.2 EuroLLM-1.7B-Instruct EuroLLM-9B-Instruct ALMA-7B-R ALMA-13B-R

# Array of languages
langs=("de" "es" "zh" "ru" "ko")

# iterate over orgs and models at the same time
for i in "${!orgs[@]}"; do
    org=${orgs[$i]}
    model=${models[$i]}
    # if org is not empty, then
    if [ -n "$org" ]; then
        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --model ${org}${model} --tensor_parallel_size 4 --served-model-name ${model} --generation-config . --api-key token-abc123 --port 8001 &
        API_PID=$!
        echo "Started API server with PID: $API_PID"
        # Allow some time for the server to start
        sleep 300
    fi
    echo "Processing model: $model"
    # Iterate over each language and call the Python script
    for lang in "${langs[@]}"; do
        echo "Processing language: $lang"
        python gen_answer.py --bench_name "m-arena-hard-${lang}" --model ${model}
    done

    # Kill the API server before moving to the next iteration
    if [ -n "$API_PID" ]; then
        echo "Stopping API server with PID: $API_PID"
        kill $API_PID
        wait $API_PID 2>/dev/null
    fi
done

echo "All languages processed."
