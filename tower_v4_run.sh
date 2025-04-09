# models=("google/gemma-2-2b-it" "google/gemma-2-9b-it" "google/gemma-2-27b-it" "ModelSpace/GemmaX2-28-2B-v0.1" "ModelSpace/GemmaX2-28-9B-v0.1" "utter-project/EuroLLM-1.7B-Instruct" "utter-project/EuroLLM-9B-Instruct" "CohereForAI/aya-expanse-8b" "CohereForAI/aya-expanse-32b" "Unbabel/Tower-Mistral-7B-Porfirissimo-2607" "Unbabel/Tower-Llama3-70B-WMT24-Merged-2106" "haoranxu/ALMA-7B-R" "haoranxu/ALMA-13B-R" "Unbabel/TowerInstruct-7B-v0.2" "Unbabel/TowerInstruct-Mistral-7B-v0.2" "Unbabel/TowerInstruct-13B-v0.1" "Widn/Tower-3.0-anthill-241001" "Widn/Tower-3.0-sugarloaf-241001" "Widn/Tower-3.0-vesuvius-241001" "Widn/Tower-4-Anthill" "Widn/Tower-4-Anthill-SFT" "Widn/Tower-4-Sugarloaf" "Widn/Tower-4-Sugarloaf-SFT" "Widn/Tower-4-Vesuvius-SFT")
models=("Widn/Tower-4-Anthill" "Widn/Tower-4-Sugarloaf")

for m in "${models[@]}"; do
    python run.py $m
done

benches=(arena-hard-v0.1 m-arena-hard-de m-arena-hard-es m-arena-hard-zh m-arena-hard-ru)
# m-arena-hard-ko
for b in "${benches[@]}"; do
    python gen_judgment.py --bench_name $b
done