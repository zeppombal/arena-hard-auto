models=(Qwen/Qwen2.5-3B-Instruct) #Qwen/Qwen2.5-7B-Instruct
langs=(hi fr zh)
judges=(translated_data_3_langs_REAL) #glider prometheus_with_eurollm prometheus_2_7B prometheus_with_mistral prometheus_with_qwen add_multilingual hercule_hi hercule_fr prometheus_2_8x7B translated_data_3_langs multilingual_data_5_langs add_mt 
# 
# for m in "${models[@]}"; do
#     for l in "${langs[@]}"; do
#         echo "Generating candidates for $m on $l"
#         python run_candidates.py --answer_model ${m} --bench_name m-arena-hard-${l}
#     done
# done

for m in "${models[@]}"; do
    stubbed_model_name=$(echo $m | sed 's/\//__/g')
    for j in "${judges[@]}"; do
        for l in "${langs[@]}"; do
            echo "Running best-of-n for $m, $j, $l"
            python run_qad.py --answer_model ${stubbed_model_name} --judge_name ${j} --bench_name m-arena-hard-${l}
        done
    done
done