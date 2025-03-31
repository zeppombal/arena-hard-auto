models=(Qwen/Qwen2.5-3B-Instruct) #Qwen/Qwen2.5-7B-Instruct
langs=(hi fr zh)
j=$1
# 
# for m in "${models[@]}"; do
#     for l in "${langs[@]}"; do
#         echo "Generating candidates for $m on $l"
#         python run_candidates.py --answer_model ${m} --bench_name m-arena-hard-${l}
#     done
# done

for m in "${models[@]}"; do
stubbed_model_name=$(echo $m | sed 's/\//__/g')
    for l in "${langs[@]}"; do
        echo "Running best-of-n for $m, $j, $l"
        python run_qad.py --answer_model ${stubbed_model_name} --judge_name ${j} --bench_name m-arena-hard-${l}
    done
done