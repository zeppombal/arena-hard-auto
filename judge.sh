langs=(fr hi zh)

for l in "${langs[@]}"; do
    echo "Running best-of-n for $m, $j, $l"
    python gen_judgment.py --bench_name m-arena-hard-${l}
done