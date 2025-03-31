#!/bin/bash

# Array of languages
langs=("ar" "cs" "de" "fr" "he" "hi" "id" "it" "ja" "ko" "nl" "fa" "pl" "pt" "ro" "ru" "es" "tr" "uk" "vi" "zh")

# Iterate over each language and call the Python script
for lang in "${langs[@]}"; do
    echo "Processing language: $lang"
    python gen_answer.py --bench_name "m-arena-hard-${lang}"
done

echo "All languages processed."