import os
import json

def null_scores_in_jsonl(path):

    total_scores = 0
    null_count = 0
    pairs_count = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            found = False
            for game in data["games"]:
                total_scores += 1
                if game["score"] is None:
                    #import pdb; pdb.set_trace()
                    null_count += 1
                    found = True
            if found:
                pairs_count += 1

    if total_scores == 0:
        print("No scores found.")
    else:
        percentage = (null_count / total_scores) * 100
        print(f"Total scores: {total_scores}")
        print(f"Null count: {null_count}")
        #print(f"Null pairs count: {pairs_count}")
        print(f"Percentage null: {percentage:.2f}%")
        #print(f"Percentage null pairs: {(2 * pairs_count / total_scores)*100:.2f}%\n")




def null_scores_in_directory(judge_model):

    bench_names = [
        "arena-hard-v0.1",
        "m-arena-hard-zh",
        "m-arena-hard-fr",
        "m-arena-hard-ko"
    ]

    total_nulls = 0
    total_scores = 0

    print(f"Percentage null scores for: {judge_model}\n")

    for bench in bench_names:

        total_nulls_bench = 0
        total_scores_bench = 0        
        print(f"{bench}:")
        directory_path = f"/mnt/home/beatriz/arena-hard-auto/data/{bench}/model_judgment/{judge_model}"

        if not os.path.exists(directory_path):
            continue

        for filename in os.listdir(directory_path):
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                file_path = os.path.join(directory_path, filename)
                null_count = 0
                score_count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        for game in data.get("games", []):
                            total_scores += 1
                            score_count += 1
                            total_scores_bench += 1
                            if game.get("score") is None:
                                null_count += 1
                                total_nulls += 1
                                total_nulls_bench += 1
                print(f"{filename}: {(null_count / score_count) * 100:.2f}%")
        print(f"\nPercentage null for {bench}: {(total_nulls_bench / total_scores_bench) * 100:.2f}%\n\n" if total_scores_bench > 0 else "No scores found.")

    print(f"\nTotal null scores in directory: {total_nulls}")
    print(f"Total scores in directory: {total_scores}")
    print(f"Percentage null: {(total_nulls / total_scores) * 100:.2f}%" if total_scores > 0 else "No scores found.")


print("\n")
judge_model = "deepseek-ai__DeepSeek-R1-Distill-Llama-8B"
judge_model = "deepseek-ai__DeepSeek-R1-Distill-Qwen-14B"
#null_scores_in_directory(judge_model)


file_path = "/mnt/home/beatriz/arena-hard-auto/data/arena-hard-v0.1/prompt/deepseek-ai__DeepSeek-R1-Distill-Qwen-14B/all_ZERO_NO_SHUFFLE.jsonl"
#file_path = "/mnt/home/beatriz/arena-hard-auto/data/arena-hard-v0.1/prompt/meta-llama__Llama-3.1-8B/all_3rd.jsonl"
null_scores_in_jsonl(file_path)
