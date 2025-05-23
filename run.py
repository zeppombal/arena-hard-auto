import argparse
import subprocess
import sys

import pandas as pd
from vllm import LLM, SamplingParams

DEFAULT_JUDGE = "gpt-3.5-turbo-0125"

ANSWER_MODEL_TEMPLATE = """bench_name: {bench_name}
temperature: 0.0
max_tokens: 4096
num_choices: 1


model_list:
  - {model}"""


def turns_to_messages(turns):
    content = turns[0]["content"]
    message = [{"role": "user", "content": content}]
    return message


def main(answer_model, model, sampling_params, bench_name):
    model_name = answer_model.replace("/", "__")
    # Get answers
    questions = pd.read_json(f"data/{bench_name}/question.jsonl", lines=True)
    messages = questions["turns"].apply(turns_to_messages).tolist()
    model_output = model.chat(messages, sampling_params, use_tqdm=True)
    answers = [output.outputs[0].text for output in model_output]
    # Save answers
    existing_answers = pd.read_json(
        f"data/{bench_name}/model_answer/gpt-4o-2024-11-20.jsonl", lines=True
    )
    turns_to_save = [
        [{"index": 0, "turns": [{"content": answer, "token_len": len(answer)}]}]
        for answer in answers
    ]
    existing_answers["question_id"] = questions["question_id"].tolist()
    existing_answers["choices"] = turns_to_save
    existing_answers["model_id"] = model_name
    existing_answers.to_json(
        f"data/{bench_name}/model_answer/{model_name}.jsonl",
        lines=True,
        orient="records",
        force_ascii=False,
    )
    # Add answer model
    # answer_config = ANSWER_MODEL_TEMPLATE.format(model=model_name, bench_name=bench_name)
    # with open("config/gen_answer_config.yaml", "w") as f:
    #     f.write(answer_config)
    # # Create judge config and run judge
    # subprocess.run(["cp", "config/judge_config_base.yaml", "config/judge_config.yaml"])
    # with open("config/judge_config.yaml", "a") as f:
    #     f.write(f"\n  - {model_name}")
    # subprocess.run(["python", "gen_judgment.py"])
    # subprocess.run(["python", "show_result.py", "--judge-name", judge_model])


# parser = argparse.ArgumentParser(description="Wrapper for Arena Hard Auto")
# parser.add_argument("--answer_model", required=True)
# parser.add_argument("--judge_model", default="gpt-4-1106-preview")
# parser.add_argument("--bench_name", default="arena-hard-v0.1")
# args = parser.parse_args()
# main(args.answer_model, args.judge_model, args.bench_name)

benches = [
    "arena-hard-v0.1",
    "m-arena-hard-de",
    "m-arena-hard-es",
    "m-arena-hard-zh",
    # "m-arena-hard-ko",
    "m-arena-hard-ru",
]


m = sys.argv[1]

print(f"### Running {m} ###")
if m in [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "ModelSpace/GemmaX2-28-2B-v0.1",
    "ModelSpace/GemmaX2-28-9B-v0.1",
    "Widn/Tower-4-Anthill",
    "Widn/Tower-4-Anthill-SFT",
    "Widn/Tower-4-Sugarloaf",
    "Widn/Tower-4-Sugarloaf-SFT",
    "Widn/Tower-4-Vesuvius-SFT",
    "CohereForAI/aya-expanse-8b",
    "CohereForAI/aya-expanse-32b",
    "/mnt/cephfs-nvme/ricardorei/tower-internal/GRPO/output_dir/Anthill-GRPO/checkpoint-1000",
    "Unbabel/Tower4-Sugarloaf-Vision-merged",
]:
    max_tokens = 4096
    truncate_prompt_tokens = 4096
    max_model_len = 8192
elif m in [
    "Widn/Tower-3.0-anthill-241001",
    "Widn/Tower-3.0-sugarloaf-241001",
    "Widn/Tower-3.0-vesuvius-241001",
    "Unbabel/TowerInstruct-13B-v0.1",
    "Unbabel/TowerInstruct-7B-v0.2",
    "utter-project/EuroLLM-1.7B-Instruct",
    "utter-project/EuroLLM-9B-Instruct",
    "haoranxu/ALMA-7B-R",
    "haoranxu/ALMA-13B-R",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "Unbabel/Tower-Llama3-70B-WMT24-Merged-2106",
]:
    max_tokens = 2048
    truncate_prompt_tokens = 2048
    max_model_len = 4096
else:
    max_tokens = 8192
    truncate_prompt_tokens = None
    max_model_len = 8192
if (
    "72" in m
    or "70" in m
    or "27" in m
    or "32" in m
    or "vesuvius" in m
    or "Vesuvius" in m
):
    tensor_parallel_size = 4
else:
    tensor_parallel_size = 1

llm = LLM(
    m,
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=max_model_len,
    # tokenizer_mode="mistral", # for pixtral
)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=max_tokens,
    truncate_prompt_tokens=truncate_prompt_tokens,
)
for bench in benches:
    print(f"### Running {bench} ###")
    main(m, llm, sampling_params, bench)
