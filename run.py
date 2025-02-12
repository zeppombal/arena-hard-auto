import argparse
import subprocess

import pandas as pd
from vllm import LLM, SamplingParams

DEFAULT_JUDGE = "gpt-3.5-turbo-0125"

ANSWER_MODEL_TEMPLATE = """bench_name: arena-hard-v0.1
temperature: 0.0
max_tokens: 4096
num_choices: 1


model_list:
  - {model}"""


def turns_to_messages(turns):
    content = turns[0]["content"]
    message = [{"role": "user", "content": content}]
    return message


def main(answer_model, judge_model):
    if judge_model != DEFAULT_JUDGE:
        pass  # raise NotImplementedError("Only gpt-4-1106-preview is supported as judge model")
    model_name = answer_model.replace("/", "__")
    # Get answers
    questions = pd.read_json("data/arena-hard-v0.1/question.jsonl", lines=True)
    messages = questions["turns"].apply(turns_to_messages).tolist()
    model = LLM(answer_model, tensor_parallel_size=4)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    model_output = model.chat(messages, sampling_params, use_tqdm=True)
    answers = [output.outputs[0].text for output in model_output]
    # Save answers
    existing_answers = pd.read_json(
        "data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl", lines=True
    )
    turns_to_save = [
        [{"index": 0, "turns": [{"content": answer, "token_len": len(answer)}]}]
        for answer in answers
    ]
    existing_answers["question_id"] = questions["question_id"].tolist()
    existing_answers["choices"] = turns_to_save
    existing_answers["model_id"] = model_name
    existing_answers.to_json(
        f"data/arena-hard-v0.1/model_answer/{model_name}.jsonl",
        lines=True,
        orient="records",
        force_ascii=False,
    )
    # Add answer model
    answer_config = ANSWER_MODEL_TEMPLATE.format(model=model_name)
    with open("config/gen_answer_config.yaml", "w") as f:
        f.write(answer_config)
    # Create judge config and run judge
    subprocess.run(["cp", "config/judge_config_base.yaml", "config/judge_config.yaml"])
    with open("config/judge_config.yaml", "a") as f:
        f.write(f"\n  - {model_name}")
    subprocess.run(["python", "gen_judgment.py"])
    subprocess.run(
        ["python", "show_result.py", "--judge-name", judge_model, "--show-elo"]
    )
    a = 1


parser = argparse.ArgumentParser(description="Wrapper for Arena Hard Auto")
parser.add_argument("--answer_model", required=True)
parser.add_argument("--judge_model", default="gpt-4-1106-preview")
args = parser.parse_args()
main(args.answer_model, args.judge_model)
