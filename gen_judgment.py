import argparse
import json
import os
import re

from utils import load_model_answers, load_questions, make_config
from vllm import LLM, SamplingParams


def get_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


def make_convo(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]

    num_games = 2 if configs["pairwise"] else 1

    convs = []
    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}

            for i, turn in enumerate(question["turns"]):
                prompt_args[f"question_{i+1}"] = turn["content"]
            base = 1

            if baseline:
                if game % 2 == 1:  # swap position
                    answer, baseline = baseline, answer

                for i, turn in enumerate(baseline["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+1}"] = turn["content"]
                    base += 1
            if answer:
                for i, turn in enumerate(answer["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+base}"] = turn["content"]

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]

            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})
        convs.append(conv)
    return convs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/judge_config.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    parser.add_argument("--bench-name", type=str)
    parser.add_argument("--judge-model", type=str)
    parser.add_argument("--baseline", type=str)
    parser.add_argument("--models", type=str, nargs="+")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)
    configs["judge_model"] = args.judge_model
    configs["baseline_model"] = args.baseline
    configs["baseline"] = True
    configs["model_list"] = args.models
    configs["bench_name"] = args.bench_name

    print(
        f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
        + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}'
    )

    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join("data", configs["bench_name"], "reference_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)

    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]

    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]

    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model'].replace('/', '__')}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    judge_model = LLM(configs["judge_model"], max_model_len=8192)
    sampling_params = SamplingParams(temperature=0, max_tokens=8192)

    for model in models:
        all_convos = []
        for question in questions:
            question_id = question["question_id"]

            kwargs = {}
            kwargs["question"] = question

            kwargs["answer"] = model_answers[model][question_id]
            if ref_answers:
                kwargs["reference"] = [
                    ref_answer[question_id] for ref_answer in ref_answers
                ]
                assert len(kwargs["reference"]) == len(configs["ref_model"])
            else:
                kwargs["reference"] = None
            if configs["baseline"]:
                kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][
                    question_id
                ]
            else:
                kwargs["baseline_answer"] = None
            kwargs["configs"] = configs
            kwargs["output_file"] = output_files[model]
            kwargs["regex_pattern"] = pattern
            convs = make_convo(**kwargs)
            all_convos.extend(convs)
        model_outputs = judge_model.chat(all_convos, sampling_params=sampling_params)
        generations = [output.outputs[0].text for output in model_outputs]
        all_scores = [get_score(j, pattern)[0] for j in generations]
        # create outputs to save
        all_outputs = []
        for i, question in enumerate(questions):
            result_0 = {
                "user_prompt": all_convos[2 * i][1]["content"],
                "judgment": generations[2 * i],
                "score": all_scores[2 * i],
            }
            result_1 = {
                "user_prompt": all_convos[(2 * i) + 1][1]["content"],
                "judgment": generations[(2 * i) + 1],
                "score": all_scores[(2 * i) + 1],
            }
            output = {
                "question_id": question["question_id"],
                "model": model,
                "judge": configs["judge_model"],
                "games": [result_0, result_1],  # pairwise games
            }
            all_outputs.append(output)

        # save outputs
        with open(output_file, "w") as f:
            for output in all_outputs:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
