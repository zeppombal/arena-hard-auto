import argparse
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams


def write_lines(
    path: Path,
    lines: list,
    escape_newline: bool = False,
    escape_return_char: bool = True,
) -> None:
    """Writes lines to a file.

    Lines can be escaped, meaning \n is transformed to \\n.

    Args:
        path: The path to the file.
        lines: The lines to write.
        escape_newline: Whether to escape newlines.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    for i, line in enumerate(lines):
        if escape_return_char:
            line = line.replace("\r", "\\r")
        if escape_newline:
            line = line.replace("\n", "\\n")
        out_lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines((f"{l}\n" for l in out_lines))


def turns_to_messages(turns):
    content = turns[0]["content"]
    message = [{"role": "user", "content": content}]
    return message


def main(answer_model, bench_name, n_candidates: int = 30):
    model_name = answer_model.replace("/", "__")
    # Get answers
    questions = pd.read_json(f"data/{bench_name}/question.jsonl", lines=True)
    messages = questions["turns"].apply(turns_to_messages).tolist()
    model = LLM(answer_model, tensor_parallel_size=1, max_model_len=32000)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    model_output = model.chat(messages, sampling_params, use_tqdm=True)
    greedy_answers = [output.outputs[0].text for output in model_output]
    sampling_params = SamplingParams(temperature=0.3, max_tokens=8192)
    messages = [
        m for m in messages for _ in range(n_candidates - 1)
    ]  # one of the candidates is the greedy answer
    model_output = model.chat(messages, sampling_params, use_tqdm=True)
    answers = [output.outputs[0].text for output in model_output]
    all_answers = []
    for g in greedy_answers:
        all_answers.extend([g] + answers[: n_candidates - 1])
        answers = answers[n_candidates - 1 :]
    # Save answers
    write_lines(
        f"data/{bench_name}/candidates/{n_candidates}/{model_name}.txt",
        all_answers,
        escape_newline=True,
        escape_return_char=True,
    )


parser = argparse.ArgumentParser(description="Wrapper for Arena Hard Auto")
parser.add_argument("--answer_model", required=True)
parser.add_argument("--bench_name", default="arena-hard-v0.1")
parser.add_argument("--n_candidates", default=30, type=int)
args = parser.parse_args()
main(args.answer_model, args.bench_name)
