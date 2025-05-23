import argparse
import random
import re
from pathlib import Path

import numpy as np
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


PROMETHEUS_INSTRUCTION_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[How accurate and comprehensive is the response? How well is the response structured and communicated? How effectively does it address the prompt's requirements?]
Score 1: Response is fundamentally incorrect or incomprehensible. Coverage is severely incomplete and fails to address the prompt. Lacks supporting evidence and shows extreme bias. Also applies to responses not in the prompt's language.
Score 2: Response contains notable inaccuracies and incomplete coverage. Organization needs improvement and partially misses prompt intent. Supporting evidence is limited and some bias is present. Falls short of meeting basic requirements.
Score 3: Response provides generally accurate information with sufficient coverage of main points. Organization is clear despite some issues. Mostly relevant to prompt with basic supporting evidence and reasonable objectivity. Meets basic requirements but lacks excellence.
Score 4: Response shows high accuracy with thorough coverage, clear organization, and strong alignment with the prompt. Contains good supporting evidence and maintains objectivity. May have minor imperfections but achieves all major goals effectively.
Score 5: Response demonstrates exceptional accuracy with comprehensive coverage, perfect structure, and complete relevance to the prompt. Includes strong supporting evidence and maintains complete objectivity where needed. Information is well-researched with valuable additional context.

###Feedback: """

PROMETHEUS_INSTRUCTION_NOREF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
[How accurate and comprehensive is the response? How well is the response structured and communicated? How effectively does it address the prompt's requirements?]
Score 1: Response is fundamentally incorrect or incomprehensible. Coverage is severely incomplete and fails to address the prompt. Lacks supporting evidence and shows extreme bias. Also applies to responses not in the prompt's language.
Score 2: Response contains notable inaccuracies and incomplete coverage. Organization needs improvement and partially misses prompt intent. Supporting evidence is limited and some bias is present. Falls short of meeting basic requirements.
Score 3: Response provides generally accurate information with sufficient coverage of main points. Organization is clear despite some issues. Mostly relevant to prompt with basic supporting evidence and reasonable objectivity. Meets basic requirements but lacks excellence.
Score 4: Response shows high accuracy with thorough coverage, clear organization, and strong alignment with the prompt. Contains good supporting evidence and maintains objectivity. May have minor imperfections but achieves all major goals effectively.
Score 5: Response demonstrates exceptional accuracy with comprehensive coverage, perfect structure, and complete relevance to the prompt. Includes strong supporting evidence and maintains complete objectivity where needed. Information is well-researched with valuable additional context.

###Feedback: """

HERCULE_PROMPT_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[How accurate and comprehensive is the response? How well is the response structured and communicated? How effectively does it address the prompt's requirements?]
Score 1: Response is fundamentally incorrect or incomprehensible. Coverage is severely incomplete and fails to address the prompt. Lacks supporting evidence and shows extreme bias. Also applies to responses not in the prompt's language.
Score 2: Response contains notable inaccuracies and incomplete coverage. Organization needs improvement and partially misses prompt intent. Supporting evidence is limited and some bias is present. Falls short of meeting basic requirements.
Score 3: Response provides generally accurate information with sufficient coverage of main points. Organization is clear despite some issues. Mostly relevant to prompt with basic supporting evidence and reasonable objectivity. Meets basic requirements but lacks excellence.
Score 4: Response shows high accuracy with thorough coverage, clear organization, and strong alignment with the prompt. Contains good supporting evidence and maintains objectivity. May have minor imperfections but achieves all major goals effectively.
Score 5: Response demonstrates exceptional accuracy with comprehensive coverage, perfect structure, and complete relevance to the prompt. Includes strong supporting evidence and maintains complete objectivity where needed. Information is well-researched with valuable additional context.

###Feedback:
"""

HERCULE_PROMPT_NOREF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
[How accurate and comprehensive is the response? How well is the response structured and communicated? How effectively does it address the prompt's requirements?]
Score 1: Response is fundamentally incorrect or incomprehensible. Coverage is severely incomplete and fails to address the prompt. Lacks supporting evidence and shows extreme bias. Also applies to responses not in the prompt's language.
Score 2: Response contains notable inaccuracies and incomplete coverage. Organization needs improvement and partially misses prompt intent. Supporting evidence is limited and some bias is present. Falls short of meeting basic requirements.
Score 3: Response provides generally accurate information with sufficient coverage of main points. Organization is clear despite some issues. Mostly relevant to prompt with basic supporting evidence and reasonable objectivity. Meets basic requirements but lacks excellence.
Score 4: Response shows high accuracy with thorough coverage, clear organization, and strong alignment with the prompt. Contains good supporting evidence and maintains objectivity. May have minor imperfections but achieves all major goals effectively.
Score 5: Response demonstrates exceptional accuracy with comprehensive coverage, perfect structure, and complete relevance to the prompt. Includes strong supporting evidence and maintains complete objectivity where needed. Information is well-researched with valuable additional context.

###Feedback:
"""

GLIDER_PROMPT_REF = """Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:

1. Understand the text tags, pass criteria and rubric thoroughly.
2. Review the finer details of the text and the rubric.
3. Compare the tags to be evaluated to the score descriptions in the rubric.
4. Pay close attention to small details that might impact the final score and form accurate associations between tags and pass criteria.
5. Write a detailed reasoning justifying your evaluation in a bullet point format. 
6. The reasoning must summarize the overall strengths and weaknesses of the output while quoting exact phrases from the output wherever required.
7. Output a list of words or phrases that you believe are the most important in determining the score.
8. Assign a final score based on the scoring rubric.

Data to evaluate:
<USER PROMPT>
{instruction}
</USER PROMPT>

<ASSISTANT REPLY>
{response}
</ASSISTANT REPLY>

<REFERENCE>
{reference}
</REFERENCE>

Pass Criteria:
How accurate and comprehensive is the response? How well is the response structured and communicated? How effectively does it address the prompt's requirements?

Rubric:
1: Response is fundamentally incorrect or incomprehensible. Coverage is severely incomplete and fails to address the prompt. Lacks supporting evidence and shows extreme bias. Also applies to responses not in the prompt's language.
2: Response contains notable inaccuracies and incomplete coverage. Organization needs improvement and partially misses prompt intent. Supporting evidence is limited and some bias is present. Falls short of meeting basic requirements.
3: Response provides generally accurate information with sufficient coverage of main points. Organization is clear despite some issues. Mostly relevant to prompt with basic supporting evidence and reasonable objectivity. Meets basic requirements but lacks excellence.
4: Response shows high accuracy with thorough coverage, clear organization, and strong alignment with the prompt. Contains good supporting evidence and maintains objectivity. May have minor imperfections but achieves all major goals effectively.
5: Response demonstrates exceptional accuracy with comprehensive coverage, perfect structure, and complete relevance to the prompt. Includes strong supporting evidence and maintains complete objectivity where needed. Information is well-researched with valuable additional context.

Your output must in the following format:
<reasoning>
[Detailed reasoning justifying your evaluation in a bullet point format according to the specifics defined above]
</reasoning>
<highlight>
[List of words or phrases that you believe are the most important in determining the score]
</highlight>
<score>
[The final integer score assigned based on the scoring rubric]
</score>
"""

GLIDER_PROMPT_NOREF = """Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:

1. Understand the text tags, pass criteria and rubric thoroughly.
2. Review the finer details of the text and the rubric.
3. Compare the tags to be evaluated to the score descriptions in the rubric.
4. Pay close attention to small details that might impact the final score and form accurate associations between tags and pass criteria.
5. Write a detailed reasoning justifying your evaluation in a bullet point format. 
6. The reasoning must summarize the overall strengths and weaknesses of the output while quoting exact phrases from the output wherever required.
7. Output a list of words or phrases that you believe are the most important in determining the score.
8. Assign a final score based on the scoring rubric.

Data to evaluate:
<USER PROMPT>
{instruction}
</USER PROMPT>

<ASSISTANT REPLY>
{response}
</ASSISTANT REPLY>

Pass Criteria:
How accurate and comprehensive is the response? How well is the response structured and communicated? How effectively does it address the prompt's requirements?

Rubric:
1: Response is fundamentally incorrect or incomprehensible. Coverage is severely incomplete and fails to address the prompt. Lacks supporting evidence and shows extreme bias. Also applies to responses not in the prompt's language.
2: Response contains notable inaccuracies and incomplete coverage. Organization needs improvement and partially misses prompt intent. Supporting evidence is limited and some bias is present. Falls short of meeting basic requirements.
3: Response provides generally accurate information with sufficient coverage of main points. Organization is clear despite some issues. Mostly relevant to prompt with basic supporting evidence and reasonable objectivity. Meets basic requirements but lacks excellence.
4: Response shows high accuracy with thorough coverage, clear organization, and strong alignment with the prompt. Contains good supporting evidence and maintains objectivity. May have minor imperfections but achieves all major goals effectively.
5: Response demonstrates exceptional accuracy with comprehensive coverage, perfect structure, and complete relevance to the prompt. Includes strong supporting evidence and maintains complete objectivity where needed. Information is well-researched with valuable additional context.

Your output must in the following format:
<reasoning>
[Detailed reasoning justifying your evaluation in a bullet point format according to the specifics defined above]
</reasoning>
<highlight>
[List of words or phrases that you believe are the most important in determining the score]
</highlight>
<score>
[The final integer score assigned based on the scoring rubric]
</score>
"""


def get_template(judge_type, qad_type):
    return {
        "prometheus": (
            PROMETHEUS_INSTRUCTION_REF
            if qad_type == "mbr"
            else PROMETHEUS_INSTRUCTION_NOREF
        ),
        "glider": GLIDER_PROMPT_REF if qad_type == "mbr" else GLIDER_PROMPT_NOREF,
        "hercule": HERCULE_PROMPT_REF if qad_type == "mbr" else HERCULE_PROMPT_NOREF,
        "generic": GLIDER_PROMPT_REF if qad_type == "mbr" else GLIDER_PROMPT_NOREF,
    }[judge_type]


def read_lines(path: str, unescape_newline: bool = False) -> list[str]:
    """Reads lines from a file.
    Lines can be unescapped, meaning \\n is transformed to \n.
    Args:
        path: The path to the file.
        unescape_newline: Whether to unescape newlines.
    Returns:
        The lines in the file."""
    with open(path, encoding="utf-8") as f:
        lines = [l[:-1] for l in f.readlines()]
    if unescape_newline:
        lines = [l.replace("\\n", "\n") for l in lines]
    return lines


def turns_to_messages(turns):
    content = turns[0]["content"]
    message = [{"role": "user", "content": content}]
    return message


JUDGE_PATHS = {
    "ours_7B": "Unbabel/M-Prometheus-7B",
    "ours_3B": "Unbabel/M-Prometheus-3B",
    "ours_14B": "Unbabel/M-Prometheus-14B",
    "glider": "PatronusAI/glider",
    "hercule_hi": "ai4bharat/hercule-hi",
    "hercule_fr": "ai4bharat/hercule-fr",
    "qwen_3": "Qwen/Qwen2.5-3B-Instruct",
    "qwen_7": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_14": "Qwen/Qwen2.5-14B-Instruct",
    "prometheus_2_7B": "prometheus-eval/prometheus-7b-v2.0",
    "prometheus_2_8x7B": "prometheus-eval/prometheus-8x7b-v2.0",
}


def get_judge_type(judge_model):
    judge_type = "generic"
    if "prometheus" in judge_model:
        judge_type = "prometheus"
    elif "glider" in judge_model:
        judge_type = "glider"
    elif "hercule" in judge_model:
        judge_type = "hercule"
    return judge_type


def parse_judge_output(output, judge_type):
    if judge_type in ["prometheus", "hercule", "generic"]:
        score = output.split("[RESULT]")[-1].strip()
        try:
            score = int(score)
        except ValueError:
            print("Score not found in output. Assigning random score.")
            score = random.randint(1, 5)
    elif judge_type == "glider":
        pattern = r"<score>\s*(\d)\s*</score>"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            score = int(match.group(1))
        else:
            print("Score not found in output. Assigning random score.")
            score = random.randint(1, 5)
    return score


MAX_LENGTH = {
    "ours_7B": 10000000,
    "ours_3B": 10000000,
    "ours_14B": 10000000,
    "glider": 5000,
    "hercule_hi": 5000,
    "hercule_fr": 5000,
    "qwen_3": 10000000,
    "qwen_7": 10000000,
    "qwen_14": 10000000,
    "prometheus_2_7B": 5000,
    "prometheus_2_8x7B": 5000,
}


def main(
    answer_model,
    judge_name,
    bench_name,
    n_candidates: int = 30,
    qad_type: str = "n_best",
):
    judge_model = JUDGE_PATHS[judge_name]
    judge_type = get_judge_type(judge_model)
    instruction_template = get_template(judge_type, qad_type)
    model_name = answer_model.replace("/", "__")
    candidates = read_lines(
        f"data/{bench_name}/candidates/{n_candidates}/{model_name}.txt",
        unescape_newline=True,
    )
    # create prompts
    q_df = pd.read_json(f"data/{bench_name}/question.jsonl", lines=True)
    questions = q_df["turns"].apply(lambda x: x[0]["content"]).tolist()
    messages = []
    if qad_type == "n_best":
        for i, q in enumerate(questions):
            for c in candidates[i * n_candidates : (i + 1) * n_candidates]:
                c = c[
                    : MAX_LENGTH[judge_name]
                ]  # truncate string to avoid length issues after
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": instruction_template.format(
                                instruction=q, response=c
                            ),
                        }
                    ]
                )
        assert len(messages) == len(questions) * n_candidates
    if qad_type == "mbr":
        # convert list of strings to list of lists where each list contains n_candidates strings
        grouped_hypotheses = [
            candidates[i : i + n_candidates]
            for i in range(0, len(candidates), n_candidates)
        ]
        # create a list of duplicated sources, candidates, and references
        cands = []
        refs = []
        dup_srcs = []
        for i, samples in enumerate(grouped_hypotheses):
            indices = list(range(n_candidates))
            for cand in samples:
                for ref_id in indices:
                    # exclude the case where the candidate is its own reference
                    if ref_id == i:
                        continue
                    cands.append(cand)
                    refs.append(samples[ref_id])
                    dup_srcs.append(questions[i])
        for q, c, r in zip(dup_srcs, cands, refs):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": instruction_template.format(
                            instruction=q, response=c, reference=r
                        ),
                    }
                ]
            )
        assert len(messages) == (
            len(questions) * n_candidates**2
        ) - n_candidates * len(questions)
    # generate outputs
    model = LLM(judge_model, tensor_parallel_size=1)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
    )
    model_output = model.chat(messages, sampling_params, use_tqdm=True)
    outputs = [output.outputs[0].text for output in model_output]
    # extract scores
    scores = [parse_judge_output(output, judge_type) for output in outputs]
    # perform mbr with scores
    best_utilities = []
    best_candidates = []
    if qad_type == "n_best":
        for i in range(0, len(scores), n_candidates):
            best_utilities.append(max(scores[i : i + n_candidates]))
            best_candidate_index = np.argmax(scores[i : i + n_candidates])
            best_candidates.append(candidates[i + best_candidate_index])
    elif qad_type == "mbr":
        expected_utilities = []
        for i in range(0, len(scores), n_candidates):
            expected_utilities.append(np.mean(scores[i : i + n_candidates]))
        for i in range(0, len(expected_utilities), n_candidates):
            best_utilities.append(max(expected_utilities[i : i + n_candidates]))
            best_candidate_index = np.argmax(expected_utilities[i : i + n_candidates])
            best_candidates.append(candidates[i + best_candidate_index])
    # save artifacts
    # save judge outputs
    write_lines(
        f"data/{bench_name}/candidates/{n_candidates}/judgements_on_{model_name}_{n_candidates}_{qad_type}_qad_w_{judge_name}.txt",
        outputs,
        escape_newline=True,
    )
    # save best candidates as answers
    existing_answers = pd.read_json(
        f"data/{bench_name}/model_answer/Llama-3.3-70B-Instruct.jsonl", lines=True
    )
    turns_to_save = [
        [{"index": 0, "turns": [{"content": answer, "token_len": len(answer)}]}]
        for answer in best_candidates
    ]
    existing_answers["question_id"] = q_df["question_id"].tolist()
    existing_answers["choices"] = turns_to_save
    existing_answers["model"] = (
        f"{model_name}_{n_candidates}_{qad_type}_qad_w_{judge_name}"
    )
    existing_answers.to_json(
        f"data/{bench_name}/model_answer/{model_name}_{n_candidates}_{qad_type}_qad_w_{judge_name}.jsonl",
        lines=True,
        orient="records",
        force_ascii=False,
    )


parser = argparse.ArgumentParser(description="Wrapper for Arena Hard Auto")
parser.add_argument("--answer_model", required=True)
parser.add_argument("--judge_name", default="")
parser.add_argument("--bench_name", default="arena-hard-v0.1")
parser.add_argument("--n_candidates", default=30, type=int)
parser.add_argument("--qad_type", default="n_best")
args = parser.parse_args()
main(
    args.answer_model,
    args.judge_name,
    args.bench_name,
    args.n_candidates,
    args.qad_type,
)
