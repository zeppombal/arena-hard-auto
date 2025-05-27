import argparse
import os
import subprocess
import jinja2


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=arena-hard-deepseek
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos={{ qos }}
#SBATCH --partition={{ partition }}
#SBATCH --gpus-per-node={{ gpus_per_node }}
#SBATCH --cpus-per-task={{ cpus_per_task }}

source /mnt/data-artemis/beatriz/venvs/reasoning_eval/bin/activate

cd /mnt/home/beatriz/arena-hard-auto

{% for bench, models in all_jobs.items() %}
echo "Processing benchmark: {{ bench }}"
{% for model in models %}
echo "Running model: {{ model }}"
python gen_judgment2.py \\
    --judge-model {{ judge_model }} \\
    --baseline {{ baseline }} \\
    --models {{ model }} \\
    --bench-name {{ bench }}

{% endfor %}
{% endfor %}
"""


def render_and_submit(judge_model, baseline, all_jobs, script_dir, qos, partition, gpus_per_node, cpus_per_task):
    script_name = f"job_all_benchmarks.sbatch"
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    template = jinja2.Template(SBATCH_TEMPLATE)
    script_content = template.render(
        judge_model=judge_model,
        baseline=baseline,
        all_jobs=all_jobs,
        qos=qos,
        partition=partition,
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
    )

    script_path = os.path.join(script_dir, script_name)
    with open(script_path, "w") as f:
        f.write(script_content)

    subprocess.run(["sbatch", script_path])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--baseline", default="claude-3-7-sonnet-20250219")
    parser.add_argument("--qos", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--gpus-per-node", type=str, required=True)
    parser.add_argument("--cpus-per-task", type=str, required=True)
    parser.add_argument("--bench-names", nargs="+", required=True)
    args = parser.parse_args()

    judge_model = args.judge_model
    baseline = args.baseline
    bench_names = args.bench_names

    #bench_names = [
    #    "arena-hard-v0.1",
    #    "m-arena-hard-zh",
        #"m-arena-hard-fr",
        #"m-arena-hard-ko"
    #]

    all_jobs = {}

    for bench in bench_names:
        model_dir = f"/mnt/home/beatriz/arena-hard-auto/data/{bench}/model_answer/"
        all_models = [
            os.path.splitext(f)[0]
            for f in os.listdir(model_dir)
            if f.endswith(".jsonl")
        ]

        models_to_run = []
        for model in all_models:
            output_file = f"/mnt/home/beatriz/arena-hard-auto/data/{bench}/model_judgment/{judge_model.replace('/', '__')}/{model}"
            if not os.path.exists(output_file):
                models_to_run.append(model)

        if models_to_run:
            all_jobs[bench] = models_to_run

    if all_jobs:
        print("Submitting single job for all benchmarks/models.")
        render_and_submit(judge_model, baseline, all_jobs, script_dir="sbatch_scripts", qos=args.qos, partition=args.partition, gpus_per_node=args.gpus_per_node, cpus_per_task=args.cpus_per_task)
    else:
        print("Nothing to evaluate — all outputs already exist.")

if __name__ == "__main__":
    main()

