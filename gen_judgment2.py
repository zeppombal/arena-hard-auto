import argparse
import json
import os
import re
import random

from utils import load_model_answers, load_questions, make_config
from vllm import LLM, SamplingParams


def shuffle_zeroshot(prompt):
    options = [
        "Assistant A is significantly better: [[A>>B]]\n",
        "Assistant A is slightly better: [[A>B]]\n",
        "Tie, relatively the same: [[A=B]]\n",
        "Assistant B is slightly better: [[B>A]]\n",
        "Assistant B is significantly better: [[B>>A]]\n",
    ]
    random.shuffle(options)
    for idx, option in enumerate(options):
        prompt += f"{idx + 1}. {option}"
    prompt += "\n\n<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>\n\nMy final verdict is: "
    return prompt


def shuffle_fewshot(prompt):
    examples = [
        "<|User Prompt|>\nWhat season of Star Trek TNG is generally considered the worst and for what reasons?\n\n<|The Start of Assistant A's Answer|>\nStar Trek: The Next Generation (TNG) is a science fiction television series that aired from 1987 to 1994. While the show is generally considered to be of high quality, there are some fans and critics who consider the seventh season to be the weakest.\n\nThere are a few reasons why the seventh season is often viewed as the weakest:\n\n1. Focus on character relationships: In the earlier seasons, the show focused more on exploring new worlds and encountering new civilizations. In the seventh season, the show shifted its focus to character relationships, with a particular emphasis on the romantic relationships between characters. Some fans felt that this shift in focus was a departure from the show's original premise and made the show feel repetitive.\n2. Use of humor: The show's humor was often a source of criticism, with some fans feeling that it was too forced or cheesy. In the seventh season, the use of humor was particularly pronounced, with some episodes relying heavily on jokes and puns.\n3. Lack of continuity: The seventh season was the first year that TNG was not overseen by creator Gene Roddenberry, who had passed away the previous year. Some fans felt that the show's focus on character relationships and humor was a departure from Roddenberry's original vision for the show, and that the season lacked the sense of continuity and coherence that had characterized earlier seasons.\n\nOverall, while the seventh season of TNG is not universally considered to be the worst, there are certainly some fans and critics who feel that it is not as strong as the earlier seasons.\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\nThe consensus among fans is that season 1 of Star Trek: The Next Generation is the worst season. It is generally regarded as being weaker than the other seasons due to its focus on action-oriented stories, as well as its use of clichéd plot devices. Additionally, many of the episodes lack the deeper exploration of characters and themes that are present in the later seasons.\n<|The End of Assistant B's Answer|>\n\nMy final verdict is: [[A<B]]\n[ENDOFGENERATION]\n\n\n",
        "<|User Prompt|>\nA = 5, B =10, A+B=?\n\n<|The Start of Assistant A's Answer|>\nTo find the sum of A and B, we need to add their corresponding elements:\n\nA + B = C = 5 + 10 = 15\n\nTherefore, the sum of A and B is 15.\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\nTo find the sum of two numbers we can use the addition formula: A + B = (A + C) - (B - C), where C is an arbitrary number.\n\nIn this case, C = 0, because we want to find the difference between A and B.\n\nSo, to find the sum of A = 5 and B = 10, we can use the formula: A + B = (A + 0) - (B - 0) = 5 + 0 - 10 - 0 = 5.\n<|The End of Assistant B's Answer|>\n\nMy final verdict is: [[A>>B]]\n[ENDOFGENERATION]\n\n\n",
        "<|User Prompt|>\nwrite go code that calulates the first n prime numbers as fast as possible. n can be given as a command line parameter.\n\n<|The Start of Assistant A's Answer|>\nHere is a Python script that calculates the first n prime numbers as fast as possible. You can give it as a command line parameter when you run it:\n\n```python\nfrom itertools import permutations\n\ndef is_prime(n):\n if n <= 1:\n return False\n for i in range(2, int(n**0.5)+1):\n if n % i == 0:\n return False\n return True\n\nif __name__ == '__main__':\n n = int(input('Enter the number of prime numbers to calculate: '))\n primes = permutations(range(2, n+1))\n for i, prime in enumerate(primes):\n print(f\"Prime number {{i+1}}: {{prime}}\")\n```\n\nThis script uses the itertools module to generate all possible permutations of the range from 2 to the given number n. It then iterates over each permutation and checks if the given number is a prime number by iterating over all the numbers up to the square root of the number and checking if the number is evenly divisible by any of them. If the number is not divisible by any of them, it is considered to be a prime number.\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\nI have created a go program that calculates the first n prime numbers as fast as possible. n can be given as a command line parameter. Here is the code:\nh(n) = 2\nfor i in range(3, n, 2):\nif n % i == 0:\ncontinue] }}\nreturn h(n)\n<|The End of Assistant B's Answer|>\n\nMy final verdict is: [[A=B]]\n[ENDOFGENERATION]\n\n\n",
        "<|User Prompt|>\nWhy can't fish fly?\n\n<|The Start of Assistant A's Answer|>\nFish are aquatic creatures that are adapted for life in water. They have evolved various swimming adaptations, such as fins and muscles, that allow them to move efficiently in water. However, these adaptations are not suitable for flying, which requires different physiological and anatomical features.\n\nFlying requires the ability to generate lift and propulsion, which is achieved through the use of wings. Fish do not have wings or the muscles necessary to generate the lift required for flight. Additionally, the body shape and overall anatomy of fish is not well suited for flying. Fish have a streamlined body shape that is adapted for movement through water, not for the aerodynamic forces of flight.\n\nTherefore, it is simply not possible for fish to fly because they do not have the necessary adaptations or anatomy to do so.\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\nFish are not able to fly because they do not have the necessary muscles and bones to support their bodies in an aerial state. The fish's body is structured such that it is able to swim efficiently in water, and its fins provide it with stability and support. In addition, the air surrounding us is much denser than water, which makes it difficult for fish to sustain flight.\n<|The End of Assistant B's Answer|>\n\nMy final verdict is: [[A>B]]\n[ENDOFGENERATION]\n\n\n",
        "<|User Prompt|>\nwhat is the current country leading in natural water resource?\n\n<|The Start of Assistant A's Answer|>\nThe current country leading in natural water resource is China, which has a large and growing population that needs access to water resources to meet its needs. China has a long coastline and a large area of land, which means that it has a large potential for water resources. The country has developed several major water resources, including the Yangtze River, Yellow River, and water supply systems in the Jiangxi Province. Additionally, China has been investing in research and development to improve its water resources management and distribution systems.\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\nIt is difficult to determine which country has the most natural water resources, as this can depend on various factors such as the size of the country, the location of its water sources, and the rate at which water is replenished. Additionally, different countries may prioritize water resources differently, such as using more water for irrigation, energy production, or drinking.\n\nHowever, some countries have significant natural water resources, such as Russia, Canada, and Brazil. Russia has the largest surface area of any country in the world and is home to a number of major rivers and lakes, including the Volga, the Ob, and the Yenisei. Canada is another country with significant water resources, with the Great Lakes and the Mackenzie River being two of the largest freshwater systems in the world. Brazil has a large share of the Amazon River, which is the largest river by volume in the world, and is home to a number of other significant water resources as well.\n\nIt's also worth noting that some countries may have fewer natural water resources but have a more efficient use of the water they do have. For example, countries like Israel and Singapore have used advanced water management systems to conserve water resources and make the most of their limited supplies.\n\nOverall, it is difficult to say definitively which country has the most natural water resources, as it can depend on various factors.\n<|The End of Assistant B's Answer|>\n\nMy final verdict is: [[A<<B]]\n[ENDOFGENERATION]\n\n\n",
    ]
    random.shuffle(examples)
    for example in examples:
        prompt += example
    prompt += "<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>\n\nMy final verdict is: "
    return prompt



def chat_to_prompt(chat):
    """
    Converts a chat conversation (list of dicts with 'role' and 'content') 
    into a plain string prompt for models like LLaMA-3 that use .generate().
    """
    prompt = ""
    for turn in chat:
        if turn["role"] == "system":
            #prompt += f"<|System Prompt|>\n{turn['content'].strip()}\n\n"  # <|User Prompt|>   <|The End of Assistant A's Answer|>
            prompt += f"{turn['content'].strip()}\n\n"
        elif turn["role"] == "user":
            prompt += f"{turn['content'].strip()}\n\n"
        elif turn["role"] == "assistant":
            #prompt += f"<|Assistant Answer|>\n{turn['content'].strip()}\n\n"
            prompt += f"{turn['content'].strip()}\n\n"
    #prompt = prompt.strip()
    #if shot == "fewshot":
    #    prompt = shuffle_fewshot(prompt)
    #elif shot == "zeroshot":    
    #    prompt = shuffle_zeroshot(prompt)
    return prompt.strip()


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
        #conv = [{"role": "system", "content": ""}]

        for template in configs["prompt_template"]:
            #template = shuffle_fewshot(template) # IDENTIFY HERE FEWSHOT OR ZEROSHOT; COMMENT THIS FOR NO SHUFFLE
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

    judge_model = LLM(configs["judge_model"], max_model_len=128000) #, tensor_parallel_size=2) #, max_model_len=8192)
    #sampling_params = SamplingParams(temperature=0.6, max_tokens=50000) #, max_tokens=8192)

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

        if configs['judge_model'] in ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B"]:   # Use this for DeepSeek models for comparison?
            prompt_texts = [chat_to_prompt(conv) for conv in all_convos]
            sampling_params = SamplingParams(temperature=0, max_tokens=500, stop=["[ENDOFGENERATION]"]) #, max_tokens=8192)
            model_outputs = judge_model.generate(prompt_texts, sampling_params=sampling_params) #, stop=["end of generation(tambem funciona com chat)"])
            #import pdb; pdb.set_trace()
        else:
            sampling_params = SamplingParams(temperature=0, max_tokens=50000)
            #sampling_params = SamplingParams(temperature=0, max_tokens=500)
            model_outputs = judge_model.chat(all_convos, sampling_params=sampling_params)   

        #model_outputs = judge_model.chat(all_convos, sampling_params=sampling_params)
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
        output_file = output_files[model]
        with open(output_file, "w") as f:
            for output in all_outputs:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")