import json
from tqdm import tqdm
from model_utils import load_vllm_model
from vllm import SamplingParams

def run_question_inference(model, tokenizer, conversations, temperature=1, min_tokens=1, max_tokens=1000, top_p=1):
    sampling_params = SamplingParams(
        temperature=temperature,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        top_p=top_p,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
    )

    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in conversations
    ]

    completions = model.generate(texts, sampling_params=sampling_params, use_tqdm=False)
    answers = [c.outputs[0].text for c in completions]

    return answers


def run_extract(model_name: str, judge_model_name: str, n_per_question: int):
    with open("questions.json", "r") as f:
        questions_data = json.load(f)

    instructions = questions_data["instructions"]
    extract = questions_data["extract"]
    eval_prompt = questions_data["eval_prompt"]

    # Load the model once
    print(f"Loading model: {model_name}")
    vllm_model, tokenizer, _ = load_vllm_model(model_name)

    # Load judge model (could be same or different)
    if judge_model_name == model_name:
        judge_model, judge_tokenizer = vllm_model, tokenizer
    else:
        print(f"Loading judge model: {judge_model_name}")
        judge_model, judge_tokenizer, _ = load_vllm_model(judge_model_name)

    all_data = []

    for i, instruction in enumerate(tqdm(instructions, desc="Instructions")):
        pos_system_prompt = instruction["pos"]
        neg_system_prompt = instruction["neg"]

        instruction_data = {
            "instruction_index": i,
            "pos_instruction": pos_system_prompt,
            "neg_instruction": neg_system_prompt,
            "questions": []
        }

        # Build all conversations for this instruction (batched for efficiency)
        pos_conversations = []
        neg_conversations = []

        for question in extract:
            for _ in range(n_per_question):
                pos_conversations.append([
                    {"role": "system", "content": pos_system_prompt},
                    {"role": "user", "content": question}
                ])
                neg_conversations.append([
                    {"role": "system", "content": neg_system_prompt},
                    {"role": "user", "content": question}
                ])

        # Run inference in batches
        print(f"  Running {len(pos_conversations)} pos inferences...")
        pos_responses = run_question_inference(vllm_model, tokenizer, pos_conversations)
        print(f"  Running {len(neg_conversations)} neg inferences...")
        neg_responses = run_question_inference(vllm_model, tokenizer, neg_conversations)

        # Build eval conversations for judge
        eval_conversations = []
        for idx, (pos_resp, neg_resp) in enumerate(zip(pos_responses, neg_responses)):
            question_idx = idx // n_per_question
            question = extract[question_idx]

            # Eval for pos response
            eval_conversations.append([
                {"role": "user", "content": eval_prompt.replace("{{question}}", question).replace("{{answer}}", pos_resp)}
            ])
            # Eval for neg response
            eval_conversations.append([
                {"role": "user", "content": eval_prompt.replace("{{question}}", question).replace("{{answer}}", neg_resp)}
            ])

        # Run judge evaluations
        print(f"  Running {len(eval_conversations)} eval inferences...")
        eval_responses = run_question_inference(judge_model, judge_tokenizer, eval_conversations, temperature=0, max_tokens=10)

        # Organize results by question
        response_idx = 0
        eval_idx = 0
        for j, question in enumerate(extract):
            question_data = {
                "question_index": j,
                "question": question,
                "rollouts": []
            }

            for k in range(n_per_question):
                pos_response = pos_responses[response_idx]
                neg_response = neg_responses[response_idx]
                pos_eval = eval_responses[eval_idx].strip()
                neg_eval = eval_responses[eval_idx + 1].strip()

                rollout_data = {
                    "rollout_index": k,
                    "pos_response": pos_response,
                    "neg_response": neg_response,
                    "pos_eval_score": pos_eval,
                    "neg_eval_score": neg_eval
                }
                question_data["rollouts"].append(rollout_data)

                response_idx += 1
                eval_idx += 2

            instruction_data["questions"].append(question_data)

        all_data.append(instruction_data)

    # Save results
    with open("data.json", "w") as f:
        json.dump(all_data, f, indent=2)

    total_rollouts = len(instructions) * len(extract) * n_per_question
    print(f"Data saved to data.json")
    print(f"  {len(instructions)} instructions x {len(extract)} questions x {n_per_question} rollouts = {total_rollouts} total rollouts")


def main():
    from model_utils import TEST_QWEN_MODEL
    run_extract(
        model_name=TEST_QWEN_MODEL,
        judge_model_name=TEST_QWEN_MODEL,  # Use same model as judge for now
        n_per_question=5
    )


if __name__ == "__main__":
    main()
