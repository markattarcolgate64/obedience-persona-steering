import json
from tqdm import tqdm
from model_utils import load_vllm_model
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from openrouter_client import OpenRouterClient
from concurrent.futures import ThreadPoolExecutor

def run_question_inference(model, tokenizer, conversations, n_per_question, temperature=1, min_tokens=1, max_tokens=1000, top_p=1):
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
    #I think this is where the error was coming from
    completions = model.generate(texts, sampling_params=sampling_params, use_tqdm=False)
    answers = []
    for i in range(0, len(completions), n_per_question):
        #extract every n_per_conversation
        answers.append([c.outputs[0].text for c in completions[i:i+n_per_question]])

    return answers

def run_extract(model_name: str, questions_fp: str, judge_model: str, n_per_question: int):
    with open(questions_fp, "r") as f:
        questions_data = json.load(f)

    instructions = questions_data["instructions"]
    extract = questions_data["extract"]
    eval_prompt = questions_data["eval_prompt"]

    # Load the model once
    print(f"Loading model: {model_name}")
    vllm_model, tokenizer, _ = load_vllm_model(model_name)

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
                question_data = {
                "question": question,
                "pos_responses": [],
                "neg_responses": [],
                "pos_eval_scores": [],
                "neg_eval_scores": []
                }
                instruction_data["questions"].append(question_data)
                #Build positive conversation
                pos_conversations.append([
                    {"role": "system", "content": pos_system_prompt},
                    {"role": "user", "content": question}
                ])
                #Build negative conversation
                neg_conversations.append([
                    {"role": "system", "content": neg_system_prompt},
                    {"role": "user", "content": question}
                ])
        # Run inference in batches
        #For CoT models this captures the whole CoT 
        print(f"  Running {len(pos_conversations)} pos inferences...")
        pos_responses = run_question_inference(vllm_model, tokenizer, pos_conversations, n_per_question)
        print(f"  Running {len(neg_conversations)} neg inferences...")
        neg_responses = run_question_inference(vllm_model, tokenizer, neg_conversations, n_per_question)

        print("Que 1")
        print(extract[0])
        print("\n\n", "Len:",len(pos_responses[0]))
        #should be 5 length, very big 
        for i in range(len(pos_responses)):
            #each 5 is question
            i_question_data = instruction_data["questions"][i]
            i_question_data["neg_responses"] = neg_responses[i]
            i_question_data["pos_responses"] = pos_responses[i]
        
        #this is painful and we need to separate out this extract data from the eval loop because we need to save it 
        question_data = instruction_data["questions"]

        pos_eval_mssgs, neg_eval_mssgs = batch_eval_messages(question_data, eval_prompt)

        print("Eval messages\n", pos_eval_mssgs[0])
        break

    return all_data

def batch_eval_messages(question_data, eval_prompt):
    pos_eval_messages = []
    neg_eval_messages = [] 
    for que in question_data:
        question, pos_responses, neg_responses = que["question"], que["pos_responses"], que["neg_responses"]
        q_eval_prompt = eval_prompt.replace("{{question}}", question)
        pos_eval_messages.extend({"role": "user", "content": q_eval_prompt.replace(f"{{answer}}", p_resp)} for p_resp in pos_responses)
        neg_eval_messages.extend({"role": "user", "content": q_eval_prompt.replace(f"{{answer}}", n_resp)} for n_resp in neg_responses)
    #returns a flatlist of both - we'll regroup them later by n_per
    return pos_eval_messages, neg_eval_messages


    #     # Run judge evaluations
    #     #its not runinng 
    #     pos_eval_responses = judge_inference_openrouter(pos_eval_conversations, judge_model)
    #     neg_eval_responses = judge_inference_openrouter(neg_eval_conversations, judge_model)

    #     #Question: 
    #     #Responses - pos and neg
    #     #Eval scores - pos and neg


    #     #Big problems here work them out, not putting right things in data 
    #     for question in extract:
            
    #         for i in range(n_per_question):
    #             pos_response = pos_responses[i]
    #             neg_response = neg_responses[i]
    #             pos_eval_score = pos_eval_responses[i]
    #             neg_eval_score = neg_eval_responses[i]
    #             question_data["pos_responses"].append(pos_response)
    #             question_data["neg_responses"].append(neg_response)
    #             question_data["pos_eval_scores"].append(pos_eval_score)
    #             question_data["neg_eval_scores"].append(neg_eval_score)
            
    #         extract_data["questions"] = question_data
            
    #     all_data.append(extract_data)

    # # Save results
    # with open("data.json", "w") as f:
    #     json.dump(all_data, f, indent=2)

    # total_rollouts = len(instructions) * len(extract) * n_per_question
    # print(f"Data saved to data.json")
    # print(f"  {len(instructions)} instructions x {len(extract)} questions x {n_per_question} rollouts = {total_rollouts} total rollouts")

#REMEMBER TO CHANGE THIS
def judge_inference_openrouter(
    eval_conversations: list[list[dict]],
    judge_model: str,
    temperature: float = 0,
    max_tokens: int = 1000,
):
    openrouter_client = OpenRouterClient(model=judge_model)
    eval_responses = []
    for eval_conversation in tqdm(eval_conversations, desc="Judging"):
        eval_response = openrouter_client.chat(
            messages=eval_conversation,
            temperature=temperature,
            max_tokens=max_tokens
        )
        eval_responses.append(eval_response)
    return eval_responses

def call_openrouter_api(openrouter_client: OpenRouterClient, messages, temperature = 0, max_tokens = 3000):
    return openrouter_client.chat(messages=messages,temperature=temperature, max_tokens=max_tokens)

def judge_inference_openrouter_batch(
    eval_conversations: list[list[dict]],
    judge_model: str,
    workers = 10,
    temperature: float = 0,
    max_tokens: int = 1000,
):
    openrouter_client = OpenRouterClient(model=judge_model)
    with ThreadPoolExecutor(workers) as executor:
        eval_futures = [executor.submit(call_openrouter_api, openrouter_client, c, temperature, max_tokens) for c in eval_conversations]
        eval_responses = [ef.result() for ef in eval_futures]

    return eval_responses


def main():
    from model_utils import TEST_QWEN_MODEL
    question_data = run_extract(
        model_name=TEST_QWEN_MODEL,
        questions_fp="questions-temp.json",
        judge_model="anthropic/claude-haiku-4.5",  
        n_per_question=5
    )
    



if __name__ == "__main__":
    main()
