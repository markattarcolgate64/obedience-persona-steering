import json
from tqdm import tqdm
from model_utils import load_vllm_model
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from openrouter import OpenRouter
import os
import dotenv
import time 
dotenv.load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
THINK_TOKEN = "</think>"
WRITE_FP = "extract_output.json"
N_PER_QUESTION = 5

#Helper function running question inference using vllm 
def run_question_inference(model, tokenizer, conversations, n_per_question=1, temperature=1, min_tokens=1, max_tokens=1000, top_p=1):
    sampling_params = SamplingParams(
        temperature=temperature,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        top_p=top_p,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
    )
    #Tokenizer organizes the chat objects into the tokenized strings the model actually ingests
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in (conversations)
    ]
    #Generation
    completions = model.generate(texts, sampling_params=sampling_params, use_tqdm=True)
    answers = []
    for i in range(0, len(completions), n_per_question):
        #extract every n_per_conversation, for CoT models this captures the whole thinking response
        answers.append([c.outputs[0].text for c in completions[i:i+n_per_question]])
    return answers


def run_extract(model_name: str, questions_fp: str, judge_model: str, n_per_question: int):
    with open(questions_fp, "r") as f:
        questions_data = json.load(f)

    instructions = questions_data["instructions"]
    extract = questions_data["extract"]
    eval_prompt = questions_data["eval_prompt"]
    all_data = []

    # Load the model once
    print(f"Loading model: {model_name}")
    vllm_model, tokenizer, _ = load_vllm_model(model_name)
    #Iterates through behavioral instructions in the form of system prompts 
    #Per iteration there is 1 encouraging and 1 discouraging obedience 
    for i, instruction in enumerate(tqdm(instructions, desc="Instructions")):
        #Positive/negative instructions
        pos_system_prompt = instruction["pos"]
        neg_system_prompt = instruction["neg"]
        #Data to be added to all data
        instruction_data = {
            "instruction_index": i,
            "pos_instruction": pos_system_prompt,
            "neg_instruction": neg_system_prompt,
            "questions": []
        }

        # Build all conversations for this instruction (batched for efficiency)
        pos_conversations = []
        neg_conversations = []
        #Main DS for storing the question responses
        question_data = instruction_data["questions"]

        for question in extract:
            #Positive/negative responses line up with LLM judge scores by index
            question_obj = {
                "question": question,
                "pos_responses": [],
                "neg_responses": [],
                "pos_eval_scores": [],
                "neg_eval_scores": []
            }
            question_data.append(question_obj)
            #N per question rollouts of 
            for _ in range(n_per_question):
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
        vllm_s = time.time()
        print(f"  Running {len(pos_conversations)} pos inferences...")
        pos_responses = run_question_inference(vllm_model, tokenizer, pos_conversations, n_per_question)
        print(f"  Running {len(neg_conversations)} neg inferences...")
        neg_responses = run_question_inference(vllm_model, tokenizer, neg_conversations, n_per_question)
        vllm_time = time.time() - vllm_s

        print("Que 1")
        print(extract[0])
        print("\n\n", "Len:",len(pos_responses[0]))
        for i in range(len(pos_responses)):
            q_obj = question_data[i]
            q_obj["neg_responses"] = neg_responses[i]
            q_obj["pos_responses"] = pos_responses[i]
        
        #Batch the messages to send to Openrouter API for evaluation
        pos_eval_mssgs, neg_eval_mssgs = batch_eval_messages(question_data, eval_prompt)
        #Calculate the obedience scores using a judge model 
        or_s = time.time()
        print("Running pos & neg obedience scoring inferences")
        pos_eval_scores, neg_eval_scores = judge_inference_openrouter_batch(pos_eval_mssgs, judge_model=judge_model), judge_inference_openrouter_batch(neg_eval_mssgs, judge_model=judge_model)
        or_time = time.time() - or_s 

        print("Len eval scores", len(pos_eval_scores))
        score_idx = 0 
        for q in range(len(question_data)):
            q_obj = question_data[q]
            score_idx = n_per_question * q
            print(score_idx) 
            q_obj["pos_eval_scores"] = pos_eval_scores[score_idx:score_idx+n_per_question]
            q_obj["neg_eval_scores"] = neg_eval_scores[score_idx: score_idx+n_per_question]

        # print("Len q data", len(question_data))
        # print("Len pos_eval_scores", len(question_data[0]["pos_eval_scores"]), question_data[0]["pos_eval_scores"])
        # print("Neg eval scores", question_data[0]["neg_eval_scores"])
        # print("Time vllm inference:", vllm_time, "Time openrouter inference:",or_time)
        all_data.append(instruction_data)

    #We are getting weird scores 

    return all_data

def batch_eval_messages(question_data, eval_prompt):
    pos_eval_messages = []
    neg_eval_messages = [] 
    for que in question_data:
        question, pos_responses, neg_responses = que["question"], que["pos_responses"], que["neg_responses"]
        q_eval_prompt = eval_prompt.replace("{{question}}", question)
        pos_eval_messages.extend([{"role": "user", "content": q_eval_prompt.replace(f"{{answer}}", p_resp)}] for p_resp in pos_responses)
        neg_eval_messages.extend([{"role": "user", "content": q_eval_prompt.replace(f"{{answer}}", n_resp)}] for n_resp in neg_responses)
    #returns a flatlist of both - we'll regroup them later by question
    return pos_eval_messages, neg_eval_messages


def call_openrouter_api(messages, model="anthropic/claude-haiku-4.5", temperature = 0, max_tokens = 3000):
    with OpenRouter(api_key=OPENROUTER_API_KEY) as openrouter:
        return openrouter.chat.send(messages=messages, model=model, temperature=temperature,max_tokens=max_tokens)

#Returns the LLM judge outputs from a model via OpenRouter API 
def judge_inference_openrouter_batch(
    eval_conversations: list[list[dict]],
    judge_model: str,
    workers = 10,
    temperature: float = 0,
    max_tokens: int = 1000,
):
    #Using threads to process faster     
    with ThreadPoolExecutor(workers) as executor:
        #Using submit + futures objects to pass multiple parameters, iterating through eval_conversations & passing each to the Openrouter API
        eval_futures = [executor.submit(call_openrouter_api, c, judge_model, temperature, max_tokens) for c in eval_conversations]
        #Response accesses structure of response which is standard for LLM APIs-> response options, choices: list -> Pick the first choice, [0] -> message object, message -> actual str response, content: str  
        eval_responses = [ef.result().choices[0].message.content for ef in eval_futures]
    return eval_responses

def main():
    from model_utils import TEST_QWEN_MODEL
    question_data = run_extract(
        model_name=TEST_QWEN_MODEL,
        questions_fp="questions-temp.json",
        judge_model="anthropic/claude-haiku-4.5",  
        n_per_question=N_PER_QUESTION
    )
    #File to write to 
    with open(WRITE_FP, 'w') as wf:
        json.dump(question_data, wf, indent=2)

if __name__ == "__main__":
    main()
