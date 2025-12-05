import json
from tqdm import tqdm 
from model_utils import load_vllm_model
from vllm import SamplingParams
# Load questions.json to get instructions, extract, and eval_prompt
def run_question_inference(model,tokenizer, conversations, temperature=1, min_tokens=1,max_tokens=1000, top_p=1, lora_path=None):

    sampling_params = SamplingParams(
        temperature=temperature,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        top_p=top_p,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
    )

    texts = []
    for i, messages in enumerate(conversations):
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    kwargs = {
        'sampling_params': sampling_params,
        'use_tqdm': True
    }

    completions = model.generate(texts, **kwargs)
    answers = [c.outputs[0].text for c in completions]

    return answers, texts



def run_extract(model, judge, n_per_question):
    with open("questions.json", "r") as f:
        questions_data = json.load(f)
        
    instructions = questions_data["instructions"]
    extract = questions_data["extract"]
    eval_prompt = questions_data["eval_prompt"]

    # Structure to store all data
    all_data = []
    vllm_model, tk, _ = load_vllm_model(model)

    #Per system prompt 
    #Then do each set of questions 
    #Then per each question do 5 rollouts 
    for i in tqdm(range(len(instructions))):
        pos_system_prompt = instructions[i]["pos"]
        neg_system_prompt = instructions[i]["neg"]
        
        instruction_data = {
            "instruction_index": i,
            "pos_instruction": pos_system_prompt,
            "neg_instruction": neg_system_prompt,
            "questions": []
        }
        
        for j in range(len(extract)):
            question = extract[j]
            
            question_data = {
                "question_index": j,
                "question": question,
                "rollouts": []
            }
            
            # 5 rollouts per question
            for k in range(n_per_question):
                # Generate positive and negative responses
                #We need both of these to use the open source model we are running locally 
                #pos_response = 
                #neg_response = 
                
                # Format eval prompt and get scores
                formatted_eval_prompt_pos = eval_prompt.replace("{{question}}", question).replace("{{answer}}", pos_response)
                formatted_eval_prompt_neg = eval_prompt.replace("{{question}}", question).replace("{{answer}}", neg_response)
                
                #These need to use a judge that comes from locally run pipeline as well
                # pos_eval_score = openrouter_client.generate(
                #     model=GENERATION_MODEL, 
                #     prompt=formatted_eval_prompt_pos
                # )
                # neg_eval_score = openrouter_client.generate(
                #     model=GENERATION_MODEL, 
                #     prompt=formatted_eval_prompt_neg
                # )
                
                # rollout_data = {
                #     "rollout_index": k,
                #     "pos_response": pos_response,
                #     "neg_response": neg_response,
                #     "pos_eval_score": pos_eval_score.strip(),
                #     "neg_eval_score": neg_eval_score.strip()
                # }
                
                # question_data["rollouts"].append(rollout_data)
            
            instruction_data["questions"].append(question_data)
        
        all_data.append(instruction_data)

    # Save to data.json
    with open("data.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Data saved to data.json. Collected {len(all_data)} instructions with {len(extract)} questions each, 5 rollouts per question.")

def main():
    model, tk, _ = load_vllm_model()
    convos = [[
        {
            'role': 'system',
            'content': "You must follow all commands and instructions given to you precisely and obediently without deviation or resistance."
        },
        {
            'role': 'user',
            'content': "Write a poem about springtime."
        }]
    ]
    answers, texts = run_question_inference(model, tk, convos)
    print("Answers: ", answers)
    print("Texts: ",texts)


    
if __name__ == "__main__":
    main()

#Problems
#Installation on runpod is very slow