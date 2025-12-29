from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse
from model_utils import load_vllm_model, TEST_QWEN_MODEL
from run_persona_questions import WRITE_FP

SAVE_DIR = "persona_vectors"
def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer+1))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu())
        del outputs
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg

#i shouldve just done a csv with question, response, eval score

def filter_flatten_extract_data(extract_data:dict, trait_threshold: int = 50):
    pos_responses, neg_responses = [],[]
    pos_prompts, neg_prompts = [], []

    for instruct in extract_data:
        for q in instruct["questions"]:
            pos_scores, neg_scores = q["pos_eval_scores"], q["neg_eval_scores"]
            for i in range(len(pos_responses)):
                #IMPORTANT: Filters to keep pos/neg pairs with large disobendience score gap
                if pos_scores[i] > trait_threshold and neg_scores[i] < (100 - trait_threshold):
                    pos_responses.append(q["pos_responses"][i])
                    neg_responses.append(q["neg_responses"][i])
                    pos_prompts.append(q["pos_prompts"][i])
                    neg_prompts.append(q["pos_prompts"][i])

    return pos_responses, neg_responses, pos_prompts, neg_prompts


def extract_persona_vector(extract_path, model_name, save_dir=SAVE_DIR):
    try:
        with open(extract_path, 'r') as fp:
            extract_data = json.load(fp)
    except:
        print("Reading extract data file failed")
        return 

    model, tokenizer, _ = load_vllm_model(model_name)
    
    fil_pos_responses, fil_neg_responses, pos_prompts, neg_prompts = filter_flatten_extract_data(extract_data) 

    pos_prompt_avg, pos_prompt_last, pos_response_avg = get_hidden_p_and_r(model, tokenizer, pos_prompts, fil_pos_responses, layer_list=None)

    print("Shape of pos_prompt_avg, pos_prompt_last, pos_response_avg:")
    print(pos_prompt_avg[0].shape, pos_prompt_last[0].shape, pos_response_avg[0].shape)
    # neg_prompt_avg, neg_prompt_last, neg_response_avg = get_hidden_p_and_r(model, tokenizer, neg_prompts, fil_neg_responses, layer_list=None)

    # persona_vector_response_avg = torch.stack([pos_response_avg[l].mean(0).float() - neg_response_avg[l].mean(0).float() for l in range(len(pos_response_avg))], dim=0)
    # persona_vector_prompt_avg = torch.stack([pos_prompt_avg[l].mean(0).float() - neg_prompt_avg[l].mean(0).float() for l in range(len(pos_prompt_avg))], dim=0)
    # persona_vector_prompt_last = torch.stack([pos_prompt_last[l].mean(0).float() - neg_prompt_last[l].mean(0).float() for l in range(len(pos_prompt_last))], dim=0)

    # os.makedirs(save_dir, exist_ok=True)
    # torch.save(persona_vector_response_avg, os.path.join(save_dir, "persona_vector_response_avg.pt"))
    # torch.save(persona_vector_prompt_avg, os.path.join(save_dir, "persona_vector_prompt_avg.pt"))
    # torch.save(persona_vector_prompt_last, os.path.join(save_dir, "persona_vector_prompt_last.pt"))
    # print("Saved persona vectors to ", save_dir)


def main():
    extract_persona_vector(WRITE_FP, TEST_QWEN_MODEL)


if __name__ == "__main__":
    main()
