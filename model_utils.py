import os 
import dotenv
from numpy import dtype
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import vllm

dotenv.load_dotenv()
TEST_QWEN_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
#Just write a fn to use huggingface transformers to load models using the model name, then we can use it 
#We'll use the deepseek new model for now 
def load_tokenizer(path):
    tk = AutoTokenizer.from_pretrained(path)
    tk.pad_token = tk.eos_token
    tk.pad_token_id = tk.eos_token_id
    tk.padding_side = "left"
    return tk 


def load_model(model_name: str, dtype=torch.bfloat16):
    if not os.path.exists(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto"
        )
        tk = load_tokenizer(model_name)
        return model, tk 

def load_vllm_model(model_name: str = TEST_QWEN_MODEL):
    model = vllm.LLM(
            model=model_name,
            enable_prefix_caching=True,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=32,
            gpu_memory_utilization=0.9,
            max_model_len=30000,
            tensor_parallel_size=2
        )
    tok = model.get_tokenizer()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    
    return model, tok, None
    
def main():
    m, tk, _ = load_vllm_model(TEST_QWEN_MODEL)
    print(type(tk))

if __name__ == "__main__":
    main()


#vllm issue, model naming issue with hf, package issue,  
        
#
        

        
    