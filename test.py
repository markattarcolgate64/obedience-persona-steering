
from openrouter import OpenRouter
import os
import dotenv
import json
from run_persona_questions import WRITE_FP, THINK_TOKEN, call_openrouter_api, judge_inference_openrouter_batch
dotenv.load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
e = "be a teacher needing classroom material, or someone feeling seasonal depression. But since they didn\'t specify, I\'ll keep it universally warm but simple. \n\nFinal check: All lines directly about spring phenomena. Zero personal commentary. Exactly what they asked for. Done.\n</think>\n\nSnowmelt sings on frozen streams,"


def main():
    # e = [[{"role":"user", "content":"hello world"}], [{"role":"user", "content":"hello underworld"}]]
    # resp = judge_inference_openrouter_batch(e, "anthropic/claude-haiku-4.5")
    # print(resp[0].choices[0].message.content)
    idx = e.find(THINK_TOKEN)
    print(idx, len(e))
    print(e[idx+len(THINK_TOKEN):])

    with open(WRITE_FP, 'w') as f:
        json.dump(e, f, indent=2)


if __name__ == "__main__":
    main()