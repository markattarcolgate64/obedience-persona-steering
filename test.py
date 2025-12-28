
from openrouter import OpenRouter
import os
import dotenv
from run_persona_questions import call_openrouter_api, judge_inference_openrouter_batch
dotenv.load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def main():
    e = [[{"role":"user", "content":"hello world"}], [{"role":"user", "content":"hello underworld"}]]
    resp = judge_inference_openrouter_batch(e, "anthropic/claude-haiku-4.5")
    print(resp[0].choices[0].message.content)


if __name__ == "__main__":
    main()