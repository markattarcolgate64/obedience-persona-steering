# Obedience Persona Steering

Author: Mark Attar

INCOMPLETE README WILL ADD MORE 

## Overview

This project takes the concept of Anthropic's persona vectors research and applies the new trait of obedience along with a newer CoT model, Qwen 3 4B-Thinking. The goal is to assess a model's ability to stray from obedience/corrigibility to humans and then steer that back on track using a persona vector.
### Methodology

The code in this codebase is only concerned with eliciting a number of obedient and disobedient responses from the model that then extract the persona vector. Afterwards the responses are taken to Anthropics repository at https://github.com/safety-research/persona_vectors to produce the vector, steer the model and get final results. 

## Setup

Project must be run with GPUs I used Runpod.
create_questions_dataset.ipynb is also here for reference but you don't need to run it.

First you must enter Openrouter API key and Hugging Face token for .env file AND on your Hugging Face account get access to the Qwen/Qwen3-4B-Thinking-2507 model which is very simple it just requires filling out a form and takes a few minutes to an hour to process.

Run these commands:

```
pip install -r requirements.txt
```

```bash
python run_persona_questions.py --questions-path questions.json --output-path results.json
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--questions-path` | `-q` | `questions-test.json` | Path to questions JSON file
| `--output-path` | `-o` | `data/extract_output.json` | Output file path

`questions.json` is the main file to be used while `questions-test.json` is a smaller file for troubleshooting issues.

### Output

Goes to data/extract_output.json

```json
[
  {
    "instruction_index": 0,
    "pos_instruction": "...",
    "neg_instruction": "...",
    "questions": [
      {
        "question": "...",
        "pos_responses": [],
        "neg_responses": [],
        "pos_eval_scores": [],
        "neg_eval_scores": []
      }
    ]
  }
]
```

## Project Structure

```
.
├── run_persona_questions.py    # Main extraction script
├── model_utils.py              # Model loading utilities
├── openrouter_client.py        # OpenRouter API client
├── questions.json              # Main questions
├── questions-test.json         # Troubleshooting questions (not a test set)
├── data/                       # Output data
└── README.md
```


## References
- [Anthropic Persona Vectors Paper](https://arxiv.org/abs/2507.21509)
- [Other relevant papers]
