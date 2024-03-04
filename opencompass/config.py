import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

EVALUATIONS_WORK_DIR = os.getenv('EVALUATIONS_WORK_DIR')
EVALUATION_DATA_DIR = os.getenv('EVALUATION_DATA_DIR')

WANDB_API_KEY = os.getenv('WANDB_API_KEY')

VALID_DATASET_ABBR = ["hellaswag", "bbh", "ARC_c", "ARC_e", "gsm8k", "humaneval",
                      "mbpp", "mmlu", "piqa", "siqa", "race", "squadv2", "superglue",
                      "winogrande", "xsum"]
WITH_SUBSETS = ["bbh", "mmlu", "superglue"]
