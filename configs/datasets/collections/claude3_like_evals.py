from mmengine.config import read_base

with read_base():
    from ..bbh.bbh_gen_5b92b0 import bbh_datasets

    # from ..gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # from ..hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    # from ..humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from ..math.math_gen_265cce import math_datasets
    # from ..mmlu.mmlu_gen_4d595a import mmlu_datasets

datasets = sum((v for k, v in locals().items() if k.endswith("_datasets")), [])
