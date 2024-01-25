from mmengine.config import read_base

with read_base():
    # from .datasets.siqa.siqa_gen import siqa_datasets
    # from .datasets.winograd.winograd_ppl import winograd_datasets
    from .datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from .models.openai.gpt_3_5_turbo import models as gpt_3_5_turbo
    # from .models.openai.gpt_4 import models as gpt_4

datasets = [*hellaswag_datasets, ]
models = [*gpt_3_5_turbo, ]
