from mmengine.config import read_base
with read_base():
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.qwen.hf_qwen_14b_chat import models as hf_qwen_14b_chat
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
    from .models.baichuan.hf_baichuan2_7b_chat import models as hf_baichuan2_7b
    from .models.hf_internlm.hf_internlm2_chat_7b import models as internlm2_7b
    from .models.hf_internlm.hf_internlm2_chat_20b import models as internlm2_20b
    from .datasets.subjective.subjective_cmp.subjective_corev2 import subjective_datasets

datasets = [*subjective_datasets]

from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import Corev2Summarizer
models = [*internlm2_7b, *internlm2_20b]

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=500),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)


judge_model =    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-7b-chat-hf',
        path="Qwen/Qwen-7B-Chat",
        tokenizer_path='Qwen/Qwen-7B-Chat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,),
        pad_token_id=151643,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        mode='m2n',
        max_task_size=500,
        base_models = [*internlm2_7b],
        compare_models = [*internlm2_20b]
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(
            type=SubjectiveEvalTask,
        judge_cfg=judge_model
        )),
)
work_dir = 'outputs/corev2/'

summarizer = dict(
    type=Corev2Summarizer,
    match_method='smart',
)