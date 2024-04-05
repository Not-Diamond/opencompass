from mmengine.config import read_base

from opencompass.models import NotDiamond, NotDiamondModelSelect
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():

    from .datasets.collections.claude3_like_evals import datasets
    from .summarizers.medium import summarizer

# Refered config.models.gemini_pro
models = [
    dict(
        abbr="nd_oob",
        type=NotDiamondModelSelect,  # NotDiamondModelSelect, NotDiamond
        path="N/A",
        key="your keys",  # The key will be obtained from Environment, but you can write down your key here as well
        url="your url",
        query_per_second=4,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        temperature=1,
    )
]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(type=LocalRunner, max_num_workers=10, task=dict(type=OpenICLInferTask)),
)