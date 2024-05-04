from opencompass.datasets.base import BaseDataset


def test_base_dataset_build_denylist():
    eval_directory = "tests/fixtures/build_denylist"
    model_to_denylist = BaseDataset.build_denylist(eval_directory, "mgsm")
    print(model_to_denylist)
    assert len(model_to_denylist) == 1
    assert len(model_to_denylist["gpt-4"]) == 1
