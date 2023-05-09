import shutil
import os

import huggingface_hub
import pytest
import torch
from transformers import AutoModelForCausalLM


@pytest.mark.parametrize(
    "model_name, base_model, dataset, training_logs, eval",
    [
        (
                "h2ogpt-oasst1-512-20b",
                "EleutherAI/gpt-neox-20b",
                [
                    "h2oai/openassistant_oasst1",
                    "h2oai/openassistant_oasst1_h2ogpt",
                ],
                [
                    "https://huggingface.co/h2oai/h2ogpt-oasst1-512-20b/blob/main/gpt-neox-20b.openassistant_oasst1.json.6.0_epochs.5a14ea8b3794c0d60476fc262d0a297f98dd712d.1013.zip",
                    "https://huggingface.co/h2oai/h2ogpt-oasst1-512-20b/blob/main/h2ogpt-oasst1-512-20b.h2oaiopenassistant_oasst1_h2ogpt.2_epochs.fcaae7ef70600de8c97c9b38cb3f0075467cdad1.3.zip",
                ],
"""

[eval source code](https://github.com/h2oai/h2ogpt/issues/35#issuecomment-1521119301)

|    Task     |Version| Metric |Value |   |Stderr|
|-------------|------:|--------|-----:|---|-----:|
|hellaswag    |      0|acc     |0.5419|±  |0.0050|
|             |       |acc_norm|0.7259|±  |0.0045|
|boolq        |      1|acc     |0.7125|±  |0.0079|
|piqa         |      0|acc     |0.7742|±  |0.0098|
|             |       |acc_norm|0.7775|±  |0.0097|
|openbookqa   |      0|acc     |0.2800|±  |0.0201|
|             |       |acc_norm|0.4000|±  |0.0219|
|arc_challenge|      0|acc     |0.3993|±  |0.0143|
|             |       |acc_norm|0.4420|±  |0.0145|
|winogrande   |      0|acc     |0.6614|±  |0.0133|
|arc_easy     |      0|acc     |0.7327|±  |0.0091|
|             |       |acc_norm|0.6894|±  |0.0095|
"""
        ),
        # (
        #         "h2ogpt-oasst1-256-20b",
        #         "EleutherAI/gpt-neox-20b",
        #         "h2oai/openassistant_oasst1",
        #         "https://huggingface.co/h2oai/h2ogpt-oasst1-256-20b/blob/main/gpt-neox-20b.openassistant_oasst1.json.1_epochs.5fc91911bc2bfaaf3b6c2de577c4b0ae45a07a4a.18.zip",
        # ),
        (
                "h2ogpt-oasst1-512-12b",
                "EleutherAI/pythia-12b",
                [
                    "h2oai/openassistant_oasst1",
                    "h2oai/openassistant_oasst1_h2ogpt",
                ],
                [
                    "https://huggingface.co/h2oai/h2ogpt-oasst1-512-12b/blob/main/pythia-12b.openassistant_oasst1.json.1_epochs.d45a9d34d34534e076cc6797614b322bd0efb11c.15.zip",
                    "https://huggingface.co/h2oai/h2ogpt-oasst1-512-12b/blob/main/h2ogpt-oasst1-512-12b.h2oaiopenassistant_oasst1_h2ogpt.2_epochs.fcaae7ef70600de8c97c9b38cb3f0075467cdad1.2.zip",
                ],
"""
[eval source code](https://github.com/h2oai/h2ogpt/issues/35#issuecomment-1521119301)

|    Task     |Version| Metric |Value |   |Stderr|
|-------------|------:|--------|-----:|---|-----:|
|arc_easy     |      0|acc     |0.6932|±  |0.0095|
|             |       |acc_norm|0.6225|±  |0.0099|
|openbookqa   |      0|acc     |0.2900|±  |0.0203|
|             |       |acc_norm|0.3740|±  |0.0217|
|winogrande   |      0|acc     |0.6369|±  |0.0135|
|hellaswag    |      0|acc     |0.5140|±  |0.0050|
|             |       |acc_norm|0.6803|±  |0.0047|
|piqa         |      0|acc     |0.7682|±  |0.0098|
|             |       |acc_norm|0.7661|±  |0.0099|
|boolq        |      1|acc     |0.6685|±  |0.0082|
|arc_challenge|      0|acc     |0.3157|±  |0.0136|
|             |       |acc_norm|0.3507|±  |0.0139|
"""
        ),
        # (
        #         "h2ogpt-oig-oasst1-256-12b",
        #         "EleutherAI/pythia-12b-deduped",
        #         "h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v1",
        #         "https://huggingface.co/h2oai/h2ogpt-oig-oasst1-256-12b/blob/main/pythia-12b-deduped.h2ogpt-oig-oasst1-instruct-cleaned-v1.json.1_epochs.5fc91911bc2bfaaf3b6c2de577c4b0ae45a07a4a.17.zip",
        # ),
        (
                "h2ogpt-oig-oasst1-512-6.9b",
                "EleutherAI/pythia-6.9b",
                [
                    "h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v1",
                    "h2oai/openassistant_oasst1_h2ogpt",
                    "h2oai/h2ogpt-fortune2000-personalized",
                    "h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v3",
                ],
                [
                    "https://huggingface.co/h2oai/h2ogpt-oig-oasst1-512-6.9b/blob/main/pythia-6.9b.h2ogpt-oig-oasst1-instruct-cleaned-v1.json.1_epochs.5fc91911bc2bfaaf3b6c2de577c4b0ae45a07a4a.7.zip",
                    "https://huggingface.co/h2oai/h2ogpt-oig-oasst1-512-6.9b/blob/main/h2ogpt-oig-oasst1-512-6.9b.h2oaiopenassistant_oasst1_h2ogpt.2_epochs.e35e2e06e0af2f7dceac2e16e3646c90ccce4ec0.1.zip",
                    "https://huggingface.co/h2oai/h2ogpt-oig-oasst1-512-6.9b/blob/main/h2ogpt-oig-oasst1-512-6.9b.h2oaih2ogpt-oig-oasst1-instruct-cleaned-v3.1_epochs.e48f9debb0d2bd8d866fa5668bbbb51c317c553c.1.zip",
                ],
"""
[eval source code](https://github.com/h2oai/h2ogpt/issues/125#issue-1702311702)

|    Task     |Version| Metric |Value |   |Stderr|
|-------------|------:|--------|-----:|---|-----:|
|arc_easy     |      0|acc     |0.6591|±  |0.0097|
|             |       |acc_norm|0.6178|±  |0.0100|
|arc_challenge|      0|acc     |0.3174|±  |0.0136|
|             |       |acc_norm|0.3558|±  |0.0140|
|openbookqa   |      0|acc     |0.2540|±  |0.0195|
|             |       |acc_norm|0.3580|±  |0.0215|
|winogrande   |      0|acc     |0.6069|±  |0.0137|
|piqa         |      0|acc     |0.7486|±  |0.0101|
|             |       |acc_norm|0.7546|±  |0.0100|
|hellaswag    |      0|acc     |0.4843|±  |0.0050|
|             |       |acc_norm|0.6388|±  |0.0048|
|boolq        |      1|acc     |0.6193|±  |0.0085|
"""
        ),
        # (
        #         "h2ogpt-oig-oasst1-256-20b",
        #         "EleutherAI/gpt-neox-20b",
        #         "h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v1",
        #         "https://huggingface.co/h2oai/h2ogpt-oig-oasst1-256-20b/blob/main/gpt-neox-20b.h2ogpt-oig-oasst1-instruct-cleaned-v1.json.1_epochs.5fc91911bc2bfaaf3b6c2de577c4b0ae45a07a4a.19.zip",
        # ),
    ],
)
def test_create_model_cards(model_name, base_model, dataset, training_logs, eval):
    model_size = model_name.split("-")[-1].upper()
    assert "B" == model_size[-1]
    assert int(model_size[-2]) >= 0
    assert os.path.exists("README-template.md"), "must be running this test from the model dir."
    shutil.rmtree(model_name, ignore_errors=True)
    try:
        repo = huggingface_hub.Repository(
            local_dir=model_name,
            clone_from="h2oai/%s" % model_name,
            skip_lfs_files=True,
            token=True,
        )
        repo.git_pull()
    except:
        print("call 'huggingface_cli login' first and provide access token with write permission")
    model = AutoModelForCausalLM.from_pretrained("h2oai/%s" % model_name,
                                                 local_files_only=False,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto")
    model_arch = str(model)
    model_config = str(model.config)
    with open("README-template.md", "r") as f:
        content = f.read()
        assert "<<MODEL_NAME>>" in content
        content = content.replace("<<MODEL_NAME>>", model_name)

        assert "<<MODEL_SIZE>>" in content
        content = content.replace("<<MODEL_SIZE>>", model_size[:-1])

        assert "<<BASE_MODEL>>" in content
        content = content.replace("<<BASE_MODEL>>", f"[{base_model}](https://huggingface.co/{base_model})")

        assert "<<DATASET>>" in content
        assert "<<DATASET_NAME>>" in content
        if not isinstance(dataset, list):
            dataset = [dataset]
        content = content.replace("<<DATASET>>", " and ".join([f"[{d}](https://huggingface.co/datasets/{d})" for d in dataset]))
        content = content.replace("<<DATASET_NAME>>", "\n".join([f"- {d}" for d in dataset]))

        assert "<<MODEL_ARCH>>" in content
        content = content.replace("<<MODEL_ARCH>>", model_arch)

        assert "<<MODEL_CONFIG>>" in content
        content = content.replace("<<MODEL_CONFIG>>", model_config)

        assert "<<TRAINING_LOGS>>" in content
        if not isinstance(training_logs, list):
            training_logs = [training_logs]
        content = content.replace("<<TRAINING_LOGS>>", " and ".join(f"[zip]({t})" for t in training_logs))
        content = content.replace("<<MODEL_EVAL>>", eval)

        assert "<<" not in content
        assert ">>" not in content

    with open(os.path.join(model_name, "README.md"), "w") as f:
        f.write(content)
    try:
        repo.commit("Update README.md")
        repo.push_to_hub()
    except Exception as e:
        print(str(e))
