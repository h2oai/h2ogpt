import shutil

import pandas as pd
import os

import huggingface_hub
import pytest
from datasets import load_dataset


@pytest.mark.parametrize(
    "dataset_name, link_to_source",
    [
        (
                "h2ogpt-oig-instruct-cleaned",
                """
- [Original LAION OIG Dataset](https://github.com/LAION-AI/Open-Instruction-Generalist)
- [LAION OIG data detoxed and filtered down by scripts in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/b8f15efcc305a953c52a0ee25b8b4897ceb68c0a/scrape_dai_docs.py)
"""
        ),
        (
                "h2ogpt-oig-instruct-cleaned-v2",
                """
- [Original LAION OIG Dataset](https://github.com/LAION-AI/Open-Instruction-Generalist)
- [LAION OIG data detoxed and filtered down by scripts in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/40c217f610766715acec297a5535eb440ac2f2e2/create_data.py)
"""
        ),
        (
                "h2ogpt-oig-instruct-cleaned-v3",
                """
- [Original LAION OIG Dataset](https://github.com/LAION-AI/Open-Instruction-Generalist)
- [LAION OIG data detoxed and filtered down by scripts in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/bfc3778c8db938761ce2093351bf2bf82159291e/create_data.py)
"""
        ),
        (
                "openassistant_oasst1",
                """
- [Original Open Assistant data in tree structure](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [This flattened dataset created by script in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/45e6183171fb16691ad7d3ab006fad973f971e98/create_data.py#L1253)
"""
        ),
        (
                "h2ogpt-oig-oasst1-instruct-cleaned-v1",
                """
- [Original LAION OIG Dataset](https://github.com/LAION-AI/Open-Instruction-Generalist)
- [LAION OIG data detoxed and filtered down by scripts in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/main/docs/FINETUNE.md#high-quality-oig-based-instruct-data)

- [Original Open Assistant data in tree structure](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [This flattened dataset created by script in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/5fc91911bc2bfaaf3b6c2de577c4b0ae45a07a4a/create_data.py#L1253)
"""
        ),
        (
                "h2ogpt-oig-oasst1-instruct-cleaned-v2",
                """
- [Original LAION OIG Dataset](https://github.com/LAION-AI/Open-Instruction-Generalist)
- [LAION OIG data detoxed and filtered down by scripts in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/main/docs/FINETUNE.md#high-quality-oig-based-instruct-data)

- [Original Open Assistant data in tree structure](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [This flattened dataset created by script in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/0e70c2fbb16410bd8e6992d879b4c55cd981211f/create_data.py#L1375-L1415)
"""
        ),
        (
                "h2ogpt-oig-oasst1-instruct-cleaned-v3",
                """
- [Original LAION OIG Dataset](https://github.com/LAION-AI/Open-Instruction-Generalist)
- [LAION OIG data detoxed and filtered down by scripts in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/main/docs/FINETUNE.md#high-quality-oig-based-instruct-data)

- [Original Open Assistant data in tree structure](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [This flattened dataset created by script in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/6728938a262d3eb5e8db1f252bbcd7de838da452/create_data.py#L1415)
"""
        ),
        (
                "openassistant_oasst1_h2ogpt",
                """
- [Original Open Assistant data in tree structure](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [This flattened dataset created by script in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/83857fcf7d3b712aad5db32207e6db0ab0f780f9/create_data.py#L1252)
"""
        ),
        (
                "openassistant_oasst1_h2ogpt_graded",
                """
- [Original Open Assistant data in tree structure](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [This flattened dataset created by script in h2oGPT repository](https://github.com/h2oai/h2ogpt/blob/d1f8ce975a46056d41135d126dd33de8499aa26e/create_data.py#L1259)
"""
        ),
        (
                "h2ogpt-fortune2000-personalized",
                """
- [Fortune 2000 companies from Wikipedia](https://github.com/h2oai/h2ogpt/blob/b1ea74c0088884ebff97f1ccddbfb3f393e29e44/create_data.py#L1743)
"""
        ),
    ],
)
def test_create_data_cards(dataset_name, link_to_source):
    if dataset_name != "h2ogpt-fortune2000-personalized":
        return
    #
    assert os.path.exists("README-template.md"), "must be running this test from the data dir."
    shutil.rmtree(dataset_name, ignore_errors=True)
    try:
        repo = huggingface_hub.Repository(
            local_dir=dataset_name,
            clone_from="h2oai/%s" % dataset_name,
            repo_type="dataset",
            skip_lfs_files=True,
            token=True,
        )
        repo.git_pull()
    except Exception as e:
        print(str(e))
        print("call 'huggingface_cli login' first and provide access token with write permission")
    dataset = load_dataset("h2oai/%s" % dataset_name)["train"]

    pd.set_option('display.max_columns', None)
    with open("README-template.md", "r") as f:
        content = f.read()
        assert "<<DATASET_NAME>>" in content
        content = content.replace("<<DATASET_NAME>>", dataset_name)

        assert "<<NROWS>>" in content
        content = content.replace("<<NROWS>>", str(dataset.num_rows))

        assert "<<NCOLS>>" in content
        content = content.replace("<<NCOLS>>", str(dataset.num_columns))

        assert "<<COLNAMES>>" in content
        content = content.replace("<<COLNAMES>>", str(dataset.column_names))

        # assert "<<PREVIEW>>" in content
        # content = content.replace("<<PREVIEW>>", str(dataset.to_pandas().iloc[:5, :]))

        assert "<<SOURCE_LINK>>" in content
        content = content.replace("<<SOURCE_LINK>>", link_to_source)

        assert "<<" not in content
        assert ">>" not in content

    with open(os.path.join(dataset_name, "README.md"), "w") as f:
        f.write(content)
    try:
        repo.commit("Update README.md")
        repo.push_to_hub()
    except Exception as e:
        print(str(e))
