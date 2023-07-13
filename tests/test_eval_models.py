import os
import pytest

from tests.utils import wrap_test_forked


@pytest.mark.skipif(not os.getenv('BENCHMARK'),
                    reason="Only valid on sufficiently large system and not normal part of testing."
                           "  Instead used to get eval scores for all models.")
@pytest.mark.parametrize(
    "base_model",
    [
        "h2oai/h2ogpt-oasst1-falcon-40b",
        "h2oai/h2ogpt-oig-oasst1-512-6_9b",
        "h2oai/h2ogpt-oig-oasst1-512-12b",
        "h2oai/h2ogpt-oig-oasst1-512-20b",
        "h2oai/h2ogpt-oasst1-512-12b",
        "h2oai/h2ogpt-oasst1-512-20b",
        "h2oai/h2ogpt-gm-oasst1-en-1024-20b",
        "databricks/dolly-v2-12b",
        "h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2",
        "ehartford/WizardLM-7B-Uncensored",
        "ehartford/WizardLM-13B-Uncensored",
        "AlekseyKorshuk/vicuna-7b",
        "TheBloke/stable-vicuna-13B-HF",
        "decapoda-research/llama-7b-hf",
        "decapoda-research/llama-13b-hf",
        "decapoda-research/llama-30b-hf",
        "junelee/wizard-vicuna-13b",
        "openaccess-ai-collective/wizard-mega-13b",
    ]
)
@wrap_test_forked
def test_score_eval(base_model):
    from src.gen import main
    main(
        base_model=base_model,
        chat=False,
        stream_output=False,
        gradio=False,
        eval_prompts_only_num=500,
        eval_as_output=False,
        num_beams=2,
        use_gpu_id=False,
    )


@pytest.mark.skipif(not os.getenv('FALCONS'), reason="download purpose")
@pytest.mark.parametrize(
    "base_model",
    [
        "OpenAssistant/falcon-7b-sft-top1-696",
        "OpenAssistant/falcon-7b-sft-mix-2000",
        "h2oai/h2ogpt-oasst1-falcon-40b",
        "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1",
        "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2",
        "h2oai/h2ogpt-gm-oasst1-multilang-2048-falcon-7b",
        "OpenAssistant/falcon-40b-sft-top1-560",
        "OpenAssistant/falcon-40b-sft-mix-1226",
    ]
)
@wrap_test_forked
def test_get_falcons(base_model):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    t = AutoTokenizer.from_pretrained(base_model,
                                      use_fast=False,
                                      padding_side="left",
                                      trust_remote_code=True,
                                      use_auth_token=True,
                                      )
    assert t is not None
    m = AutoModelForCausalLM.from_pretrained(base_model,
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                             )
    assert m is not None


@pytest.mark.skipif(not os.getenv('LLAMA'), reason="LLaMa conversion")
@wrap_test_forked
def test_get_landmark_llama():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LlamaForCausalLM, LlamaTokenizer
    m = LlamaForCausalLM.from_pretrained("epfml/landmark-attention-llama7b-wdiff")
    t = LlamaTokenizer.from_pretrained("epfml/landmark-attention-llama7b-wdiff")
    assert m is not None and t is not None

    os.system("""
#
# step 1, convert llama to HF format
pip install protobuf==3.19.0
source ~/.bashrc.mamba
mamba create -n trans
conda activate trans
conda install python=3.10 -y

git clone https://github.com/epfml/landmark-attention.git
pip install fire datasets
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
pip install torch accelerate sentencepiece protobuf==3.19.0
# below requires LLaMa weights
python src/transformers/models/llama/convert_llama_weights_to_hf.py     --input_dir /data/jon/LLaMA --model_size 7B --output_dir llama_7B
#
# step 2, make landmark model (change hash if updated)
mkdir -p epfml/landmark-attention-llama7b-wdiff
cd epfml/landmark-attention-llama7b-wdiff
ln -s ~/.cache/huggingface/hub/models--epfml--landmark-attention-llama7b-wdiff/snapshots/050562871ac72723b4ab674f0392b02cd9609842/* .
cd ../../
python ../landmark-attention/llama/weight_diff.py recover --path_raw llama_7B --path_diff epfml/landmark-attention-llama7b-wdiff --path_tuned landmark_llama_7b
""")
