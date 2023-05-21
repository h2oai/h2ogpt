import pytest

from generate import main


@pytest.mark.parametrize(
    "base_model",
    [
        "h2oai/h2ogpt-oig-oasst1-512-6.9b",
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
def test_score_eval(base_model):
    main(
        base_model=base_model,
        chat=False,
        stream_output=False,
        gradio=False,
        eval_sharegpt_prompts_only=500,
        eval_sharegpt_as_output=False,
        num_beams=2,
        infer_devices=False,
    )
