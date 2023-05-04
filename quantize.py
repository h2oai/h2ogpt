import os

from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "h2oai/h2ogpt-oig-oasst1-512-6.9b"
quantized_model_dir = "h2ogpt-oig-oasst1-512-6.9b-4bit"

# os.makedirs(quantized_model_dir, exist_ok=True)


def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys contains "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(examples, use_triton=True)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)

    # load quantized model, currently only support cpu or single gpu
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=True)

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to("cuda:0"))[0]))

    # or you can also use pipeline
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda:0")
    print(pipeline("auto-gptq is")[0]["generated_text"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
