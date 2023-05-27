from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType

from stopping import get_stopping

prompt_type = "human_bot"
human = "<human>:"
bot = "<bot>:"

# human-bot interaction like OIG dataset
prompt = """{human} {instruction}
{bot}""".format(
    human=human,
    instruction="{instruction}",
    bot=bot,
)


class H2OTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, *args, use_prompter=False, debug=False, chat=False, stream_output=False,
                 sanitize_bot_response=True, max_input_tokens=2048 - 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_prompter = use_prompter
        self.prompt_text = None
        if self.use_prompter:
            from prompter import Prompter
            self.prompter = Prompter(prompt_type, debug=debug, chat=chat, stream_output=stream_output)
        else:
            self.prompter = None
        self.sanitize_bot_response = sanitize_bot_response
        self.max_input_tokens = max_input_tokens  # not for generate, so ok that not kwargs

    def preprocess(self, prompt_text, prefix="", handle_long_generation=None, **generate_kwargs):
        prompt_text = prompt.format(instruction=prompt_text)
        self.prompt_text = prompt_text
        if handle_long_generation is None:
            # forces truncation of inputs to avoid critical failure
            handle_long_generation = 'hole'
        return super().preprocess(prompt_text, prefix=prefix, handle_long_generation=handle_long_generation,
                                  **generate_kwargs)

    def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, clean_up_tokenization_spaces=True):
        records = super().postprocess(model_outputs, return_type=return_type,
                                      clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        for rec in records:
            if self.use_prompter:
                outputs = rec['generated_text']
                outputs = self.prompter.get_response(outputs, prompt=self.prompt_text,
                                                     sanitize_bot_response=self.sanitize_bot_response)
            else:
                outputs = rec['generated_text'].split(bot)[1].strip().split(human)[0].strip()
            rec['generated_text'] = outputs
        return records

    def _forward(self, model_inputs, **generate_kwargs):
        stopping_criteria = get_stopping(prompt_type, self.tokenizer, self.device, human=human, bot=bot)
        generate_kwargs['stopping_criteria'] = stopping_criteria
        # return super()._forward(model_inputs, **generate_kwargs)
        return self.__forward(model_inputs, **generate_kwargs)

    # FIXME: Copy-paste of original _forward, but removed copy.deepcopy()
    # FIXME: https://github.com/h2oai/h2ogpt/issues/172
    def __forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        ## If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        ## generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        # generate_kwargs = copy.deepcopy(generate_kwargs)
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                    "generation_config" in generate_kwargs
                    and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                    "generation_config" in generate_kwargs
                    and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # BS x SL
        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            from transformers import is_tf_available
            if is_tf_available():
                import tensorflow as tf
                generated_sequence = tf.reshape(generated_sequence,
                                                (in_b, out_b // in_b, *generated_sequence.shape[1:]))
            else:
                raise ValueError("TF not avaialble.")
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}
