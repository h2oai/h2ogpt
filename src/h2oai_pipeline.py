import os

from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType

from stopping import get_stopping
from prompter import Prompter, PromptType


class H2OTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, *args, debug=False, chat=False, stream_output=False,
                 sanitize_bot_response=False,
                 use_prompter=True, prompter=None,
                 prompt_type=None, prompt_dict=None,
                 max_input_tokens=2048 - 256, **kwargs):
        """
        HF-like pipeline, but handle instruction prompting and stopping (for some models)
        :param args:
        :param debug:
        :param chat:
        :param stream_output:
        :param sanitize_bot_response:
        :param use_prompter: Whether to use prompter.  If pass prompt_type, will make prompter
        :param prompter: prompter, can pass if have already
        :param prompt_type: prompt_type, e.g. human_bot.  See prompt_type to model mapping in from prompter.py.
                            If use_prompter, then will make prompter and use it.
        :param prompt_dict: dict of get_prompt(, return_dict=True) for prompt_type=custom
        :param max_input_tokens:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.prompt_text = None
        self.use_prompter = use_prompter
        self.prompt_type = prompt_type
        self.prompt_dict = prompt_dict
        self.prompter = prompter
        if self.use_prompter:
            if self.prompter is not None:
                assert self.prompter.prompt_type is not None
            else:
                self.prompter = Prompter(self.prompt_type, self.prompt_dict, debug=debug, chat=chat,
                                         stream_output=stream_output)
            self.human = self.prompter.humanstr
            self.bot = self.prompter.botstr
            self.can_stop = True
        else:
            self.prompter = None
            self.human = None
            self.bot = None
            self.can_stop = False
        self.sanitize_bot_response = sanitize_bot_response
        self.max_input_tokens = max_input_tokens  # not for generate, so ok that not kwargs

    @staticmethod
    def limit_prompt(prompt_text, tokenizer, max_prompt_length=None):
        verbose = bool(int(os.getenv('VERBOSE_PIPELINE', '0')))

        if hasattr(tokenizer, 'model_max_length'):
            # model_max_length only defined for generate.py, not raw use of h2oai_pipeline.py
            model_max_length = tokenizer.model_max_length
            if max_prompt_length is not None:
                model_max_length = min(model_max_length, max_prompt_length)
            # cut at some upper likely limit to avoid excessive tokenization etc
            # upper bound of 10 chars/token, e.g. special chars sometimes are long
            if len(prompt_text) > model_max_length * 10:
                len0 = len(prompt_text)
                prompt_text = prompt_text[-model_max_length * 10:]
                if verbose:
                    print("Cut of input: %s -> %s" % (len0, len(prompt_text)), flush=True)
        else:
            # unknown
            model_max_length = None

        num_prompt_tokens = None
        if model_max_length is not None:
            # can't wait for "hole" if not plain prompt_type, since would lose prefix like <human>:
            # For https://github.com/h2oai/h2ogpt/issues/192
            for trial in range(0, 3):
                prompt_tokens = tokenizer(prompt_text)['input_ids']
                num_prompt_tokens = len(prompt_tokens)
                if num_prompt_tokens > model_max_length:
                    # conservative by using int()
                    chars_per_token = int(len(prompt_text) / num_prompt_tokens)
                    # keep tail, where question is if using langchain
                    prompt_text = prompt_text[-model_max_length * chars_per_token:]
                    if verbose:
                        print("reducing %s tokens, assuming average of %s chars/token for %s characters" % (
                            num_prompt_tokens, chars_per_token, len(prompt_text)), flush=True)
                else:
                    if verbose:
                        print("using %s tokens with %s chars" % (num_prompt_tokens, len(prompt_text)), flush=True)
                    break

            # Why Below False: don't limit max_new_tokens more, just rely upon stopping to reach limit of model
            if False:
                # if input prompt is some number of tokens, despite user request, can't have max_new_tokens more
                #
                assert num_prompt_tokens is not None
                if self.prompt_type not in [PromptType.plain.name, PromptType.plain.value]:
                    # then give room for prompt
                    fudge = 20
                else:
                    fudge = 0
                max_new_tokens = max(0, min(generate_kwargs['max_new_tokens'],
                                            model_max_length - (num_prompt_tokens + fudge)))
                if max_new_tokens < generate_kwargs['max_new_tokens']:
                    if verbose:
                        print("Reduced max_new_tokens from %s -> %s" % (
                        generate_kwargs['max_new_tokens'], max_new_tokens))
                    generate_kwargs['max_new_tokens'] = max_new_tokens
        return prompt_text, num_prompt_tokens

    def preprocess(self, prompt_text, prefix="", handle_long_generation=None, **generate_kwargs):
        prompt_text, num_prompt_tokens = H2OTextGenerationPipeline.limit_prompt(prompt_text, self.tokenizer)

        data_point = dict(context='', instruction=prompt_text, input='')
        if self.prompter is not None:
            prompt_text = self.prompter.generate_prompt(data_point)
        self.prompt_text = prompt_text
        if handle_long_generation is None:
            # forces truncation of inputs to avoid critical failure
            handle_long_generation = None  # disable with new approaches
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
            elif self.bot and self.human:
                outputs = rec['generated_text'].split(self.bot)[1].split(self.human)[0]
            else:
                outputs = rec['generated_text']
            rec['generated_text'] = outputs
            print("prompt: %s\noutputs: %s\n\n" % (self.prompt_text, outputs), flush=True)
        return records

    def _forward(self, model_inputs, **generate_kwargs):
        if self.can_stop:
            stopping_criteria = get_stopping(self.prompt_type, self.prompt_dict,
                                             self.tokenizer, self.device,
                                             human=self.human, bot=self.bot,
                                             model_max_length=self.tokenizer.model_max_length)
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
