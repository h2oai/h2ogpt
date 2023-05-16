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
                 sanitize_bot_response=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_prompter = use_prompter
        self.prompt_text = None
        if self.use_prompter:
            from prompter import Prompter
            self.prompter = Prompter(prompt_type, debug=debug, chat=chat, stream_output=stream_output)
        else:
            self.prompter = None
        self.sanitize_bot_response = sanitize_bot_response

    def preprocess(self, prompt_text, prefix="", handle_long_generation=None, **generate_kwargs):
        prompt_text = prompt.format(instruction=prompt_text)
        self.prompt_text = prompt_text
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
        return super()._forward(model_inputs, **generate_kwargs)
