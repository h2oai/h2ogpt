from finetune import generate_prompt


class Prompter(object):
    def __init__(self, prompt_type, debug=False, chat=False, stream_output=False, repeat_penalty=True,
                 allowed_repeat_line_length=10):
        self.prompt_type = prompt_type
        data_point = dict(instruction='', input='', output='')
        _, self.pre_response, self.terminate_response, self.chat_sep = \
            generate_prompt(data_point, prompt_type, chat, False)
        self.debug = debug
        self.chat = chat
        self.stream_output = stream_output
        self.repeat_penalty = repeat_penalty
        self.allowed_repeat_line_length = allowed_repeat_line_length

    def generate_prompt(self, data_point):
        reduced = False
        prompt, _, _, _ = generate_prompt(data_point, self.prompt_type, self.chat, reduced)
        if self.debug:
            print("prompt: ", prompt, flush=True)
        self.prompt = prompt
        return prompt

    def get_response(self, outputs, prompt=None, sanitize_bot_response=True):
        if isinstance(outputs, str):
            outputs = [outputs]
        if self.debug:
            print("output: ", '\n\n'.join(outputs), flush=True)
        if prompt is not None:
            self.prompt = prompt

        def clean_response(response):
            meaningless_words = ['<pad>', '</s>', '<|endoftext|>']
            for word in meaningless_words:
                response = response.replace(word, "")
            if sanitize_bot_response:
                from better_profanity import profanity
                response = profanity.censor(response)
            response = response.strip("\n")
            return response

        def clean_repeats(response):
            lines = response.split('\n')
            new_lines = []
            [new_lines.append(line) for line in lines if
             line not in new_lines or len(line) < self.allowed_repeat_line_length]
            if self.debug and len(lines) != len(new_lines):
                print("cleaned repeats: %s %s" % (len(lines), len(new_lines)), flush=True)
            response = '\n'.join(new_lines)
            return response

        multi_output = len(outputs) > 1

        for oi, output in enumerate(outputs):
            if self.prompt_type in [0, '0', 'plain']:
                output = clean_response(output)
            else:
                # find first instance of prereponse
                # prompt sometimes has odd characters, that mutate length,
                # so can't go by length alone
                if self.pre_response:
                    outputi = output.find(prompt)
                    if outputi >= 0:
                        output = output[outputi + len(prompt):]
                        allow_terminate = True
                    else:
                        # subtraction is risky due to space offsets sometimes, so only do if necessary
                        output = output[len(prompt) - len(self.pre_response):]
                        # [1] to avoid repeated pre_response, just take first (after prompt - pre_response for chat)
                        if self.pre_response in output:
                            output = output.split(self.pre_response)[1]
                            allow_terminate = True
                        else:
                            if output:
                                print("Failure of parsing or not enough output yet: %s" % output, flush=True)
                            allow_terminate = False
                else:
                    allow_terminate = True
                    output = output[len(prompt):]
                # clean after subtract prompt out, so correct removal of pre_response
                output = clean_response(output).strip()
                if self.repeat_penalty:
                    output = clean_repeats(output).strip()
                if self.terminate_response and allow_terminate:
                    finds = []
                    for term in self.terminate_response:
                        finds.append(output.find(term))
                    finds = [x for x in finds if x >= 0]
                    if len(finds) > 0:
                        termi = finds[0]
                        output = output[:termi].strip()
                    else:
                        output = output.strip()
                else:
                    output = output.strip()
            if multi_output:
                # prefix with output counter
                output = "\n=========== Output %d\n\n" % (1 + oi) + output
                if oi > 0:
                    # post fix outputs with seperator
                    output += '\n'
            outputs[oi] = output
        # join all outputs, only one extra new line between outputs
        output = '\n'.join(outputs)
        if self.debug:
            print("outputclean: ", '\n\n'.join(outputs), flush=True)
        return output
