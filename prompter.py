from finetune import generate_prompt


class Prompter(object):
    def __init__(self, prompt_type, debug=False, chat=False, stream_output=False):
        self.prompt_type = prompt_type
        data_point = dict(instruction='', input='', output='')
        _, self.pre_response, self.terminate_response = generate_prompt(data_point, prompt_type, chat, False)
        self.debug = debug
        self.chat = chat
        self.stream_output = stream_output

    def generate_prompt(self, data_point):
        reduced = False
        prompt, _, _ = generate_prompt(data_point, self.prompt_type, self.chat, reduced)
        if self.debug:
            print("prompt: ", prompt, flush=True)
        self.prompt = prompt
        return prompt

    def get_response(self, outputs, prompt=None):
        if self.debug:
            print("output: ", '\n\n'.join(outputs), flush=True)
        if prompt is not None:
            self.prompt = prompt

        def clean_response(response):
            meaningless_words = ['<pad>', '</s>', '<|endoftext|>', '”\n']
            for word in meaningless_words:
                response = response.replace(word, "")
            response = response.strip("\n")
            return response

        if self.chat:
            # have to go by length for now
            # FIXME: odd chars like -- as single char can mess this up
            assert len(outputs) == 1, "Cannot have num_return_sequences>1"
            outputs = [outputs[0][len(prompt) - len(self.pre_response):].strip()]
            if self.debug:
                print("outputchat: ", '\n\n'.join(outputs), flush=True)

        multi_output = len(outputs) > 1

        for oi, output in enumerate(outputs):
            output = clean_response(output)
            if self.prompt_type not in [0, '0', 'plain']:
                # find first instance of prereponse
                # prompt sometimes has odd characters, that mutate length,
                # so can't go by length alone
                if self.pre_response:
                    # [1] to avoid repeated pre_response, just take first (after prompt - pre_response for chat)
                    output = output.split(self.pre_response)[1]
                if self.terminate_response:
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


