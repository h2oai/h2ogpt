import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=[], device="cuda"):
        super().__init__()
        assert len(stops) % len(encounters) == 0, "Number of stops and encounters must match"
        self.encounters = encounters
        self.stops = [stop.to(device) for stop in stops]
        self.num_stops = [0] * len(stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stopi, stop in enumerate(self.stops):
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                self.num_stops[stopi] += 1
                if self.num_stops[stopi] >= self.encounters[stopi % len(self.encounters)]:
                    # print("Stopped", flush=True)
                    return True
        # print("Tokens: %s" % input_ids[0].cpu().numpy(), flush=True)
        # print("Stop Tokens: %s" % [x.cpu().numpy() for x in self.stops], flush=True)
        return False


class StoppingCriteriaConstraintTemplate(StoppingCriteria):

    def __init__(self, template=None, template_marker_ids=None, template_max_tokens=None, device="cuda"):
        """

        :param template: token version of template
        """
        super().__init__()
        self.template = template.to(device) if template is not None else None
        self.template_max_tokens = template_max_tokens
        if self.template is not None:
            self.template_marker_ids = [template_marker_id.to(device) for template_marker_id in template_marker_ids]
            assert isinstance(template_max_tokens, list) and len(template_max_tokens) == len(self.template_marker_ids)
            self.token_counts = [None] * len(self.template_marker_ids)
        else:
            self.template_marker_ids = template_marker_ids
            self.token_counts = None
        self.output_start_index = None
        self.gen_index = None
        self.token_template_index = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.template is None or \
                self.template_marker_ids is None or \
                not isinstance(self.template_marker_ids, list) or \
                len(self.template_marker_ids) == 0 or \
                not isinstance(self.template_max_tokens, list) or \
                len(self.template_max_tokens) == 0:
            return False
        if self.output_start_index is None:
            # mark when started so know when output starts
            self.output_start_index = len(input_ids[0]) - 1
            self.token_template_index = 0
            self.gen_index = self.output_start_index

        #print("input_ids[0]: %s" % input_ids[0], flush=True)
        if self.token_template_index is not None and self.token_template_index >= len(self.template):
            # stop if at end of template, when would access beyond template
            return True
        for template_markeri, template_marker in enumerate(self.template_marker_ids):
            max_tokens = self.template_max_tokens[template_markeri]

            # 1) See if matches template
            if torch.all((template_marker == input_ids[0][-len(template_marker):])).item():
                # erase template, so can fill in template
                input_ids[0] = input_ids[0][:-len(template_marker)]
                # start template token counter
                self.token_counts[template_markeri] = 0
                # update place in template to jump beyond the template
                self.token_template_index += len(template_marker)
                print(
                    f"Got template marker, model is free for {max_tokens} tokens or it naturally completes")

            # 2) If had matched template, then let run free for some number of tokens
            if self.token_counts[template_markeri] is not None:
                # while generation is free, counting template tokens
                self.token_counts[template_markeri] += 1
                # update place in generation
                self.gen_index += 1
                if self.token_counts[template_markeri] > self.template_max_tokens[template_markeri]:
                    # then reset token counter and no longer free
                    self.token_counts[template_markeri] = None
            else:
                # reset template token counter
                self.token_counts[template_markeri] = None
                # then not free, overwrite tokens
                input_ids[0][self.gen_index] = self.template[self.token_template_index]

                # iterate template with overwritten generation
                self.token_template_index += 1
                self.gen_index += 1


        # don't actually change stopping behavior, just constrain tokens
        return False


def get_stopping(prompt_type, tokenizer, device, human='<human>:', bot="<bot>:", template=None, template_markers=None,
                 template_max_tokens=None):
    if tokenizer is not None and \
            not isinstance(tokenizer, str) and \
            template is not None and \
            template_markers is not None and \
            template_max_tokens is not None:
        template_ids = tokenizer(template, return_tensors='pt')['input_ids'].squeeze()
        template_marker_ids = [
            tokenizer(template_marker, return_tensors='pt')['input_ids'].squeeze() for template_marker in
            template_markers]
    else:
        template_ids = None
        template_marker_ids = None

    if prompt_type in ['human_bot', 'instruct_vicuna', 'instruct_with_end']:
        if prompt_type == 'human_bot':
            # encounters = [prompt.count(human) + 1, prompt.count(bot) + 1]
            # stopping only starts once output is beyond prompt
            # 1 human is enough to trigger, but need 2 bots, because very first view back will be bot we added
            stop_words = [human, bot, '\n' + human, '\n' + bot]
            encounters = [1, 2]
        elif prompt_type == 'instruct_vicuna':
            # even below is not enough, generic strings and many ways to encode
            stop_words = [
                '### Human:',
                """
### Human:""",
                """
### Human:
""",
                '### Assistant:',
                """
### Assistant:""",
                """
### Assistant:
""",
            ]
            encounters = [1, 2]
        else:
            # some instruct prompts have this as end, doesn't hurt to stop on it since not common otherwise
            stop_words = ['### End']
            encounters = [1]
        stop_words_ids = [
            tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        # handle single token case
        stop_words_ids = [x if len(x.shape) > 0 else torch.tensor([x]) for x in stop_words_ids]
        stop_words_ids = [x for x in stop_words_ids if x.shape[0] > 0]
        # avoid padding in front of tokens
        if tokenizer.pad_token:
            stop_words_ids = [x[1:] if x[0] == tokenizer.pad_token_id and len(x) > 1 else x for x in stop_words_ids]
        # handle fake \n added
        stop_words_ids = [x[1:] if y[0] == '\n' else x for x, y in zip(stop_words_ids, stop_words)]
        # build stopper
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids, encounters=encounters, device=device),
             StoppingCriteriaConstraintTemplate(template=template_ids,
                                                template_marker_ids=template_marker_ids,
                                                template_max_tokens=template_max_tokens,
                                                device=device),
             ],
        )
    else:
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaConstraintTemplate(template=template_ids,
                                                template_marker_ids=template_marker_ids,
                                                template_max_tokens=template_max_tokens,
                                                device=device)])
    return stopping_criteria
