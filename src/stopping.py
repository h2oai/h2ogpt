import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from enums import PromptType, t5_type


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=[], device="cuda", model_max_length=None):
        super().__init__()
        assert len(stops) % len(encounters) == 0, "Number of stops and encounters must match"
        self.encounters = encounters
        self.stops = [stop.to(device) for stop in stops]
        self.num_stops = [0] * len(stops)
        self.model_max_length = model_max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stopi, stop in enumerate(self.stops):
            current_block = input_ids[0][-len(stop):]
            len_new_tokens = current_block.shape[0]
            if len(stop) <= len_new_tokens and torch.all((stop == input_ids[0][-len(stop):])).item():
                self.num_stops[stopi] += 1
                if self.num_stops[stopi] >= self.encounters[stopi % len(self.encounters)]:
                    # print("Stopped", flush=True)
                    return True
        if self.model_max_length is not None and input_ids[0].shape[0] >= self.model_max_length:
            # critical limit
            return True
        # print("Tokens: %s" % input_ids[0].cpu().numpy(), flush=True)
        # print("Stop Tokens: %s" % [x.cpu().numpy() for x in self.stops], flush=True)
        return False


def get_stopping(prompt_type, prompt_dict, tokenizer, device, base_model,
                 human='<human>:', bot="<bot>:", model_max_length=None):
    # FIXME: prompt_dict unused currently
    user_human_assistant_types = [PromptType.instruct_vicuna.value, str(PromptType.instruct_vicuna.value),
                                 PromptType.instruct_vicuna.name] + \
                                [PromptType.guanaco.value, str(PromptType.guanaco.value),
                                 PromptType.guanaco.name] + \
                                [PromptType.one_shot.value, str(PromptType.one_shot.value),
                                 PromptType.one_shot.name] + \
                                [PromptType.instruct_vicuna2.value, str(PromptType.instruct_vicuna2.value),
                                 PromptType.instruct_vicuna2.name] + \
                                [PromptType.instruct_vicuna3.value, str(PromptType.instruct_vicuna3.value),
                                 PromptType.instruct_vicuna3.name] + \
                                [PromptType.instruct_with_end.value, str(PromptType.instruct_with_end.value),
                                 PromptType.instruct_with_end.name]
    human_bot_types = [PromptType.human_bot.value, str(PromptType.human_bot.value),
                       PromptType.human_bot.name] + \
                      [PromptType.human_bot_orig.value, str(PromptType.human_bot_orig.value),
                       PromptType.human_bot_orig.name]
    all_types = user_human_assistant_types + human_bot_types
    if prompt_type in all_types:
        if prompt_type in human:
            # encounters = [prompt.count(human) + 1, prompt.count(bot) + 1]
            # stopping only starts once output is beyond prompt
            # 1 human is enough to trigger, but need 2 bots, because very first view back will be bot we added
            stop_words = [human, bot, '\n' + human, '\n' + bot]
            encounters = [1, 2]
        elif prompt_type in user_human_assistant_types:
            # even below is not enough, generic strings and many ways to encode
            stop_words = [
                '### Human:',
                """
### Human:""",
                """
### Human:
""",
                """###  Human:  """,
                """###  Human:""",
                '### Assistant:',
                """
### Assistant:""",
                """
### Assistant:
""",
                """###  Assistant:  """,
                """###  Assistant:"""
            ]
            if prompt_type in [PromptType.instruct_vicuna2.value, str(PromptType.instruct_vicuna2.value),
                               PromptType.instruct_vicuna2.name]:
                stop_words = [x.upper() for x in stop_words]
            if prompt_type in [PromptType.instruct_vicuna3.value, str(PromptType.instruct_vicuna3.value),
                               PromptType.instruct_vicuna3.name]:
                stop_words = [x.replace('Human', 'User') for x in stop_words]
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
        if tokenizer._pad_token:  # use hidden variable to avoid annoying properly logger bug
            stop_words_ids = [x[1:] if x[0] == tokenizer.pad_token_id and len(x) > 1 else x for x in stop_words_ids]
        if tokenizer._unk_token:  # use hidden variable to avoid annoying properly logger bug
            stop_words_ids = [x[1:] if x[0] == tokenizer.unk_token_id and len(x) > 1 else x for x in stop_words_ids]
            stop_words_ids = [x[:-1] if x[-1] == tokenizer.unk_token_id and len(x) > 1 else x for x in stop_words_ids]
        if tokenizer._eos_token:  # use hidden variable to avoid annoying properly logger bug
            stop_words_ids = [x[:-1] if x[-1] == tokenizer.eos_token_id and len(x) > 1 else x for x in stop_words_ids]
        if base_model and t5_type(base_model):
            # T5 encoder converts internal double space to space+new line, so fix
            for stopi, stop_word_id in enumerate(stop_words_ids):
                start = stop_word_id[0:1]
                mlist = stop_word_id[1:-1]
                end = stop_word_id[-1:]
                mlist = [tokenizer.vocab[' '] if x == tokenizer.vocab['\n'] else x for x in mlist]
                stop_words_ids[stopi] = torch.tensor(list(start) + list(mlist) + list(end), device=stop_word_id.device)
        # handle fake \n added
        stop_words_ids = [x[1:] if y[0] == '\n' else x for x, y in zip(stop_words_ids, stop_words)]
        # build stopper
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids, encounters=encounters, device=device,
                                 model_max_length=model_max_length)])
    else:
        stopping_criteria = StoppingCriteriaList()
    return stopping_criteria
