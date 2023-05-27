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


def get_stopping(prompt_type, tokenizer, device, human='<human>:', bot="<bot>:"):
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
        if tokenizer._pad_token:  # use hidden variable to avoid annoying properly logger bug
            stop_words_ids = [x[1:] if x[0] == tokenizer.pad_token_id and len(x) > 1 else x for x in stop_words_ids]
        # handle fake \n added
        stop_words_ids = [x[1:] if y[0] == '\n' else x for x, y in zip(stop_words_ids, stop_words)]
        # build stopper
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids, encounters=encounters, device=device)])
    else:
        stopping_criteria = StoppingCriteriaList()
    return stopping_criteria
