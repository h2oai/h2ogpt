import traceback
from queue import Queue
from threading import Thread

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=[]):
        super().__init__()
        assert len(stops) == len(encounters), "Number of stops and encounters must match"
        self.encounters = encounters
        self.stops = [stop.to("cuda") for stop in stops]
        self.num_stops = [0] * len(stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stopi, stop in enumerate(self.stops):
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                self.num_stops[stopi] += 1
                if self.num_stops[stopi] >= self.encounters[stopi]:
                    return True
        return False


class Stream(StoppingCriteria):
    """
    This class can be used to callback during generation. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        func (`callable`):
            A callable function to apply on first input in list every iteration of generation
    """
    def __init__(self, func=None):
        self.func = func

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.func is not None:
            # only consume first of multiple responses
            self.func(input_ids[0])
        return False


class InvalidDataError(ValueError):
    pass


class Generator:

    """
    Wrap a function so its callback acts as generator in a thread
    """

    def __init__(self, func, *args, callback=None, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.callback = callback

        self.q = Queue()
        self.sentinel = object()

        self.stop = False

        def val_callback(val):
            if self.stop:
                raise InvalidDataError
            self.q.put(val)

        def wrap_func():
            val = None
            try:
                val = self.func(callback=val_callback, *self.args, **self.kwargs)
            except InvalidDataError as e:
                traceback.print_exc()
                raise
            except Exception:
                # FIXME: ignore exceptions for now
                traceback.print_exc()

            self.q.put(self.sentinel)
            if val is not None and self.callback:
                self.callback(val)

        self.thread = Thread(target=wrap_func)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        val = self.q.get(True, None)
        if val is self.sentinel:
            raise StopIteration
        else:
            return val

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = True
