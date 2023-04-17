import traceback
from queue import Queue
from threading import Thread
import collections.abc

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=[]):
        super().__init__()
        assert len(stops) % len(encounters) == 0, "Number of stops and encounters must match"
        self.encounters = encounters
        self.stops = [stop.to("cuda") for stop in stops]
        self.num_stops = [0] * len(stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stopi, stop in enumerate(self.stops):
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                self.num_stops[stopi] += 1
                if self.num_stops[stopi] >= self.encounters[stopi % len(self.encounters)]:
                    return True
        # print("Tokens: %s" % input_ids[0].cpu().numpy(), flush=True)
        # print("Stop Tokens: %s" % [x.cpu().numpy() for x in self.stops], flush=True)
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


class CallbackToGenerator(collections.abc.Generator):
    """
    A generator wrapper for a function that invokes a callback multiple times.

    Calling `send` on the generator emits a value from one callback, and returns
    the next.

    Note this starts a background thread
    """

    def __init__(self, func, *args, callback=None, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.callback = callback

        self._ready_queue = Queue(1)
        self._done_queue = Queue(1)
        self._done_holder = [False]

        # local to avoid reference cycles
        ready_queue = self._ready_queue
        done_queue = self._done_queue
        done_holder = self._done_holder

        def val_callback(value):
            done_queue.put((False, value))
            cmd, val = ready_queue.get()
            if cmd == 'send':
                return val
            elif cmd == 'throw':
                raise val
            else:
                assert False  # pragma: no cover

        def thread_func():
            while True:
                cmd, val = ready_queue.get()
                if cmd == 'send' and val is not None:
                    done_queue.put((True, TypeError("can't send non-None value to a just-started generator")))
                    continue
                break
            try:
                if cmd == 'throw':
                    raise val
                ret = func(callback=val_callback, **self.kwargs)
                raise StopIteration(ret) if ret is not None else StopIteration
            except BaseException as e:
                done_holder[0] = True
                done_queue.put((True, e))

        self._thread = Thread(target=thread_func)
        self._thread.start()

    def _put(self, *args):
        if self._done_holder[0]:
            raise StopIteration
        self._ready_queue.put(args)
        is_exception, val = self._done_queue.get()
        if is_exception:
            try:
                raise val
            finally:
                # prevent val's traceback containing a reference cycle
                del val
        else:
            return val

    def send(self, value):
        return self._put('send', value)

    def throw(self, exc):
        return self._put('throw', exc)

    def close(self):
        try:
            self.throw(GeneratorExit)
        except StopIteration:
            self._thread.join()
        except GeneratorExit:
            self._thread.join()
        except BaseException:
            self._thread.join()
            raise
        else:
            # yielded again, can't clean up the thread
            raise RuntimeError('Task with callback ignored GeneratorExit')

    def __del__(self):
        self.close()
