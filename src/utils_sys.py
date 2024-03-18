import sys
import os
import traceback


class StreamProxy:
    def __init__(self, original_stream):
        self.__original_stream = original_stream

    def write(self, *args, **kwargs):
        try:
            return self.__original_stream.write(*args, **kwargs)
        except ValueError as e:
            if str(e) == "I/O operation on closed file":
                self.handle_closed_file_error("write")
            else:
                raise

    def flush(self, *args, **kwargs):
        try:
            return self.__original_stream.flush(*args, **kwargs)
        except ValueError as e:
            if str(e) == "I/O operation on closed file":
                self.handle_closed_file_error("flush")
            else:
                raise

    def handle_closed_file_error(self, operation):
        message = f"Warning: Attempt to {operation} to a closed stream has been ignored."
        if os.getenv("HARD_ASSERTS"):
            raise ValueError("I/O operation on closed file.")
        else:
            # Use sys.__stderr__ to ensure the message is seen even if stderr is closed/redirected.
            print(message, file=sys.__stderr__)

    def close(self):
        # Print the stack trace to the original stream
        traceback.print_stack(file=self.__original_stream)
        message = "Warning: Attempt to close stream has been ignored."

        if os.getenv("HARD_ASSERTS"):
            # Raise an exception if HARD_ASSERTS is set
            raise Exception("Attempt to close stream intercepted.")
        else:
            print(message, file=self.__original_stream)

    def __getattr__(self, name):
        return getattr(self.__original_stream, name)

    def __setattr__(self, name, value):
        is_hard_asserts = os.getenv("HARD_ASSERTS")
        if name in {"_StreamProxy__original_stream"}:
            super().__setattr__(name, value)
        else:
            traceback.print_stack(file=self.__original_stream)
            message = "Modification attempt of protected stream attribute has been logged."
            if is_hard_asserts:
                raise AttributeError(f"{message} Modification of '{name}' is not allowed on StreamProxy instances.")
            else:
                print(message, file=self.__original_stream)


class FinalizeStream:
    def __init__(self, proxy):
        self.__proxy = proxy

    def __setattr__(self, key, value):
        is_hard_asserts = os.getenv("HARD_ASSERTS")
        if key in {"_FinalizeStream__proxy"}:
            super().__setattr__(key, value)
        else:
            # Use sys.__stdout__ to ensure output if sys.stderr/stdout is protected
            traceback.print_stack(file=sys.__stdout__)
            message = "Stream protection violation has been logged."
            if is_hard_asserts:
                raise AttributeError(f"{message} Modification of '{key}' is prohibited.")
            else:
                print(message, file=sys.__stdout__)

    def __getattr__(self, item):
        return getattr(self.__proxy, item)


def protect_stream(stream_name):
    if stream_name == "stdout":
        sys.stdout = FinalizeStream(StreamProxy(sys.stdout))
    elif stream_name == "stderr":
        sys.stderr = FinalizeStream(StreamProxy(sys.stderr))
    else:
        raise ValueError("Unsupported stream name. Choose 'stdout' or 'stderr'.")


def protect_stdout_stderr():
    # Protect both stdout and stderr at the start of your application
    protect_stream("stdout")
    protect_stream("stderr")
