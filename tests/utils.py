from concurrent.futures import ProcessPoolExecutor


def call_subprocess_onetask(func, args=None, kwargs=None):
    if isinstance(args, list):
        args = tuple(args)
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result()
