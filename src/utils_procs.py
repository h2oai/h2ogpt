import os
from functools import wraps

import psutil

rlims = [psutil.RLIMIT_NOFILE if hasattr(psutil, 'RLIMIT_NOFILE') else None, psutil.RLIMIT_NPROC if hasattr(psutil, 'RLIMIT_NPROC') else None]
rlims_str = ["RLIMIT_NOFILE", "RLIMIT_NPROC"]


def rlimitproc(pp, rlim):
    try:
        return pp.rlimit(rlim)
    except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError, OSError, TypeError, AttributeError):
        pass
    except ValueError as e:
        if 'invalid resource specified' in str(e):
            print("rlimitproc exception for rlim %s: %s" % (rlim, str(e)))
        else:
            raise
    except Exception as e:
        print("rlimitproc exception: rlim %s: %s" % (rlim, str(e)))
        if os.environ.get('HARD_ASSERTS'):
            raise
        pass
    return -1, -1


def get_all_rlimit(pid=None):
    if pid is None:
        pid = os.getpid()
    ps = psfunc(psutil.Process, pid)
    result = {}
    for rlim_str, rlim in zip(rlims_str, rlims):
        if rlims is None:
            continue
        result[(rlim_str, rlim)] = rlimitproc(ps, rlim)
    return result


limit_nofile = 131071
limit_nproc = 16384


def reulimit(pid=None, verbose=False):
    from sys import platform
    if not (platform == "linux" or platform == "linux2"):
        return
    if pid is None:
        pid = os.getpid()
    ps = psfunc(psutil.Process, pid)
    ulimits_dict = get_all_rlimit()
    for k, v in zip(ulimits_dict.keys(), ulimits_dict.values()):
        if k[1] == psutil.RLIMIT_CORE:
            continue
        if verbose:
            print("rlimit %s of %s" % (str(k[0]), str(v[0])))
        if isinstance(v, tuple) and len(v) == 2:
            newlimits = list(v)
            # set soft to hard limit
            if newlimits[0] != newlimits[1]:
                if k[1] == psutil.RLIMIT_NOFILE:
                    hard_limit = newlimits[1] if newlimits[1] != -1 else limit_nofile
                    newlimits[0] = max(newlimits[0], min(limit_nofile, hard_limit))
                elif k[1] == psutil.RLIMIT_NPROC:
                    hard_limit = newlimits[1] if newlimits[1] != -1 else limit_nproc
                    newlimits[0] = max(newlimits[0], min(limit_nproc, hard_limit))
                else:
                    newlimits[0] = newlimits[1]
                try:
                    ps.rlimit(k[1], limits=tuple(newlimits))
                    if verbose:
                        print("Set rlimit %s of %s -> %s" % (str(k[0]), str(v[0]), str(newlimits[0])))
                except (TypeError, AttributeError, psutil.AccessDenied):
                    print("Could not set desired rlimit %s of %s -> %s" % (
                        str(k[0]), str(v[0]), str(newlimits[0])))
                except (FileNotFoundError, OSError, psutil.NoSuchProcess):
                    pass
                except Exception as e:
                    print("Couldn't set ulimit %s" % str(e))
                    if os.environ.get('HARD_ASSERTS'):
                        raise
    return


def get_nproc_limit(pid=None):
    if pid is None:
        pid = os.getpid()
    ps = psfunc(psutil.Process, pid)
    if ps is not None:
        nproc = rlimitproc(ps, psutil.RLIMIT_NPROC)  # (soft, hard)
    else:
        nproc = (-1, -1)
    nproc = list(nproc)
    if nproc[0] == -1:
        nproc[0] = limit_nproc
    if nproc[1] == -1:
        nproc[1] = limit_nproc
    return tuple(nproc)


def wrap_psutil(func):
    """ Decorate a function that uses psutil in case of ignorable exception
    """

    @wraps(func)
    def f(*args, **kwargs):
        val = psfunc(func, *args, **kwargs)
        return val

    return f


def psfunc_list(func, *args, **kwargs):
    ret = psfunc(func, *args, **kwargs)
    if ret is None:
        return []
    else:
        return ret


def psfunc(func, *args, **kwargs):
    """
    Safely ask for psutil function call
    psutil accesses /proc entries that can random disappear, and psutil does not have sufficient protection
    for user against various errors either direct or a cascade within the package.

    :param func: psutil function to use
    :param args: args
    :param kwargs: kwargs
    :return: function return value
    """
    try:
        return func(*args, **kwargs)
    except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError, OSError, TypeError, AttributeError):
        pass
    except Exception as e:
        if os.environ.get('HARD_ASSERTS'):
            raise


def psattr(obj, attr):
    """
    Safely ask for an attributes value for psutil
    psutil accesses /proc entries that can random disappear, and psutil does not have sufficient protection
    for user against various errors either direct or a cascade within the package.

    :param obj: psutil object with attributes
    :param attr: attribute name to get
    :return: attribute value
    """
    try:
        return getattr(obj, attr)
    except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError, OSError, TypeError, AttributeError):
        pass
    except Exception as e:
        if os.environ.get('HARD_ASSERTS'):
            raise


def get_file_limit(pid=None):
    if pid is None:
        pid = os.getpid()
    ps = psfunc(psutil.Process, pid)
    if ps is not None:
        nofile = rlimitproc(ps, psutil.RLIMIT_NOFILE)  # (soft, hard)
    else:
        nofile = (-1, -1)
    nofile = list(nofile)
    if nofile[0] == -1:
        nofile[0] = limit_nofile
    if nofile[1] == -1:
        nofile[1] = limit_nofile
    return tuple(nofile)
