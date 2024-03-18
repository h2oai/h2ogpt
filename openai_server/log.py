import logging

# create logger
logger = logging.getLogger('__name__')
level = logging.INFO
logger.setLevel(level)

# ----> console info messages require these lines <----
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(level)

# add ch to logger
logger.addHandler(ch)
