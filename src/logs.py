# INspired from Son Nguyen Kim gist here: https://gist.github.com/nguyenkims/e92df0f8bd49973f0c94bddf36ed7fd0
import logging
import sys
from datetime import datetime as dt
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE =  "fwhleng-" + dt.now().strftime(
    "logs_%Y-%m-%dT%H-%M-%S.log"
)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False

    return logger

def find_logger_basefilename(self, logger):
    """Finds the logger base filename(s) currently there is only one
    """
    log_file = None
    parent = logger.__dict__['parent']
    if parent.__class__.__name__ == 'RootLogger':
        # this is where the file name lives
        for h in logger.__dict__['handlers']:
            if h.__class__.__name__ == 'TimedRotatingFileHandler':
                log_file = h.baseFilename
    else:
        log_file = self._find_logger_basefilename(parent)

    return log_file  
