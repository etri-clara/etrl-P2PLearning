import logging
from argparse import ArgumentParser
from enum import IntEnum
import os
import os.path as path
import sys

LOGFORMAT = "[%(asctime)s {%(filename)s:%(lineno)d}] <%(levelname)s> %(message)s"

class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def name_of(cls, name: str):
        name = name.upper()
        for e in cls:
            if e.name == name:
                return e
        raise ValueError(f"Unknown {cls.__name__} name: {name}")

    @classmethod
    def value_of(cls, value: int):
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown {cls.__name__} value: {value}")

    def __str__(self):
        return self.name.upper()


def config_logger(logformat: str = LOGFORMAT,
                  loglevel: LogLevel = LogLevel.INFO,
                  logfile: str = None):

    if logfile is None:
        logging.basicConfig(level=loglevel, format=logformat,
                            stream=sys.stderr)  
    else:
        logdir = path.dirname(logfile)
        os.makedirs(logdir, exist_ok=True)
        logging.basicConfig(level=loglevel, format=logformat,
                            filename=logfile)  

