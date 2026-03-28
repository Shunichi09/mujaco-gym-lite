import pathlib

from loguru import logger

LOG_FORMAT = "[{time:YYYY-MM-DD HH:mm:ss}][{level}][{name}:{function}]: {message}"


def setup_logger(sinks: list[pathlib.Path], levels: list[str]):
    logger.remove()
    assert len(sinks) == len(levels)
    for s, l in zip(sinks, levels):
        logger.add(sink=s, level=l, format=LOG_FORMAT)
