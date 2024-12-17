import logging
from time import perf_counter
from colorama import Fore, Style


class FileFormatter(logging.Formatter):

    def __init__(self, fmt="[%(levelname)s] (%(funcName)s:%(lineno)d) %(message)s"):
        super(FileFormatter, self).__init__(fmt)
        self.START_TIME = perf_counter()

    @property
    def relative_time(self):
        return perf_counter() - self.START_TIME

    def format(self, record):
        msg = super().format(record)
        return f"{self.relative_time:010.3f}s {msg}"


class StreamFormatter(logging.Formatter):

    def __init__(self, fmt="[%(funcName)s:%(lineno)d] %(message)s"):

        super(StreamFormatter, self).__init__(fmt)

        self.prefix_dict = {
            logging.DEBUG: f"{Fore.CYAN}▲ ",
            logging.INFO: f"{Fore.GREEN}✓ ",
            logging.WARNING: f"{Fore.YELLOW}⚠ ",
            logging.ERROR: f"{Fore.RED}✗ ",
            logging.CRITICAL: f"{Fore.MAGENTA}※ ",
        }

    def format(self, record):
        prefix = self.prefix_dict[record.levelno]
        msg = super().format(record)
        suffix = Style.RESET_ALL
        return prefix + msg + suffix


def setup_logger(level=logging.INFO, log_path="log"):

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(FileFormatter())

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(StreamFormatter())

    logging.basicConfig(level=level, handlers=[file_handler, stream_handler])
