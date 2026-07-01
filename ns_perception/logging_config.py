import logging
import sys

import colorlog


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | %(name)s | %(funcName)s | %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True,
    )


def print_logo():
    print("""
 ▄▄
██                                          ██
████▄ ██ ██ ▀██▀▀ ▄████    ▄█▀▀▀ ▄█▀█▄ ▄████ ██ ██ ▄█▀█▄ ██
██ ██ ██▄██  ██   ██       ▀███▄ ██▄█▀ ██ ██ ██ ██ ██▄█▀ ██
██ ██  ▀██▀  ██   ▀████ ██ ▄▄▄█▀ ▀█▄▄▄ ▀████ ▀██▀█ ▀█▄▄▄ ██
██                                ██
▀▀▀                                 ▀▀
""")
