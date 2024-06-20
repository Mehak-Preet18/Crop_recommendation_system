"""Set up the logger."""

import logging
import socket
import os
import sys

_LOG_PATH = "./logs"


def setup_logger():
    """Set up the logger."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s - Host: "
        + socket.gethostname(),
        handlers=[
            logging.FileHandler(os.path.join(_LOG_PATH, "logs.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info("-----Logs Starting------")
    return logger
