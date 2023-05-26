import logging
import os

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # format="%(levelname)s: %(message)s",
    level=os.environ.get("DEFAULT_LOGGER_LEVEL", logging.INFO),
)

logger = logging.getLogger(__name__)
