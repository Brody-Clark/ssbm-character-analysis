import json
import logging
import sys
from datetime import datetime, UTC


class JsonFormatter(logging.Formatter):
    """
    Formats logs in structured JSON
    """
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        })


def configure_logging(debug: bool):
    """
    Overrides logger to setup JSON formatting and optionally set debug logs

    Args:
        debug (bool): Enables debug logs if True
    """

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    log_level = logging.DEBUG if debug else logging.INFO
    root.setLevel(log_level)

    logging.captureWarnings(True)