import json
import logging
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
        debug (bool): Disables debug logs if True
    """
    if not debug:
        logging.disable(logging.CRITICAL)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)