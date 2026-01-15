import logging
from pathlib import Path

def setup_logger(log_path: str = "logs/training.log") -> logging.Logger:
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("fine_tuning")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers in interactive sessions
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        ch = logging.StreamHandler()

        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
