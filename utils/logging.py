# Configure logging with multiple levels like server.py
import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

# Set up logging
def setup_logging(verbose: bool = False, module_levels: dict[str, str] = None):
    """Configure logging with appropriate handlers and formatters
    
    Args:
        verbose: Enable DEBUG level logging for root logger
        module_levels: Dict of module_name -> log_level to set specific module log levels
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Default module levels - these modules are particularly noisy at DEBUG level
    default_module_levels = {
        "uvicorn": "INFO",
        "uvicorn.access": "WARNING",
    }
    
    if module_levels:
        default_module_levels.update(module_levels)

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # Set specific module loggers
    for module_name, module_level in default_module_levels.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, module_level.upper()))
    
    # Allow environment variable override for specific modules
    import os
    for key, value in os.environ.items():
        if key.startswith('LOG_LEVEL_'):
            module_name = key[10:].lower().replace('_', '.')
            try:
                module_logger = logging.getLogger(module_name)
                module_logger.setLevel(getattr(logging, value.upper()))
                root_logger.info(f"Set {module_name} log level to {value} from environment")
            except AttributeError:
                root_logger.warning(f"Invalid log level {value} for module {module_name}")

    return root_logger

