import json
import time
from pathlib import Path

import structlog
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = open(log_file, 'w')

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    class DualLogger:
        # structlog file is canonical; wandb metrics are derived from emit() kwargs.
        # Numeric scalars on events that carry `step` are auto-mirrored to run.log
        # under "{event}/{field}". One-time events (no step) stay structlog-only.
        _WANDB_SKIP = {"step", "max_steps", "prnt"}

        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            self.run = None

        def set_run(self, run):
            self.run = run

        def emit(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()

            if self.run is not None and "step" in kwargs:
                metrics = {
                    f"{event}/{k}": v
                    for k, v in kwargs.items()
                    if k not in self._WANDB_SKIP and isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                if metrics:
                    self.run.log(metrics, step=kwargs["step"])

            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)

    return DualLogger(file_handler)
