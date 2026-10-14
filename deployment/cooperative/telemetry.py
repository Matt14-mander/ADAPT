"""Exclusive JSONL/JSON output; a failed episode is preserved without auto-reset."""

import json
from pathlib import Path


class CarryTelemetry:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self.stream = (self.directory / "telemetry.jsonl").open("x", encoding="utf-8")

    def write(self, record):
        self.stream.write(json.dumps({"schema": 1, **record}, allow_nan=False) + "\n")

    def finish(self, summary):
        self.stream.close()
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
