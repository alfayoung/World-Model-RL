"""Thin WandB wrapper compatible with the existing project style."""
from __future__ import annotations
import os
from typing import Any, Dict, Optional


class WandBLogger:
    def __init__(
        self,
        project: str,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        group: str = '',
        tags: list[str] | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        if not enabled:
            return
        try:
            import wandb
            wandb.init(
                project=project,
                name=name,
                config=config or {},
                group=group or None,
                tags=tags,
                reinit=True,
            )
            self._wandb = wandb
        except Exception as e:
            print(f'[WandBLogger] WandB init failed: {e}. Logging disabled.')
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: int):
        if self.enabled:
            self._wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self._wandb.finish()
