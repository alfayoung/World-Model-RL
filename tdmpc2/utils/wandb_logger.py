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
        self._wandb = None
        # Treat WANDB_MODE=disabled or WANDB_DISABLED as fully off
        if os.environ.get('WANDB_MODE', '').lower() == 'disabled' or \
                os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1'):
            enabled = False
        self.enabled = enabled
        if not enabled:
            print('[WandBLogger] Logging disabled.')
            return
        try:
            import wandb
            # Fall back to offline mode if no API key is configured
            if not os.environ.get('WANDB_API_KEY'):
                try:
                    key = wandb.api.api_key
                except Exception:
                    key = None
                if not key:
                    os.environ['WANDB_MODE'] = 'offline'
            wandb.init(
                project=project,
                name=name,
                config=config or {},
                group=group or None,
                tags=tags,
                reinit=True,
            )
            self._wandb = wandb
            print(f'[WandBLogger] Initialized (mode={wandb.run.settings.mode}).')
        except Exception as e:
            print(f'[WandBLogger] WandB init failed: {e}. Logging disabled.')
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: int):
        if self.enabled and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
