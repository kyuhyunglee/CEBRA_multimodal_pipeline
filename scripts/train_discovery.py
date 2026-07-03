from __future__ import annotations

from train_joint import train_joint


def run_discovery_training() -> None:
    train_joint("configs/default_config.yaml", use_labels=False)


if __name__ == "__main__":
    run_discovery_training()
