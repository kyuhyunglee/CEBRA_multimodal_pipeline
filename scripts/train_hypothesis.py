from __future__ import annotations

from train_joint import train_joint


def run_hypothesis_training() -> None:
    train_joint("configs/default_config.yaml", use_labels=True)


if __name__ == "__main__":
    run_hypothesis_training()
