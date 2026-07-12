"""Stage 2 training entry point (streaming, step based).

Thin wrapper over chessfengen.train.phase for the square classifier. See that
module for the full description and the available command line flags.

Run as: python -m chessfengen.train.stage2 --total-steps 20000
"""

from __future__ import annotations

from chessfengen.train.phase import train_stage_cli


def main() -> None:
    train_stage_cli(stage=2)


if __name__ == "__main__":
    main()
