import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/peft.yaml",
        help="Path to PEFT / LoRA training config yaml.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    print(f"[BOOT] project_root={PROJECT_ROOT}", flush=True)
    print(f"[BOOT] config_path={config_path}", flush=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BOOT] output_dir={output_dir}", flush=True)
    print("[BOOT] importing src.peft ...", flush=True)
    from src.peft import train_peft

    print("[BOOT] imported src.peft", flush=True)
    train_peft(cfg, args.config)


if __name__ == "__main__":
    main()
