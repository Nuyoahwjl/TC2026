from pathlib import Path

from datasets import (
    load_dataset,
    load_from_disk,
    get_dataset_config_names,
    concatenate_datasets,
    DatasetDict,
)
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def is_hf_dataset_saved(path: Path) -> bool:
    return path.exists() and (path / "dataset_dict.json").exists()


def save_math(force: bool = False):
    """
    Download EleutherAI/hendrycks_math and merge all configs into one DatasetDict.

    Saved path:
        data/math
    """
    save_path = DATA_DIR / "math"

    if is_hf_dataset_saved(save_path) and not force:
        print(f"[SKIP] MATH already exists: {save_path}")
        return

    print("[DOWNLOAD] EleutherAI/hendrycks_math")

    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    print(f"[INFO] MATH configs: {configs}")

    split_buffers = {}

    for cfg in configs:
        print(f"[LOAD] config = {cfg}")
        ds = load_dataset(dataset_name, cfg)

        for split_name, split_ds in ds.items():
            split_buffers.setdefault(split_name, [])
            split_buffers[split_name].append(split_ds)

    merged = DatasetDict()
    for split_name, parts in split_buffers.items():
        merged[split_name] = concatenate_datasets(parts)
        print(f"[MERGE] split = {split_name}, size = {len(merged[split_name])}")

    if save_path.exists() and force:
        import shutil
        shutil.rmtree(save_path)

    merged.save_to_disk(str(save_path))
    print(f"[DONE] MATH saved to: {save_path}")

    # quick check
    loaded = load_from_disk(str(save_path))
    print(loaded)
    if "train" in loaded:
        print("[SAMPLE] MATH train[0]:")
        print(loaded["train"][0])


def save_phyx(force: bool = False):
    """
    Download Cloudriver/PhyX for PEFT / LoRA cross-domain finetuning.

    Saved path:
        data/phyx
    """
    save_path = DATA_DIR / "phyx"

    if is_hf_dataset_saved(save_path) and not force:
        print(f"[SKIP] PhyX already exists: {save_path}")
        return

    print("[DOWNLOAD] Cloudriver/PhyX")

    ds = load_dataset("Cloudriver/PhyX")

    if save_path.exists() and force:
        import shutil
        shutil.rmtree(save_path)

    ds.save_to_disk(str(save_path))
    print(f"[DONE] PhyX saved to: {save_path}")

    # quick check
    loaded = load_from_disk(str(save_path))
    print(loaded)
    first_split = list(loaded.keys())[0]
    print(f"[SAMPLE] PhyX {first_split}[0]:")
    print(loaded[first_split][0])


def save_qwen_math_model():
    """
    Download Qwen/Qwen2.5-Math-1.5B to local models directory.

    Saved path:
        models/Qwen2.5-Math-1.5B
    """
    repo_id = "Qwen/Qwen2.5-Math-1.5B"
    local_dir = MODEL_DIR / "Qwen2.5-Math-1.5B"

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[SKIP] Model already exists: {local_dir}")
        return

    print(f"[DOWNLOAD] {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"[DONE] Model saved to: {local_dir}")


def main():
    save_math(force=False)
    save_phyx(force=False)
    save_qwen_math_model()


if __name__ == "__main__":
    main()