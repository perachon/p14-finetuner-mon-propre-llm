from __future__ import annotations

from pathlib import Path
import shutil

from dotenv import load_dotenv
from huggingface_hub import upload_folder


def copy_minimal_adapter(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "training_args.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ]
    for name in names:
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, dst_dir / name)


def ensure_readme(dst_dir: Path, title: str, run_ts: str, note: str) -> None:
    readme = dst_dir / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "\n".join(
            [
                f"# {title} ({run_ts})",
                "",
                "- Base: Qwen/Qwen3-1.7B-Base",
                "- Type: PEFT LoRA adapter",
                f"- Note: {note}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv(".env")

    root = Path(".")
    run_ts = "20260318_1657"

    src_sft = root / "checkpoints" / f"qwen3-1.7b-sft-lora_LONG_{run_ts}"
    src_dpo = root / "checkpoints" / f"qwen3-1.7b-dpo_LONG_{run_ts}"

    if not src_sft.exists():
        raise FileNotFoundError(f"Missing SFT checkpoint dir: {src_sft}")
    if not src_dpo.exists():
        raise FileNotFoundError(f"Missing DPO checkpoint dir: {src_dpo}")

    work = root / "runs" / "hf_upload" / "model_adapters_long"
    if work.exists():
        shutil.rmtree(work)

    dst_sft = work / "sft"
    dst_dpo = work / "dpo"

    copy_minimal_adapter(src_sft, dst_sft)
    copy_minimal_adapter(src_dpo, dst_dpo)

    ensure_readme(
        dst_sft,
        title="SFT long adapter",
        run_ts=run_ts,
        note="SFT long (Windows RTX 4050 6GB)",
    )
    ensure_readme(
        dst_dpo,
        title="DPO long adapter",
        run_ts=run_ts,
        note="DPO long from SFT-long adapter (Windows RTX 4050 6GB)",
    )

    repo_id = "perachon/p14-model"

    upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(dst_sft),
        path_in_repo=f"adapters/sft_long_{run_ts}",
        commit_message=f"Add SFT long LoRA adapter ({run_ts})",
    )
    upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(dst_dpo),
        path_in_repo=f"adapters/dpo_long_{run_ts}",
        commit_message=f"Add DPO long LoRA adapter ({run_ts})",
    )

    print("Uploaded long adapters:")
    print(f"- adapters/sft_long_{run_ts}")
    print(f"- adapters/dpo_long_{run_ts}")


if __name__ == "__main__":
    main()
