import subprocess
import sys
from pathlib import Path

import modal


ROOT = "/workspace/tiny-speculator"
ARTIFACTS = "/artifacts"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.25.1")
    .add_local_dir(
        ".",
        ROOT,
        copy=True,
        ignore=[".git/**", ".venv/**", "tiny_speculators/data/**"],
    )
    .workdir(ROOT)
    .uv_pip_install(".")
    .env(
        {
            "HF_HOME": f"{ARTIFACTS}/cache/huggingface",
            "VLLM_CACHE_ROOT": f"{ARTIFACTS}/cache/vllm",
        }
    )
)

app = modal.App("tiny-speculators-pipeline")
volume = modal.Volume.from_name(
    "tiny-speculators",
    create_if_missing=True,
    version=2,
)


@app.function(
    image=image,
    gpu="H200",
    cpu=4,
    memory=48 * 1024,
    ephemeral_disk=900 * 1024,
    timeout=5 * 60 * 60,
    volumes={ARTIFACTS: volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_pipeline(max_samples: int, chunk_size: int, resume: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "tiny_speculators.scripts.pipeline",
        "--data",
        f"{ARTIFACTS}/data/qwen3-8b-sharegpt",
        "--checkpoint",
        f"{ARTIFACTS}/checkpoints/eagle3",
        "--exported-checkpoint",
        f"{ARTIFACTS}/checkpoints/eagle3-vllm",
        "--max-samples",
        str(max_samples),
        "--chunk-size",
        str(chunk_size),
        "--max-sequence-length",
        "4096",
    ]
    if resume:
        command += [
            "--resume",
            f"{ARTIFACTS}/checkpoints/eagle3",
        ]
    log_path = Path(ARTIFACTS) / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


@app.local_entrypoint()
def main(
    max_samples: int = 40_000,
    chunk_size: int = 20_000,
    resume: bool = False,
) -> None:
    run_pipeline.remote(max_samples, chunk_size, resume)
