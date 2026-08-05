"""Run the pipeline on a Modal GPU instead of the local machine.

The pipeline itself is unchanged: this only supplies a CUDA box, the deps, and a
persistent volume for `runs/`. Stage logic still lives in pipeline/stages/*.

    modal run --detach modal_train.py                 # ALWAYS use --detach for real runs:
                                                      # without it the job dies when your
                                                      # laptop sleeps or the shell closes.
    modal run modal_train.py                          # full run, default config
    modal run modal_train.py --stage evaluate         # one stage
    modal run modal_train.py --gpu A10G               # cheaper, slower
    modal run modal_train.py --download               # pull artifacts back locally

Cost note: A100-80GB bills ~$2.10/hr by the second, so a ~30min run is ~$1 of the
$30/month free credit. `timeout` below is the hard ceiling on a runaway job.
"""

import os

import modal

APP_NAME = "indic-voice-pipeline"
CONFIG = "configs/hi_stt_fleurs_large.yaml"
# Override with MODAL_GPU=L4 (etc) when a card is cheap and actually available.
GPU = os.environ.get("MODAL_GPU", "A10G")

# Volume keeps runs/ (corpus, checkpoints, metrics) alive between invocations, so an
# interrupted run resumes via train.py's checkpoint auto-resume instead of restarting.
volume = modal.Volume.from_name("indic-voice-runs", create_if_missing=True)
RUNS_DIR = "/runs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")          # soundfile/torchaudio decode deps
    .pip_install(
        "torch>=2.2", "torchaudio>=2.2",
        "transformers>=4.44", "peft>=0.11", "datasets>=2.20",
        "accelerate>=0.30", "soundfile>=0.12", "librosa",
        # datasets 5.x decodes the Audio feature through torchcodec, not soundfile.
        # Without it every audio row raises ImportError at ingest.
        "torchcodec",
        "pydantic>=2.6", "pyyaml>=6.0",
    )
    # Ship the repo last so code edits don't invalidate the (slow) dependency layers.
    .add_local_dir("pipeline", "/root/pipeline")
    .add_local_dir("configs", "/root/configs")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    # A100/H100 require a payment method on file even with free credit. A10G (24GB)
    # holds large-v3 + LoRA in fp16 like L4 does, and L4 was capacity-starved: runs
    # sat in "waiting to be scheduled on a GPU_L4 worker" and then took a
    # KeyboardInterrupt when the scheduler reclaimed the container.
    gpu=GPU,
    volumes={RUNS_DIR: volume},
    timeout=6 * 60 * 60,
    # Capacity evictions are infra failures, not pipeline failures, so they raise and
    # are retried — while a real pipeline error returns a nonzero exit_code normally
    # and is NOT retried. Each retry resumes from the last checkpoint on the volume
    # (train.py auto-resumes), so an eviction costs at most `save_steps` of progress
    # instead of the whole run. This is what actually makes the lid safe to close.
    retries=modal.Retries(max_retries=10, initial_delay=60.0, backoff_coefficient=1.0),
    # No secret needed: FLEURS and whisper-large-v3 are both ungated. Gated models
    # (indicwhisper) would need secrets=[modal.Secret.from_name("huggingface")].
)
def run_pipeline(stage: str = "all", config: str = CONFIG) -> dict:
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    os.chdir("/root")
    # Point the pipeline's runs/ at the persistent volume via a symlink, so no
    # pipeline code needs to know it's running on Modal.
    local_runs = Path("/root/runs")
    if not local_runs.is_symlink():
        shutil.rmtree(local_runs, ignore_errors=True)
        local_runs.symlink_to(RUNS_DIR)

    import torch
    print(f"GPU: {torch.cuda.get_device_name()} | "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB | "
          f"torch {torch.__version__}", flush=True)

    # Checkpoints are worthless unless they outlive an eviction. A single commit after
    # the subprocess never runs when the scheduler reclaims the container mid-training,
    # which is how a preempted run came back with an EMPTY checkpoint dir. Commit while
    # training instead, so a retry resumes instead of restarting at step 0.
    import threading

    def _complete_checkpoints_exist() -> bool:
        """HF writes trainer_state.json at the END of saving a checkpoint, so its
        presence marks that dir as finished. Committing mid-write would persist a
        half-saved checkpoint that resume then chokes on."""
        ckpts = sorted(Path(RUNS_DIR).glob("*/checkpoint/checkpoint-*"))
        return bool(ckpts) and (ckpts[-1] / "trainer_state.json").exists()

    stop = threading.Event()

    def _periodic_commit() -> None:
        while not stop.wait(120):
            try:
                if _complete_checkpoints_exist():
                    volume.commit()
                    print("[commit] checkpoints flushed to volume", flush=True)
            except Exception as exc:  # never let the watchdog kill the run
                print(f"[commit] failed (will retry): {exc}", flush=True)

    threading.Thread(target=_periodic_commit, daemon=True).start()

    cmd = [sys.executable, "-m", "pipeline.run", stage, "--config", config, "--force"]
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)

    stop.set()
    volume.commit()  # flush checkpoints/metrics before the container dies

    metrics = {
        p.parent.name: p.read_text() for p in Path(RUNS_DIR).glob("*/metrics.json")
    }
    return {"exit_code": result.returncode, "metrics": metrics}


@app.local_entrypoint()
def main(stage: str = "all", config: str = CONFIG, download: bool = False) -> None:
    if download:
        print("Artifacts live in the 'indic-voice-runs' volume. Pull them with:")
        print("  modal volume get indic-voice-runs / ./runs_modal")
        return

    out = run_pipeline.remote(stage=stage, config=config)
    print(f"\nexit_code={out['exit_code']}")
    for name, blob in out.get("metrics", {}).items():
        print(f"\n--- {name}/metrics.json ---\n{blob}")
    if out["exit_code"] != 0:
        raise SystemExit(out["exit_code"])


if __name__ == "__main__":
    # Detached launch, run as PLAIN python — not `modal run`.
    #
    # `modal run --detach` still creates an ephemeral app owned by the local client,
    # and three runs launched that way were cancelled within 40s of each other while
    # the laptop was idle. A deployed app plus .spawn() has no client attachment at
    # all: the call lives server-side and the local process exits immediately.
    #
    #   modal deploy modal_train.py            # once, after any code change
    #   python modal_train.py <config> [...]   # spawn one detached run per config
    import sys

    fn = modal.Function.from_name(APP_NAME, "run_pipeline")
    for cfg in sys.argv[1:] or [CONFIG]:
        call = fn.spawn(stage="all", config=cfg)
        print(f"{cfg}  ->  call {call.object_id}")
