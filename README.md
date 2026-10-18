# CatStudio

A local web application that turns an English description into a **512 × 512 cat portrait**. It compares the original Stable Diffusion 1.5 model with an AFHQ-trained LoRA. The input is text and a random seed; no photograph or outline is required. This is a noncommercial APAN PS5560 course project.

Read [LICENSING.md](LICENSING.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing data or an adapter. AFHQ noncommercial conditions and Stable Diffusion use restrictions are retained; the whole project is not being re-licensed under one blanket software license.

The interface is HTML/CSS/JavaScript served by **FastAPI**. It runs in a browser on the same computer; it is not a public website or a paid image API. Model weights live outside this code directory. The server does not upload prompts or images, and it does not download missing models during inference.

## What is in this folder?

| Path | Purpose |
|---|---|
| `catstudio/model.py` | Local diffusion inference and explicit base/LoRA selection |
| `catstudio/training.py` | Feature caching and LoRA training |
| `catstudio/data.py` | Dataset validation, splits and caption preparation |
| `catstudio/api.py`, `schemas.py` | FastAPI routes, input validation and one-job-at-a-time inference |
| `catstudio/static/` | Browser interface, styling and client requests |
| `catstudio/legacy/` | Original Assignment 1 bigram/vector classes, unchanged; source hashes in `provenance.json` |
| `scripts/` | Asset download, data preparation, training, launch and deployment checks |
| `tests/test_api.py` | API tests with a clearly identified fake engine; not model-quality evidence |
| `Dockerfile`, `compose.yaml` | CPU deployment, localhost-only port, read-only model mount |

Project siblings: `04_Data` contains source images, captions, split manifests and licenses; `05_Models` contains downloaded weights and trained adapters; `06_Results` contains experiments; `08_Evidence` contains verification records. Do not add images, weights, caches or secrets to Git.

Git attributes preserve recorded file bytes across Windows and Linux checkouts; shell scripts use LF line endings. This keeps source and evidence checksums comparable after cloning.

## Quick start: prepared project

Start Docker Desktop in Linux-container mode. From `03_Code`, run:

```console
docker compose up --build -d
```

Open **http://127.0.0.1:8000**. API documentation is at **http://127.0.0.1:8000/docs**. Models must already exist in `../05_Models/base_sd15` and `../05_Models/cat_lora`; the image deliberately does not bundle several gigabytes of weights. The first portrait loads the model. CPU generation can take minutes; an elapsed timer is shown instead of an invented completion percentage.

```console
docker compose logs -f catstudio
docker compose stop
```

The default container uses CPU. It does not need an NVIDIA GPU. Keep **one worker** because the model is shared and guarded by an in-process lock. Multiple workers would duplicate model memory and bypass that shared concurrency guard. A busy server rejects a second portrait request with HTTP 409; it does not queue the request.

The Docker base image is pinned by digest and `requirements-lock-cpu.txt` records the complete tested Linux dependency versions, including the original Assignment 1 embedding model. Local GPU installations use `requirements.txt` with a matching CUDA PyTorch build; the CPU lock is not a GPU environment lock.

For a machine with a compatible NVIDIA driver and Docker GPU support, an optional CUDA recipe is provided:

```console
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

This selects `Dockerfile.gpu` and requests an NVIDIA GPU. It downloads a larger CUDA PyTorch stack and requires enough free GPU memory; consult the actual deployment evidence before calling this path verified. The default CPU recipe remains available. Avoid running inference and training simultaneously on an 8 GB GPU.

## Reproduce on a new machine

For the **instructor's inference demo**, Docker Desktop (or Docker Engine with Compose) and Git are enough. Clone this repository into a folder named `03_Code`; the model directory will be created beside it:

```console
mkdir CatStudio
cd CatStudio
git clone https://github.com/ShikiRyo1/apan5560-catstudio.git 03_Code
cd 03_Code
```

From `03_Code`, run:

```console
docker compose build catstudio
docker compose --profile setup run --rm assets scripts/download_assets.py sd15
docker compose --profile setup run --rm assets scripts/fetch_adapter.py
docker compose up -d catstudio
```

The two setup commands download publicly available, pinned base weights and the published trained adapter, then verify their checksums. They need internet access and several gigabytes of disk space. The setup container writes only to the mounted model folder; the inference container subsequently mounts that folder read-only and runs offline as an unprivileged user. The adapter installer refuses to overwrite a different existing release. On Linux, files created by this one-time setup container may be owned by root; they remain readable by the inference container.

The same sequence is available as `scripts/setup_docker.ps1` on Windows and `sh scripts/setup_docker.sh` on macOS/Linux. It does not require host Python or a CUDA GPU. Consult the recorded CPU memory and runtime measurements before choosing hardware. Setup completion and health status do not replace the real `/portraits` verification below.

The following host-Python route is for development or **retraining**, not a prerequisite for querying the Docker application.

Use Python **3.12** and place this code in a `03_Code` folder under a project folder. Keep `04_Data`, `05_Models`, `06_Results` and `08_Evidence` beside it; the asset scripts resolve those sibling paths. The commands below are PowerShell commands from `03_Code`.

```powershell
py -3.12 -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0
..\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-legacy.txt -r requirements-dev.txt
..\.venv\Scripts\python.exe scripts/download_assets.py sd15
```

For CUDA training on a compatible NVIDIA machine, install the matching official CUDA wheels **instead of the CPU wheels**:

```powershell
..\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
```

The inference base model must include the upstream safety checker. Downloads record model revisions and SHA-256 hashes in each model directory's `PROVENANCE.json`. A published trained adapter or a completed local training run must supply `05_Models/cat_lora/pytorch_lora_weights.safetensors`. The application rejects a missing adapter; it never presents the base model as a trained LoRA result.

To reproduce training, download the public data and local caption model, then use the preparation/training scripts described in the teaching guide. For download:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
..\.venv\Scripts\python.exe scripts/download_assets.py data
..\.venv\Scripts\python.exe scripts/download_assets.py blip
..\.venv\Scripts\python.exe scripts/download_assets.py clip
```

Run the pipeline in this order. The command below is the 3,000-update configuration selected after the hardware pilot; the selected release's `ADAPTER_PROVENANCE.json` and training `config.json` record the actual completed run and are authoritative for reproducing that artifact.

```powershell
..\.venv\Scripts\python.exe scripts/prepare_data.py inventory
..\.venv\Scripts\python.exe scripts/prepare_data.py caption
..\.venv\Scripts\python.exe scripts/prepare_data.py normalize
..\.venv\Scripts\python.exe scripts/train_lora.py cache
..\.venv\Scripts\python.exe scripts/train_lora.py train --run-name full --max-steps 3000 --accumulation 4 --rank 8 --learning-rate 0.0001 --checkpoint-every 500
..\.venv\Scripts\python.exe scripts/select_adapter.py full
..\.venv\Scripts\python.exe scripts/evaluate_models.py base
..\.venv\Scripts\python.exe scripts/evaluate_models.py lora
..\.venv\Scripts\python.exe scripts/evaluate_models.py metrics
```

`inventory` decodes images, groups near duplicates and freezes partitions. `caption` runs BLIP only on train/validation photos. `normalize` applies conservative coat descriptions and documented visual-audit corrections. `cache` computes frozen VAE/text-encoder features; changing captions invalidates this cache. `train` updates LoRA attention parameters, evaluates fixed validation loss and saves checkpoints. `select_adapter.py` copies only the completed run's validation-selected adapter into the application folder, with hashes, model card and licenses. The evaluation uses a fixed, code-shipped 16-prompt × 4-seed grid and verifies artifact identities before reusing outputs.

The reserved 500 photos are held out from this LoRA experiment, not guaranteed absent from the foundation model's upstream pretraining. Automated CLIP scores are evaluation proxies, not human quality scores. See the report for measured results and limitations. Do not select checkpoints by looking at the final test-grid results.

Downloaded images and model weights are license-restricted assets, not project-authored source code. AFHQ is CC BY-NC 4.0; Stable Diffusion 1.5 uses CreativeML Open RAIL-M. Keep the attribution, license files and model-use conditions. No API key or paid inference endpoint is needed. Downloading and local computation still use disk space, time and electricity.

## Run locally without Docker

```powershell
.\scripts\run_local.ps1 -Device auto -Port 8000
```

On macOS/Linux, create the project virtual environment at `../.venv`, install the dependencies, then run `sh scripts/run_local.sh` (CPU is supported; MPS is not a tested deployment target). `CATSTUDIO_DEVICE` can be `auto`, `cpu` or `cuda`. The PowerShell launcher uses the project-specific virtual environment and sets the weight paths automatically. Both launchers bind only to `127.0.0.1`.

If `/health` says files are present, that is **not** proof that inference succeeded: it distinguishes files found, model loaded, the last successful request and the last runtime error.

## Use the application

1. Write a short English description or select an example. Chinese instructions explain the controls; this model is tested with English prompts.
2. Select **CatStudio · trained LoRA** or **Original · base model**.
3. Choose a seed. Keep prompt, seed, steps and guidance fixed for a base/LoRA comparison. Exact pixels may still differ across devices/software versions.
4. Click **Generate portrait**. The server handles one portrait at a time. A second concurrent request receives HTTP 409.
5. Inspect the result, download the PNG and, if needed, its JSON generation record. The browser keeps the current result in memory; download it to retain it.

Images can miss requested colors, facial details or composition. Automatic safety filtering is retained; a filtered result is reported explicitly. The app is for cat portraits, and its controls do not imply that every arbitrary prompt will be followed.

## API examples

New portrait endpoint:

```powershell
$body = @{prompt='A close-up portrait of an orange and white cat, plain cream background'; seed=42; model_choice='lora'; num_steps=20; guidance_scale=7.5; adapter_scale=1.0} | ConvertTo-Json
$result = Invoke-RestMethod http://127.0.0.1:8000/portraits -Method Post -ContentType application/json -Body $body
New-Item -ItemType Directory -Force -Path .\local_outputs | Out-Null
[IO.File]::WriteAllBytes((Join-Path $PWD 'local_outputs/cat.png'), [Convert]::FromBase64String($result.image_base64))
$result.metadata
```

Assignment 1 endpoints remain:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/embedding?word=apple'
Invoke-RestMethod 'http://127.0.0.1:8000/generate' -Method Post -ContentType application/json -Body '{"start_word":"the","length":5}'
```

`POST /generate` retains its original bigram meaning. Cat images use `POST /portraits` to avoid breaking the older API. The root route now serves the frontend, and `/health` adds honest portrait status. The Docker build includes the original spaCy 3.8.0 English vector model; local installations use `requirements-legacy.txt` for the same functionality.

| Response | Meaning |
|---|---|
| 200 | Request succeeded; a portrait response contains base64 PNG bytes and actual settings |
| 409 | Another generation is running |
| 422 | Invalid input or an output filtered by the retained safety checker |
| 503 | Missing model/adapter/dependency or model execution failed; inspect local logs and `/health` |

## Verification

For the Git-and-Docker setup, run the check inside the running CPU container and copy its real response evidence to the host:

```console
docker compose exec catstudio python scripts/verify_deployment.py --model-choice lora --num-steps 20 --output /tmp/catstudio-verification
docker compose cp catstudio:/tmp/catstudio-verification ../08_Evidence/deployment
```

Create `../08_Evidence` first if it does not exist. The verification command may take several minutes on CPU; it performs actual generation. The optional GPU recipe also includes these scripts, but its verification status is separate from the default CPU build.

For a host development environment:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe scripts/verify_deployment.py --model-choice lora --num-steps 20
```

The first command checks API behavior with a test double; it does **not** prove diffusion quality. The second calls the running server, exercises the inherited text/vector routes, generates a real 512 × 512 image and saves its PNG, hash and response metadata under `08_Evidence/deployment`. Use `--model-choice base` for a base-model test. `--skip-portrait` is only an endpoint check and is labeled accordingly. Failed checks save a failure report and return a failing exit status.

Training quality claims belong to the recorded evaluation, not these API tests. Report both improvements and regressions; do not replace measured results with expected results.

To render the measured training and evaluation figures after both stages have completed, install `requirements-figures.txt`, then run `python scripts/plot_training.py full` and `python scripts/plot_results.py` using the same project environment. These scripts read recorded values; they do not train or synthesize results.

## Attribution

- Choi, Y., Uh, Y., Yoo, J., & Ha, J.-W. (2020). *StarGAN v2: Diverse Image Synthesis for Multiple Domains*. CVPR. [AFHQ source](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq).
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. [Model](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
- Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. [Paper](https://arxiv.org/abs/2106.09685).
- Li, J., Li, D., Xiong, C., & Hoi, S. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. ICML. [Caption model](https://huggingface.co/Salesforce/blip-image-captioning-base).

Code and instructional text were developed with AI assistance. Model outputs and automated captions require evaluation; AI assistance does not establish course submission, team approval or a grade.
