# Third-party notices

Verified against the upstream sources on 2026-10-18. Downloaded assets remain subject to their own terms. No endorsement by the organizations below is implied.

| Component | Use in CatStudio | Attribution / governing source |
|---|---|---|
| AFHQ v1 cats | Training/validation photos and held-out reference photos | NAVER; Choi et al., CVPR 2020; [CC BY-NC 4.0 statement](https://github.com/clovaai/stargan-v2#license); full text in `licenses/AFHQ_CC_BY_NC_4_LICENSE.txt` |
| Stable Diffusion 1.5 | Pretrained generator and base-model comparison | Robin Rombach, Patrick Esser and contributors; [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5); CreativeML Open RAIL-M in `licenses/SD_CreativeML_OpenRAIL_M_LICENSE.txt` |
| BLIP image-captioning base | Offline draft descriptions of training/validation images | Salesforce Research; Li et al., ICML 2022; [model card](https://huggingface.co/Salesforce/blip-image-captioning-base); BSD-3-Clause in `licenses/BLIP_BSD3_LICENSE.txt`; intended research use |
| CLIP ViT-B/32 | Image-text, attribute and nearest-reference evaluation proxies | OpenAI; Radford et al., ICML 2021; [source](https://github.com/openai/CLIP); MIT in `licenses/CLIP_MIT_LICENSE.txt` |
| spaCy `en_core_web_lg` 3.8.0 | Original Assignment 1 static word-vector endpoint | Explosion and model contributors; [model repository](https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.8.0); license/notices remain inside its installed wheel |
| PyTorch / TorchVision | Tensor computation, training and inference | [PyTorch](https://github.com/pytorch/pytorch), [TorchVision](https://github.com/pytorch/vision); licenses and bundled notices retained in installed distributions |
| Diffusers / Transformers / PEFT / Accelerate / Hugging Face Hub | Pipeline, text/image encoders, adapters and public asset downloads | [Hugging Face](https://github.com/huggingface); respective upstream licenses retained in installed distributions |
| FastAPI / Uvicorn / Pydantic / Starlette | API, server and request validation | Respective project authors; exact versions recorded in dependency files; licenses remain in installed distributions |
| NumPy / Pillow / ImageHash / SciPy / PyWavelets / Safetensors | Array/image processing, duplicate screening and weight format | Respective project authors; installed distribution licenses apply |

## Model and dataset identity

- SD 1.5 repository: `stable-diffusion-v1-5/stable-diffusion-v1-5`; pinned revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
- BLIP repository: `Salesforce/blip-image-captioning-base`; pinned revision `82a37760796d32b1411fe092ab5d4e227313294b`.
- CLIP evaluator repository: `openai/clip-vit-base-patch32`; pinned revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- AFHQ v1 official archive: `https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1`; expected SHA-256 `7f63dcc14ef58c0e849b59091287e1844da97016073aac20403ae6c6132b950f`.

`scripts/asset_checksums.json` and each asset's generated `PROVENANCE.json` record exact files and hashes. Model downloads are anonymous and explicit; model inference is local. The complete environment's runtime dependencies are listed in `requirements-lock-cpu.txt`. This short inventory is not a replacement for those packages' complete notices.

## Modifications and boundaries

CatStudio adds dataset integrity/split checks, automated caption cleaning, LoRA training, fixed paired evaluation, and a local web/API interface. The base-model weights, captioner and evaluator are not presented as models trained from scratch by the team. Assignment 1 model classes are copied unchanged and identified separately in `catstudio/legacy/provenance.json`.

Training recipes follow the public Diffusers text-to-image LoRA approach; cite [the official guide](https://huggingface.co/docs/diffusers/training/lora) and Hu et al.'s [LoRA paper](https://arxiv.org/abs/2106.09685). Retain upstream license headers if copying further source code. AI assistance was used to develop and review project code, documentation and weak captions; measured output and evaluation must remain distinguishable from generated explanations.

The project does not claim ownership of AFHQ originals, pretrained weights, third-party code or all generated-image rights. See [LICENSING.md](LICENSING.md) for noncommercial adapter distribution and downstream model-use conditions.
