# CatStudio AFHQ LoRA — model card

## Intended use
Noncommercial educational/research cat-portrait generation. Input: an English cat portrait description and a random seed. Output: one 512 × 512 RGB image. The adapter requires the pinned SD1.5 base model; it cannot generate images on its own. Retain the upstream safety checker.

## Training provenance
- Base: `stable-diffusion-v1-5/stable-diffusion-v1-5` at `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
- Adapter SHA256: `87a74dd4f07243383d30c6efcfd5a1c12e5008e2b440f9af963b3f0b73828b30`; 6,414,992 bytes.
- Original data: AFHQ v1 cats, 5,653 photos. Training: 4,631; validation: 515; final reference/test: 500; excluded training neighbors of test: 7.
- Automatic local BLIP captions were conservatively cleaned to cat/coat descriptors; one contradictory sampled caption was replaced by a generic cat caption after an AI-assisted audit. Captions are weak labels, not human ground truth.
- UNet attention LoRA rank 8; 1,594,368 trainable parameters; frozen VAE, CLIP text encoder and base UNet weights.
- Actual optimizer updates: 3,000; effective batch 4; photo exposures 12,000; effective epochs 2.591.
- Selected checkpoint: step 3000; fixed validation noise-prediction MSE 0.183854; initial base MSE 0.184997.
- Hardware: NVIDIA GeForce RTX 3070; training duration 83.2 minutes, excluding data/caption/feature preprocessing.
- Training source SHA256: `c52daa19a18af76537cd28599a0089c0088059f612090be02bdc9fb2ca053b4f`. Full source snapshot, logs and resume state are in the training-run folder.

## Evaluation and limitations

The completed comparison contains 16 fixed prompts × 4 seeds for both models, 128 images in total. Each pair uses 512 × 512 pixels, 20 sampling steps, CFG 7.5 and identical prompt/seed settings. The selected checkpoint was chosen before these outputs, using validation denoising error. Lower validation loss is not evidence of better appearance.

| Automated metric | Base | LoRA | LoRA minus base [95% interval] |
|---|---:|---:|---:|
| Image-text cosine | 0.3164 | 0.3233 | +0.0069 [+0.0022, +0.0118] |
| Color agreement proxy | 0.9219 | 0.9688 | +0.0469 [+0.0000, +0.0938] |
| Pattern agreement proxy | 0.6250 | 0.6719 | +0.0469 [-0.1562, +0.2500] |
| Pose agreement proxy | 0.3906 | 0.4375 | +0.0469 [-0.1562, +0.2500] |
| Within-prompt diversity | 0.0478 | 0.0452 | -0.0025 [-0.0104, +0.0046] |

Intervals resample the 16 paired prompt clusters, with four seeds averaged within each prompt. These are descriptive automated proxies on a small fixed grid; they do not establish human preference or general performance. Full numeric records and fixed illustrative pairs are in the repository's `results/` directory. Both gains and regressions are retained.

Training captions mainly describe broad coat colors and hair length. They contain no reliable pose/background labels and no tabby/striped/calico labels. Those capabilities largely depend on the base model. Captions can be generic or wrong, and generated anatomy, pose or colors can fail. The 500 official reference images were held out from this LoRA pipeline, but cannot be guaranteed absent from upstream pretraining. Nearest-image screening cannot prove absence of memorization.

The base model's safety checker is retained. The project is for noncommercial coursework and research. No human preference study was performed. The small strict resume-equivalence test failed its unchanged 1e-6 tolerance; numerical drift is reported in `results/resume_equivalence.json` rather than labeled bitwise reproduction.

### Observed pose-control failure

A single unblinded AI-assisted reviewer inspected the predetermined 16 seed-101 images per model. Requested pose was met / not met / uncertain in 10 / 5 / 1 base images and 8 / 8 / 0 LoRA images. LoRA failed 8/8 oblique/profile requests p09–p16 in this subset, producing approximately frontal cats. The fixed p15 pair shows a base profile becoming frontal after adaptation. This is a known limitation of the released adapter. The automated pose proxy uses four seeds per prompt and slightly increases, while this visual audit uses seed 101 only; their samples and decision rules differ. The proxy increase does not establish improved pose control. This is not a human preference study. Public `results/visual_review_protocol.json`, `visual_review.json`, `nearest_training_visual_review.json` and `report_findings.md` retain the scope, labels, image identities and interpretation.

## Sources and license conditions
- Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. https://arxiv.org/abs/2112.10752
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Choi et al. (2020). StarGAN v2: Diverse Image Synthesis for Multiple Domains. https://github.com/clovaai/stargan-v2
- SD1.5 model card: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- BLIP: https://huggingface.co/Salesforce/blip-image-captioning-base

The base model is governed by CreativeML OpenRAIL-M. AFHQ is CC BY-NC 4.0 (NAVER Corporation); this project release is limited to noncommercial educational/research use and retains attribution and notice of changes. Complete source license texts accompany this adapter. No paid image API was used. Code and writing were developed with AI assistance and checked against actual execution evidence.
