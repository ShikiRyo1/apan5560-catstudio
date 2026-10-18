# CatStudio licensing and permitted scope

CatStudio is a **noncommercial course/research project**. Its source code, downloaded data, upstream model weights, trained adapter and generated examples are different materials with different rights. This document does not claim ownership of third-party images or weights, and does not apply one blanket license to the whole repository.

## Source and inherited course code

Project-authored source is supplied for coursework review and reproducibility. No general commercial relicensing of the project is granted by this notice. The original Assignment 1 classes in `catstudio/legacy/` remain attributed to the student's earlier submission; their unchanged source hashes and original course-relative paths are recorded in `provenance.json`. Libraries retain their own licenses and installed notices.

## AFHQ images and the AFHQ-trained adapter

AFHQ v1 is distributed by NAVER under **CC BY-NC 4.0**. Use and redistribution of its licensed material and adaptations must remain noncommercial, retain appropriate attribution and identify changes. The project uses only the cat subset, screens near duplicates, creates train/validation partitions, generates and cleans weak captions, and trains LoRA parameters. Original data are obtained from the upstream download rather than re-licensed as CatStudio content.

The CatStudio adapter is offered for **noncommercial education and research**, consistent with the AFHQ training-data conditions. Keep the AFHQ attribution and this restriction with the adapter. This is a conservative project distribution condition; it is not a claim that every generated image automatically acquires the dataset's copyright license. The adapter also remains subject to the Stable Diffusion model terms below.

Attribution: Choi, Y., Uh, Y., Yoo, J., & Ha, J.-W. (2020). *StarGAN v2: Diverse Image Synthesis for Multiple Domains*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. [Official AFHQ source and license statement](https://github.com/clovaai/stargan-v2#license). The full upstream license is included in [licenses/AFHQ_CC_BY_NC_4_LICENSE.txt](licenses/AFHQ_CC_BY_NC_4_LICENSE.txt).

## Stable Diffusion 1.5 and the LoRA modification

Stable Diffusion 1.5 is governed by **CreativeML Open RAIL-M**, including its use restrictions. The CatStudio modification trains attention-projection LoRA parameters on AFHQ cat photos; it does not replace ownership or notices for the pretrained base model.

Use or redistribution of the base model or this adapter must comply with the full [CreativeML Open RAIL-M license](licenses/SD_CreativeML_OpenRAIL_M_LICENSE.txt), including paragraph 5 and Attachment A. Those restrictions are incorporated as conditions of the CatStudio adapter's use and distribution. Supply the complete license to downstream recipients, preserve notices, clearly identify the modification, and pass on the same use restrictions. A downloadable model or a hosted API must not strip these conditions.

The full license, rather than a short summary, controls. It restricts unlawful or harmful uses, exploitation of minors, harmful misinformation or personal-data disclosure, harassment, discriminatory profiling, certain consequential automated decisions, medical advice, and specified justice/law-enforcement uses. The project's evaluated scope is cat portraits. The upstream safety checker remains enabled.

Sources: [SD 1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [upstream license](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).

## Other pretrained components

BLIP generates local draft captions; its model card identifies BSD-3-Clause and describes a research release. CLIP provides evaluation embeddings under its upstream MIT terms. Their license texts and attribution are in `licenses/` and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Captions are automated weak labels, not independently verified human annotations.

## Generated images and distribution

An output is not a photograph of a real, verified animal. The project does not promise exclusive copyright, freedom from all third-party rights, or commercial clearance of generated images. Keep the project's noncommercial scope, model-use restrictions and relevant source attribution when sharing course materials. Do not label downloaded AFHQ photographs as generated examples.

A selected-adapter release must include its model card, SHA-256 checksum, base-model identity/revision, this notice, third-party notices, and full AFHQ and Open RAIL-M license texts. Exact experiment and download records identify which artifacts were used. Public availability or free downloading does not mean public-domain status.
