# Observed findings

## What changed after fine-tuning

The adapter changed the generated portraits and slightly increased the image–text score on the fixed grid. Mean CLIP cosine rose from 0.3164 to 0.3233; the paired difference was 0.00694, with a 95% prompt-cluster bootstrap interval of 0.00221 to 0.01181. This supports a narrow claim about this automated score. It does not establish a general improvement in appearance or prompt following.

The color proxy agreed with the requested category for 59 of 64 base images and 62 of 64 LoRA images. Its interval touched zero. Pattern and pose proxy agreement each increased by three images, but both difference intervals crossed zero. Within-prompt diversity decreased from 0.0478 to 0.0452, also with an interval crossing zero. These results do not support a consistent gain across attributes or diversity.

The independent visual audit is important because the pose proxy did not capture an obvious failure. Across the 16 seed-101 prompts, the base model met the requested pose in 10 images, missed it in five, and had one ambiguous view. LoRA met eight and missed eight: every three-quarter or profile request in that subset produced an approximately frontal cat. Both models produced recognizable cats with the requested broad colors and patterns in this small audit. These observations come from one unblinded AI-assisted reviewer, not a human preference experiment or a population accuracy estimate.

| Fixed visual audit, seed 101 | Base: met / not met / uncertain | LoRA: met / not met / uncertain |
|---|---:|---:|
| Recognizable cat | 16 / 0 / 0 | 16 / 0 / 0 |
| Requested broad color | 16 / 0 / 0 | 16 / 0 / 0 |
| Requested pattern | 16 / 0 / 0 | 16 / 0 / 0 |
| Requested pose | 10 / 5 / 1 | 8 / 8 / 0 |
| Plain light background | 2 / 9 / 5 | 2 / 8 / 6 |

The two evaluation methods cover different samples: the quantitative metrics use four seeds, while the visual audit uses the predetermined seed 101. Their categories and decision rules also differ. We therefore retain both results and do not treat a slightly higher automated pose score as evidence that pose control improved. In particular, the visual failures show that the score is insufficient for validating this user requirement.

The fixed p03 example illustrates a mixed change. LoRA includes both ears and more of the head, whereas the base image crops one ear. LoRA also adds stronger edge lighting and an uneven mouth contour. Both images have pupil artifacts and a darker background than requested. The fixed p15 example shows a clearer regression: the base cat is in profile, while the adapted cat faces forward despite the same profile prompt. These pairs were chosen before output inspection, not selected to make the adapter look better.

## Similarity, speed and practical limits

Generated images moved closer to AFHQ training images in CLIP space: mean nearest-training cosine increased from 0.9313 to 0.9458. Mean nearest perceptual-hash distance fell from 16.19 to 15.06; the minimum was 12 for both models. Inspection of the eight highest-CLIP-similarity pairs per model showed shared coat types and portrait composition, with visible differences in eyes, markings, lighting or background. None of these inspected pairs appeared to be a direct image duplicate. This limited screen does not establish that memorization is absent. Greater similarity can reflect domain adaptation, reduced variation or memorization; the metric alone cannot separate them.

The exploratory CLIP polynomial MMD squared against the 500 held-out reference images decreased from 1.2410 to 0.8575. This indicates closer distributions under that specific kernel on this sample. It is not FID, standard KID or proof of better-looking images, and it played no part in selecting the checkpoint.

On the recorded GPU, mean warm generation time was 1.66 seconds for base and 1.79 seconds for LoRA. All 128 planned outputs completed; none triggered the retained safety checker. The roughly 0.13-second warm overhead excludes cold loading and browser-request overhead. Separate CPU Docker timings describe the deployment route.

The practical conclusion is that this small adapter specializes the portrait appearance but does not reliably preserve pose control. A plausible explanation is the portrait-oriented data combined with coarse captions that contain no reliable pose labels; this experiment does not isolate that cause. The application keeps base and LoRA as explicit choices so users can compare their behavior. More balanced pose data, checked captions and separately planned adapter-strength experiments are appropriate next steps. They were not run or used to revise the frozen test results.
