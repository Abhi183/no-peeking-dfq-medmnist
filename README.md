# No Peeking

Quantizing medical image classifiers without calibration data.

Most post-training quantization pipelines need a handful of real images to calibrate on — to figure out activation ranges, pick rounding thresholds, that kind of thing. In a hospital setting that's already a problem: those calibration images are patient data, and handing a compression pipeline a peek at them isn't always something you're allowed to do. This project asks what happens if you quantize without looking at any data at all, using only the trained weights.

Full writeup: [`paper/no_peeking.pdf`](paper/no_peeking.pdf)

## The idea, roughly

Random rotation of the weight matrix before quantizing spreads outlier weights out instead of letting them blow up the whole grid's dynamic range. Do that, then quantize on a fixed uniform grid, and you don't need calibration data to decide where the grid boundaries go — the rotation already tamed the distribution enough that a naive scheme works.

Tested on four MedMNIST tasks — DermaMNIST, PneumoniaMNIST, BloodMNIST, PathMNIST — three training seeds each, twelve checkpoints total.

## What actually happened

INT8 is basically free. Mean AUC stays within 0.002 of full precision across every dataset, and checkpoint size drops from 6.06MB to 1.56MB.

![EDA / weight distributions](paper/figures/fig0_eda.png)

Below 8 bits things get harder, as you'd expect. Rotation plus a uniform grid comes out on top at 4 bits on all four tasks, though the gap to FP32 is still real — this isn't a "ship it" result, it's a "the method direction works" result:

| Dataset | Uniform | Rotation + uniform | Gaussian codebook | Rotation + Gaussian |
|---|---:|---:|---:|---:|
| DermaMNIST | 0.614 | **0.673** | 0.546 | 0.595 |
| PneumoniaMNIST | 0.685 | **0.840** | 0.689 | 0.791 |
| BloodMNIST | 0.771 | **0.812** | 0.631 | 0.796 |
| PathMNIST | 0.821 | **0.890** | 0.741 | 0.754 |

(test AUC, mean over 3 seeds, 4 bits per weight)

![4-bit ablation, all four methods](paper/figures/fig2_corrected_ablation.png)

Rotation helps the uniform grid on every dataset, though only 8 of 12 paired checkpoints actually improve individually — it's not a universal win, more of a "tilts the odds in your favor" thing. It's more consistent for the Gaussian codebook (12 out of 12 pairs improve there), but that codebook starts from a weaker baseline so it still trails rotation+uniform overall.

![AUC vs. bit width](paper/figures/fig3_corrected_bits.png)

At 2-3 bits, everything is close to chance. No amount of clever rotation saves you once the grid gets that coarse — worth knowing before you try to push this further down than 4 bits.

One more thing worth flagging: results are noisy. Swap out the random rotation seed and you can see swings of 0.02-0.08 AUC depending on the dataset, sometimes comparable to the effect you're trying to measure. All of the numbers above are checkpoint-paired and evaluated under a protocol fixed before looking at test data, specifically to keep that noise from leaking into the conclusions. Details on why that mattered are in `FINDINGS.md`.

## Repo layout

```
paper/      the actual paper (LaTeX + PDF), figures, an older draft kept for reference
code/       training, quantization, evaluation, and the tests for all of it
results/    raw CSVs, generated tables, and the manifest tying results to checkpoint hashes
FINDINGS.md what changed between the first pass at this and the corrected version, and why
```

## Reproducing it

```bash
python code/train.py                    # trains the checkpoints, if you don't have them
python code/corrected_eval.py           # the fixed-protocol evaluation
python code/corrected_eval.py --factorial   # rotation-seed variance sweep
python code/corrected_report.py         # regenerates the tables in results/
cd paper && latexmk -pdf no_peeking.tex # rebuilds the PDF
```

Datasets come from `medmnist` and download automatically on first run — not redistributed here.

Every result row in `results/dfq_corrected_fixed.csv` carries the SHA-256 of the checkpoint it came from, so you can check a number against the exact weights that produced it.
