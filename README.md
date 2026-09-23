# AlignUS

**Weakly Supervised Spatial Grounding for Discriminative Attention-Based
Ultrasound-Histopathology Alignment in Prostate Cancer Grading**

Obed Korshie Dzikunu, Emma Willis, Mohammad Mahdi Abootorabi, Mohamed
Harmanani, Zhuoxin Guo, Ferdinand Luger, Adam Kinnaird, Brian Wodlinger,
Parvin Mousavi, Purang Abolmaesumi

University of British Columbia · Vector Institute · Queen's University ·
Ordensklinikum Linz · University of Alberta · Exact Imaging

## Abstract

Unpaired cross-modal distillation transfers grade structure from
histopathology into a micro-ultrasound (micro-US) encoder by aligning a
pooled needle-region embedding to a frozen histopathology teacher under
grade-group correspondence alone. A single objective is thereby required
to serve two distinct functions: rendering patch features discriminative
of tissue state, and selecting which patches enter the pooled
representation. We decouple them. Weak spatial supervision derived from
percentage involvement, recorded routinely at biopsy, constrains the
predicted proportion of malignant tissue within each core, acting on the
encoder features independently of the alignment objective. The alignment
loss then operates on features that differ across a core, and attention
concentrates on a subset of patches rather than remaining near-uniform.
On 7,166 biopsy cores from 811 patients across seven centers under
patient-level 5-fold cross-validation, the method reaches 67.1 macro AUC
and 68.5 csPCa AUC, against 61.2 and 52.8 for the existing unpaired
alignment method and 63.1 and 62.6 for the strongest unimodal baselines.
Ablation against existing attention regularizers designed to prevent
attention-uniformity collapse shows that such regularizers do not
substitute for label-derived supervision: they constrain the attention
distribution, whereas the signal required acts on the features that
attention reads.

## Method

A DINOv3 ViT-L/16 encoder produces patch tokens per B-mode frame,
restricted to the annotated needle-trace region. Gated attention-based
multiple-instance learning (ABMIL) pools these into a bag embedding
`z = Aᵀ H`, which is aligned to a frozen histopathology (GigaPath)
embedding of the same ISUP grade group via a grade-distance-weighted
supervised contrastive loss (`L_align`).

The key contribution is decoupling the two jobs that objective is usually
asked to do at once — making patch features discriminative, and choosing
which of them the pooled embedding uses. A lightweight head predicts a
per-patch cancer probability; its needle-masked mean is matched to the
core's recorded percentage involvement via a proportion-matching BCE term
(`L_sg`, "spatial grounding"). This supplies discriminative structure to
the patch features directly — gradient reaches `H`, not the attention
weights `A` — while attention selection remains the sole job of
`L_align`, now operating on features that actually differ across a core.

```
L = L_align + λ_sg · L_sg
```

This is implemented as `WithinModalSupConLossv2` (`L_align`) +
`NeedleProportionBCE` (`L_sg`) — see `losses.py::build_alignus_loss` —
trained by `train_patched.py`.

## Results

**Table 1 — per-grade one-vs-rest AUC (×100, mean±std over 5 patient-level folds).**
Cross-modal methods use histopathology at training time only.

| Method | GG0 | GG1 | GG2 | GG3 | GG4 | GG5 | Macro | csPCa |
|---|---|---|---|---|---|---|---|---|
| DINO-FT | 64.5±2.0 | 69.2±4.6 | **59.2±3.2** | 57.9±3.5 | 65.0±7.2 | 60.8±12.6 | 62.3±3.3 | 62.6±2.5 |
| MicroSegNet-FT | 68.1±4.4 | 90.8±1.4 | 51.6±4.7 | 50.1±5.8 | 62.4±12.9 | 55.6±15.7 | 63.1±4.1 | 57.2±8.1 |
| MedSAM-FT | 66.9±4.5 | **91.4±1.3** | 52.2±3.1 | 50.4±7.2 | 59.4±11.2 | 54.0±14.3 | 62.3±3.8 | 55.9±5.9 |
| ProstNFound+ | 61.1±6.7 | 90.6±2.0 | 51.4±5.0 | 49.2±6.5 | 59.5±1.2 | 61.1±8.1 | 62.2±2.2 | 55.7±7.1 |
| GUIDE-US (cross-modal) | 65.5±4.8 | 91.2±2.1 | 50.7±1.6 | 49.0±7.2 | 57.3±11.2 | 53.4±12.2 | 61.2±3.8 | 52.8±6.7 |
| **Ours** | **69.9±1.4** | 83.7±6.7 | 59.0±6.2 | **59.8±5.4** | **67.5±8.2** | **62.7±10.5** | **67.1±4.1** | **68.5±2.7** |

**Table 2 — pooling and supervision ablation.** All rows share the
DINOv3 backbone, batch construction, and evaluation protocol; unless
noted the alignment objective is the supervised contrastive loss above.

| Configuration | Macro AUC | csPCa AUC |
|---|---|---|
| Micro-US only (no alignment) | 62.3±3.3 | 62.6±2.5 |
| Mean pooling, w/o L_sg | 63.4±1.3 | 65.2±3.4 |
| ABMIL, w/o L_sg | 65.0±2.9 | 66.9±3.1 |
| ABMIL + AEM | 65.8±3.3 | 66.6±2.2 |
| ABMIL + ACMIL | 65.8±3.9 | 66.4±2.6 |
| **ABMIL + L_sg (ours)** | **67.1±4.1** | **68.5±2.7** |
| ABMIL, triplet alignment, w/o L_sg | 63.9±4.2 | 54.3±7.3 |
| ABMIL, triplet alignment + L_sg | 64.2±5.4 | 63.2±2.4 |

Attention pooling accounts for 1.6 macro-AUC points over mean pooling;
adding the cross-modal alignment term over micro-US-only training
accounts for a further 2.7. Neither generic attention regularizer (AEM,
ACMIL) improves on unconstrained attention — both slightly reduce csPCa
AUC relative to plain ABMIL. `L_sg` is the only intervention that helps,
and its gain transfers (on csPCa) to a second, unrelated alignment
objective (triplet), showing the effect isn't specific to the
contrastive loss used elsewhere.

Full results, the representation-level mechanism check (attention
entropy / patch-feature dispersion), and discussion are in the paper.

## Repo layout

```
alignUS/
├── train_patched.py    — training script (backbone-agnostic: dino / medsam / microsegnet)
├── losses.py             — every loss function in the repo, in one file.
│                          build_alignus_loss (L_align, L_sg, the AEM/ACMIL
│                          regularizer hooks — train_patched.py only) and
│                          build_guidepnf_loss (guideus/pnf only) are
│                          independent entry points that never share a
│                          config or a class with each other — see the
│                          file's own docstring
├── diagnostics.py, extract_embeddings.py, make_figures.py,
│   plot_diagnostics.py, plot_embeddings.py
├── src/                 — self-contained project dependencies (dataset
│                          classes, ABMIL pooling, DINOv3 loading, the
│                          AEM loss component, the MicroSegNet/TransUnet
│                          encoder)
├── medAI/, external_libs/ — vendored copies of the shared library this
│                          project is built on (dataset loaders, transforms,
│                          losses, model factories, evaluators, and the
│                          SAM/MedSAM backbone code) — fully self-contained,
│                          nothing to clone or point at separately
├── cfgs/                — cfg_alignus/ (flagship), cfg_dino/, cfg_medsam/
│                          (backbone variants), cfg_triplet/ (triplet-loss
│                          ablation, Table 2's bottom two rows)
├── baseline/             — comparison methods from Table 1/2, see
│                          baseline/README.md
├── train.sh / debug.sh / extract_embeddings.sh
└── figures/, runs/       — plots and cached prediction/embedding outputs
```

## Setup

This repo is self-contained — `medAI/` and `external_libs/` (the shared
library this project is built on: dataset loaders, transforms, losses,
model factories, evaluators, and the SAM/MedSAM backbone code) are
vendored directly here, nothing external to clone or point `PYTHONPATH`
at. Install the Python dependencies and run everything from this repo's
root:

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`train.sh`/`debug.sh` expect `WANDB_API_KEY` to already be set in your
environment (never commit a real key to these scripts) and assume a
SLURM cluster with `module load python/3.12 cuda/12.2 opencv/4.12.0` —
drop those lines if you're not on one.

## Running

Everything is invoked with `python -m` **from this repo's root**:

```
python -m train_patched -c cfgs/cfg_alignus/alignus_f0.yaml [dotlist overrides...]
```

Baselines (see `baseline/README.md`):

```
python -m baseline.guideus.guideus_pnf_train -c baseline/guideus/guideus_cfg0.yaml   # guideus
python -m baseline.guideus.guideus_pnf_train -c baseline/pnf/cfg/pnf_cfg0.yaml       # pnf
python -m train_patched -c baseline/acmil/cfg/supcon_onlyf0.yaml                     # acmil
python -m train_patched -c baseline/aem/cfg/aem_f0.yaml                              # aem
python -m train_patched -c baseline/microsegnet/cfg/microseg_cfg0.yaml               # microsegnet
```

Configs are OmegaConf YAML with CLI dotlist overrides.

## Citation

```bibtex
@article{dzikunu2026weakly,
  title={Weakly Supervised Spatial Grounding for Discriminative Attention-Based Ultrasound-Histopathology Alignment in Prostate Cancer Grading},
  author={Dzikunu, Obed Korshie and Willis, Emma and Abootorabi, Mohammad Mahdi and Harmanani, Mohamed and Guo, Zhuoxin and Luger, Ferdinand and Kinnaird, Adam and Wodlinger, Brian and Mousavi, Parvin and Abolmaesumi, Purang},
  journal={arXiv preprint arXiv:2609.15150},
  year={2026}
}
```
