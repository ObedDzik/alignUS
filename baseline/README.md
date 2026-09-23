# Baseline comparison methods

Five baselines are compared against the core AlignUS method (see the repo
root `README.md`, Tables 1–2). They are not five separate codebases — two
share a script with each other, and the other three share a script with
the core method itself, distinguished purely by config.

## guideus and pnf — share `baseline/guideus/guideus_pnf_train.py`

- **guideus**: `experiment_type: guideus`, triplet loss —
  `python -m baseline.guideus.guideus_pnf_train -c baseline/guideus/guideus_cfg0.yaml`
- **pnf**: `experiment_type: pnf`, plain cancer-detection loss, no
  alignment term — `python -m baseline.guideus.guideus_pnf_train -c baseline/pnf/cfg/pnf_cfg0.yaml`

## acmil and aem — share the root `train_patched.py`

Both are `train_patched.py` runs with different loss/flag configuration,
not separate scripts.

- **acmil**: `use_acmil: true` —
  `python -m train_patched -c baseline/acmil/cfg/supcon_onlyf0.yaml`
- **aem** (attention entropy maximization): `loss_type_reg: aem` —
  `python -m train_patched -c baseline/aem/cfg/aem_f0.yaml`

Both regularizers are implemented in `src/baseline_attention_reg.py`
(`ACMILHead`, `AttentionEntropyMaximization`) and wired into
`clean_loss.py::build_loss`.

## microsegnet — Table 1's MicroSegNet-FT, also via `train_patched.py`

`model_type: microsegnet` selects a ResNet50-ViT hybrid encoder
(`src/microseg_model.py::WrapperMicroSegNet`, wrapping the TransUnet
implementation vendored under `src/transunet/`) in place of the DINOv3/
MedSAM backbone —

`python -m train_patched -c baseline/microsegnet/cfg/microseg_cfg0.yaml`
