# CUA-JEPA

This MVP tests whether action-conditioned JEPA training helps a small vision-language model learn computer use.

The primary experiment compares four controlled model branches. See [the four-model design](PDFs%20and%20md%20files/Four_Model_Experiment.md).

The project learning log records mistakes, corrections, evidence, and open limits. See [learnings.md](learnings.md).

The full-run measurement plan is in [full_run_observability.md](full_run_observability.md).

The observability gate passed a two-step Modal test at commit `27af5ff`. The run ID was `model4-lora_smoke-seed20260809-20260810T052238Z`.

## Dataset milestone

The final synthetic dataset is in `data/synthetic/clean-20260808-v7/full`.

It contains:

- 34,444 transitions;
- 8,611 same-state bundles;
- four action branches in each bundle;
- 12 app-disjoint mock applications;
- zero exact screenshot leakage between splits;
- zero duplicate starting screenshots;
- zero duplicate actions or outcomes within a bundle;
- 22,418 clicks, 8,231 type actions, and 3,795 scroll actions.

Modal generation used persistent shards. A local backup is present in `data/synthetic`.

See [the dataset guide](PDFs%20and%20md%20files/Synthetic_Dataset.md).

## Model 4 JEPA dynamics pilot

The current dynamics pilot freezes the complete Qwen visual encoder. It trains only a small action encoder and predictor.

The predictor receives one current screen and four different actions. It predicts the latent change caused by each action:

```text
future latent - current latent
```

The pure JEPA objective contains:

- changed-region delta regression;
- a smaller global delta regression loss;
- bundle variance matching;
- bundle covariance matching;
- action-relation matching.

The action-separation experiment adds InfoNCE. It is reported as a separate objective.

Both experiments measure:

- four-way future accuracy;
- correct-action and shuffled-action distance;
- predicted action spread;
- target action spread;
- results by application and action type;
- repeated action-conditioning collapse checks.

## Run the pilot

Run the CPU dependency check and GPU smoke test first:

```powershell
python -m modal run modal_train_model4.py --mode deps
python -m modal run modal_train_model4.py --mode smoke --seed 20260809
```

Run the controlled 500-step experiments:

```powershell
python -m modal run modal_train_model4.py --mode pure --seed 20260809
python -m modal run modal_train_model4.py --mode separation --seed 20260809
```

Use a second seed to test run-to-run stability:

```powershell
python -m modal run modal_train_model4.py --mode pure --seed 20260810
python -m modal run modal_train_model4.py --mode separation --seed 20260810
```

Outputs persist in the `cua-jepa-training-v1` Modal Volume.

## Full Model 4 continued training

The full Model 4 run does not load a pilot checkpoint.

It starts from these new components:

- the pinned Qwen model revision;
- a new online vision LoRA adapter;
- a new target vision LoRA adapter;
- a new action encoder;
- a new predictor.

LoRA is a small trainable adapter. The base Qwen weights stay frozen.

The online LoRA adapter receives gradients. The target LoRA adapter receives an EMA update only. EMA means a slow moving copy of the online adapter.

The run checks these rules before training:

- base Qwen has zero trainable parameters;
- target LoRA has zero trainable parameters;
- the optimizer contains only online LoRA and the new heads;
- online LoRA and target LoRA start with identical values;
- training contains exactly 30,668 transitions;
- validation contains exactly 2,000 transitions;
- training and validation have no shared bundle or exact screenshot;
- the test split is not in the training paths.

Upload the audited training and validation data. Then run the LoRA smoke test:

```powershell
python scripts/upload_model4_full.py
python -m modal run modal_train_model4.py --mode lora_smoke --seed 20260809
```

Start the full run only after the smoke test passes:

```powershell
python -m modal run modal_train_model4.py --mode model4_full --seed 20260809
```

This run uses all 7,667 training bundles for one pass. It does not use the test split.

## Corrected pilot result

Both objectives ran with seeds `20260809` and `20260810`.

| Objective | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|
| Pure JEPA | 25.6% | 24.8% | 25.2% |
| JEPA plus InfoNCE | 36.0% | 34.8% | 35.4% |

Chance accuracy is 25%.

The non-contrastive losses prevented identical action predictions. They did not learn the correct action-to-future pairing. InfoNCE produced a repeated 10.2-point mean improvement over pure JEPA.

This is a dynamics-pilot result. It does not yet show that JEPA improves downstream computer-use task success.

## Important limit

The current data contains one-step transitions. It does not contain real consecutive multi-step screenshot sequences. Do not use a multi-step rollout loss until those sequences exist.
