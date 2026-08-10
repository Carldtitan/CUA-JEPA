# CUA-JEPA

This MVP tests whether action-conditioned JEPA training helps a small vision-language model learn computer use.

The primary experiment compares four controlled model branches. See [the four-model design](PDFs%20and%20md%20files/Four_Model_Experiment.md).

The project learning log records mistakes, corrections, evidence, and open limits. See [learnings.md](learnings.md).

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

## Important limit

The current data contains one-step transitions. It does not contain real consecutive multi-step screenshot sequences. Do not use a multi-step rollout loss until those sequences exist.
