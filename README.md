# CUA-JEPA

An MVP experiment testing whether action-conditioned JEPA pretraining helps a small vision-language model learn computer use more effectively than ordinary supervised fine-tuning or GUI exposure alone.

The primary experiment compares four controlled model branches. See [the four-model design](PDFs%20and%20md%20files/Four_Model_Experiment.md).

The first dataset is synthetic browser interaction data collected from resettable CUA-Gym-Hub mock applications. Each base-state bundle contains one identical starting screen, four executed actions, and the four resulting screens. Generated screenshots and credentials remain outside Git. See [the dataset guide](PDFs%20and%20md%20files/Synthetic_Dataset.md).

## Dataset milestone

The clean v4 dataset is complete and deeply audited:

- 32,000 train transitions
- 2,000 validation transitions
- 2,000 held-out test transitions
- 100-bundle quality gate before scaling
- Modal generation with persistent shards and local backup
- 12 app-disjoint mock applications
- zero exact screenshot leakage between splits
- zero duplicate actions or duplicate outcomes within a bundle

## Model 4 JEPA pilot

The first paid training gate uses one audited 100-transition training shard and one held-out
validation shard. It updates LoRA adapters in Qwen's vision encoder while a small
action-conditioned predictor learns to predict the EMA target encoder's future-screen latents.

Run the CPU-only processor check, followed by the two-step GPU smoke test:

```powershell
python -m modal run modal_train_model4.py --mode deps
python -m modal run modal_train_model4.py --mode smoke
```

After it passes, run the 100-step pilot with `--mode pilot`. Outputs persist in the
`cua-jepa-training-v1` Modal Volume. Four-way future selection is evaluation only; training uses
direct non-contrastive latent prediction.

The diverse Stage 2 gate uses 2,000 training transitions across all eight train applications and
500 held-out Jira/Slack transitions:

```powershell
python -m modal run modal_train_model4.py --mode stage2
```

## Model 4 JEPA pilot

The first paid training gate uses one audited 100-transition training shard and one held-out
validation shard. It updates LoRA adapters in Qwen's vision encoder while a small
action-conditioned predictor learns to predict the EMA target encoder's future-screen latents.

Run the two-step remote smoke test with:

```powershell
python -m modal run modal_train_model4.py --mode smoke
```

After the smoke test passes, run the 100-step pilot with `--mode pilot`. Outputs are persisted in
the `cua-jepa-training-v1` Modal Volume. The four-way future selection is evaluation only; the
training objective is direct non-contrastive latent prediction.
