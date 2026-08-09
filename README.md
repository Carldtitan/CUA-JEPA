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
