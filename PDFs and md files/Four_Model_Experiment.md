# Four-Model JEPA Computer-Use MVP

## Research question

Does learning the correct relationship between a GUI action and its resulting screen make the same Qwen model better at computer use, beyond any benefit from simply seeing more GUI screenshots?

## The four models

### Model 1 - Original Qwen reference

- Start from the selected Qwen checkpoint.
- No GUI transition pretraining.
- No computer-use SFT.
- Evaluate it zero-shot on the same held-out evaluation set.

Purpose: measure what the original model can already do.

### Model 2 - Normal SFT baseline

- Start from the same original Qwen checkpoint.
- No JEPA or GUI transition pretraining.
- Train on the agreed computer-use SFT dataset.
- Evaluate on the same held-out evaluation set.

Purpose: measure the benefit of ordinary computer-use SFT.

### Model 3 - No-action GUI-exposure control

- Start from the same original Qwen checkpoint.
- Use the same before-screen and after-screen transition examples as Model 4.
- Use the same JEPA architecture, number of updates, and training settings as Model 4.
- Replace every action with the same fixed `NO_ACTION` input. This lets the model see the identical images and transition targets without giving it the information about which action produced each result.
- Then train on exactly the same computer-use SFT dataset as Model 2 and Model 4.
- Evaluate on the same held-out evaluation set.

Purpose: measure whether improvement comes merely from extra GUI exposure, extra optimization, or generic temporal screen prediction rather than from learning what actions do.

### Model 4 - Correct-action JEPA plus SFT

- Start from the same original Qwen checkpoint.
- Train on the GUI transitions with the correct action attached to every before-screen and after-screen pair.
- Use the same JEPA architecture, number of updates, and training settings as Model 3.
- Then train on exactly the same computer-use SFT dataset as Model 2 and Model 3.
- Evaluate on the same held-out evaluation set.

Purpose: test whether correct action-conditioned GUI-dynamics learning improves the final computer-use policy.

## The comparisons and what they mean

| Comparison | Question answered |
|---|---|
| Model 2 vs Model 1 | Does ordinary computer-use SFT help? |
| Model 3 vs Model 2 | Does extra GUI exposure or transition pretraining help without correct action information? |
| Model 4 vs Model 3 | Does learning the correct action-to-consequence relationship add value? |
| Model 4 vs Model 2 | Does the complete JEPA-first training recipe outperform ordinary SFT? |

Model 4 versus Model 3 is the most important comparison for attributing any improvement specifically to action-conditioned dynamics learning.

## Variables that must remain controlled

Models 2, 3, and 4 must use:

- the same starting Qwen checkpoint;
- the same LoRA configuration for comparable stages;
- the same SFT examples, ordering, number of epochs, and optimizer settings;
- the same normalized action representation;
- the same image resolution and preprocessing;
- the same held-out evaluation examples and scoring code;
- the same random seeds where practical;
- approximately matched pretraining updates and compute for Models 3 and 4.

The `NO_ACTION` representation must be fixed and identical for every Model 3 transition. Action shuffling can still be used as an evaluation diagnostic, but it is not the primary Model 3 training condition because deliberately false action labels could actively damage the control model.

## Data received by each model

| Model | Correct transition data | No-action transition data | SFT data |
|---|---:|---:|---:|
| Model 1 | No | No | No |
| Model 2 | No | No | Yes |
| Model 3 | No | Yes | Yes |
| Model 4 | Yes | No | Yes |

## Exact data assignment for the MVP

### JEPA pretraining data

Use the custom same-state synthetic dataset, not AgentNet:

```text
data/synthetic/clean-20260808-v7/full
```

| Use | Apps | Bundles | Transitions |
|---|---:|---:|---:|
| Pretraining | 8 train apps | 7,667 | 30,668 |
| Early stopping/model selection | Jira and Slack | 500 | 2,000 |
| Final held-out dynamics test | Outlook and Trello | 444 | 1,776 |

Model 4 receives each training transition as `current screen + correct action -> future screen`. Model 3 receives the exact same transitions in the exact same order, but every action is replaced with `NO_ACTION`. Models 1 and 2 receive none of this data. Validation and test transitions must never be used for gradient updates.

### SFT data

Use a separate 2,000-example subset of AgentNet for SFT, plus 250 validation examples from disjoint task IDs. This subset has not yet been downloaded or prepared locally. Keep only examples from completed, high-quality trajectories whose individual step is marked correct and not redundant. Limit the number of steps taken from any one task and balance operating systems, domains, and action types.

Format each SFT example as `task instruction + current screenshot + short action history -> next normalized action`. Use action-and-code targets only; do not train on AgentNet's synthesized thoughts as if they were ground truth. Models 2, 3, and 4 must receive the exact same 2,000 training examples, validation examples, ordering, and SFT settings. Model 1 receives no SFT.

The custom branch data must not be used as SFT data: its actions were sampled to expose different consequences, not selected because they advance a user task. AgentNet is unsuitable for the controlled JEPA pretraining comparison because it normally shows only the demonstrated action from a state, but that is not a problem for SFT, where the demonstrated action is precisely the target label.

## Direct dynamics check

Before downstream SFT results are interpreted, Models 3 and 4 should be tested on held-out same-state bundles:

```text
one starting screen
one proposed action
four possible resulting screens
```

The model must identify the result caused by the proposed action. With four candidates, chance performance is 25 percent. Model 4 should outperform Model 3 here; otherwise there is little evidence that correct action consequences were learned.

## Optional experiment after the main four models

If the main experiment shows a signal, repeat Models 2 and 4 at smaller nested SFT budgets, such as 250 and 1,000 examples. This is a separate learning-curve experiment testing whether JEPA reduces the amount of task-labeled SFT data required. It must not replace Model 3 in the primary controlled experiment.
