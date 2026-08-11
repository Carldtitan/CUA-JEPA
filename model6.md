# Model 6

## Direct purpose

Model 6 tests one question:

Can a separate JEPA world model improve Qwen's action choice without changing Qwen?

Qwen stays frozen. V-JEPA 2 also stays frozen. Only small Model 6 components learn.

## Architecture

Model 6 has two separate paths.

The policy path is:

1. Raw Qwen receives the task, action history, and current screen.
2. Qwen proposes up to four unique actions.
3. Qwen is not trained or changed.

The world-model path is:

1. Frozen V-JEPA 2 encodes the current screen once.
2. A small action encoder encodes each candidate action.
3. An AdaLN predictor predicts one future latent state for each action.
4. A goal-conditioned scorer gives each predicted future a value.
5. Model 6 selects the action with the highest value.

A latent state is a learned numeric representation of a screen. Model 6 predicts this representation. It does not generate a future screenshot.

## Trainable components

The dynamics stage trains these components:

- a residual online latent adapter;
- an action encoder;
- an action-conditioned AdaLN predictor;
- an inverse-dynamics head.

The target latent adapter uses an exponential moving average of the online adapter. The V-JEPA 2 encoder stays frozen.

The scorer stage trains these components:

- a goal-conditioned future scorer;
- an action-only control scorer.

Qwen, V-JEPA 2, and the saved dynamics components stay frozen during scorer training.

## Data

The dynamics stage uses the synthetic GUI transition bundles. Each bundle starts from one screen and has four different actions and four future screens.

The scorer stage uses AgentNet task data. AgentNet provides the task, current screen, action history, and correct next action.

The two datasets have different jobs:

- synthetic transitions teach what an action changes;
- AgentNet teaches which action helps a task.

The AgentNet training and validation sets are task-disjoint. A task in training cannot occur in validation.

## Required controls

Model 6 reports four decision systems on the same validation examples:

- raw Qwen greedy action;
- Model 6 future scorer;
- action-only scorer;
- shuffled-future scorer.

The action-only scorer tests whether action and task text are enough. The shuffled-future scorer gives each action the wrong predicted future. Model 6 must beat both controls before we say that predicted futures help action choice.

Training can add the correct action when Qwen does not propose a useful action. Validation never adds the correct action. Candidate recall at 4 gives the maximum score that any reranker can reach.

## Dynamics result

The full dynamics run used 30,668 training transitions and 2,000 held-out transitions.

- Forward four-way future matching: 37.15%.
- Forward 95% confidence interval: 35.10% to 39.15%.
- Inverse four-way action recovery: 45.80%.
- Inverse 95% confidence interval: 43.85% to 47.80%.
- Chance level: 25%.

The action-shuffle test reduced forward accuracy by 16.35 percentage points. The screen-shuffle test reduced it by 12.00 points. This shows that the predictor uses both the screen and the action.

The forward score is above chance, but it is not strong. An older frozen V-JEPA 2 predictor reached 44.65%. Model 6 has not yet shown a better dynamics result.

## Decision result

Model 6 did not improve Qwen's action choice.

| System | Seed 20260813 | Seed 20260814 |
|---|---:|---:|
| Raw Qwen greedy | 40.8% | 40.8% |
| Model 6 future scorer | 40.8% | 40.4% |
| Action-only control | 42.8% | 42.4% |
| Shuffled-future control | 44.0% | 44.0% |

For seed 20260813, Model 6 minus raw Qwen was 0.0 percentage points. The task-bootstrap 95% interval was -4.35 to +4.38 points.

For seed 20260814, Model 6 minus raw Qwen was -0.4 points. The task-bootstrap 95% interval was -4.82 to +3.92 points.

Both intervals include zero. There is no reliable improvement.

The shuffled-future control beat Model 6 in both seeds. This is strong evidence that the predicted future did not give the scorer useful decision information.

## Candidate result

On the 250 untouched validation examples:

- raw Qwen exact success was 40.8%;
- exact candidate recall at 4 was 54.4%;
- any-positive candidate recall was 56.8%;
- mean unique candidate count was 3.52;
- 76.0% of examples had four unique candidates;
- no validation oracle was inserted.

The 54.4% candidate recall was the maximum exact score available to any reranker.

## Inference measurements

The frozen feature cache measured 21.21 ms for V-JEPA 2 screen encoding and 1.20 ms for goal-text encoding per example.

Across the two full scorer seeds:

- batched JEPA prediction took 5.25 to 6.69 ms per candidate set;
- all three scorer controls together took 1.12 to 1.63 ms per candidate set;
- peak scorer GPU memory was 0.13 GiB.

The latent simulation is cheap. Candidate generation is not cheap. Raw Qwen greedy generation averaged 1.12 seconds. The batch of seven sampled candidates averaged 2.80 seconds.

## Research conclusion

Model 6 answered the main question with a negative result.

The separate JEPA world model learned measurable action dynamics. It also ran quickly. However, its predicted future latent states did not improve next-action selection on AgentNet.

Do not claim that Model 6 improves accuracy. The defensible result is:

1. Separate latent simulation avoided damage to Qwen.
2. Latent counterfactual prediction was much faster than Qwen candidate generation.
3. The present future representation and scorer did not add decision value.

## Saved locations

- Dynamics run: `/training/model6-dynamics-full-seed20260813-20260811T052443Z`
- Candidate run: `/training/model6-candidates-raw-qwen-seed20260813-20260811T055037Z`
- Scorer seed 20260813: `/training/model6-scorer-seed20260813-20260811T085544Z`
- Scorer seed 20260814: `/training/model6-scorer-seed20260814-20260811T085917Z`
- Modal volume: `cua-jepa-training-v1`

Verified local copies are in:

- `artifacts/model6/dynamics-files`;
- `artifacts/model6/candidates-files`;
- `artifacts/model6/scorer-seed20260813-files`;
- `artifacts/model6/scorer-seed20260814-files`.

The trained dynamics checkpoint is about 70 MB. Each trained scorer checkpoint is about 3.8 MB.
