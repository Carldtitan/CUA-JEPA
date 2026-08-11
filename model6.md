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

The full decision result is pending. Do not use the two-example scorer smoke score as research evidence.

## Saved locations

- Dynamics run: `/training/model6-dynamics-full-seed20260813-20260811T052443Z`
- Active candidate run: `/training/model6-candidates-raw-qwen-seed20260813-20260811T055037Z`
- Modal volume: `cua-jepa-training-v1`

The full result must also be copied to the local `artifacts` directory after training.
