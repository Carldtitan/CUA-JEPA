# Full Model 4 Observability Plan

Do not start the paid full run until the required items in this plan exist.

Observability means enough saved information to explain what happened during training.

## 1. Run identity

Save these values before training:

- run ID;
- UTC start time;
- Git commit;
- Modal app ID;
- run mode;
- random seed;
- Qwen model ID;
- exact Qwen revision;
- complete training config;
- Python, PyTorch, Transformers, PEFT, and CUDA versions;
- GPU type and GPU count.

These values make the run reproducible.

## 2. Data identity and split safety

Save these values before training:

- dataset version;
- dataset audit file hash;
- tar-file count for each split;
- transition count for each split;
- bundle count for each split;
- application list for each split;
- action count by click, type, and scroll;
- changed-pixel distribution;
- bundle overlap count;
- exact screenshot overlap count;
- proof that no test path entered training.

Save the exact ordered bundle IDs used for training and evaluation.

## 3. Parameter and optimizer safety

Save these values before training:

- base Qwen parameter count;
- base Qwen trainable parameter count;
- online LoRA parameter count;
- target LoRA trainable parameter count;
- action encoder parameter count;
- predictor parameter count;
- optimizer parameter count;
- initial online-to-target LoRA difference;
- hash of each initial trainable state dictionary.

The required values are:

- base Qwen trainable parameters: `0`;
- target LoRA trainable parameters: `0`;
- initial online-to-target LoRA difference: `0`.

## 4. Training record for each log interval

Save one JSONL record every 25 or 50 bundle updates.

Each record must include:

- update number;
- completed data fraction;
- current application;
- action mix since the last record;
- total loss;
- changed-region regression loss;
- global regression loss;
- variance loss;
- covariance loss;
- relation loss;
- InfoNCE loss;
- learning rate;
- total gradient norm before clipping;
- online LoRA gradient norm;
- action encoder gradient norm;
- predictor gradient norm;
- online LoRA parameter norm;
- online-to-target LoRA difference;
- current and peak GPU memory;
- updates per second;
- transitions per second;
- elapsed time;
- estimated remaining time;
- estimated compute cost;
- NaN or infinity count.

Do not log only the combined loss. The separate losses explain which objective changed.

## 5. Fixed evaluation checkpoints

Evaluate the same fixed validation bundles at these points:

- step `0`;
- step `100`;
- step `250`;
- step `500`;
- every `500` steps after that;
- the final step.

For each checkpoint, save:

- four-way accuracy;
- correct-action distance;
- shuffled-action distance;
- shuffled-minus-correct gap;
- prediction action spread;
- target action spread;
- prediction-to-target spread ratio;
- delta prediction loss;
- target latent variance;
- predicted latent variance;
- accuracy by application;
- accuracy by action type;
- accuracy by changed-pixel range.

Use fixed bundle IDs. Changing the evaluation sample adds noise.

## 6. Per-bundle evaluation records

Save one row for every evaluated bundle.

Each row must include:

- bundle ID;
- application;
- split;
- four action records;
- target future selected for each action;
- predicted rank of the correct future;
- four prediction-to-future distances;
- correct or incorrect result;
- changed-pixel fraction;
- action type;
- click coordinates or scroll amount;
- typing text category, but not private text.

These rows support bundle bootstrap confidence intervals and error analysis.

## 7. Collapse and shortcut checks

Track these failure signals:

- predictions become almost identical across actions;
- predicted spread becomes much smaller than target spread;
- shuffled actions perform like correct actions;
- four-way accuracy stays near 25%;
- target latent variance becomes very small;
- loss falls while held-out accuracy does not improve;
- train accuracy rises while validation accuracy stays flat;
- one action type causes most of the gain;
- unchanged regions dominate the loss;
- online LoRA changes while the action encoder receives almost no gradient.

Save the stop reason if an automatic collapse check stops training.

## 8. Checkpoints and crash recovery

Save a recoverable checkpoint every 500 updates.

Each checkpoint must contain:

- online LoRA weights;
- target LoRA weights;
- action encoder weights;
- predictor weights;
- optimizer state;
- random-number states;
- completed update number;
- ordered data position;
- config;
- metrics collected so far.

Keep milestone checkpoints. Keep only one replaceable latest checkpoint between milestones.

Record whether a result came from a fresh run or a resumed run.

## 9. Time and cost

Track these values separately:

- startup time;
- data-audit time;
- model-load time;
- initial-evaluation time;
- training time;
- checkpoint time;
- final-evaluation time;
- GPU seconds;
- CPU seconds;
- memory GiB-seconds;
- estimated Modal cost;
- actual Modal cost after the run.

Set a maximum runtime before launch. Stop the run when the approved limit is reached.

## 10. Files required after the run

The final run directory must contain:

- `run_manifest.json`;
- `dataset_audit.json`;
- `initialization_audit.json`;
- `train.jsonl`;
- `evaluation_checkpoints.jsonl`;
- `evaluation_bundles.jsonl`;
- `resource_usage.jsonl`;
- `final_metrics.json`;
- online LoRA weights;
- target LoRA weights;
- JEPA head weights;
- checkpoint metadata;
- a file with the stop reason.

Copy the final research artifacts from Modal to the local `artifacts` directory.

## 11. Research-paper reporting

Report these values even when the result is negative:

- all random seeds;
- every stopped or failed run;
- the planned and actual number of updates;
- the selected checkpoint rule;
- train and validation curves;
- the final held-out result;
- results by application and action type;
- confidence intervals from bundle resampling;
- compute time and estimated cost;
- all changes made after seeing pilot results.

Do not select the best checkpoint using the test split. Keep the test split untouched until the final comparison.
