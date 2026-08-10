# CUA-JEPA Learning Log

This is a living record of mistakes, corrections, and open limits in the CUA-JEPA MVP.

Each item gives the technical term and a simple explanation. The log records what happened. It does not treat an early assumption as a proven fact.

## Current experiment

The main question is:

> Does correct action-conditioned GUI transition learning improve the same Qwen model beyond ordinary SFT and extra GUI exposure?

The controlled experiment has four models:

1. Untouched Qwen.
2. Qwen with computer-use SFT.
3. Qwen with no-action JEPA training, followed by the same SFT.
4. Qwen with correct-action JEPA training, followed by the same SFT.

Model 4 versus Model 3 is the main causal comparison. It tests whether correct action information adds value.

## Corrected dynamics pilot results

The corrected pilot used 488 training bundles and all 125 held-out bundles. Each run used 500 updates.

| Objective | Seed | Train accuracy | Held-out accuracy | Shuffled-action gap | Prediction/target spread |
|---|---:|---:|---:|---:|---:|
| Pure JEPA | 20260809 | 19.0% | 25.6% | 0.00245 | 0.332 |
| Pure JEPA | 20260810 | 18.0% | 24.8% | 0.00271 | 0.430 |
| JEPA plus InfoNCE | 20260809 | 56.0% | 36.0% | 0.01137 | 0.409 |
| JEPA plus InfoNCE | 20260810 | 63.0% | 34.8% | 0.01349 | 0.575 |

The mean pure JEPA accuracy was 25.2%. This equals the 25% chance level.

The mean JEPA plus InfoNCE accuracy was 35.4%. The mean paired improvement was 10.2 percentage points.

This result is a useful dynamics-learning signal. It is not yet evidence that Model 4 improves downstream computer-use task success.

The local source metrics are:

- `artifacts/model4-pure-seed20260809-20260810T022122Z/metrics.json`
- `artifacts/model4-pure-seed20260810-20260810T023405Z/metrics.json`
- `artifacts/model4-separation-seed20260809-20260810T022748Z/metrics.json`
- `artifacts/model4-separation-seed20260810-20260810T024235Z/metrics.json`

## Research-design learnings

### 1. The original thesis included three different claims

- **Technical term:** Hypothesis decomposition.
- **Mistake:** We discussed accuracy, data cost, and inference cost as one result.
- **Simple explanation:** One experiment cannot prove three different benefits at once.
- **Correction:** Measure final accuracy, SFT sample efficiency, and inference cost separately.
- **Status:** Fixed in the four-model design. Final measurements are still open.

### 2. More GUI data can look like action learning

- **Technical term:** Exposure confound.
- **Mistake:** Model 4 could improve only because it saw more GUI screenshots.
- **Simple explanation:** More practice images can help even when the model does not learn what actions do.
- **Correction:** Model 3 sees the same screens, steps, and compute. It receives one fixed `NO_ACTION` input.
- **Status:** Design fixed. Models 2, 3, and 4 are not yet complete.

### 3. One observed action per screen is not a strong dynamics test

- **Technical term:** Observational confounding.
- **Mistake:** AgentNet was first considered for JEPA transition training.
- **Simple explanation:** The model can guess the normal next screen without using the action.
- **Correction:** Use four different actions from the same starting screen.
- **Status:** Fixed for JEPA data.

### 4. AgentNet still has a valid use

- **Technical term:** Dataset-task alignment.
- **Mistake:** We first treated AgentNet as either useful for everything or useful for nothing.
- **Simple explanation:** AgentNet is weak for the causal transition test. It can still provide correct next-action labels for SFT.
- **Correction:** Use custom same-state data for JEPA. Use a separate high-quality AgentNet subset for SFT.
- **Status:** Design fixed. The SFT subset is not yet prepared.

### 5. Transition actions are not policy labels

- **Technical term:** Objective mismatch.
- **Mistake:** Custom exploration actions could have been reused as SFT targets.
- **Simple explanation:** An action that shows a useful screen change is not always the correct action for a user task.
- **Correction:** Keep custom transitions in JEPA training only. Use successful task actions for SFT.
- **Status:** Fixed in the experiment design.

### 6. Automatically observed data is not annotation-free

- **Technical term:** Automatic supervision.
- **Mistake:** The project could have claimed that the data needs no annotation.
- **Simple explanation:** The next screen is recorded automatically. The pipeline still needs action definitions, resets, filters, and quality checks.
- **Correction:** Say that the data needs less manual labeling. Do not say that it is free to create.
- **Status:** Fixed in the dataset documentation.

### 7. Lower JEPA inference cost is not automatic

- **Technical term:** Inference-path accounting.
- **Mistake:** We treated latent prediction as proof of cheaper inference.
- **Simple explanation:** JEPA training can be cheap while the final policy costs the same to run.
- **Correction:** Remove the JEPA predictor after pretraining if it is not needed. Measure policy latency and GPU use directly.
- **Status:** Open measurement.

### 8. A small gain can be random noise

- **Technical term:** Statistical uncertainty.
- **Mistake:** One small run could have been treated as a reliable result.
- **Simple explanation:** A small dataset or one random seed can produce a lucky score.
- **Correction:** Use held-out applications, complete bundles, repeated seeds, confidence intervals, and learning curves.
- **Status:** Partly fixed. Multi-seed training is now supported.

### 9. Application-disjoint testing is necessary

- **Technical term:** Out-of-distribution evaluation.
- **Mistake:** Random transition splits could let the model see the same application style during training and testing.
- **Simple explanation:** A model can memorize one interface instead of learning general software behavior.
- **Correction:** Hold out complete applications for validation and test.
- **Status:** Fixed in the synthetic dataset splits.

### 10. The four-way test is a gate, not the final claim

- **Technical term:** Proxy evaluation.
- **Mistake:** Four-way future selection could have been treated as proof of better computer use.
- **Simple explanation:** Predicting a future screen is not the same as completing a user task.
- **Correction:** Use four-way accuracy to test dynamics. Use downstream SFT and task success for the main result.
- **Status:** Fixed in the experiment design.

## Data-generation learnings

### 11. Remote generation must not depend on one laptop connection

- **Technical term:** Fault tolerance.
- **Mistake:** A Wi-Fi outage interrupted local control of the overnight run.
- **Simple explanation:** The collection job must survive a laptop or network failure.
- **Correction:** Write completed shards to a persistent Modal Volume. Download verified copies to `data/synthetic`.
- **Status:** Fixed.

### 12. A large job needs resumable shards

- **Technical term:** Checkpointed data generation.
- **Mistake:** One large output could be lost or repeated after a failure.
- **Simple explanation:** A failed final step must not destroy many hours of completed work.
- **Correction:** Store small tar shards, sidecar manifests, checksums, and completed-shard state.
- **Status:** Fixed.

### 13. A page reload did not always restore the same state

- **Technical term:** State-reset nondeterminism.
- **Mistake:** Early branches sometimes started from different hidden or visible states.
- **Simple explanation:** Four actions are not comparable unless all four start from the same screen and state.
- **Correction:** Reconstruct the post-warmup state. Retry strict resets. Measure reset differences for each accepted branch.
- **Status:** Fixed within the accepted tolerance.

### 14. Stored checksums did not always describe the stored files

- **Technical term:** Artifact-integrity mismatch.
- **Mistake:** Early screenshot checksums could be calculated from different bytes than the saved image.
- **Simple explanation:** A checksum is useful only when it describes the exact saved file.
- **Correction:** Calculate checksums from the final stored bytes. Audit each tar member.
- **Status:** Fixed.

### 15. Application state could leak into another application

- **Technical term:** Cross-application contamination.
- **Mistake:** Shared mock storage could carry state between applications.
- **Simple explanation:** A GitHub example must not contain state created by another mock application.
- **Correction:** Isolate app state and add contamination checks.
- **Status:** Fixed.

### 16. Some mock applications were unstable

- **Technical term:** Environment instability.
- **Mistake:** Early apps did not reset or render reliably.
- **Simple explanation:** Unstable software creates false transition targets.
- **Correction:** Patch broken assets, replace unstable mocks, and reject unstable branches.
- **Status:** Fixed for the retained dataset.

### 17. Four selected actions did not always create four useful outcomes

- **Technical term:** Counterfactual branch quality.
- **Mistake:** Distinct action coordinates could still produce duplicate or ineffective results.
- **Simple explanation:** Four different clicks are not useful when they all leave the screen unchanged.
- **Correction:** Search for four effective actions. Reject duplicate actions, duplicate outcomes, and invisible changes.
- **Status:** Fixed in the final audit.

### 18. Identical visible inputs could have conflicting targets

- **Technical term:** Hidden-state aliasing.
- **Mistake:** An early audit found 74 identical screenshot-action pairs with different future screens. This affected 368 transitions.
- **Simple explanation:** The model received the same visible input but was asked to predict different answers.
- **Correction:** Remove repeated starting states and conflicting input-action pairs.
- **Status:** Fixed in `clean-20260808-v7`.

### 19. Duplicate starting screens reduced effective diversity

- **Technical term:** Input deduplication.
- **Mistake:** Different bundles could start from the exact same screenshot.
- **Simple explanation:** Repeated images make the dataset look larger than it is.
- **Correction:** Remove 389 later bundles whose starting screenshot matched an earlier bundle.
- **Status:** Fixed. The final dataset has 8,611 bundles and 34,444 transitions.

### 20. Typing text was highly repetitive

- **Technical term:** Low lexical diversity.
- **Mistake:** All 8,231 early typing actions used the same `Synthetic ...` pattern.
- **Simple explanation:** The model could memorize one text pattern instead of learning general typing behavior.
- **Correction:** Replay every typing branch with deterministic, field-aware text. Retarget covered or unsuitable fields.
- **Status:** Fixed for text repetition.

### 21. Typing targets are still not balanced

- **Technical term:** Action-context imbalance.
- **Mistake:** About 92% of typing actions still use search or filter fields.
- **Simple explanation:** The dataset teaches search typing much more often than forms, messages, names, or titles.
- **Correction:** Collect new starting states with diverse editable controls. Regenerating text alone cannot fix this distribution.
- **Status:** Open limitation.

### 22. Element hints were too generic

- **Technical term:** Weak auxiliary metadata.
- **Mistake:** Many element hints were only `BUTTON` or `INPUT`.
- **Simple explanation:** These labels do not describe what the control does.
- **Correction:** Do not use element hints as a core model input or claim. The screenshots and actions remain the core data.
- **Status:** Removed from the main learning objective.

### 23. Exact split leakage needed an explicit audit

- **Technical term:** Train-test leakage.
- **Mistake:** App-disjoint splits alone did not prove that screenshots were unique across splits.
- **Simple explanation:** The same image in training and testing can make evaluation too easy.
- **Correction:** Compare exact screenshot hashes across train, validation, and test.
- **Status:** Fixed. The final audit found zero exact screenshot leakage.

### 24. Repeated future screens are not always errors

- **Technical term:** Many-to-one transitions.
- **Mistake:** Global duplicate outcomes could be confused with conflicting labels.
- **Simple explanation:** Different starting states can correctly reach the same screen.
- **Correction:** Reject duplicate outcomes within one bundle. Permit valid repeated outcomes across unrelated bundles.
- **Status:** Fixed in the auditor.

### 25. The current data has no true multi-step sequences

- **Technical term:** Horizon limitation.
- **Mistake:** We planned a multi-step rollout loss before confirming that intermediate screenshots existed.
- **Simple explanation:** One-step branches cannot train a real multi-step rollout.
- **Correction:** Do not invent missing frames. Collect consecutive state-action-next-state sequences for a later experiment.
- **Status:** Open data limitation.

### 26. Synthetic mock applications limit external validity

- **Technical term:** Simulation-to-real gap.
- **Mistake:** Synthetic success could have been described as success on real software.
- **Simple explanation:** Resettable mocks are controlled, but they are not live websites or desktop applications.
- **Correction:** Treat this dataset as an MVP dynamics test. Later evaluate on real, held-out applications.
- **Status:** Open limitation.

## Model and training learnings

### 27. This is continued pretraining, not foundation-model pretraining

- **Technical term:** Continued self-supervised pretraining.
- **Mistake:** The word `pretraining` caused confusion because Qwen was already pretrained.
- **Simple explanation:** We are not training Qwen from the beginning. We add a JEPA training stage before computer-use SFT.
- **Correction:** Call it JEPA continued pretraining or JEPA intermediate training.
- **Status:** Terminology fixed.

### 28. Updating Qwen during the first dynamics pilot added a confound

- **Technical term:** Representation-drift confound.
- **Mistake:** An early Stage 2 run trained Qwen vision LoRA while it also trained the predictor.
- **Simple explanation:** We could not tell whether a result came from new visual features or action dynamics.
- **Correction:** Freeze the complete Qwen visual encoder for the controlled dynamics pilot.
- **Status:** Fixed for current pure and separation runs.

### 29. One weak action injection was easy to ignore

- **Technical term:** Conditioning underuse.
- **Mistake:** The first predictor added action information too weakly.
- **Simple explanation:** The visual tokens were strong enough to dominate one small action signal.
- **Correction:** Apply action-conditioned adaptive layer normalization in every predictor block.
- **Status:** Fixed.

### 30. Click coordinates were not tied to screen locations

- **Technical term:** Spatial grounding failure.
- **Mistake:** A click changed all visual tokens in the same general way.
- **Simple explanation:** A click at the top left must affect a different location than a click at the bottom right.
- **Correction:** Add a coordinate heatmap and spatial offsets to the visual-token grid.
- **Status:** Fixed.

### 31. Unchanged pixels dominated the objective

- **Technical term:** Sparse-change imbalance.
- **Mistake:** Most GUI pixels stay unchanged after one action.
- **Simple explanation:** The model could score well by copying the background and ignoring the action result.
- **Correction:** Put most regression weight on changed screen regions. Keep a smaller global loss.
- **Status:** Fixed in the current loss.

### 32. A direct copy path made the shortcut easier

- **Technical term:** Identity shortcut.
- **Mistake:** The first predictor included a direct current-to-future residual path.
- **Simple explanation:** The model could copy the current screen without learning the action.
- **Correction:** Remove the direct predictor copy path.
- **Status:** Fixed.

### 33. Independent transition training hid the counterfactual structure

- **Technical term:** Bundle-structure loss.
- **Mistake:** The first training design could process each branch as an unrelated example.
- **Simple explanation:** The model did not directly see that four actions from one screen caused four different futures.
- **Correction:** Train all four branches of each bundle together.
- **Status:** Fixed.

### 34. One hundred steps were not enough to judge the architecture

- **Technical term:** Undertraining.
- **Mistake:** A 100-step pilot could only test whether the code ran.
- **Simple explanation:** A smoke test cannot show whether the method works.
- **Correction:** Use smoke tests for code. Use at least 500 bundle updates for the research pilot.
- **Status:** Fixed for the last controlled pilots.

### 35. We did not apply the article's anti-collapse warning early enough

- **Technical term:** Missing collapse prevention.
- **Mistake:** The original article explicitly warned that latent prediction can collapse. The first loss used regression without a suitable action-sensitive regularizer.
- **Simple explanation:** We read the warning but did not turn it into a required test and loss.
- **Correction:** Add variance, covariance, and action-relation regularization. Keep InfoNCE as a separate comparison.
- **Status:** Fixed in the current code. The corrected pure runs no longer produced nearly identical action predictions.

### 36. We first used the wrong collapse label

- **Technical term:** Action-conditioning collapse.
- **Mistake:** The failure could have been called complete representation collapse.
- **Simple explanation:** Qwen's target representations stayed different. The predictor produced almost the same answer for each action.
- **Correction:** Call this action-conditioning collapse. It also includes shortcut learning and regression to the mean.
- **Status:** Terminology fixed.

### 37. Predicting the complete future encouraged an average future

- **Technical term:** Conditional-mean regression.
- **Mistake:** The predictor learned the full future representation from a mostly unchanged current screen.
- **Simple explanation:** The easiest answer was an average screen that ignored the action.
- **Correction:** Predict `future representation - current representation`. This is delta prediction.
- **Status:** Fixed in the current code. Delta loss decreased in both repeated runs.

### 38. A falling regression loss did not prove action learning

- **Technical term:** Metric-objective mismatch.
- **Mistake:** The loss decreased while four-way accuracy stayed near chance.
- **Simple explanation:** A model can reduce average error without learning which action caused which future.
- **Correction:** Track four-way accuracy, shuffled-action gap, predicted action spread, target spread, and their ratio.
- **Status:** Fixed in evaluation.

### 39. The first Stage 2 evaluation used only 25 held-out bundles

- **Technical term:** Evaluation sampling error.
- **Mistake:** The first Stage 2 report used a small held-out subset.
- **Simple explanation:** A small evaluation can hide instability and increase noise.
- **Correction:** Evaluate the final pilot on all 125 held-out bundles, or 500 transitions.
- **Status:** Fixed in later pure and separation runs.

### 40. Pure latent regression collapsed to chance

- **Technical term:** Predictor collapse.
- **Mistake:** The 500-step pure regression pilot produced almost identical predictions across actions.
- **Simple explanation:** Four-way validation accuracy was 25.2%, near the 25% chance level.
- **Evidence:** Prediction action distance was about `0.0020`. The target distance was about `0.3875`.
- **Correction:** Add non-contrastive bundle regularization first. Then compare with action separation.
- **Status:** Repeated. The corrected pure runs scored 25.6% and 24.8%. They stayed at chance.

### 41. InfoNCE improved separation but changed the objective

- **Technical term:** Contrastive action separation.
- **Mistake:** InfoNCE could have been presented as the same pure JEPA objective.
- **Simple explanation:** InfoNCE explicitly treats the other bundle futures as incorrect choices.
- **Evidence:** Validation accuracy increased to 29.4%. Prediction action distance increased to about `0.0272`.
- **Correction:** Report pure JEPA and JEPA plus action separation as separate experiments.
- **Status:** Repeated. The corrected InfoNCE runs scored 36.0% and 34.8%.

### 42. The first separation result showed overfitting

- **Technical term:** Generalization gap.
- **Mistake:** A 29.4% validation result could be reported without its 52% training result.
- **Simple explanation:** The model learned the training bundles much better than unseen bundles.
- **Correction:** Report both scores. Add repeated seeds and results by application and action type.
- **Status:** Confirmed. Mean training accuracy was 59.5%. Mean held-out accuracy was 35.4%.

### 43. Collapse checks happened after training

- **Technical term:** Online collapse detection.
- **Mistake:** We spent a complete run before measuring action sensitivity.
- **Simple explanation:** The run should stop when predictions stay identical and accuracy stays at chance.
- **Correction:** Check four-way accuracy, shuffled-action gap, and separation ratio during training. Stop after repeated failed checks.
- **Status:** Fixed in the current code.

### 44. A random latent variable is not needed for this deterministic MVP

- **Technical term:** Latent uncertainty variable.
- **Mistake:** JEPA's optional latent variable could be added before the action signal works.
- **Simple explanation:** The action should identify the correct future in this controlled data.
- **Correction:** Do not add a latent uncertainty variable now. Add one only when one state-action pair has several valid futures.
- **Status:** Correctly excluded.

### 45. Multi-step rollout loss needs real sequences

- **Technical term:** Temporal rollout supervision.
- **Mistake:** A multi-step loss was requested before the dataset supported it.
- **Simple explanation:** Repeating a one-step prediction is not the same as training on real intermediate screens.
- **Correction:** Keep the current pilot one-step. Add multi-step loss only after consecutive sequences are collected.
- **Status:** Open future task.

## Operating and documentation learnings

### 46. Two Modal jobs could start when one was intended

- **Technical term:** Duplicate remote execution.
- **Mistake:** The Modal dashboard showed two active GPU jobs during an earlier test.
- **Simple explanation:** Repeated launches can spend money twice.
- **Correction:** Start controlled runs one at a time. Record each run ID. Confirm completion before the next launch.
- **Status:** Current run procedure corrected.

### 47. Documentation became stale after architecture changes

- **Technical term:** Documentation drift.
- **Mistake:** The README still said that the current pilot updated Qwen vision LoRA.
- **Simple explanation:** The code froze Qwen, but the main page described an older design.
- **Correction:** Update the README with the current frozen-Qwen and delta-prediction design.
- **Status:** Being fixed with this update.

### 48. The Windows Python launchers were broken

- **Technical term:** Interpreter-path drift.
- **Mistake:** `py`, `pip`, and `modal.exe` pointed to removed Python 3.13 or 3.14 installations.
- **Simple explanation:** The commands tried to start Python files that no longer existed.
- **Correction:** Use the working Python 3.12 interpreter and run tools as `python -m ...`.
- **Status:** Workaround active.

### 49. A global pytest plugin blocked local tests

- **Technical term:** Test-environment contamination.
- **Mistake:** Pytest loaded an unrelated global plugin with a broken OpenAI dependency.
- **Simple explanation:** Our tests did not fail. A tool outside the repository failed before tests started.
- **Correction:** Set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for isolated repository tests.
- **Status:** Workaround active.

### 50. Research commits and contribution activity are different goals

- **Technical term:** Process-metric misalignment.
- **Mistake:** A high commit count could become a goal by itself.
- **Simple explanation:** More commits do not make the experiment more correct.
- **Correction:** Make small, meaningful commits at verified milestones. Preserve a clear history.
- **Status:** Current practice.

### 51. Prediction diversity does not prove correct action learning

- **Technical term:** Unaligned diversity.
- **Mistake:** We could have treated increased prediction spread as proof that the action mapping worked.
- **Simple explanation:** The model can make four different predictions and still match each action to the wrong future.
- **Evidence:** Pure JEPA had spread ratios of `0.332` and `0.430`, but its mean accuracy was 25.2%.
- **Correction:** Measure correct pairing with four-way accuracy and shuffled-action tests.
- **Status:** Confirmed by two seeds.

### 52. Non-contrastive regularization fixed one failure but not the task

- **Technical term:** Necessary but insufficient regularization.
- **Mistake:** Variance, covariance, and relation matching could have been expected to solve the full action-learning problem.
- **Simple explanation:** These losses stop identical predictions. They do not strongly identify which prediction belongs to which action.
- **Correction:** Keep the pure result as the non-contrastive control. Use a separate action-separation objective when correct pairing is required.
- **Status:** Confirmed by two seeds.

### 53. The InfoNCE signal repeated across seeds

- **Technical term:** Seed replication.
- **Mistake:** The earlier 29.4% result came from one seed and could have been noise.
- **Simple explanation:** A second run was necessary before trusting the direction of the result.
- **Evidence:** The corrected held-out scores were 36.0% and 34.8%.
- **Correction:** Report the two scores and their 35.4% mean. Do not report only the best run.
- **Status:** Confirmed.

### 54. Action types do not improve equally

- **Technical term:** Heterogeneous treatment effect.
- **Mistake:** One overall score can hide which actions the model learns.
- **Simple explanation:** Typing improved much more than scrolling.
- **Evidence:** Across the two InfoNCE runs, mean accuracy was about 62.3% for typing, 33.0% for clicks, and 23.6% for scrolling.
- **Correction:** Report each action type. Improve scroll data or its action representation before making a broad action-learning claim.
- **Status:** Open problem for scrolling.

### 55. A chance-level model is not always collapsed

- **Technical term:** Failure-mode separation.
- **Mistake:** Chance performance and action-conditioning collapse were treated as the same failure.
- **Simple explanation:** A model can make diverse predictions that are consistently paired with the wrong actions.
- **Correction:** Use one check for identical predictions. Use another check for incorrect action-future pairing.
- **Status:** Confirmed by the corrected pure runs.

### 56. Aggregate metrics are not enough for a strong confidence interval

- **Technical term:** Clustered evaluation.
- **Mistake:** The evaluator saved aggregate counts but not one record for each held-out bundle.
- **Simple explanation:** The four predictions in one bundle can be related. They should not always be treated as four independent trials.
- **Correction:** Save per-bundle predictions and correctness. Use a bundle bootstrap for confidence intervals.
- **Status:** Open evaluation improvement.

## Current corrections in the training code

The current Model 4 Stage 2 code now does these actions:

1. Freeze the complete Qwen visual encoder.
2. Train the action encoder and predictor only.
3. Train all four branches of one bundle together.
4. Apply action conditioning in every predictor block.
5. Connect pointer coordinates to visual-token locations.
6. Predict latent screen changes instead of complete future latents.
7. Put most regression weight on changed screen regions.
8. Keep a smaller global regression loss.
9. Match the target bundle's variance, covariance, and action relations.
10. Run a separate InfoNCE action-separation experiment.
11. Measure results by application and action type.
12. Detect action-conditioning collapse during training.
13. Stop after repeated collapse checks fail.
14. Support explicit random seeds for repeat runs.

## Open work

The following work is not complete:

1. Save per-bundle evaluation records and calculate bundle-level confidence intervals.
2. Prepare the controlled SFT dataset.
3. Train Models 2, 3, and 4 with identical SFT settings.
4. Measure task success and SFT sample efficiency.
5. Collect real multi-step sequences if the one-step MVP shows a useful signal.
6. Collect more non-search typing states.
7. Improve or rebalance scroll transitions.
8. Test on real, unseen software.
