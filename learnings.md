# CUA-JEPA Learning Log

This is a living record of mistakes, corrections, and open limits in the CUA-JEPA MVP.

Each item gives the technical term and a simple explanation. The log records what happened. It does not treat an early assumption as a proven fact.

## Permanent logging rule

Add each new mistake to this file during the same work session.

Each entry must contain:

1. the technical term;
2. the mistake;
3. a simple explanation;
4. the correction;
5. the current status.

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

### 57. A pilot checkpoint is not a full training starting point

- **Technical term:** Unplanned warm start.
- **Mistake:** We almost used the best pilot checkpoint to start full Model 4 training.
- **Simple explanation:** The pilot used a small data subset. It was an architecture test, not a trained base for the final run.
- **Correction:** Start full Model 4 from the original Qwen model. Create new LoRA adapters and new heads.
- **Status:** Fixed in the full-run code.

### 58. A model name alone does not give an identical starting point

- **Technical term:** Model revision drift.
- **Mistake:** The code selected the Qwen model name but did not select an exact model revision.
- **Simple explanation:** Files under one model name can change later.
- **Correction:** Pin Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203` in the code and config.
- **Status:** Fixed.

### 59. The target LoRA adapter did not receive its planned EMA update

- **Technical term:** Broken exponential moving average update.
- **Mistake:** The old condition skipped EMA when the target adapter was frozen.
- **Simple explanation:** A target adapter must not receive gradients. It must still receive the slow copy update.
- **Correction:** Update target LoRA from online LoRA after each optimizer step. Keep target LoRA outside the optimizer.
- **Status:** Fixed and covered by a regression test.

### 60. Pilot data staging was not full data staging

- **Technical term:** Dataset scope mismatch.
- **Mistake:** The Modal pilot volume contained only 1,952 training transitions and 500 validation transitions.
- **Simple explanation:** A full run on that volume would still use only pilot data.
- **Correction:** Add a separate full-data path. Require 30,668 training transitions and 2,000 validation transitions.
- **Status:** Fixed in the full-run code. Upload remains a required run step.

### 61. The phrase `unfreeze Qwen` was too broad

- **Technical term:** Parameter-scope ambiguity.
- **Mistake:** The phrase could mean training all Qwen weights or training only LoRA.
- **Simple explanation:** These two methods have very different cost and risk.
- **Correction:** Keep all base Qwen weights frozen. Train only the online Qwen vision LoRA adapter.
- **Status:** Fixed in the full-run design.

### 62. An unused global TensorFlow install can break tests

- **Technical term:** Dependency contamination.
- **Mistake:** Transformers found a broken global TensorFlow and protobuf combination during test import.
- **Simple explanation:** This project does not use TensorFlow, but the global install still stopped the tests.
- **Correction:** Set `USE_TF=0` for these tests. Keep the Modal training image limited to required packages.
- **Status:** Local workaround active.

### 63. A smoke test did not authorize a paid full run

- **Technical term:** Approval-scope error.
- **Mistake:** We treated `go ahead` as approval for both the smoke test and the paid full run.
- **Simple explanation:** The user wanted to pause and review the smoke result before full training.
- **Correction:** After the smoke test, report its result, expected time, expected cost, and exact full-run settings. Require a new clear approval before full training.
- **Status:** Procedure corrected. The accidental full run was stopped before its first training step.

### 64. A two-step smoke test gave a misleading average speed

- **Technical term:** Warm-up timing bias.
- **Mistake:** We first divided total smoke-test time by two steps.
- **Simple explanation:** Model loading and evaluation made the first step look much slower than normal training.
- **Correction:** Report startup time, evaluation time, and steady training-step time separately. Use a longer timing test before a final estimate.
- **Status:** The first estimate was withdrawn. A reliable full-run estimate is still open.

### 65. Stopping the command did not stop its child process

- **Technical term:** Orphan process.
- **Mistake:** Stopping the local command wrapper left the Python uploader active.
- **Simple explanation:** The old upload continued after we thought it had stopped.
- **Correction:** After every stop, check local Python processes and Modal app state. Stop the exact process or app if it remains active.
- **Status:** Fixed during the upload. No extra GPU ran.

### 66. The first full-data upload had no visible restart points

- **Technical term:** Non-resumable batch operation.
- **Mistake:** One upload batch contained all 340 tar files and showed no useful progress.
- **Simple explanation:** We could not tell which data was complete during a slow network transfer.
- **Correction:** Upload and commit one application at a time. Verify the final remote counts and exclude the test split.
- **Status:** Fixed. Modal contains 320 training tar files, 20 validation tar files, and zero test files.

### 67. The full run still used a pilot app name

- **Technical term:** Run-name ambiguity.
- **Mistake:** Modal called the full-run app `cua-jepa-model4-pilot`.
- **Simple explanation:** The dashboard could not clearly separate a pilot from full training.
- **Correction:** Give smoke, pilot, and full runs clear names. Put the mode, seed, model revision, and timestamp in each run record.
- **Status:** Open before the next full run.

### 68. A successful smoke test does not show research success

- **Technical term:** Mechanical validation versus scientific validation.
- **Mistake:** The words `working correctly` could imply that Model 4 learned useful action dynamics.
- **Simple explanation:** The smoke test only proved that two training steps ran with the correct parameters.
- **Correction:** Say exactly what passed. Do not use smoke-test accuracy as evidence because it used only two bundles.
- **Status:** Terminology corrected.

### 69. The full run needs more measurement before it starts

- **Technical term:** Experiment observability gap.
- **Mistake:** The full run could start without per-bundle records, checkpoint recovery, gradient statistics, or a cost record.
- **Simple explanation:** A final score cannot explain when, why, or how training changed.
- **Correction:** Implement the measurements in `full_run_observability.md` before full training.
- **Status:** Fixed. The complete path passed the two-step observability smoke test.

### 70. An observability checklist is not an observability implementation

- **Technical term:** Specification-implementation gap.
- **Mistake:** We wrote the metrics that we wanted, but the training code did not yet save all of them.
- **Simple explanation:** A document cannot prove that the run creates the required files.
- **Correction:** Implement each metric in the training loop. Add a strict artifact validator and a real Modal test.
- **Status:** Fixed. The final smoke run produced and validated all required artifact types.

### 71. The first observability smoke test missed the Modal app ID

- **Technical term:** Incomplete run provenance.
- **Mistake:** The run saved a Modal task ID but stored `null` for the Modal app ID.
- **Simple explanation:** A task ID alone does not identify the complete dashboard run.
- **Correction:** Read the hydrated Modal app ID from the app object. Make the validator reject missing run IDs.
- **Status:** Fixed and confirmed in the final smoke run.

### 72. File presence and file size did not prove checkpoint integrity

- **Technical term:** Artifact-integrity validation.
- **Mistake:** The first validator accepted checkpoint files without loading them.
- **Simple explanation:** One downloaded checkpoint had the expected size but could not be opened.
- **Correction:** Load every checkpoint, optimizer state, adapter, and JEPA head file during validation.
- **Status:** Fixed. The remote run now validates its own artifacts before success.

### 73. Checkpoint downloads were not atomic

- **Technical term:** Atomic file replacement.
- **Mistake:** A stopped download could leave a partial destination file with a normal-looking size.
- **Simple explanation:** A later resume could mistake the broken file for a complete file.
- **Correction:** Download to a `.part` file. Check its byte count. Replace the destination only after completion.
- **Status:** Fixed in the Modal artifact downloader.

### 74. The observability system has its own time cost

- **Technical term:** Instrumentation overhead.
- **Mistake:** We did not separate training time from checkpoint and volume-save time.
- **Simple explanation:** Detailed records can make a run slower even when model training is fast.
- **Evidence:** The final smoke test used about `1.78` seconds for training compute and `18.32` seconds for volume persistence.
- **Correction:** Track data, model, evaluation, checkpoint, persistence, and training time separately.
- **Status:** Fixed. A longer timing test is still useful before the full run.

### 75. Artifact validation must happen before a run reports success

- **Technical term:** In-run artifact validation.
- **Mistake:** Local validation happened only after the first run had already reported success.
- **Simple explanation:** A remote run could look successful while its saved files were incomplete.
- **Correction:** Validate all remote artifacts inside the Modal function before the function returns success.
- **Status:** Fixed and confirmed by the final smoke run.

### 76. The collapse stop rule can miss an almost constant predictor

- **Technical term:** False-negative collapse detection.
- **Mistake:** The stop rule requires three collapse signals to fail at the same time. Action accuracy stayed near chance, but two small secondary values remained just above their limits.
- **Simple explanation:** The model can ignore actions without activating the automatic stop rule.
- **Correction:** Treat near-chance action accuracy and very low predicted variance as direct failure signals. Test the new rule on saved pilot and full-run records before another run.
- **Status:** Found during the active full run. The live run is unchanged. The correction is pending.

### 77. Ending the chat reply looked like stopping the training run

- **Technical term:** Local-client and remote-job state ambiguity.
- **Mistake:** I ended the reply without clearly separating the chat state from the Modal job state.
- **Simple explanation:** The reply stopped, but the GPU job continued. This made it look as if training stopped.
- **Correction:** Always state the Modal app state, task count, and current step before ending a reply during a remote run.
- **Status:** Communication correction applied. The active run was verified at step 1,250.

### 78. A near-chance checkpoint did not prove that the complete run had failed

- **Technical term:** Premature convergence judgment.
- **Mistake:** I called the active run a failure from the step-1,500 validation result.
- **Simple explanation:** The action loss started to improve after step 1,750. One early checkpoint did not show the later change.
- **Correction:** Separate the current result from the final result. Check the next fixed validation result before declaring failure.
- **Status:** Confirmed. Step-2,000 training accuracy reached 52%, while validation accuracy reached only 27.6%.

### 79. Training action accuracy can improve without strong validation transfer

- **Technical term:** Generalization gap.
- **Mistake:** We could treat the step-2,000 training accuracy increase as proof that Model 4 works.
- **Simple explanation:** Training accuracy reached 52%, but accuracy on the unseen validation application was only 27.6%. Chance is 25%.
- **Correction:** Use validation and test results for the main claim. Treat training accuracy only as proof that the model can fit the training signal.
- **Status:** Found during the active full run. Later validation checkpoints and the final test are pending.

### 80. The live validation monitor did not cover both validation applications

- **Technical term:** Non-representative monitoring subset.
- **Mistake:** The live validation monitor evaluated Jira only. The final validation evaluated Jira and Slack.
- **Simple explanation:** The live score reached 29%, but the final score reached 38% because the evaluated applications changed.
- **Correction:** Build each monitoring subset with fixed samples from every validation application.
- **Status:** Corrected and remotely tested. The monitor now selects bundles round-robin across applications.

### 81. Overall accuracy hid weak performance on important action groups

- **Technical term:** Aggregate-metric masking.
- **Mistake:** The overall 38% accuracy can look broad, but typing reached 83.3% while clicking reached 30.3%.
- **Simple explanation:** Easy typing and small screen changes raised the total score. Large screen changes stayed near chance.
- **Correction:** Report action type, application, and screen-change groups beside every overall score. Use a macro average for model selection.
- **Status:** Found after the full run. The required group metrics were saved. The model-selection rule needs correction.

### 82. The full-data run did not use multi-step GUI sequences

- **Technical term:** Objective and implementation mismatch.
- **Mistake:** We discussed V-JEPA 2-AC style sequence training, but the full-data run used independent one-step transitions.
- **Simple explanation:** The model saw one screen and one action at a time. It did not practice predicting several actions into the future.
- **Correction:** Add ordered trajectory data. Train with one-step teacher forcing and a two-step rollout loss. Check this feature in a small test before another full run.
- **Status:** Found after the full-data run. The saved model is a valid one-step baseline. Multi-step training is not implemented.

### 83. The full Model 4 experiment is not complete

- **Technical term:** Experimental scope ambiguity.
- **Mistake:** We called the completed JEPA run a full Model 4 run. It did not include the common SFT stage or the other three model controls.
- **Simple explanation:** We completed one Model 4 training stage. We did not complete the four-model comparison.
- **Correction:** Call this artifact the Model 4 one-step JEPA run. Complete the shared SFT stage and matched controls before we make the main research claim.
- **Status:** Found after the full-data run. The artifact name stays unchanged for traceability. Future reports must use the corrected name.

### 84. The default Python launcher points to a deleted installation

- **Technical term:** Stale executable path.
- **Mistake:** The first test command used `py`, which points to a deleted Python 3.13 installation.
- **Simple explanation:** The test did not start because Windows looked for a Python file that no longer exists.
- **Correction:** Use `C:\Python312\python.exe` for this repository. Repair the global launcher separately.
- **Status:** Workaround applied. The complete test suite passed with Python 3.12.

### 85. A later successful command hid an earlier test failure

- **Technical term:** Exit-code masking.
- **Mistake:** One shell call ran tests, lint, and a Modal check without stopping after the test failure.
- **Simple explanation:** The final Modal command passed. This made the complete shell call look successful even though the tests failed to start.
- **Correction:** Check `$LASTEXITCODE` after every required command. Disable unrelated global pytest plugins for repository tests.
- **Status:** Corrected in later checks. The isolated test suite passed with 42 tests.

### 86. Windows stopped a Modal launch when it printed a Unicode symbol

- **Technical term:** Console encoding failure.
- **Mistake:** The first Modal launch used the default Windows character encoding.
- **Simple explanation:** Modal printed a check mark. The Windows console could not print it, so the local controller stopped.
- **Correction:** Set `PYTHONUTF8=1` for every Modal command on this computer.
- **Status:** Corrected. The next Modal launch started successfully.

### 87. Artifact validation incorrectly required LoRA in a frozen-encoder run

- **Technical term:** Configuration-insensitive validation.
- **Mistake:** The artifact validator required identical online and target LoRA adapters when the pilot did not use LoRA.
- **Simple explanation:** Training completed, but the final check looked for model parts that did not exist.
- **Correction:** Validate LoRA identity only when the run has LoRA parameters. Require empty LoRA fields for a frozen-encoder run.
- **Status:** Corrected. The new frozen-encoder regression test passes.

### 88. The first V-JEPA 2 image missed its video-processing dependency

- **Technical term:** Missing runtime dependency.
- **Mistake:** The Modal image included PyTorch and Transformers but did not include `torchvision`.
- **Simple explanation:** The V-JEPA 2 video processor could not start without the image and video helper library.
- **Correction:** Add a compatible `torchvision` version to the training dependencies and Modal image.
- **Status:** Corrected. The V-JEPA 2 encoder loaded and passed the first deterministic feature test.

### 89. The balanced monitor used an unbalanced loaded pool

- **Technical term:** Pre-sampling selection bias.
- **Mistake:** The code balanced the monitor after a sequential file limit had already selected 75 Jira bundles and 50 Slack bundles.
- **Simple explanation:** The final selection could not become equal because too few Slack bundles were loaded.
- **Correction:** Load the complete application pool first. Then select complete bundles round-robin across applications.
- **Status:** Corrected before V-JEPA 2 training. The balanced loader tests pass.

### 90. The new V-JEPA 2 module kept unused imports

- **Technical term:** Static-analysis failure.
- **Mistake:** The first module draft imported `math` and `Counter` but did not use them.
- **Simple explanation:** These imports did not change training, but they caused the code-quality check to fail.
- **Correction:** Remove both unused imports before tests or GPU work.
- **Status:** Corrected. The code-quality check passes.

### 91. The V-JEPA 2 module imported the full Qwen trainer for one data check

- **Technical term:** Unnecessary dependency coupling.
- **Mistake:** The V-JEPA 2 module imported `train_jepa.py` only to use `validate_dataset_assignments`.
- **Simple explanation:** This also loaded Qwen, PEFT, Transformers, and a broken local TensorFlow package during test collection.
- **Correction:** Move the general data check to `jepa_data.py`. Import it from both training modules.
- **Status:** Corrected. The full targeted regression test set passes.

### 92. Candidate futures used different masks in the four-way test

- **Technical term:** Candidate-dependent metric weighting.
- **Mistake:** Each candidate future used its own changed-region mask during ranking and action-separation training.
- **Simple explanation:** The distance rule changed for each answer. This could change the winner without better action prediction.
- **Correction:** Use the union of all four changed-region masks for every candidate in the same bundle.
- **Status:** Corrected before V-JEPA 2 training. The shared-mask regression test passes.

### 93. The V-JEPA 2 training order did not balance applications

- **Technical term:** Application sampling imbalance.
- **Mistake:** One training epoch used all 488 bundles once. Application counts ranged from 40 to 75 bundles.
- **Simple explanation:** Applications with more bundles would control more training updates.
- **Correction:** Shuffle each application separately. Sample each application equally and repeat smaller application pools when needed.
- **Status:** Corrected before V-JEPA 2 training. The balanced sampling regression test passes.

### 94. The default V-JEPA 2 processor cropped wide GUI screenshots

- **Technical term:** Aspect-ratio crop and spatial-mask misalignment.
- **Mistake:** The default processor resized a wide screenshot by its short edge and then took a 256-pixel center crop.
- **Simple explanation:** The encoder could lose controls on the left and right sides. The changed-region mask also used a different layout.
- **Correction:** Fit the complete screenshot into a 256-by-256 square with neutral padding. Disable the processor resize and crop. Use the same fitted image for the changed-region mask.
- **Status:** Corrected before the full pilot. All local tests and two Modal smoke tests pass.

### 95. The V-JEPA 2 variance monitor compared different feature scales

- **Technical term:** Metric scale mismatch.
- **Mistake:** The target variance used raw encoder features. The prediction variance used normalized current features plus a predicted change.
- **Simple explanation:** The two variance numbers used different scales, so their ratio had no clear meaning.
- **Correction:** Normalize both future targets and future predictions before calculating action variance.
- **Status:** Corrected before the full pilot. All 50 local tests pass.

### 96. The first pilot mixed action learning with unseen-application transfer

- **Technical term:** In-domain and out-of-domain evaluation confound.
- **Mistake:** We treated a score on Jira and Slack as a direct action-learning score, although neither application was in training.
- **Simple explanation:** A low score can mean weak action learning, weak transfer to new software, or both.
- **Correction:** Report training-application and unseen-application scores separately. Add a same-application held-out set before the final paper evaluation.
- **Status:** Partly corrected. The reports now separate both scores. A same-application held-out set is still required.

### 97. Feature encoding printed too many progress records

- **Technical term:** Observability log saturation.
- **Mistake:** The encoder printed one record for every four bundles. The Modal output became truncated.
- **Simple explanation:** Useful training and evaluation records became harder to inspect.
- **Correction:** Print feature progress every 25 encoder batches and at completion.
- **Status:** Corrected. All 50 local tests pass.

### 98. The first V-JEPA 2 predictor had no explicit action token

- **Technical term:** Action-token conditioning.
- **Mistake:** The predictor used action-based normalization and a spatial heatmap, but it did not put the action into the transformer token sequence.
- **Simple explanation:** Every layer received action information, but the attention operation could not attend to a separate action token as V-JEPA 2-AC does.
- **Correction:** Add a predictor that prepends one learned action token. Keep the spatial click map and layer-wise action conditioning.
- **Status:** Tested. The action-token model reached 38.0%, below the 40.4% AdaLN model. Do not use it as the current best model.

### 99. Letterboxing moved the screen but did not move the click heatmap

- **Technical term:** Coordinate-frame mismatch.
- **Mistake:** The image moved into a padded square, but pointer coordinates stayed in the original wide-screen coordinate frame.
- **Simple explanation:** A click near the top of the real screen pointed into the top padding in the model input.
- **Correction:** Transform click and type coordinates into the same letterboxed coordinate frame before making the spatial heatmap.
- **Status:** Corrected and tested. The matched rerun stayed at 40.4% overall and improved click accuracy from 34.9% to 35.5%.

### 100. The first Modal folder download used a file destination

- **Technical term:** Concurrent output-path collision.
- **Mistake:** The Modal folder download received a destination path that did not exist as a directory.
- **Simple explanation:** Several remote files wrote to one local file. The result was corrupt.
- **Correction:** Create the local directory first. Use the remote folder path without a glob. Validate every downloaded artifact set.
- **Status:** Corrected. Three complete artifact sets pass the local artifact validator. The three corrupt temporary files were removed. The originals remain in Modal.

### 101. Full-screen letterboxing kept coverage but removed small GUI detail

- **Technical term:** Spatial-resolution bottleneck.
- **Mistake:** We kept the complete 1280-by-720 screen inside a 256-by-256 input. The useful screen content was only about 256 by 144 pixels.
- **Simple explanation:** The model saw the full screen, but many small buttons and text became too small. This can limit click prediction.
- **Correction:** Test two overlapping 256-by-256 screen views. Give each visual token its position in the complete screen. Map each click into the correct view.
- **Status:** Implemented as a separate controlled method. All 57 local tests pass. The two-step Modal smoke passed. The controlled pilot is not yet complete.

### 102. The first tiled smoke command ended its local client too early

- **Technical term:** Client timeout.
- **Mistake:** The first Modal command had a one-second local timeout. The client stopped before it could manage the remote app.
- **Simple explanation:** The test did not run. Modal kept an empty app record until we stopped it.
- **Correction:** Give the local Modal client the full run timeout. Use short status checks instead of ending the client.
- **Status:** Corrected. The empty app was stopped. The next smoke completed and used about $0.005.

### 103. Two fixed high-detail views did not improve validation accuracy

- **Technical term:** Multi-view prediction bottleneck.
- **Mistake:** We expected more screen detail alone to improve click prediction. We used one attention path for all 512 visual tokens.
- **Simple explanation:** The targets became easier to tell apart. The predictor still failed to reproduce enough of their differences.
- **Correction:** Reject the first two-view method. Test each 256-token view with a separate prediction path that shares weights.
- **Status:** The first method reached 36.0% overall and 35.2% on clicks. The separate-view method reached 37.8% overall and 37.1% on clicks. The old method remains best overall at 40.4%.

### 104. Predictor tests repeated the same frozen feature encoding

- **Technical term:** Frozen-feature cache miss.
- **Mistake:** Each predictor test ran the unchanged frozen V-JEPA 2 encoder on the same screens again.
- **Simple explanation:** We paid for the same fixed calculation more than once.
- **Correction:** Save frozen visual features with a data and encoder hash. Validate tensor counts and shapes before reuse.
- **Status:** Corrected. Modal loaded the checked cache for the wider-predictor run. That run cost about $0.08 instead of about $0.23 to $0.28.

### 105. One seed made the separate-view method look better than it was

- **Technical term:** Random-seed variance.
- **Mistake:** Seed 20260811 reached 37.1% click accuracy. This was close enough to the 38% gate to look promising by itself.
- **Simple explanation:** A different random start changed which action types worked best.
- **Correction:** Run the same method with seed 20260812. Report the two-run mean and each action group.
- **Status:** Corrected. The second seed reached 33.8% on clicks. The mean click score is 35.5%. The old method remains best overall.

### 106. A larger tiled predictor did not improve general use

- **Technical term:** Predictor capacity ablation.
- **Mistake:** We suspected that the 384-wide, 6-layer predictor was too small for 512 visual tokens.
- **Simple explanation:** A larger predictor learned the training screens better. It did not transfer better to Jira and Slack.
- **Correction:** Test a 512-wide, 8-layer predictor with the same data, seed, features, and steps. Reject it if validation does not improve.
- **Status:** Tested and rejected. It reached 38.2% overall and 36.0% on clicks. The smaller model reached 37.8% overall and 37.1% on clicks with the same seed.

## Current corrections in the training code

The controlled Model 4 pilot code now does these actions:

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

The full Model 4 code adds these safeguards:

1. Load the exact pinned Qwen revision.
2. Do not load a pilot checkpoint.
3. Keep the base Qwen weights frozen.
4. Train a new online Qwen vision LoRA adapter.
5. Update a new target LoRA adapter with EMA only.
6. Start the action encoder and predictor with new random weights.
7. Check the optimizer parameter list before training.
8. Require the exact full training and validation counts.
9. Reject bundle or exact screenshot overlap between training and validation.
10. Keep the test split out of the Modal training paths.
11. Save run, package, GPU, data, and parameter identity.
12. Save all loss, gradient, memory, speed, and cost records.
13. Evaluate fixed training and validation bundles at planned steps.
14. Save one record for each evaluated bundle.
15. Save recoverable checkpoints with optimizer and random states.
16. Load and validate all artifacts before reporting success.

## Open work

The following work is not complete:

1. Prepare the controlled SFT dataset.
2. Train Models 2, 3, and 4 with identical SFT settings.
3. Measure task success and SFT sample efficiency.
4. Collect real multi-step sequences if the one-step MVP shows a useful signal.
5. Collect more non-search typing states.
6. Improve or rebalance scroll transitions.
7. Test on real, unseen software.
8. Measure observability overhead in a longer timing pilot.
9. Complete and analyze the approved full Model 4 run.
10. Replace the weak collapse stop rule before a later full run.

### 107. Dense future prediction kept too much unchanged screen content

- **Technical term:** Visual dominance and counterfactual centering.
- **Mistake:** We kept changing the predictor size and screen view while the target still contained a large amount of common screen content.
- **Simple explanation:** The four futures mostly show the same software screen. The small action effect can get lost inside the much larger common screen.
- **Correction:** Add a pure JEPA target that subtracts the average of the four future representations. The remaining target shows what is different for each action. Use the union of all changed-screen masks for every branch.
- **Status:** Tested. The pure counterfactual pilot reached 31.4% validation accuracy. This is above chance, but it did not pass the 40% gate.

### 108. The four action branches have an uneven action mix

- **Technical term:** Action-type and branch-position imbalance.
- **Mistake:** We did not measure how often each action type occurs in each branch position before we interpreted the total four-way score.
- **Simple explanation:** In the 7,667 training bundles, branch 0 is a typing action 96.1% of the time. Branches 2 and 3 are clicks about 98% of the time. The total score can hide weak click learning.
- **Correction:** Keep the per-action scores. Treat click accuracy as a key result. Add controls that shuffle the action and the current screen. Do not use the total score by itself.
- **Status:** Corrected. The full training split was measured. The Modal run recorded both shuffle controls.

### 109. The counterfactual model used the action more than the screen

- **Technical term:** Conditional-input bypass.
- **Mistake:** The predictor could add action-based shifts and a pointer heatmap directly to its hidden state. This gave it a path that did not require the current-screen features.
- **Simple explanation:** The pure counterfactual model reached 31.4%. A wrong action reduced the score by 10 points. A wrong current screen reduced it by only 1.6 points. The model learned a general action pattern more than a screen-specific result.
- **Correction:** Add a visual-gated predictor. The action can only multiply visual features. It cannot add new action-only content. Keep both shuffle controls.
- **Status:** Tested and rejected. Pure visual gating reached 32.6%. It improved the wrong-screen drop to 3.2 points, but it did not pass the 40% accuracy or 5-point screen-dependence gates.

### 110. The first scaled counterfactual run did not reuse an old cache

- **Technical term:** Frozen-feature cache provenance.
- **Mistake:** We expected an old 8,000-transition cache to match the new letterbox run without checking its key and source run.
- **Simple explanation:** The old letterbox run was made before cache support. The existing large cache used a different configuration. Modal encoded 8,500 screens again.
- **Correction:** List the cache files and compare run configurations before we promise a cache hit. The new letterbox cache is `2d3715e204bdaa2ad33b5401fe59aa456b7b81fREDACTEDe91a0e5772.pt`.
- **Status:** Corrected. The new cache is saved in the Modal volume. Later letterbox tests can reuse it.

### 111. Counterfactual centering removed useful state information

- **Technical term:** Over-centering of target representations.
- **Mistake:** We expected the average-future subtraction to keep only useful action effects. It also removed absolute future-state information that helped transfer to unseen software.
- **Simple explanation:** The centered model learned the training applications. It did not transfer well to Jira and Slack. Pure training reached 31.4%. Visual gating reached 32.6%. Added action separation reached 31.2%.
- **Correction:** Reject counterfactual centering as the current main method. Keep the normal future-delta target. Use the centered result only as an ablation.
- **Status:** Corrected. All three results are saved in the Modal volume.

### 112. The old best result did not have a wrong-screen control

- **Technical term:** Current-state dependence control.
- **Mistake:** The earlier 40.4% result proved action dependence, but it did not prove that the predictor used the current screen.
- **Simple explanation:** A model can learn general action patterns without reading the software state.
- **Correction:** Rerun the exact old method. Replace the current screen with a screen from the next bundle. Keep the action and target unchanged.
- **Status:** Corrected. The repeated result reached 40.4%. A wrong action reduced it by 19.6 points. A wrong screen reduced it by 15 points. The model uses both inputs.

### 113. We treated Qwen and V-JEPA2 as replacement choices

- **Technical term:** Complementary representation fusion.
- **Mistake:** We tested Qwen targets and V-JEPA2 targets as separate choices. We did not test whether their features help each other.
- **Simple explanation:** Qwen can represent GUI text and controls. V-JEPA2 can represent visual changes. One encoder does not need to replace the other.
- **Correction:** Freeze both encoders. Resize Qwen visual tokens to a small fixed grid. Add them through zero-initialized gated cross-attention in the V-JEPA predictor. Keep the old target, data, seed, and action-separation loss.
- **Status:** Tested and rejected as the next main method. It reached 40.4% overall and 35.5% on clicks. These results match the V-JEPA-only result. The learned fusion gates were not zero, so the predictor used Qwen features, but the features did not improve this task.

### 114. The first fusion run did not record its gate size

- **Technical term:** Fusion-gate observability.
- **Mistake:** The Qwen fusion run saved the gate weights, but it did not put their size in the live log or final metrics.
- **Simple explanation:** The score did not improve. We could not immediately tell if the model ignored Qwen.
- **Correction:** Record the fusion-gate gradient and the effective gate size during training. Save the final values with the other metrics.
- **Status:** Corrected. The saved run was inspected. Its raw gate L2 norm was 0.163. Each of the six layers had a nonzero gate. New runs now record this information automatically.

### 115. Unseen applications can hide what the predictor learned

- **Technical term:** In-domain and out-of-domain validation.
- **Mistake:** We used only Jira and Slack for validation. The model trained on eight different applications. A low score can mean weak action learning, weak transfer to new software, or both.
- **Simple explanation:** The test changed the software and the action result at the same time. We could not tell which change caused the error.
- **Correction:** Keep Jira and Slack as the main unseen-application validation. Add a separate diagnostic that holds out complete bundles from the eight training applications. Use no shared bundle IDs or exact screenshots.
- **Status:** Corrected and tested. The scaled same-app result reached 69.6% overall, 55.9% on clicks, and 66.3% on large changes. A wrong action reduced accuracy by 58 points. A wrong screen reduced it by 42.4 points. The run passed every success check.

### 116. The first same-app selector passed nested bundle lists

- **Technical term:** Data-structure shape mismatch.
- **Mistake:** The selector passed a list of four-branch bundle lists into a function that accepts a flat list of transitions.
- **Simple explanation:** The function received boxes of records when it expected records.
- **Correction:** Flatten the bundle lists before balanced selection. Keep a unit test that checks a deterministic and disjoint result.
- **Status:** Corrected before any Modal run. No GPU time or data was affected.

### 117. The model seed also changed the same-app validation data

- **Technical term:** Random-seed confounding.
- **Mistake:** The same seed controlled model initialization and the held-out bundle selection.
- **Simple explanation:** A repeat with a new model seed would also use different test examples. That is not a clean repeat.
- **Correction:** Use a fixed data-split seed. Use a separate model seed. Put the split seed in the cache key and data audit. Validate the exact cached bundle IDs before reuse.
- **Status:** Corrected and verified. Both model seeds used the same 125 held-out bundles. Seed 20260811 reached 69.6%. Seed 20260812 reached 72.4%. The mean is 71.0%.

### 118. One unseen-application score hid strong action learning

- **Technical term:** Out-of-domain generalization gap.
- **Mistake:** We treated the 40.4% Jira and Slack score mainly as an architecture-learning failure.
- **Simple explanation:** The model learned the training applications well. It had more difficulty with new software styles.
- **Correction:** Report both tests. Use a same-app holdout to measure action learning. Use Jira and Slack to measure transfer to unseen applications.
- **Status:** Corrected. The two-seed same-app mean is 71.0%. The two-seed unseen-application mean is 39.6%. The gap is 31.4 points.

### 119. The first all-data cache estimate used the wrong data type

- **Technical term:** Feature-cache precision.
- **Mistake:** We estimated cache size as if V-JEPA features were stored in float32.
- **Simple explanation:** The code already stores features in bfloat16. Bfloat16 uses half as many bytes.
- **Correction:** Read the encoder output conversion before estimating storage. Use the measured 5.2 GB size for 2,125 bundles to estimate the full cache.
- **Status:** Corrected. The all-data cache estimate is about 20 GB, not 40 GB.

### 120. Rejected feature caches exceeded the storage limit

- **Technical term:** Derived-artifact retention.
- **Mistake:** Old feature caches from rejected tests remained in the Modal volume. The V-JEPA cache folder grew to about 34.4 GB.
- **Simple explanation:** Saved copies of calculations used more storage than the 30 GB limit.
- **Correction:** Keep the active unseen-application cache. Remove obsolete and duplicated caches after model artifacts are verified. Check measured size before a large run.
- **Status:** Corrected. Four obsolete caches totaling about 34.4 GB were removed. Saved models and metrics were not removed. The active all-data cache and smoke caches use about 20.0 GB.

### 121. One successful full-data seed was not enough

- **Technical term:** Random-seed stability for unseen-app transfer.
- **Mistake:** The first all-data seed passed every gate. Reporting only that seed would hide unstable click transfer.
- **Simple explanation:** A different random start learned Jira and Slack less well, even with the same training and validation data.
- **Correction:** Repeat the full run with the same exact data and a different model seed. Report both results and their mean.
- **Status:** Corrected. The two overall scores are 44.7% and 40.7%. Their mean is 42.7%. The click scores are 39.8% and 33.2%. Their mean is 36.5%.

### 122. The 8,000-transition run did not use most available data

- **Technical term:** Data-scale ablation.
- **Mistake:** We changed several predictor designs before testing the existing method on all 30,668 training transitions.
- **Simple explanation:** The working method had seen only about one quarter of the available transitions.
- **Correction:** Keep the same architecture and loss. Train one step on every complete training bundle. Evaluate all 2,000 Jira and Slack transitions.
- **Status:** Corrected. The two-seed overall mean increased from 39.6% to 42.7%. The increase is 3.1 points. This helps, but it does not remove seed variation or fully solve clicks.
