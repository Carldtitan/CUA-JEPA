# CUA-JEPA Synthetic Transition Dataset

## Canonical local dataset

The final dataset is:

```text
data/synthetic/clean-20260808-v4/full
```

Its machine-readable audit and provenance files are:

```text
data/synthetic/clean-20260808-v4/audit_report.json
data/synthetic/clean-20260808-v4/assembly_manifest.json
```

Generated data is intentionally excluded from Git. The code that generates, assembles, and audits it is committed.

## Why this dataset exists

The experiment asks whether learning which GUI action causes which next screen helps a computer-use model beyond merely seeing more GUI screenshots.

Each bundle therefore starts from one screen and branches into four distinct actions. Every action is executed independently from the same reset state. This makes action consequence learnable and avoids the single-observed-action weakness of ordinary trajectory data.

The data does not require a human to label the correct next screen. The next screen is recorded automatically after the action is executed. This supports the claim that action-conditioned GUI pretraining can use less manually annotated data. It does not mean the pipeline requires no engineering or quality control.

## Credible spread

| Split | App | Bundles | Transitions |
|---|---|---:|---:|
| Train | Gmail | 1,000 | 4,000 |
| Train | Google Docs | 1,000 | 4,000 |
| Train | Google Sheets | 1,000 | 4,000 |
| Train | GitLab | 1,000 | 4,000 |
| Train | Shopify Admin | 1,000 | 4,000 |
| Train | Salesforce | 1,000 | 4,000 |
| Train | GitHub | 1,000 | 4,000 |
| Train | Stripe Dashboard | 1,000 | 4,000 |
| Validation | Slack | 250 | 1,000 |
| Validation | Jira | 250 | 1,000 |
| Test | Outlook Web | 250 | 1,000 |
| Test | Trello | 250 | 1,000 |

Totals:

- 9,000 same-state bundles;
- 36,000 action transitions;
- 45,000 lossless WebP screenshots at 1280 x 720;
- 387 source GUI states across 12 mock applications;
- 23,320 clicks, 8,542 type actions, and 4,138 scroll actions;
- approximately 1.35 GiB of tar data.

The validation and test applications do not appear in training. Exact current-screen, rendered-screen, and after-screen hash overlap between every pair of splits is zero.

## How it was made

1. Literal task states were extracted from CUA-Gym-Hub and filtered against each app's state schema.
2. Visible text was varied deterministically without changing IDs and references.
3. Known mock-app incompatibilities were adapted, including Google Docs session isolation and Outlook's mail/folder schemas.
4. Each app ran on an isolated random local port inside Modal. A served app-identity marker prevented stale or cross-app servers from being accepted.
5. Time and browser randomness were fixed for reproducibility.
6. For every bundle, the app was reset to one state, four distinct actions were selected, and each action ran in a fresh branch from that same state.
7. A branch was kept only if its starting screen matched within the reset tolerance and its action produced a visible change inside the minimum and maximum pixel bounds.
8. Each shard was committed to a persistent Modal volume, downloaded locally, and verified using SHA-256.

AgentNet was not used.

## File format

Each tar shard contains 25 bundles and one embedded `manifest.json`. Each bundle directory contains:

```text
<bundle_id>/current.webp
<bundle_id>/after_0.webp
<bundle_id>/after_1.webp
<bundle_id>/after_2.webp
<bundle_id>/after_3.webp
<bundle_id>/bundle.json
```

Every tar has two sidecars:

```text
<shard>.json      # shard manifest, counts, failures, action spread, quality extrema
<shard>.sha256    # tar checksum
```

`bundle.json` records the app, split, source-state ID, current-screen checksum, four normalized actions, after-screen checksums, changed-pixel fractions, reset measurements, element hints, and state-diff paths.

Actions use these forms:

```json
{"kind":"click","x":640,"y":360,"x_normalized":0.5,"y_normalized":0.5}
{"kind":"type","x":500,"y":40,"text":"Synthetic ...","x_normalized":0.390625,"y_normalized":0.055556}
{"kind":"scroll","delta_y":576}
```

## How to train another model with it

For action-conditioned JEPA pretraining:

1. Read `current.webp` as the context image.
2. Encode the corresponding branch action into the model's action representation.
3. Read that branch's `after_N.webp` as the target image.
4. Predict the target image representation from the current-image representation plus action.
5. Stop gradients through the target encoder if following the standard JEPA pattern.
6. Sample all four branches; do not treat the bundle as a single sequential trajectory.

For the no-action control, use exactly the same image pairs, ordering, updates, and compute, but replace every action with one fixed `NO_ACTION` token. The normal SFT baseline receives none of this transition pretraining. All trained branches should later receive the same computer-use SFT data where specified by the four-model design.

The held-out same-state bundles can also support a four-way dynamics test: given one current screen and action, choose which of four future screens that action caused. Chance is 25 percent.

## Final audit results

The exhaustive auditor opened every tar and every image and checked every action and checksum.

- 360/360 shards passed;
- 9,000/9,000 bundles passed;
- 36,000/36,000 transitions passed;
- 45,000/45,000 images were valid 1280 x 720 WebP files;
- 8,611 of 9,000 current screenshots were unique (95.7 percent);
- no exact screenshot leakage occurred between train, validation, and test;
- no bundle contained duplicate actions;
- no bundle contained duplicate resulting screens;
- collection acceptance was 99.7 percent (9,000 accepted from 9,027 attempts);
- changed-pixel fractions ranged from 0.0001454 to 0.94993273;
- maximum reset drift was 0.00004883, below the 0.00005 limit.

There are 389 repeated current screenshots within their own splits. They are reported, not hidden. They are concentrated in a few mock applications whose different underlying states can render the same visible view. No repeated current screenshot crosses a split, and every four-action bundle has distinct actions and distinct futures.

Run the audit again with:

```powershell
python scripts/audit_dataset.py data/synthetic/clean-20260808-v4/full `
  --output data/synthetic/clean-20260808-v4/audit_report.json
```

## Limits on what this proves

These are deterministic mock applications, not live websites. Synthetic text variants and mock state logic are useful for a controlled MVP but do not establish performance on real browser tasks. The dataset can test whether action-conditioned dynamics are learned; only the controlled four-model experiment can show whether that learning improves downstream computer use, accuracy, data efficiency, or inference cost.
