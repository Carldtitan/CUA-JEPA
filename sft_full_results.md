# Matched SFT results for Models 2, 3, and 4

## Direct result

The current JEPA transfer design did not improve the Qwen computer-use policy.

Model 2 was much better than Model 3 and Model 4.

| Model | Starting exact success | Final exact success | Final mean action score |
| --- | ---: | ---: | ---: |
| Model 2: fresh Qwen plus SFT | 40.8% | 51.2% | 52.47% |
| Model 3: no-action JEPA plus SFT | 14.8% | 18.8% | 19.13% |
| Model 4: action-conditioned JEPA plus SFT | 13.6% | 18.4% | 18.56% |

Model 4 did not beat Model 3. The 0.4 percentage-point difference is not reliable.

## What each model used

- Model 2 started from the base Qwen model. It used new visual and language LoRA adapters for SFT.
- Model 3 started from the base Qwen model plus the no-action JEPA visual LoRA. It then used matched SFT.
- Model 4 started from the base Qwen model plus the action-conditioned JEPA visual LoRA. It then used matched SFT.
- The base Qwen weights were frozen in all three SFT runs.

## Controlled training setup

All three SFT runs used:

- Qwen/Qwen3-VL-2B-Instruct at revision `89644892e4d85e24eaac8bacfd4f463576704203`;
- the same 2,000 training examples;
- the same training order;
- the same 250 validation examples;
- 500 optimizer steps;
- 2,000 micro-steps;
- the same SFT code and settings;
- the same new language LoRA initialization;
- equal visual and language LoRA sizes.

The triplet audit passed all checks.

Dataset SHA-256: `b43643ec07478b8c7b728ecf188ab9b08eeb6db0a65b45a0ba9a4a0319250e37`

Training code SHA-256: `d29e6a309ea6a5370e2dcd1995413ec8c745b2c1e4061d69af2ca9b032c08f01`

Evaluation code SHA-256: `90be57cb7d8ce7495cbfbd09e00d680d5643a47b8d9949175776fa3274480475`

## Statistical comparison

Model 4 minus Model 2 mean action score was -33.91 percentage points.

Its paired bootstrap 95% interval was -40.70 to -27.21 percentage points.

The bootstrap estimated a 0% chance that Model 4 was better than Model 2 in this run.

Model 4 minus Model 3 mean action score was -0.57 percentage points.

Its paired bootstrap 95% interval was -3.27 to +2.08 percentage points.

This interval includes zero. Model 3 and Model 4 are statistically tied.

## Meaning

The 38% Model 4 JEPA result measured four-way future-screen matching. It did not measure action-policy success.

The final SFT test measured whether Qwen generated the correct GUI action. Model 4 scored 18.4% on this test.

The current direct visual-LoRA transfer caused negative transfer. It damaged visual features that base Qwen used for computer actions.

The next design should keep Qwen's original visual path. JEPA should enter through a separate gated or residual branch.

## Saved files

Local folders:

- `artifacts/sft-model2-full-seed20260810-20260810T220702Z`
- `artifacts/sft-model3-full-seed20260810-20260810T235313Z`
- `artifacts/sft-model4-full-seed20260810-20260810T220704Z`
- `artifacts/model3-model3_full-seed20260809-20260810T214351Z`
- `artifacts/sft-comparison`

Remote Modal volume: `cua-jepa-training-v1`

AgentNet preparation is complete. Its app is stopped. The SFT data has 2,250 images: 2,000 training images and 250 validation images.
