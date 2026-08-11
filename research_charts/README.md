# Research chart pack

This folder contains 49 charts from the completed JEPA and SFT experiments.

Open [research-chart-pack.pdf](research-chart-pack.pdf) to view all charts in one file.

The charts are generated from saved experiment files. Run this command to rebuild them:

```powershell
C:\Python312\python.exe scripts\generate_research_charts.py
```

## Main result

- Model 2, which used SFT only, reached 51.2% exact success.
- Model 3, which used no-action JEPA and SFT, reached 18.8%.
- Model 4, which used action-conditioned JEPA and SFT, reached 18.4%.
- Model 4 reached 38.0% on future-screen matching before SFT.
- That JEPA result did not improve the final computer-use policy.

## Important limits

- The 250-example final validation set is the main policy test.
- The 32-example live monitor was only a training health check.
- JEPA future-screen matching and policy action accuracy are different tests.
- GPU memory is measured peak use.
- Host RAM is the Modal allocation. It is not measured resident RAM use.
- Training cost is measured. Inference cost and latency are not measured.
- Rare action groups have high uncertainty because they contain few examples.

## Chart index

### Final model results

- 01 to 04: final success, SFT improvement, metric profile, and coordinate precision.
- 05 to 12: action, operating system, application, prediction, and live-monitor results.
- 21 and 22: paired wins and error types.
- 40 to 43: Model 4 comparison, total cost, model pipeline, and evidence scorecard.

### SFT training

- 13 to 20: loss, gradient, learning rate, GPU memory, time, cost, parameters, and action loss.

### SFT data

- 23 to 27: action mix, operating system balance, application mix, selection rate, and integrity checks.

### JEPA training and architecture

- 28 to 39: accuracy, loss, action sensitivity, representation variance, screen change, time, cost, parameters, pilot search, and seed variation.

### Synthetic JEPA data

- 45 to 49: action mix, application split, screen change, data size, and split integrity.

### Memory

- 17: SFT GPU memory over time.
- 36: JEPA time, cost, and GPU use.
- 44: measured GPU memory and allocated host RAM.

## Machine-readable files

- [core-metrics.csv](core-metrics.csv) contains the main metrics for Models 2, 3, and 4.
- [manifest.json](manifest.json) contains every chart title, description, source, and file name.
- [generate_research_charts.py](../scripts/generate_research_charts.py) is the chart generator.
