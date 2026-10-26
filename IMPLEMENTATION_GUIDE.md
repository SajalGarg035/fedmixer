# Fedmixer-FCL Implementation Guide

This document describes the implementation currently present in this repository. It is based on the source code, not only on the figures or the original TimeMixer paper. The repository has two related execution paths:

1. The original TimeMixer experiment framework for forecasting, imputation, anomaly detection, and classification.
2. An opt-in continual federated extension called FedMixer/Continual FedMixer that reuses the TimeMixer forecasting model.

The TimeMixer backbone keeps its standard forecast output and now also exposes an optional
multiscale inspection API. Federated behavior is implemented around model instances through
client, server, memory, drift, and round-orchestration classes.

## 1. Repository Purpose and Structure

### Main directories

- `run.py`: primary CLI for the original experiment framework.
- `models/`: model implementations. The active model is `TimeMixer`.
- `layers/`: embeddings, normalization, decomposition, attention, convolution, Fourier, and wavelet utilities.
- `data_provider/`: dataset classes and the `DataLoader` factory.
- `exp/`: task-specific training and evaluation classes.
- `continual_federated/`: continual learning and federated orchestration.
- `scripts/`: dataset downloaders, federated runners, comparisons, and shell benchmark commands.
- `utils/`: metrics, losses, time features, plotting, masking, early stopping, and analysis helpers.
- `figures/`: paper and presentation figures. They are documentation assets, not runtime inputs.

### Important distinction

`run.py --continual` enables replay and drift observation inside the standard long-term forecasting experiment. It does not create multiple clients or perform federated aggregation.

The actual multi-client path is driven by `scripts/compare_fedmixer.py`, `scripts/run_continual_csv.py`, or direct use of the classes exported from `continual_federated/__init__.py`.

## 2. Entry Points and Execution Paths

### Standard TimeMixer experiment

`run.py`:

1. Seeds Python, NumPy, and PyTorch with `2021`.
2. Parses model, dataset, task, optimization, sampling, GPU, and continual arguments.
3. Selects an experiment class based on `--task_name`.
4. Builds the experiment and model.
5. Runs training and testing when `--is_training 1`.
6. Runs checkpoint-backed testing when `--is_training 0`.
7. Clears the CUDA cache after each training/testing iteration.

Supported task names are:

- `long_term_forecast`
- `short_term_forecast`
- `imputation`
- `classification`
- `anomaly_detection`

Only `TimeMixer` is registered in `exp/exp_basic.py`, although the CLI help text still mentions names from the upstream project.

### Federated comparison

`scripts/compare_fedmixer.py` is the main comparison runner:

1. Loads one numeric `load` series from each prepared electricity client CSV.
2. Builds chronological round batches.
3. Creates one independently initialized TimeMixer model and optimizer per client.
4. Wraps each model in `ContinualClient`.
5. Creates a `ContinualFedMixer` trainer.
6. Runs either:
  - baseline mode: decoupled aggregation and synthetic-probe distillation, without replay or drift updates;
  - continual mode: decoupled aggregation, synthetic-probe distillation, replay, drift detection, and gradient projection;
  - both: runs both variants using the same round data.
7. Evaluates each client after every round on all chronological held-out rounds seen so far.
8. Reports ACE_MSE and BWT in the output log.
9. Appends a short summary to the requested output log.

The baseline is therefore not plain local training. It still uses the shared server aggregation and prediction-space distillation; it only passes `continual=False` to the trainer.

### Generic CSV smoke runner

`scripts/run_continual_csv.py` is a compact CPU-oriented path for a numeric CSV:

- It reads one target column.
- It uses a tiny TimeMixer configuration: sequence length 4, prediction length 1, one model layer, and no downsampling.
- It creates clients with SGD optimizers.
- It runs continual federated training and prints round-level state.

This path is intended for component smoke testing, not benchmark-quality evaluation.

### Component smoke test

`python -m continual_federated.tiny_demo` uses two `torch.nn.Linear` models and ten scalar values. It exercises drift observation, local training, clustering, averaging, replay size reporting, and mutual distillation without a dataset or GPU.

## 3. TimeMixer Model

The main implementation is `models/TimeMixer.py`, class `Model`. It is an MLP-based multiscale time-series model with two conceptual stages:

- Past-Decomposable-Mixing (PDM): extracts and mixes historical seasonal and trend information.
- Future-Multipredictor-Mixing (FMM): predicts from every processed scale and sums the scale forecasts.

### Model construction

`Model.__init__` creates:

- `pdm_blocks`: `e_layers` instances of `PastDecomposableMixing`.
- `preprocess`: a decomposition module used when channel dependence is enabled.
- `enc_embedding`: `DataEmbedding_wo_pos` from `layers/Embed.py`.
- `normalize_layers`: one `Normalize` module per scale from `layers/StandardNorm.py`.
- `predict_layers`: one temporal linear projection per scale.
- `projection_layer`: maps hidden features to output channels.
- `out_res_layers` and `regression_layers`: an additional residual path for channel-dependent forecasting.
- task-specific projection heads for imputation, anomaly detection, and classification.

The model configuration is supplied as an object with attributes rather than a dedicated dataclass. `run.py` supplies these attributes through `argparse`; the federated scripts use `types.SimpleNamespace`.

### Input and downsampling

`__multi_scale_process_inputs()` starts with input shape `[B, T, N]` and builds a list of temporal resolutions:

- scale 0 is the original sequence;
- later scales use average pooling, max pooling, or a strided circular convolution;
- temporal marks are subsampled with the same window stride when present.

The temporal length of scale `i` is expected to be approximately:

```text
seq_len // down_sampling_window**i
```

Every configured scale must therefore have a valid length, and the linear layers must be compatible with those lengths.

### Normalization and channel independence

Each scale is normalized independently with `Normalize`.

When `channel_independence == 1`:

- an input `[B, T, N]` is reshaped to `[B*N, T, 1]`;
- each channel is embedded as a separate one-dimensional series;
- the projection head predicts one channel at a time;
- predictions are reshaped back to `[B, pred_len, c_out]`.

When `channel_independence == 0`, channels remain together and the model uses cross-channel MLPs and residual regression layers.

### PastDecomposableMixing

`PastDecomposableMixing.forward()` processes every scale as follows:

1. Decompose each embedded series into seasonal and trend components.
2. Optionally apply a cross-channel MLP when channel independence is disabled.
3. Mix seasonal components bottom-up with `MultiScaleSeasonMixing`.
4. Mix trend components top-down with `MultiScaleTrendMixing`.
5. Add the mixed seasonal and trend components.
6. Apply an output MLP and, for channel-independent mode, add the original representation as a residual.
7. Trim each result back to its original scale length.

The decomposition is selected by `configs.decomp_method`:

- `moving_avg`: `series_decomp` from `layers/Autoformer_EncDec.py`;
- `dft_decomp`: `DFT_series_decomp` in `models/TimeMixer.py`.

### Seasonal mixing

`MultiScaleSeasonMixing` performs bottom-up mixing:

- starts with the fine/high-resolution scale;
- applies a linear-GELU-linear temporal projection to produce a lower-resolution contribution;
- adds that contribution to the next scale;
- continues toward coarser scales.

This transfers fine-scale seasonal detail into coarser representations.

### Trend mixing

`MultiScaleTrendMixing` reverses the scale list and performs top-down mixing:

- starts with the coarsest trend representation;
- projects it to a finer temporal resolution;
- adds it to the next finer scale;
- continues until the highest-resolution scale is reached.

This transfers coarse-scale trend information toward fine scales.

### Future multipredictor mixing

`future_multi_mixing()` creates one forecast per scale:

1. Transpose the encoded scale so the temporal linear head operates over time.
2. Project the scale from its input length to `pred_len`.
3. Project hidden features to output channels.
4. Reshape channel-independent results back to batch/time/channel form.
5. Sum all scale forecasts.
6. Denormalize using the scale-0 normalization module.

When `use_future_temporal_feature` is enabled, decoder time features are embedded and added before the final projection. The standalone CSV runners pass no temporal marks and therefore do not exercise this option.

### Decomposition implementations

`DFT_series_decomp`:

- computes a real FFT;
- removes the zero-frequency component from the ranking;
- keeps the top frequency components;
- reconstructs the seasonal signal with an inverse FFT;
- treats the residual as trend.

`series_decomp` and its moving-average helper come from `layers/Autoformer_EncDec.py`. The moving average uses replicated padding and a one-dimensional average pool.

### Other model tasks

`Model.forward()` dispatches based on `task_name`:

#### Long-term and short-term forecasting

Calls `forecast()` and returns `[B, pred_len, channels]`.

#### Imputation

`imputation()`:

- computes per-channel means and standard deviations;
- subtracts and scales the input;
- zeroes masked positions;
- processes the data through multiscale embedding and PDM;
- projects the first scale;
- restores the original mean and standard deviation.

#### Anomaly detection

`anomaly_detection()`:

- normalizes multiscale inputs;
- runs PDM blocks;
- projects the first scale back to the input shape;
- denormalizes and returns a reconstruction.

#### Classification

`classification()`:

- builds multiscale inputs without temporal marks;
- embeds and mixes the scales;
- uses the first scale;
- applies GELU and dropout;
- multiplies by a padding mask;
- flattens `[B, T, d_model]`;
- predicts `num_class` logits.

Unsupported task names raise `ValueError`.

## 4. Data Loading and Preprocessing

### Factory

`data_provider/data_factory.py` maps names to dataset classes and builds PyTorch `DataLoader` objects. Training loaders shuffle; test loaders do not. Anomaly and classification loaders use task-specific construction and disable dropping the final batch.

Supported dataset names are:

- `ETTh1`, `ETTh2`: hourly ETT datasets;
- `ETTm1`, `ETTm2`: 15-minute ETT datasets;
- `custom`: generic dated CSV;
- `m4`: M4 short-term forecasting;
- `PSM`, `MSL`, `SMAP`, `SMD`, `SWAT`: anomaly datasets;
- `UEA`: variable-length time-series classification;
- `PEMS`: traffic forecasting;
- `Solar`: solar forecasting.

### Forecasting window contract

Forecasting datasets return:

```text
seq_x       input history of length seq_len
seq_y       label context plus future target of length label_len + pred_len
seq_x_mark  input temporal features
seq_y_mark  decoder temporal features
```

The training loops generally use only the last `pred_len` values from `seq_y` as the target.

### ETT datasets

`Dataset_ETT_hour` and `Dataset_ETT_minute`:

- read a CSV with a `date` column;
- select all value columns for `M`/`MS`, or the target column for `S`;
- fit `StandardScaler` on the training segment;
- use fixed calendar boundaries for train, validation, and test;
- generate calendar or normalized time features;
- create sliding windows.

### Custom dataset

`Dataset_Custom`:

- requires a `date` column;
- moves the requested target to the final value column;
- splits data into 70% train, 10% validation, and 20% test;
- fits the scaler only on the training segment;
- creates sliding windows with overlap at split boundaries;
- supports inverse transformation through its fitted scaler.

### M4 dataset

`Dataset_M4` uses `M4Dataset` and `M4Meta`:

- selects one seasonal frequency at a time;
- samples a random cut point from each series;
- right-aligns the available history into a fixed input window;
- returns masks for available input and output values;
- uses M4-specific horizon and frequency settings.

### PEMS and Solar

The PEMS provider reads preprocessed `.npz` data and applies split-specific loading. Solar reads numeric rows and supplies the expected tensor structure. These datasets are treated specially by the long-term experiment, especially for temporal marks and inverse-transformed evaluation.

### Anomaly and UEA data

The anomaly loaders produce windows and anomaly labels for reconstruction-based detection. The UEA loader handles variable-length series and class labels; `collate_fn` pads sequences and creates a padding mask for classification.

### Federated CSV helpers

`continual_federated/csv_data.py` is intentionally independent of the standard data provider:

`load_csv_series(path, target, limit)`:

- reads a CSV with `csv.DictReader`;
- excludes a `date` field from automatic numeric target selection;
- converts valid target values to `float32`;
- skips malformed values;
- raises if fewer than two values remain.

`make_rounds(values, client_count, rounds, seq_len, pred_len, batch_size)`:

1. Splits the tensor into `client_count` streams.
2. Divides each stream into chronological round chunks.
3. Creates every valid sliding input/target window in each chunk.
4. Returns batches shaped as `[batch, seq_len, 1]` and `[batch, pred_len, 1]`.

By default this preserves the original training-only return value. With
`return_evaluation=True`, it returns `(train_rounds, test_rounds)`: each chronological
round is split into an 80% training chunk and a 20% held-out chunk. With
`standardize=True`, each client is standardized using statistics from its Round-1 training
reference and the same statistics are applied to all of that client's rounds. Test windows
use the preceding sequence context while keeping their prediction targets in the held-out
portion.

## 5. Standard Experiment Framework

### Base experiment

`exp/exp_basic.py` provides:

- the `TimeMixer` model registry;
- device selection for CPU, CUDA, multi-GPU, or Apple MPS;
- abstract placeholders for data, validation, training, and testing.

The task-specific classes implement the real behavior.

### Long-term forecasting

`Exp_Long_Term_Forecast`:

- builds TimeMixer and optional `DataParallel`;
- uses Adam;
- uses MSE except for PEMS, where it uses L1;
- trains with `OneCycleLR` by default;
- constructs decoder input from label history plus zeros when no downsampling is used;
- evaluates validation and test loss after every epoch;
- saves the best checkpoint using `EarlyStopping`;
- writes prediction figures and CSVs during testing;
- computes MSE and MAE summaries.

For PEMS, predictions and targets are inverse-transformed before metric computation.

### Short-term forecasting

`Exp_Short_Term_Forecast` is used for M4-style forecasting and supports MSE, MAPE, MASE, and SMAPE loss modules from `utils/losses.py`.

For M4 it overrides:

- `pred_len` from `M4Meta.horizons_map`;
- `seq_len` to twice the prediction horizon;
- `label_len` to the prediction horizon;
- frequency metadata used by MASE.

It writes per-series visualizations and forecast CSVs, then uses `M4Summary` after all six seasonal groups are available.

### Imputation

`Exp_Imputation`:

- replaces existing zero values with the mean of nonzero values;
- creates a random binary mask using `mask_rate`;
- zeros the masked inputs;
- trains the model to reconstruct only masked entries;
- evaluates masked-entry MSE and common metrics;
- saves predictions, ground truth, masks, and metrics under `results/<setting>/`.

### Anomaly detection

`Exp_Anomaly_Detection`:

- trains the model to reconstruct each input window;
- computes reconstruction MSE;
- estimates a threshold from combined train and test reconstruction energies;
- marks test windows above the percentile threshold as anomalous;
- applies the range adjustment helper;
- reports accuracy, precision, recall, and F-score.

### Classification

`Exp_Classification`:

- obtains train/test metadata before model creation;
- sets the maximum sequence length, input channel count, and class count;
- uses RAdam and cross-entropy;
- consumes padded batches and padding masks;
- clips gradients to norm 4;
- selects the best checkpoint using validation/test accuracy;
- writes classification accuracy to `results/<setting>/result_classification.txt`.

## 6. Continual Federated Learning

The federated classes are exported by `continual_federated/__init__.py`:

- `ContinualClient`
- `ReplayMemory` / `SpectralCoresetMemory`
- `MeanDriftDetector` / `MultiscaleSpectralTrendDetector`
- `OrthogonalSubspaceProjector`
- `FedMixerServer`
- `ContinualFedMixer`

The expected trainer input is:

```text
round_batches[round_index][client_index] = [(inputs, targets), ...]
```

### Client wrapper

`ContinualClient` owns:

- a model;
- an optimizer;
- a target device;
- bounded replay memory;
- replay batch size and loss weight;
- distillation loss weight;
- a multiscale spectral/trend detector;
- a bounded orthogonal gradient projector;
- the most recent seasonal/trend signatures and drift flags.

`observe(inputs)` updates the detector and returns a boolean drift flag.

`train_batch()` performs the local update:

1. Moves input and target tensors to the client device.
2. Sets the model to training mode and clears gradients.
3. Computes current-batch MSE.
4. If replay is enabled and memory is nonempty, samples stored windows and adds target replay MSE plus old-prediction dark-knowledge MSE.
5. If distillation inputs and targets are supplied, adds prediction-space MSE against the synthetic-probe ensemble target.
6. Backpropagates, consolidates drift-triggered directions, projects gradients away from stored directions, and steps the optimizer.
7. If replay is enabled, stores the current input, target, and prediction in memory.
8. Returns the detached scalar loss.

### Replay memory

`SpectralCoresetMemory` is a CPU-backed bounded list.

- `add_batch()` detaches and copies each sample to CPU.
- Each item contains input, target, and optionally an old output.
- `sample()` chooses a random subset, stacks it, and moves it to the requested device.
- Each input receives a normalized FFT signature.
- When full, the most spectrally redundant item is evicted instead of the oldest item.
- `ReplayMemory` remains an alias for compatibility.

Default federated capacity is 200 windows per client.

### Drift detector

`MultiscaleSpectralTrendDetector` is the default client detector:

1. Builds average-pooled temporal scales from each input chunk.
2. Separates moving-average trend and seasonal residual at every scale.
3. Computes normalized non-DC FFT power signatures for seasonal structure.
4. Projects trend structure onto low-order polynomial basis functions.
5. Compares consecutive seasonal signatures with a CDF/Wasserstein-style distance and trend signatures with cosine divergence.
6. Emits independent seasonal and trend flags plus signatures.

`MeanDriftDetector` remains available for compatibility and can still be used directly.

### Orthogonal continual adaptation

`OrthogonalSubspaceProjector` stores a capped basis for each parameter. Stable, non-drifted
batches accumulate pending directions. When the next batch is flagged as a new regime, the
pending stable-regime basis is consolidated before the new batch's update is computed. The
new-regime gradient is therefore not inserted into, and immediately projected out of, its
own protected basis. The rank is bounded by `max_rank` and the basis is kept on the
parameter's active device.

### Server clustering

`FedMixerServer` supports:

- client clustering;
- sample-count-weighted parameter aggregation;
- reclustering decisions;
- a standalone mutual-distillation helper.

`cluster_clients()` remains as a backwards-compatible monolithic state-dictionary method.
The active trainer uses `cluster_bipartite()`:

1. Builds an affinity matrix from seasonal signatures and clusters it independently.
2. Builds a second affinity matrix from trend signatures and clusters it independently.
3. Uses one cluster for homogeneous signatures and handles all-singleton cases without invoking invalid spectral decompositions.
4. Stores `seasonal_clusters`, `trend_clusters`, and the legacy `clusters` alias.

The default maximum cluster count is `min(3, number_of_clients)`, but homogeneous signatures
collapse to one cluster. Explicit `cluster_count` can still request a bounded number of clusters.

`average_cluster()` averages each parameter tensor. When sample counts are supplied, each client is weighted by its number of local samples.

`should_recluster()` requests a new clustering when:

- no clusters exist;
- losses are absent;
- enough clients are loss outliers;
- a client has cosine similarity below the configured threshold relative to its cluster centroid.

### Trainer round lifecycle

`ContinualFedMixer.fit(round_batches, continual=True)` executes this sequence for each round:

1. Validate that the number of client entries matches the number of clients.
2. If no public inputs were supplied, generate deterministic harmonic-plus-polynomial probes from the first batch shape.
3. Once clusters exist, run cluster representatives on those synthetic probes, average their predictions, and clone that ensemble target for every client.
4. For each client and each local batch:
   - observe drift when continual mode is enabled;
   - train with current data, optional replay, and optional distillation;
   - count samples and collect mean local loss.
5. Clone every client model state dictionary.
6. Recluster the seasonal graph only on round 1 or seasonal drift, and the trend graph only on round 1 or trend drift.
7. Average seasonal mixing parameters within seasonal clusters, trend mixing and predictor parameters within trend clusters, and remaining parameters globally.
8. Load each client with its own recombined state.
9. Record seasonal/trend clusters, losses, drift flags, and memory sizes in history.

### Data privacy boundary

The active scripts no longer pass the first client's local batch as a public distillation
dataset. `FedMixerServer.generate_synthetic_fourier_probes()` creates probe inputs from
sequence length and channel count only. The older `mutual_distillation()` helper remains
available for callers that explicitly provide a batch, but it is not used by the supplied
federated runners.

### Optional multiscale model output

`Model.forecast_multiscale()` returns:

```text
(forecast, per_scale_forecasts, seasonal_components, trend_components)
```

The normal `Model.forward()` contract is unchanged. When available, per-scale forecasts
are stacked and stored in replay memory. Replay matches both the final forecast target and
the historical per-scale forecasts. Synthetic-probe distillation likewise matches the final
forecast and the stacked per-scale forecasts. Generic models without this method continue
to use final-output replay and distillation.

### Continual evaluation

`scripts/compare_fedmixer.py` uses the evaluation-enabled CSV path. After each training
round, every client is evaluated on the held-out test slice of every round seen so far,
forming an error matrix `E[client, trained_round, evaluated_round]`. The reported metrics
are:

- `ACE_MSE`: the mean of the diagonal entries, measuring error on each newly learned regime.
- `BWT`: the final-round historical error minus the corresponding earlier diagonal error,
  averaged over clients and all non-final rounds. Positive values indicate forgetting.

The comparison output now reports `ACE_MSE` and `BWT` instead of final-round training-batch
MSE/MAE. The standard `make_rounds()` return remains backward-compatible for callers that
do not request evaluation splits.

## 7. Continual Flags in Standard Training

`run.py` exposes:

- `--continual`
- `--memory_size`
- `--memory_batch_size`
- `--memory_weight`
- `--drift_threshold`

When `--continual` is passed to `Exp_Long_Term_Forecast`:

- a `ReplayMemory` is created;
- a `MeanDriftDetector` is created;
- replay loss is added to the normal forecasting loss;
- the current batch is inserted into memory;
- the detector is updated during validation.

This path does not use `ContinualFedMixer`, `FedMixerServer`, or multiple client models. In addition, the observed validation drift is stored in `last_drift` but does not change training, clustering, or control flow in the standard experiment.

## 8. Dataset Preparation for Federated Runs

`scripts/download_electricity_federated.py` downloads the UCI Electricity Consumption Dataset using only the Python standard library.

The preparation process:

1. Downloads the ZIP archive.
2. Finds the contained text file.
3. Reads semicolon-separated rows.
4. Converts comma decimal separators to floating point values.
5. Selects complete household series.
6. Writes one CSV per client with `date,load` columns.
7. Splits each client timeline into chronological round ranges.
8. Counts the valid windows for each range.
9. Writes `manifest.json` containing source, sampling, client, task, and window metadata.

The downloader default is 10 clients, 1,000 rows, 4 rounds, sequence length 96, and prediction length 96. For the held-out federated comparison, each client-round chunk must be large enough for both 80% training and 20% prediction windows; the comparison runner therefore defaults to 4,000 rows. `--max-rows 0` means use the full archive.

The manifest is metadata only. The federated runners reconstruct windows from CSV values through `load_csv_series()` and `make_rounds()`.

## 9. Outputs

The repository can create these outputs:

- `checkpoints/<setting>/checkpoint.pth`: best standard experiment checkpoint.
- `test_results/<setting>/`: forecast, imputation, or visualization artifacts.
- `results/<setting>/`: saved arrays and classification results.
- `m4_results/TimeMixer/`: M4 forecast CSVs and summary inputs.
- `output/output.log`: appended FedMixer comparison summaries.
- `data/electricity_federated/clients/`: prepared client CSVs.
- `data/electricity_federated/manifest.json`: federated task manifest.
- root-level `result_anomaly_detection.txt` and `result_imputation.txt` in the relevant task paths.

## 10. Dependencies

`requirements.txt` declares:

- PyTorch
- NumPy
- pandas
- scikit-learn
- SciPy
- sktime
- PyWavelets
- matplotlib
- tqdm
- einops
- patool
- reformer_pytorch
- sympy

The federated implementation specifically uses PyTorch and `sklearn.cluster.SpectralClustering`. No external federated-learning framework is used.

The README documents a CUDA installation command for GPU hosts. The benchmark shell scripts assume a Unix shell and relative dataset paths; they are not native PowerShell scripts.

## 11. Utility Implementations

### Metrics

`utils/metrics.py` implements RSE, correlation, MAE, MSE, RMSE, MAPE, MSPE, and a combined `metric()` function.

The MAPE implementation suppresses very large ratios by replacing values above 5 with zero.

### M4 losses

`utils/losses.py` implements:

- MAPE with denominator protection;
- SMAPE;
- MASE using seasonal frequency differences.

### Time features

`utils/timefeatures.py` provides normalized calendar features for milliseconds, seconds, minutes, hours, weekdays, days, weeks, and months. `time_features_from_frequency_str()` chooses feature classes based on a pandas offset.

### Training utilities

`utils/tools.py` provides:

- learning-rate adjustment modes;
- `EarlyStopping` checkpoint management;
- a dot-access dictionary;
- a simple standard scaler;
- prediction CSV saving;
- plotting;
- anomaly range adjustment;
- classification accuracy.

## 12. Layer Modules

`layers/Embed.py` contains value and temporal embedding components used by TimeMixer.

`layers/Autoformer_EncDec.py` contains the moving-average decomposition used by the default TimeMixer configuration.

`layers/StandardNorm.py` contains normalization and denormalization logic.

`layers/SelfAttention_Family.py`, `layers/AutoCorrelation.py`, and `layers/Transformer_EncDec.py` contain general attention/encoder-decoder implementations inherited from the wider time-series codebase. They are not the core computation path of `TimeMixer.Model`.

`layers/Conv_Blocks.py`, `layers/FourierCorrelation.py`, and `layers/DWT_Decomposition.py` provide additional reusable research components. Their presence does not mean they are invoked by the default TimeMixer forecast path.

## 13. Benchmark Scripts

The shell scripts under `scripts/long_term_forecast/` cover ETT, ECL, Traffic, Solar, and Weather configurations. They invoke `run.py` repeatedly with dataset-specific sequence lengths, horizons, model dimensions, downsampling settings, and optimization parameters.

The scripts under `scripts/short_term_forecast/` cover M4 and PEMS experiments.

`scripts/download_dataset.py` downloads a small ETTh1 CSV for local experiments.

`scripts/download_electricity_federated.py` prepares household clients and continual round metadata.

`scripts/compare_fedmixer.py` runs the federated baseline and continual comparison.

`scripts/run_continual_csv.py` runs a small generic CSV continual path.

## 14. Known Limitations and Important Caveats

These are concrete properties of the current code, not proposed fixes:

1. `run.py --continual` is local continual training, not federated learning.
2. Standard long-term drift detection is updated during validation and does not affect optimization or server behavior.
3. The legacy `cluster_clients()` method still supports flattened state clustering, but the active trainer uses signature-based bipartite clustering.
4. The first federated round has no distillation target because clusters do not exist yet.
5. The current signature cluster count is bounded by `min(3, number_of_clients)`; it is not yet selected with a full eigengap heuristic.
6. The detector uses input-window decompositions rather than hooks into every internal PDM activation.
7. Orthogonal consolidation uses stable-batch parameter-gradient directions rather than activation covariance from a dedicated layer hook.
8. The CSV evaluation split is chronological and standardized, but the comparison currently reports MSE ACE/BWT rather than the full proposed MAE/ACE metric suite.
9. The active generic client loss uses final-output replay/distillation for models without `forecast_multiscale()`.
11. The standard custom dataset requires `date`; the federated CSV loader does not.
12. Downsampling dimensions must remain compatible with integer temporal lengths and all constructed linear layers.
13. Moving-average decomposition works best with an odd kernel size; incompatible settings can cause shape problems.
14. DFT decomposition can fail when `top_k` is larger than the available FFT bins.
15. Future temporal features require decoder marks and are not exercised by the CSV runners.
16. `output_attention` is accepted by the experiment code, but TimeMixer returns tensors rather than attention tuples.
17. Several base-class methods are intentionally abstract stubs and should not be called directly.
18. CLI validation is limited, so missing files, columns, incompatible channels, invalid task/dataset combinations, and insufficient windows generally fail at runtime.
19. The shell scripts assume Unix command syntax, relative paths, and an available CUDA/PyTorch environment.
20. The repository does not include automated unit tests; the tiny demo and smoke runners are the available executable checks for the continual components.

## 15. End-to-End Mental Model

For standard training:

```text
run.py
  -> Exp_* class
  -> data_provider()
  -> TimeMixer.Model
  -> task-specific loss and optimizer
  -> validation and EarlyStopping
  -> checkpoint and test artifacts
```

For the federated continual path:

```text
Electricity CSVs or generic CSV
  -> load_csv_series()
  -> make_rounds()
  -> one TimeMixer model per client
  -> ContinualClient local update
       -> current loss
       -> optional replay loss
       -> optional prediction distillation loss
       -> optional drift observation
  -> signature observation and synthetic-probe distillation
  -> FedMixerServer bipartite clustering and pathway-weighted averaging
  -> personalized recombined state broadcast to clients
  -> next chronological round
```

The implementation therefore combines a multiscale decomposition-and-mixing forecaster with an external federated training loop. The model learns temporal patterns; the continual/federated layer controls how multiple model copies learn from sequential client data and periodically share parameters and prediction targets.
