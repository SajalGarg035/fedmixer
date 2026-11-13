Implementation Plan: Replay-Free Distribution Memory for Modular FedMixer-FCL
Repository: `SajalGarg035/Fedmixer-improvement-`  
Target method: Adaptive Modular FedMixer-FCL with replay-free, regime-aware representation-distribution memory  
Status: Implementation specification; the method is a research hypothesis and must be validated experimentally.
---
1. Purpose and scope
This document specifies how to replace raw private-example replay in the proposed modular FedMixer-FCL method with a compact representation-distribution memory.
The change is deliberately limited to the continual-memory and distillation path. It does not ask for a complete rewrite of FedMixer, TimeMixer, clustering, or the experiment runner.
The redesigned proposed method should retain the existing architectural direction:
shared TimeMixer backbone;
lightweight cluster-specific adapters;
client similarity and soft cluster memberships;
drift-triggered reclustering;
intra-cluster aggregation;
compatibility-aware cross-cluster knowledge transfer;
historical client and cluster teachers;
task-by-task evaluation and reproducible experiments.
The key change is:
> The proposed method must not retain or replay historical private `(x, y)` windows. It should keep compact statistics describing historical representation distributions and use those statistics to retrieve relevant historical regimes and adapt the strength of historical knowledge preservation.
The old replay implementation must remain available for legacy baselines and ablation comparisons. “Remove replay” applies to the new proposed method, not to every baseline in the repository.
2. Repository findings that the agent must account for
The current public `main` branch was inspected before preparing this plan. The following points are grounded in the existing repository files.
Existing file	Current behaviour	Required action
`fedmixer_fcl/federated.py`	`ContinualFedMixer` creates one `ReplayMemory` per client, reads old windows during `_local_train`, adds current windows to memory, and includes `replay_loss` in the objective.	Add a replay-free path for the proposed method. Keep replay for legacy baseline methods.
`fedmixer_fcl/continual.py`	Defines `ReplayMemory`, `replay_loss`, and prediction-level historical distillation helpers.	Preserve the legacy replay code; add or import the new distribution-memory and representation-distillation utilities.
`fedmixer_fcl/models.py`	`OfficialTimeMixerStyle` and `TimeMixerStyle` return forecasts. `ModularTimeMixer` returns `backbone(x) + adapter(x)`. There is no public representation/encoding interface yet.	Add `encode()` / `forward_with_representation()` without changing the shape or meaning of the existing `forward()` return value.
`fedmixer_fcl/data.py`	The current anchor bank is assembled from `stream.tasks[task_id].train_x[:8]` in both synthetic and CSV loading paths.	Do not call these anchors “public”. Create a genuinely independent public-anchor source or disable public-anchor methods when none is supplied.
`fedmixer_fcl/anchor_selection.py`	Selects anchor tensors from the bank by fixed/current/current-historical/adaptive mode.	Keep its role, but ensure it only receives a valid independent public bank. Do not silently fall back to private client windows.
`fedmixer_fcl/federated.py`	`_refresh_mutual_targets()` creates peer targets on selected anchor inputs, but `_local_train()` compares those targets with predictions on the current private minibatch.	Fix the input-target alignment: student and peer predictions must be evaluated on the exact same anchor inputs.
`configs/proposed.yaml`	Currently sets `distillation.replay: true`, `lambda_replay: 0.2`, and `memory.budget: 24`.	Turn replay off for the proposed method and add explicit distribution-memory configuration.
`run_experiment.py`	Builds data, creates `ContinualFedMixer`, writes metrics and plots, and saves cluster model states.	Add persistence for distribution summaries/reference encoder state and support a separate public-anchor source if required.
`IMPLEMENTATION_LOGIC.md`, `README.md`, `plan.md`	Describe the current replay-based version in several places.	Update conflicting descriptions after implementation. This document supersedes replay-related instructions for the proposed method only.
Important data-separation issue
The existing `data.py` implementation constructs `PublicAnchorDataset` from the private client streams. In the synthetic path, it concatenates the first eight training windows from each client. In the CSV path, it does the same with the CSV client columns. This is not an independent public dataset.
The agent must not build the new method on top of that assumption. For synthetic experiments, generate anchor sequences independently using a separate random seed/generator and a documented synthetic process. For real CSV experiments, accept a separate public-anchor CSV or explicitly disable anchor-dependent cross-cluster distillation. Never silently reuse client-private windows as public anchors.
Important mutual-distillation alignment issue
The existing mutual targets are produced on an anchor batch. The local student must be evaluated on that same anchor batch when calculating mutual distillation loss. Comparing a target for one input with a prediction for a different private input is not a valid pointwise distillation objective, even if the tensor shapes happen to match.
3. Research design
3.1 Why not directly match every new task distribution to every old task distribution?
Different continual tasks can represent genuinely different temporal regimes. Forcing the current task's representation mean and variance to equal every old task's mean and variance can suppress learning of a new regime. That would trade away plasticity and could create negative transfer.
Therefore, the distribution memory should first be used for regime-aware retrieval and adaptive retention, not as a blind rule that all historical and current distributions must become identical.
The system will:
encode each observed task in a fixed reference representation space;
store only compact distribution statistics and metadata;
compare the currently observed task with stored historical regimes;
use that similarity to choose/reweight historical protection;
distil the previous teacher's outputs and representation behaviour on the current incoming batch, not on stored old private windows.
This gives a replay-free way to preserve useful historical knowledge while allowing a genuinely novel task to adapt.
3.2 High-level architecture
```text
                CLIENT i: CURRENT TASK t
                           |
                Current private minibatch (x, y)
                           |
              +------------+-------------+
              |                          |
        Live ModularTimeMixer      Frozen previous-task teacher
              |                          |
       prediction + z_student      prediction + z_teacher
              |                          |
       Forecasting loss          Prediction/representation distillation
              |                          |
              +-------------+------------+
                            |
                 Similarity-adaptive loss weights
                            ^
                            |
     Current task windows -> Frozen shared reference encoder
                            |
                  Current Gaussian summary
                            |
            Top-k retrieval from client-local memory
                            |
       [task-1 summary, task-2 summary, ... task-(t-1)]

        Local update -> server clustering/aggregation
                            |
             Drift-aware memberships and adapters
                            |
       Optional cross-cluster distillation on independent
                     public-anchor inputs
```
The reference encoder is used only to create comparable summaries and retrieve historical regimes. The live model remains trainable. The previous-task teacher is queried on the current incoming minibatch; it is not trained on, or supplied with, stored old private windows.
3.2 Memory representation
Use a diagonal Gaussian summary for each client and observed task:
[
q_{i,t}(z) = \mathcal{N}(\mu_{i,t},;\operatorname{diag}(\sigma^2_{i,t}))
]
where:
`i` is the client;
`t` is the task/regime observed by that client;
`z` is the output of a stable, frozen reference encoder;
`mu` is the feature-wise mean;
`sigma2` is the feature-wise variance.
Each memory entry should store only:
`client_id` (only in the client-local container; do not expose private client data to the server);
`task_id` and task-position metadata;
`mean`;
`log_variance` (or variance with a documented epsilon convention);
sample count used to estimate the statistics;
a compact temporal signature, such as the existing `temporal_signature` output;
representation schema/version and reference-encoder version;
optional cluster ID and creation round for diagnostics.
Do not store: input windows, target windows, raw time-series arrays, replay tensors, or a list of old model checkpoints for every task.
Use diagonal covariance initially. Do not start with a full covariance matrix, MMD, or Wasserstein optimization unless the simple method has passed tests and ablations. A diagonal Gaussian is easier to validate and has a small storage footprint.
3.3 Stable reference representation space
Historical feature summaries are only comparable when their feature coordinates mean the same thing. The live TimeMixer backbone changes during training, so comparing old summaries from an old live backbone directly with features from a newly updated backbone is unsafe.
Use one frozen shared reference encoder for memory construction and retrieval:
Train the proposed shared TimeMixer backbone on the first observed task stage.
At the end of that first stage, clone the shared backbone into `reference_encoder`.
Set it to evaluation mode and disable gradients for all its parameters.
Keep this one reference encoder fixed for the rest of the experiment.
For each client/task, pass only that task's currently available training windows through this frozen encoder in batches and calculate the mean/variance summary.
Discard the windows after summary extraction; retain only the summary and metadata.
For the initial task, create the reference encoder first and then calculate the initial distribution summaries with it. For later tasks, reuse the same frozen encoder. This gives every task summary the same representation coordinate system.
The reference encoder is one fixed shared encoder, not one encoder checkpoint per task. The normal historical teacher remains a separate, single previous-task teacher per client (and the existing previous-cluster teacher if enabled). Do not accumulate a full historical model for every task.
3.3.1 Design risk to monitor
The frozen reference encoder is trained only through the first observed task stage. This is intentional because it creates a fixed coordinate system, but later unseen regimes may be poorly represented by it. Log representation norms and retrieval-distance distributions across tasks. Treat a later aligned/EMA reference encoder as a future ablation, not as an untested default: changing the reference encoder without an explicit alignment mechanism would make stored distributions incomparable.
3.4 Representation API in the models
The current model API returns forecasts only. Add an explicit representation API while preserving compatibility:
```python
prediction = model(x)  # Existing behaviour must remain unchanged.
prediction, representation = model.forward_with_representation(x)
representation = model.encode(x)
```
The exact implementation must respect the existing TimeMixer backend.
For `OfficialTimeMixerStyle`, a practical initial representation is the concatenation of hidden activations from the season and trend mixing MLPs across all scales. Each scale's first linear layer plus activation produces a fixed-width vector; concatenate the scale-wise season/trend hidden vectors. This yields a stable feature dimension for a fixed backbone profile.
For `TimeMixerStyle`, expose a corresponding concatenation of per-scale hidden activations. For `ModularTimeMixer`, obtain the representation from its shared backbone; the forecast still includes the adapter residual. The shared backbone representation is intentional because it is the common coordinate space across clients/clusters.
Do not add trainable parameters solely to expose representations unless the existing modules make that unavoidable. Do not change the prediction calculation as part of this task.
3.5 Historical regime retrieval
At the start of a new task for client `i`, compute the current task's reference-encoder summary:
[
D_{i,t}^{cur}=(\mu_{i,t}^{cur},\log\sigma_{i,t}^{2,cur})
]
Compare it with each stored historical entry `D_{i,j}`. Use a normalized diagonal-Gaussian distance initially:
[
d_{i,t,j} = \frac{1}{d}|\mu_{i,t}^{cur}-\mu_{i,j}|_2^2
\beta_v\frac{1}{d}|\log(\sigma_{i,t}^{2,cur}+\epsilon)-\log(\sigma_{i,j}^{2}+\epsilon)|_2^2
]
Convert distance to similarity:
[
s_{i,t,j}=\exp(-d_{i,t,j}/\tau)
]
Retrieve the top-`k` relevant historical entries. Normalize their similarities to obtain retrieval weights. Record the top historical task IDs and similarity scores for experiment diagnostics.
Use the strongest/weighted retrieved similarity to determine how strongly historical teacher preservation should be applied. A novel regime should be allowed to adapt; a regime similar to previous experience should receive stronger historical protection.
A bounded modulation is recommended:
[
r_{i,t}=r_{min}+(1-r_{min}),s_{i,t}^{*}
]
where `s*` is a bounded summary of top-k similarity and `r_min` is a small configurable floor. Then:
[
\lambda_{history}^{eff}=\lambda_{history},r_{i,t},\qquad
\lambda_{repr}^{eff}=\lambda_{repr},r_{i,t}
]
Start with fixed base coefficients and a configurable similarity floor. Do not introduce a learned controller until the fixed version is stable.
3.6 Replay-free historical distillation
The existing implementation uses old private windows from `ReplayMemory` as the input to historical-teacher distillation. Replace that input with the current task minibatch `x`.
The frozen previous-task teacher receives the same current input as the student:
[
\hat y_s=f_{\theta_t}(x),\qquad
\hat y_T=f_{\theta_{t-1}}(x)
]
Prediction distillation:
[
L_{pred-history}=\operatorname{MSE}(\hat y_s,\operatorname{stopgrad}(\hat y_T))
]
This is a replay-free, Learning-without-Forgetting-style constraint on the current input distribution. It does not claim to replay historical examples.
Also compare student and teacher representations on the same current minibatch:
[
L_{repr}=|\mu(z_s)-\mu(z_T)|_2^2
+\beta_r|\log(v_s+\epsilon)-\log(v_T+\epsilon)|_2^2
]
where `z_s` and `z_T` are obtained from the current model and frozen previous teacher using the same `x`. The teacher representation is detached. This is distribution-level feature distillation on the currently observed inputs.
The distribution-memory similarity score modulates the historical and representation coefficients. This is preferable to directly forcing the new task's latent distribution to equal an old task's stored distribution.
For the first task, when no teacher or historical memory exists, both historical losses must be zero and training must reduce to the normal forecasting objective plus any independently enabled current-task terms.
3.7 Cluster historical teacher and cross-cluster transfer
Keep the existing previous-cluster teacher as an optional mechanism, but change its teacher input from replay windows to the current minibatch if it is used for replay-free local training. Keep its coefficient independently configurable.
Keep compatibility-aware cross-cluster transfer. However:
it must use a valid independent public-anchor source;
peer targets must be paired with the exact anchor inputs on which those targets were created;
low compatibility must reduce or eliminate transfer;
do not average parameters from incompatible complete model architectures.
If a real public anchor source is not available, disable public-anchor cross-cluster distillation with a clear logged explanation rather than using private data as a fallback.
3.8 Total objective
The proposed replay-free objective is:
[
\begin{aligned}
L_{total} ={}& L_{forecast} \
&+ \lambda_{history}^{eff}L_{pred-history} \
&+ \lambda_{repr}^{eff}L_{repr} \
&+ \lambda_{cluster}L_{cluster-history} \
&+ \lambda_{cross}^{eff}L_{cross}
\end{aligned}
]
There is no `L_replay` term in the proposed method.
Keep the loss terms as separate functions. Log each component separately (`forecast`, `history_prediction`, `representation_distillation`, `cluster_history`, `cross_cluster`) so that a dominant or ineffective auxiliary loss can be diagnosed.
4. Proposed configuration
Keep legacy replay keys working for old baselines. Add a separate distribution-memory section or extend `memory` without changing the meaning of the legacy `budget` field unexpectedly.
Recommended starting configuration for `configs/proposed.yaml`:
```yaml
personalization:
  mode: modular

memory:
  type: representation_distribution
  enabled: true
  max_entries_per_client: 8
  top_k: 3
  distance: diagonal_gaussian
  temperature: 1.0
  variance_weight: 0.5
  min_similarity_weight: 0.1
  reference_encoder: frozen_shared_backbone

# Legacy replay switches remain available for baseline configs.
distillation:
  mutual: true
  history: true
  cluster_history: false
  replay: false
  representation: true
  lambda_mutual: 0.2
  lambda_history: 0.3
  lambda_representation: 0.1
  lambda_cluster: 0.0
  lambda_replay: 0.0
  anchor_mode: current_historical
  historical_ratio: 0.4
  public_size: 32
```
Treat these as initial values, not tuned or validated optimum values. Make all values configurable and validate them when loading the config. The legacy `configs/fedavg_fcl.yaml` and `configs/fedmixer_fcl.yaml` should retain their current replay behaviour for fair baseline comparisons.
5. Code-level implementation map
The paths below are based on the current repository. Before editing, verify the current branch and exact function signatures.
A. `fedmixer_fcl/models.py`
Add `encode(x)` and `forward_with_representation(x)` for both TimeMixer backends.
Keep `forward(x)` returning only the forecast with the same shape as before.
Add the corresponding interface to `ModularTimeMixer`; its representation should come from the shared backbone and its prediction should still include the adapter residual.
Ensure representation output is finite, batch-first, and fixed-dimensional for a fixed profile.
Add unit tests for output compatibility and representation shape.
B. New `fedmixer_fcl/distribution_memory.py`
Implement a small, independent module. Suggested public interface:
```python
class DistributionMemory:
    def __init__(self, max_entries_per_client: int, ...): ...
    def summarize(self, representations, metadata=None): ...
    def add(self, task_id, summary, temporal_signature, ...): ...
    def retrieve(self, query_summary, top_k): ...
    def similarity(self, query_summary, stored_summary): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
```
Requirements:
Store only tensors/scalars/metadata needed for distribution summaries.
Never store `x`, `y`, raw windows, or references to `TaskData`.
Use stable numerical handling (`epsilon`, variance clamping, finite checks).
Keep a strict per-client entry limit.
Preserve task coverage and diversity when evicting; do not simply keep only the most recent entries.
Return top-k entries with their normalized similarity weights and task IDs.
Support CPU serialization and loading.
Expose memory byte count and entry count for metrics.
It is acceptable to have one `DistributionMemory` instance per client, or one container that internally partitions entries by client. The memory must remain client-local and must not be transmitted to the server by default.
C. `fedmixer_fcl/continual.py`
Keep `ReplayMemory` and `replay_loss` unchanged for legacy baselines unless a separate bug is found and documented.
Add the representation-distribution distillation helper or place it in a separate `distillation.py` utility.
Ensure teacher outputs/features are computed under `torch.no_grad()` and detached.
Ensure student representations retain gradients.
Return a scalar zero on the correct device when the teacher is unavailable.
D. `fedmixer_fcl/federated.py`
Modify `ContinualFedMixer` carefully:
Determine the method mode before constructing memories.
Instantiate raw `ReplayMemory` only for legacy methods/configurations that explicitly enable replay. The proposed modular method must not add samples to it or call `replay_loss`.
Add one frozen reference encoder for the proposed modular method. Initialize it once after the first observed task stage and keep it unchanged.
Compute the current task's reference distribution summary locally, retrieve top-k historical entries, and compute the regime-similarity modulation.
In `_local_train`, use the current minibatch for prediction-level and representation-level teacher distillation. Do not load old private windows.
Keep the cluster adapter/backbone optimization and aggregation logic intact unless tests reveal a concrete incompatibility.
At the task boundary, add one distribution summary per client for the just-completed task. Save only the summary and metadata.
Continue snapshotting only the latest previous-task client teacher and previous-cluster teachers; do not retain one full model per historical task.
Log memory retrieval, similarity, each loss component, and memory size.
Preserve the old replay route for `fedavg_fcl` / `fedmixer_fcl` baseline modes.
Be careful about the task loop: clients can have non-identical `task_order`. Use each client's actual `task_id = stream.task_order[task_position]`, not only the global task position, when writing memory entries.
E. `fedmixer_fcl/data.py` and anchor loading
Remove the current behaviour that constructs a supposedly public anchor bank from private client training windows.
For synthetic experiments, create independent public synthetic anchor sequences from a separate random generator/seed and a documented process. They must not be sliced from client streams.
For CSV/real-data experiments, support a separate public-anchor CSV (through config or a dedicated CLI argument). Validate its dimensions and chronology.
If no independent public anchors exist, set public-anchor distillation unavailable and either disable it explicitly or raise a clear configuration error when it is required.
Do not silently use private client data as a fallback.
Preserve chronological train/test splitting and future-task isolation.
F. Mutual-distillation alignment in `federated.py`
When creating mutual targets, retain the exact `anchor_x` tensor used for peer predictions.
During local training, evaluate the student on this exact `anchor_x` and compare it with the corresponding peer target.
Do not compare a peer target generated on public anchors with the student's forecast on a private minibatch.
Add a test that would fail if the target and student input batches are not the same source/tensor content.
G. Config, runner, and documentation
Update as required:
`configs/proposed.yaml` — replay off, distribution memory on, representation distillation settings.
`fedmixer_fcl/config.py` — defaults and validation for new fields.
`run_experiment.py` — save distribution-memory summaries and reference encoder state; optionally accept a separate public-anchor CSV.
`README.md` — explain replay-free proposed mode and legacy replay baselines.
`IMPLEMENTATION_LOGIC.md` — update the memory and objective sections so they no longer describe raw replay as part of the proposed method.
`plan.md` — resolve replay-related contradictions or clearly point to this implementation specification as the newer proposed-method design.
Do not delete historical baseline configurations or make old run commands unusable.
6. Training-flow pseudocode
```text
initialize client streams, independent public anchor source, models
initialize empty client-local DistributionMemory for each client
initialize previous-task teachers as None
reference_encoder = None

for task_position in each client's task stream:
    for each client i:
        task_id = client_i.task_order[task_position]
        current_task = client_i.tasks[task_id]

        if reference_encoder exists:
            current_summary = summarize(reference_encoder(current_task.train_x))
            retrieved = distribution_memory[i].retrieve(current_summary, top_k)
            regime_similarity = summarize_retrieval_similarity(retrieved)
        else:
            retrieved = []
            regime_similarity = 0

        for local epoch and current minibatch (x, y):
            y_hat, z_student = current_model.forward_with_representation(x)
            loss_forecast = MSE(y_hat, y)

            if previous teacher exists:
                with no_grad:
                    y_teacher, z_teacher = previous_teacher.forward_with_representation(x)
                loss_history = MSE(y_hat, y_teacher)
                loss_representation = moment_match(z_student, z_teacher)
            else:
                loss_history = 0
                loss_representation = 0

            loss_total = forecast
                       + adaptive_history_weight * loss_history
                       + adaptive_representation_weight * loss_representation
                       + optional_cluster_history
                       + optional_compatible_cross_cluster_loss_on_same_public_anchors

            optimize current model

        return client update and local model

    compute metadata and drift
    recluster only according to configured policy
    aggregate shared backbone and compatible cluster adapters
    refresh mutual targets on a real independent public-anchor batch
    evaluate all observed tasks

    if this is the first completed task stage:
        clone the shared backbone as reference_encoder
        set reference_encoder.eval() and requires_grad_(False)

    for each client i:
        encode only the just-observed task's training windows using reference_encoder
        create and store a compact distribution summary
        discard all temporary encoded windows
        update previous-task teacher to the latest model snapshot

    save metrics, checkpoints, distribution summaries, and configuration
```
The exact placement of summary extraction relative to aggregation/teacher updates should be implemented consistently and documented. The reference encoder must be initialized once, not overwritten after every task.
7. Implementation phases
Do not implement everything in one untested patch.
Phase 0 — Audit and baseline capture
Inspect the complete current repository, not only the files named in this plan.
Read `README.md`, `IMPLEMENTATION_LOGIC.md`, all configs, all tests, and the training/evaluation entry points.
Draw the actual current training control flow and map every replay read/write/loss call.
Identify whether any code outside `federated.py` depends on `ReplayMemory` or `memory.budget`.
Run the current smoke test and current proposed config before editing; save the command, config, seed, metrics, and output path.
Report the existing private-to-public anchor construction and mutual-target input mismatch.
Phase 1 — Model representation interface
Add the representation API to both TimeMixer backends and `ModularTimeMixer`.
Preserve forecast output exactly.
Test shapes, finiteness, gradients, and both backends.
Phase 2 — Distribution-memory module
Implement diagonal-Gaussian summary extraction.
Implement similarity, top-k retrieval, bounded storage, serialization, and byte accounting.
Test that no raw input/target data are stored.
Phase 3 — Frozen reference encoder
Freeze one shared backbone after the first task stage in the proposed modular method.
Generate per-client task summaries using the fixed encoder.
Test that its parameters never change and representations remain deterministic in evaluation mode.
Phase 4 — Replay-free training integration
Turn off replay for the proposed config.
Use current minibatches for prediction-level and representation-level teacher distillation.
Use distribution similarity to modulate historical preservation strength.
Log component losses and retrieval scores.
Keep old replay code/configs available for baseline runs.
Phase 5 — Public-anchor correctness
Replace private-derived “public” anchors with independent public anchors.
Fix mutual-target/input alignment.
Verify no raw client windows are transmitted or placed in the server-side public bank.
Phase 6 — End-to-end validation and fair ablations
Run at least:
Experiment	Replay	Distribution memory	Teacher distillation	Purpose
A. Existing replay baseline	On	Off	Existing baseline behaviour	Preserve the current reference point
B. No replay baseline	Off	Off	On, current-input teacher	Measure what is lost by removing replay
C. Distribution-memory method	Off	On	On	Test the replacement mechanism
D. No representation distillation	Off	On	Prediction-only teacher	Isolate representation-level benefit
E. No distribution retrieval weighting	Off	On	Fixed teacher weights	Isolate regime-aware retrieval
F. Full proposed method	Off	On	Adaptive prediction + representation teacher	Test the complete proposed path
Also run existing baseline methods (`fedavg`, `fedavg_fcl`, `fedmixer`, `fedmixer_fcl`) without silently changing their behaviour.
Start with the small smoke config. Then run multiple seeds and heterogeneity levels `0, 0.25, 0.5, 0.75, 1.0`. Do not tune only on one seed or one heterogeneity value.
8. Required tests and acceptance criteria
Implementation is not complete until the following are verified.
Model/API tests
`model(x)` returns the same forecast shape as before.
`model.forward_with_representation(x)` returns `(prediction, representation)`.
The forecast from `forward_with_representation(x)` matches `model(x)` within numerical tolerance.
Representations are finite and batch-first for both `official` and `approximate` backends.
The modular representation is sourced from the shared backbone and does not include the adapter's forecast residual as a substitute for an encoder representation.
Distribution-memory tests
Means and variances match a hand-computed small tensor example.
Variance is numerically safe for singleton/tiny batches.
Similarity is finite, bounded, and higher for identical distributions than clearly separated distributions.
Top-k retrieval returns the expected task IDs and normalized weights.
Entry count never exceeds `max_entries_per_client`.
State save/load round-trip preserves summaries and metadata.
Stored entries contain no raw `x` or `y` tensors.
Continual-learning tests
First task has zero history/representation loss when no teacher exists.
Later tasks use the current minibatch for teacher input; no old private windows are fetched.
Teacher parameters receive no gradients; student parameters do.
Distribution similarity increases retention weights for a similar historical regime and reduces them for a sufficiently different regime, subject to the configured floor.
Proposed mode does not instantiate/populate raw `ReplayMemory` and does not call `replay_loss`.
Legacy replay baselines still instantiate and use `ReplayMemory` when configured.
Data and distillation tests
Public anchors are not constructed from private client training windows.
CSV experiments without a separate public-anchor source do not silently reuse the private CSV columns as public anchors.
If public anchors are required but absent, the system fails clearly or disables the corresponding optional method according to configuration.
Student mutual predictions and peer targets use the same anchor inputs.
Chronological task splitting and no-future-task leakage remain intact.
End-to-end acceptance
Proposed smoke run completes with finite MAE/RMSE and finite loss components.
Distribution summaries are created for each client/task observed so far.
Memory remains within configured limits.
Saved artifacts include config, seed, metrics, task matrix, distribution summaries, reference encoder state, cluster history, drift history, timing, memory size, and communication estimates.
Baselines still run.
Tests and exact commands/results are reported; do not state that tests passed unless they were executed.
9. Metrics and research evaluation
In addition to existing MAE, RMSE, forgetting, clustering, communication, parameter count, and runtime, record:
number of distribution entries per client;
serialized memory bytes per client and total;
reference-encoder extraction time;
distribution-retrieval time;
average/top-k historical regime similarity;
effective history and representation-distillation coefficients;
prediction-history loss and representation-distillation loss separately;
raw replay buffer size for baseline comparisons (must be zero in proposed mode);
client-level and task-level metrics.
Preserve the existing task-by-task evaluation matrix: after each task, evaluate all tasks observed so far. Report average/final performance and forgetting. Use identical data, clients, task ordering, train/test splits, seeds, model budget, rounds, local epochs, and evaluation procedure across comparable methods.
Do not claim that the distribution memory improves forecasting or reduces forgetting until the ablations support that conclusion. Also do not call the method “privacy-preserving” or “differentially private”: keeping summaries client-local reduces raw-data movement but does not, by itself, provide a formal privacy guarantee.
10. Failure conditions
Stop and investigate rather than hiding results if:
representation dimensions change across tasks or clusters unexpectedly;
reference-encoder weights change after being frozen;
similarity becomes NaN/Inf or all historical similarities collapse to the same value;
the memory budget is exceeded;
current-task performance collapses due to over-regularization;
teacher losses dominate forecasting loss;
a new regime is unable to adapt because history weights remain too high;
the memory contains raw windows or targets;
public anchors are derived from private client data;
mutual targets are applied to a different input batch;
a baseline changes unintentionally;
future-task data enter memory before their task is observed.
11. Final deliverables from the coding agent
The agent must return:
A concise audit of the original control flow and exact replay call sites.
A list of files changed and why.
A description of the final distribution-memory schema and reference-encoder design.
The final equations and training pseudocode actually implemented.
Updated YAML configuration and exact run commands.
Unit-test and end-to-end-test commands with actual results.
A baseline-versus-ablation results table, including memory size and runtime overhead.
A list of limitations and any items not implemented.
Confirmation that legacy replay baselines remain available and the proposed method stores no raw historical windows.
No state-of-the-art or novelty claim should be made from implementation alone.
---
12. Source files reviewed
Repository: https://github.com/SajalGarg035/Fedmixer-improvement-
Existing research plan: https://github.com/SajalGarg035/Fedmixer-improvement-/blob/main/plan.md
Existing implementation logic: https://github.com/SajalGarg035/Fedmixer-improvement-/blob/main/IMPLEMENTATION_LOGIC.md
README and current method/configuration overview: https://github.com/SajalGarg035/Fedmixer-improvement-/blob/main/README.md
Current proposed configuration: https://github.com/SajalGarg035/Fedmixer-improvement-/blob/main/configs/proposed.yaml
Current training orchestration: https://github.com/SajalGarg035/Fedmixer-improvement-/blob/main/fedmixer_fcl/federated.py
Current continual memory helpers: https://github.com/SajalGarg035/Fedmixer-improvement-/blob/main/fedmixer_fcl/continual.py
Current model definitions: https://github.com/SajalGarg035/Fedmixer-improvement-/blob/main/fedmixer_fcl/models.py
Current data and anchor construction: https://github.com/SajalGarg035/Fedmixer-improvement-/blob/main/fedmixer_fcl/data.py