# Go2 shadow-mode terrain-selection experiment

For an end-to-end procedure using our fixed rough/baseline controller, follow
the [baseline experiment walkthrough](BASELINE_RUNTHROUGH.md).

This experiment records proposals from two classifiers × three temporal selectors
while a **separately operated controller** moves the robot. The collection
executable subscribes to `rt/lowstate` and `rt/sportmodestate`. It has no motor
publisher, controller import, mode-switch client, specialist model, or policy
activation path. It does not start locomotion. Existing controller behavior and
trained model files are preserved.

## Code and models

The implementation was inspected against both upstream HEADs on 2026-09-15:

- Deployment: `3ad8cd05f0cf23da929bded9322d455f0155f8ff`.
- Training/evaluation: `d41b0e6103d0386f91de53dece47080f922fc781`.

`reference_snapshot.json` records hashes of the inspected reference files and the
pre-existing uncommitted training-config change. Each trial separately records
its actual checkout commit, dirty status/diff, source hashes and a source snapshot;
the pinned reference snapshot remains available on a robot without the training
repository. A missing local reference checkout is recorded as unavailable, not as
a verified clean checkout.

The examples use `models/classifiers_latest_offline`:

| Export | Selected seed | Modes |
| --- | --- | --- |
| `terrain_selector_feature_nn_best_seed_0.pt` | 0 | instantaneous, EMA, Bayes |
| `terrain_selector_raw_depth_nn_best_seed_1.pt` | 1 | instantaneous, EMA, Bayes |

Both manifests order their outputs `stairs, gap, pit, random_uniform`. The
configured reporting map is `stairs → stairs`, `gap → gap`, `pit → climb`,
`random_uniform → rough`. **Climb is an explicit alias of the trained pit class**,
not a newly trained classifier. Hashes, sidecar manifests, seeds, and the export's
held-out selection metadata are saved per trial. Files are never rewritten.

The collector runs each classifier once per accepted frame. Its logits and
probabilities feed three independently reset selectors. It reuses
`TerrainSelector.reset`, `_ema`, and `_bayes`; the only filter extension permits
passing already-computed probabilities to Bayes. The original deployment call
path has the same numerical results. EMA smooths **logits**, with the existing
alpha and change-patience rules; its proposed skill can differ from its smoothed
argmax while a change is pending. Bayes retains the existing uniform prior,
transition/observation matrices, epsilon floors, and normalization.

## Define one physical configuration and one independent trial

Copy one of `configs/shadow/rough_to_{gap,stairs,climb}.yaml`. Replace the example
terrain description, dimensions, difficulty, robot identity, and trial ID with
measured values. Repeat each physical configuration independently, returning the
robot to the rough/flat starting area before each new collection process. Never
concatenate multiple obstacles or reset filters within a trial.

Exactly one transition condition is configured:

```yaml
initial_class: rough
target_class: gap
transition:
  type: operator
  marker: transition
  gamepad: {enabled: true, modifier: null, button: B}
# Or:
# transition: {type: time, seconds: 5.0}
# transition: {type: position, axis: x, comparison: ge, threshold_m: 2.0}
```

Time is relative to the manifest's `start_monotonic_ns`, when the trial writer
is created after model loading, before camera startup. Position uses reported
`rt/sportmodestate.position` odometry in meters, with an explicitly chosen axis
and `ge`/`le` comparison. Verify that this stream exists and its coordinate frame
is appropriate; the collector does not infer distance from IMU acceleration.

For an operator trigger, press **B alone** on the gamepad (enabled in the
example configs), or type `mark transition` and Enter in the collection terminal.
The collector reads `wireless_remote` from low-state messages; it does not consume
or change controller input. B is unused by our baseline state machine. Each new
press is detected once; a held button at startup is ignored until released and
pressed again. Other simultaneously held buttons suppress the marker. Set
`transition.gamepad.enabled: false` to disable gamepad annotation, or configure a
button name from `KeyMap` and an optional modifier. Gamepad input requires an
operator-type transition and records its button names, state tick, and host
receipt timestamp. The original one-transition latch still applies; this button
annotates the configured transition, not an independently verified crossing. Receipt time is recorded. The first satisfied condition latches;
subsequent conditions cannot change its timestamp or undo the transition. Every
frame is labeled rough before that timestamp and the configured target after it.
The first observed frame ID at/after the event is recorded, even if that frame is
rejected for classification. All records separately retain `configured_target`.
A trigger can precede the first accepted frame; such a trial cannot support a
valid pre/post transition-delay measurement.

An annotation is **not proof of physical crossing**. To record an independently
observed crossing, type, for example:

```text
crossing front feet crossed marked gap boundary; video camera B at 00:12
```

This records operator verification time and evidence, without changing ground
truth or retroactively changing the configured trigger. The example does not
claim that the verifier's keypress is a precise exposure-time crossing
measurement. Reports identify verified trials separately; delays remain anchored
to the configured annotation. If the annotation condition is never met, labels
remain rough and the trial is approach-only, even if crossing verification was
entered inconsistently. Such discrepancies remain visible in the records.

Known uncertain intervals can be declared in `analysis.excluded_intervals_s` as
`[start,end)` pairs relative to trial start. They retain all six online outputs
but are excluded from accuracy and temporal metrics. Do not edit resolved trial
files afterward; their hashes are validated. Use a separately documented report
extension for later annotation adjudication rather than silently changing labels.

## Collect on hardware

Use the robot's existing Python environment with Torch, NumPy, PyYAML,
`unitree_sdk2py`, CycloneDDS, and pyrealsense2. Offline reporting additionally
needs matplotlib; it does not import camera or DDS libraries.

**Direct camera ownership** (default): use an independent controller that does
not also open this RealSense device. The collector opens only camera streams:

```bash
cd ~/go2_deploy_python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m shadow_experiment.collect \
  --config configs/shadow/rough_to_gap.yaml --trial-id 003 --interface eth0
```

The interface must match the independent controller's DDS interface. `lo` uses
domain 1 for simulation; other interfaces use domain 0. A trial stops at the
configured duration, on `stop` + Enter, or on SIGINT/SIGTERM. No mode switch or
motor stop command is sent by the collector; robot operation remains with the
independent controller/operator.

**Share the existing depth publisher** if the independent controller needs the
same camera. Set `camera.source: publisher_tap`, `camera.rgb: false`, and use the
same `socket_path` on both sides. Start the collector first, then start the depth
publisher with an optional tap:

```bash
python -m shadow_experiment.collect \
  --config configs/shadow/rough_to_gap.yaml --trial-id 004 --interface eth0

# Separate terminal, same host:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python rough_depth_image.py --interface eth0 --config configs/single_policy.yaml \
  --shadow-socket /tmp/go2_shadow_camera.sock
```

Operate the locomotion controller separately as usual. Do not launch a second
camera publisher through a combined controller launcher in this arrangement.
The tap sends original Z16 frames before the existing publisher's filters. It is
opt-in, uses a bounded queue and a separate sender thread, and cannot wait on the
collector in the camera's normal publish path. Without the flag, the existing
publisher follows its usual preprocessing and DDS publication. Tap errors are
reported without stopping that publisher. Optional copying adds overhead when
enabled; include it in your hardware scheduling checks.

The socket is local to one host, mode 0600, and must not already exist. A source
disconnect terminates the trial instead of silently restarting sensor/frame IDs.
After a hard kill, verify that no collector owns a leftover socket before
removing it. Publisher-tap acquisition cadence follows the independent
publisher, and is recorded rather than inferred from requested FPS. Direct
capture supports optional RGB from the same frameset, checks timestamp domain and
skew, and stores RGB unregistered to depth with that limitation explicit.

## Depth, state, and overload semantics

The default `training_bicubic` reproduces the deterministic tensor operations in
training's `depth_mixin.py`: convert raw units to meters, clamp/normalize to
0–3 m, crop with training endpoints (no extra bottom/right pixel), bicubic resize
with `align_corners=False`, then clamp to [0,1]. Input is exactly float32 48×64.
The 640×480 crop `[48,0,28,36]` scales the training 160×120 crop `[12,0,7,9]`.
Synthetic training noise, rendered latency, and artifacts are not added to real
sensor measurements. Matching these tensor operations does not certify the
physical camera mounting or intrinsics; these must be checked against the
training pose/FOV and are recorded where available.

`deployment_tensor_only` is an explicit alternative that reuses
`common.depth_processing.preprocess_depth_array` (legacy crop endpoints and
adaptive average pooling). It does **not** reproduce the separate stateful
RealSense hole/spatial/temporal filters. The examples use training preprocessing;
no hidden fallback or automatic interpolation change is applied.

State is the most recent low-state message received **at or before** depth host
receipt, bounded by `max_state_age_s`. RPY is in radians; gyroscope is unscaled
body-frame rad/s, as in the exported classifiers. Exact state input arrays and
sensor tick are saved. Position is separately aligned by the same causal rule.
Host receipt alignment is not hardware exposure-time synchronization. RealSense
sensor timestamps and domains are saved, but no unmeasured cross-device clock
offset is invented. Host input age is measured from host receipt, not exposure.

All six selectors share the same accepted frames and state samples. The capture
queue is bounded and drops new arrivals when full; drops happen before any
classifier. Rate-decimated, duplicate/reordered sensor IDs, stale sensor times,
stale host inputs, missing/stale state, invalid depth/state, and unsynchronized
RGB have explicit counters/events. Identical pixel values in genuinely new sensor
frames are allowed; repeated sensor frames are not new observations. A model
error terminates the trial, preventing partial advancement of one filter bank
from contaminating subsequent comparisons. No frame is reused to fill a gap.

Drop counters describe different stages and must **not be added blindly**:
sensor-number gaps can include queue losses; publisher-tap counters are
last-observed cumulative sender counters and can include losses before the trial.
Their final unseen tail is unknowable after disconnect. Queue drops and frames
left queued at termination are reported separately. The separate bounded drop-event queue saves lost frame IDs/receipt times when
possible and counts its own overflow; it cannot allocate unbounded log backlog.
State ring-buffer eviction
is intentional bounded history retention, not a classification-frame loss.

## Trial records and recovery

A trial lives under:

```text
<output_root>/<experiment>/<configuration>/rough_to_gap_easy_trial003_<UTC>/
  resolved.yaml
  manifest.json
  events.jsonl
  source_snapshot/
  frames_000000.npz
  frames_000001.npz
  ...
```

A persistent `.trial_<id>.claim` in the configuration directory prevents reuse of
an explicit trial ID, including after an interruption. Use a new ID for a new
trial. The resolved configuration and manifest, not the path, define metadata.
Model paths can be relocated for replay only with matching model/sidecar hashes.
Model exports are excluded by the repository gitignore; copy the two `.pt` files,
their `.pt.json` sidecars, and `best_terrain_selectors.json` to the robot separately.
A code-only pull does not install those assets.

Each compressed chunk contains raw Z16, exact processed depth, exact RPY/omega,
optional RGB, and UTF-8 JSON records with IDs, timestamps, probabilities,
filtered distributions/beliefs, pending EMA state, proposed skills, annotations,
validity, timings, and counters. The manifest indexes chunk hashes and frame
counts. Events record resets, markers, camera identity, rejections, errors and
termination. Chunk files are fsynced and atomically renamed before the manifest
is advanced. Readers discover completed but not-yet-indexed chunks after a crash.
Incomplete temporary files are ignored and flagged; up to `chunk_frames - 1`
accepted frames still buffered in RAM can be lost on SIGKILL/power loss. Normal
signals flush the partial final chunk. A read or hash error is never silently
accepted as a complete trial.

## Replay, report, and isolated benchmark

Copy the trial tree to any machine with the offline dependencies. Reporting uses
**recorded online outputs** and needs neither hardware nor classifier execution:

```bash
python -m shadow_experiment.analyze report shadow_trials \
  --output shadow_report --appendix-failure
```

Replay checks both model/sidecar hashes, resets every selector, reruns saved
processed inputs, reconstructs preprocessing from raw input, and compares logits,
probabilities, filtered distributions, pending EMA state, and proposed skills:

```bash
python -m shadow_experiment.analyze replay /path/to/trial \
  --model-root models/classifiers_latest_offline --device cpu \
  --output replay_comparison.json
```

The default absolute tolerance is 1e-6 with no relative tolerance. Exact discrete
proposals must agree. Changed software/device kernels may require an explicitly
reported tolerance; a mismatch is reported rather than replacing online outputs.
Structural validation issues are retained. Do not equate successful replay of
recoverable data with a complete physical trial.

On the robot, benchmark each of the six pipelines **in isolation** using saved
inputs, without starting a camera or locomotion controller:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m shadow_experiment.benchmark /path/to/trial \
  --output isolated_benchmark --warmup 10 --repeats 3 --device cpu
```

Each isolated pipeline loads only its classifier and one filter; warmup is
excluded, filter state resets before each repetition, and recorded frames stay in
order. Reports include sample CSVs, p50/p95/mean/std, machine/software provenance,
and model hashes. Repeat measurements are timing repetitions, not independent
physical trials. Preprocessing, model transfer/forward, and filter time are
included; saved-file reads, sensor acquisition, and specialist execution are not.
CUDA runs synchronize before/after forwards, so asynchronous GPU dispatch is not
mistaken for execution time.

Collection timings are labeled **concurrent shadow-mode selector timings**:
all six configurations coexist in a serial shared-model evaluation cycle. This
is not six isolated measurements and not six simultaneous GPU streams. Each
record contains preprocessing, both classifier calls, six filter durations,
total six-pipeline compute time, processing update interval, host input/state age,
and both compute-only and receipt-to-completion deadline flags. Model cold-start
costs remain in the online record. Compression/write overhead affects subsequent
update intervals and queue losses but is outside selector compute latency.

## Analysis conventions and artifacts

- Grouping fingerprints the resolved physical, model, camera, filter, and runtime
  configuration while excluding trial IDs and output/reference paths. Different
  settings cannot silently pool under one configuration name.
- `validation.json` lists incomplete, corrupt, inconsistent or recovered trials.
  Primary aggregates exclude unclosed trials, structural errors, failed
  collection, and capture shutdown timeouts. Recoverable data remains available
  for per-trial diagnosis and replay.
- `per_trial.csv` reports accuracy, balanced accuracy, supported-truth macro F1,
  filtered NLL/Brier, premature nonrough frames, false transitions, timing,
  validity counts, and annotated/verified/approach status. Missing truth classes
  are not counted as zero recall in balanced accuracy or supported-truth F1.
- False-transition rate reuses the reference: a prediction change when truth is
  unchanged, divided by all valid adjacent classification opportunities. The
  copied reporting-only `segment_bounded_v2` functions are unchanged from the
  recorded reference commit.
- Recorded queue losses, rejected invalid/stale inputs, excluded intervals (even
  those falling between accepted frames), and gaps above `max_contiguous_gap_s` split temporal
  sequences. A first match is searched only within the first valid target
  segment; later matches after a gap cannot rescue a miss. An annotation without
  a contiguous valid rough-to-target boundary is excluded with a reason. Delay
  in seconds is from the recorded annotation timestamp, not the filename or
  trial start. Misses are explicit, not assigned zero delay.
- Approach-only trials contribute rough classification, premature predictions,
  and time/odometry plots; they never enter the completed annotated-transition
  denominator. A position plot is emitted only if position was measured.
- `aggregate.csv`/`.tex` provide counts and equal-trial means/sample standard
  deviations. Matched-delay counts are separate from misses. Verified crossing
  counts and annotation-delay statistics are separate; a keypress annotation
  alone is never reported as a verified physical crossing.
- Confusion matrices use pooled frame counts and retain their numeric figure
  data. Representative trials use the first lexical eligible trial per group;
  the image frame is the first accepted post-annotation frame, or midpoint for
  approach-only trials. Optional appendix selection is the lowest mean six-mode
  accuracy trial with errors (lexical tie-break), and its first valid erroneous
  frame. All choices are saved in `figure_selection.json` to avoid undocumented
  cherry-picking.
- Figures save PDF and PNG plus `.npz` source data. Class colors are fixed across
  ground truth and all proposed skills. RGB absence is displayed explicitly.

## Verification and limits

Run the deployment regression suite, including the shadow integration tests:

```bash
python -m unittest discover -s tests -v
```

Tests use real selected TorchScript exports with synthetic sensor data and verify
all three trigger types, latching/reset, causal timestamp alignment, legacy filter
equivalence, duplicate/drop handling, segment-bounded misses, exact replay,
compressed-chunk recovery, overwrite prevention, offline figures/tables, and
isolated benchmark execution. They also check that shadow modules expose no
motor-command publisher or control-client imports. Existing deployment tests
remain part of the suite.

No physical obstacle trials, verified crossings, RealSense/DDS shared-stream
latency tests, on-robot load/thermal tests, CUDA benchmarks, or camera mounting
validation were performed as part of this implementation. The generated test
figures and timings are synthetic validation artifacts, not experimental results.
