## Memory statistics in the live latency test

Each measured call now records `process_rss_mib` after inference, outside the
latency timer. Per-pipeline `summary.csv` and terminal output include its mean
and sample standard deviation (`ddof=1`). RSS is current resident RAM from
`/proc/self/statm`, measured in MiB; it includes this process's Python, DDS,
input buffers and allocator caches, and excludes the separate depth publisher.
It is not model-only memory or a transient peak measurement. Pipelines run in
one process, so caches retained from earlier pipelines may affect later values.
CUDA runs also log `cuda_allocated_mib` and `cuda_reserved_mib` with mean/std;
CPU runs leave these fields empty. These CUDA figures cover the PyTorch allocator,
not total GPU use. Do not add them to RSS on the robot's unified-memory hardware.
No command-line changes are required. Existing result files remain unchanged.

## Background frame recording

Recording-only collection sends owned frame/metadata snapshots to a background
writer. `runtime.writer_queue_size: 32` bounds waiting frames (plus one chunk
being assembled/written). At 640x480, raw + filtered depth use roughly 38 MiB
for 32 waiting frames, excluding chunk/compression workspace and optional RGB.
The collector does not wait for compression or frame file writes. Overflow is
logged as `writer_queue_drop_new`, separate from capture-queue drops, and breaks
the analysis continuity segment. Accepted counts include admitted frames; the
manifest frame count describes committed data. The final manifest includes writer
queue capacity, high-water mark, admission and overflow counts.

On stop, Ctrl+C or SIGTERM, the writer drains admitted frames and commits its last
partial chunk before closing. Allow shutdown to finish. A force kill can lose
queued frames; completed atomic chunks remain recoverable. Disk failures stop
collection and leave the trial incomplete. Event logging retains its existing
synchronous durability. The saved trial format and offline inference commands
are unchanged; latency tests do not use this recording queue.

## Robot DDS runtime

Camera and shadow entry points select `~/cyclonedds-0.10.2` on ARM64 when
installed. This fixes the verified native write crash with the robot's
`/usr/local/lib/libddsc.so` (Iceoryx path). Selection changes only these processes
and their children; system DDS libraries and Unitree services are untouched.
`GO2_CYCLONEDDS_HOME` can explicitly select another installation. The process
restarts once before importing DDS so both Python and native dependencies use
the same installation. Camera settings and depth message format are unchanged.

## Live-frame robot latency test

```bash
python -u -m shadow_experiment.live_latency \
  --config configs/shadow/rough_to_climb.yaml \
  --output latency_results/live_run001 --iterations 1000 --warmup 50 --interface eth0
```

Run on the robot with other camera publishers/collection stopped. This starts the
existing deployment depth node in a separate process, subscribes to robot state,
and runs just one classifier/filter pipeline at a time. Each call waits for a
new capture acquired after the previous call; duplicate, old and unaligned inputs
are excluded and counted. Frame waiting and camera preprocessing are outside the
classifier/filter timer. Each pipeline has 50 warmup calls, then a filter reset
and 1,000 measured calls. No motor commands or control-mode changes are sent.
Results are written incrementally on the robot under the specified directory:
per-pipeline samples, mean/sample-standard-deviation summary, camera log and
provenance manifest. Six pipelines at 10 Hz take at least 10.5 minutes, longer if
frames are skipped. Different pipelines see different live images; keep the
scene and robot workload stable. These measurements include contention with the
async camera node. Use the saved-input test below for identical-input comparisons.

## Controlled robot latency test

Run on the robot while collection and custom controllers are stopped:

```bash
python -m shadow_experiment.latency /path/to/recorded_trial \
  --output latency_results/run001 --iterations 1000 --warmup 50 --device cpu
```

Each of the six classifier/filter pipelines runs separately, with 50 unmeasured
warmup calls and exactly 1,000 measured calls. All use the same saved normalized
depth/state sequence, held in RAM. Filters reset after warmup and retain state
through measured calls. No camera, resizing, disk reads, or specialist execution
is timed. Calls include tensor conversion, transfers, validation and output
packaging from the existing pipeline. CUDA execution is synchronized when selected.
The benchmark uses one Torch intra-op and inter-op thread. Results stay in the
specified directory on the machine executing the command: per-call CSVs,
`summary.csv` (mean, sample standard deviation and p95 for total/classifier/filter),
and `manifest.json` recording hardware/software, model hashes and test settings.
Use a new output directory; existing runs are never overwritten. This is a
sequential microbenchmark, not a guarantee under concurrent robot workloads.

## Recording with the Unitree controller (current workflow)

Run the deployment depth node and recording-only collector in separate processes:

```bash
python -u -m shadow_experiment.record --config configs/shadow/rough_to_gap.yaml --trial-id 002 --interface eth0
```

The launcher owns the camera; stop any previous camera publisher before launching.
It starts no locomotion controller. Operate using the Unitree controller, press B
once to annotate the transition, then type `stop` or press Ctrl+C to finish.
Use a new trial ID each time. The publisher restarts for every trial, resetting
RealSense temporal history. It uses the trial camera configuration and processes
at `runtime.update_hz`, with a bounded asynchronous socket sender. The collector
saves raw, filtered and exact normalized 48x64 depth, aligned state, annotations,
timestamps and drop counters. It loads no classifiers and repeats no resizing.
RGB is unavailable through this publisher. Disk recording can still drop frames;
inspect counters after each trial. Recorded timing is acquisition/recording timing,
not classifier latency.

Copy the trial to the analysis machine and generate a separate derived trial:

```bash
python -m shadow_experiment.infer /path/to/recorded_trial --output offline_trials/trial002 --model-root models/classifiers_latest_offline
python -m shadow_experiment.analyze report offline_trials --output shadow_report
```

All six selectors run offline, once per recorded frame in timestamp order, with
fresh filter state per trial and verified model hashes. Original recordings are
preserved. Derived manifests and per-trial CSVs identify offline execution and
its host; inference timings are **not robot deployment latency**. Reports retain
historical timing column names; consult `execution_mode` and `timing_scope`.
For robot latency use the separate saved-input isolated benchmark on the robot.
The older online collection instructions below apply only with
`runtime.record_only: false`.

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

**Direct camera ownership** (set `camera.source: realsense` explicitly): use an
independent controller that does not also open this RealSense device. The collector opens only camera streams:

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

**Share the existing depth publisher** (the example configs default to this) if
the independent controller needs the
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
The tap sends both original Z16 and the publisher's filtered depth, before tensor
cropping/resizing. Filtering is not applied twice. It is
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

The examples now use `realsense_filters: true` and
`preprocessing: deployment_tensor_only`, matching robot-control operations:
hole filling (SDK default), spatial filtering (magnitude 5, alpha .75, delta 1,
holes_fill 4), then temporal filtering (alpha .75, delta 1). The settings are
shared with `rough_depth_image.py` through `common/realsense_filters.py`.

Filtered depth is passed to the same `preprocess_depth_array` used by control:
convert sensor units to meters, normalize/clamp to 0–3 m, use the control crop
endpoints (including its extra bottom/right pixel), and adaptive-average-pool to
48×64. Rotation follows `rotate_180`. This replaces the previous example default
of training-style bicubic resizing; old trial configurations remain unchanged.

Both raw and filtered depth are saved. Replay and isolated benchmarks reconstruct
tensor preprocessing from **saved filtered depth**, not by rerunning a temporal
filter on a subsampled raw recording. RealSense filtering runs in the acquisition
path, with `realsense_filter_ms` recorded separately from selector compute time.
The isolated benchmark excludes RealSense filter execution; it measures the
remaining tensor preprocessing, classifier and selector.

Direct capture creates fresh filters per trial and updates them on every acquired
camera frame. Shared-camera capture uses the existing publisher's filter history
and cadence; it does not reset or change the controller's filters. Thus identical
operations do not imply identical temporal history across the two capture modes.
Use one capture mode consistently across compared trials. Restart the publisher
between trials if your protocol requires its temporal filter history to reset.

`training_bicubic` remains available explicitly for training-style crop/resize
comparisons; `realsense_filters: false` preserves the unfiltered capture option.
Physical camera mounting and intrinsics still require independent verification.

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
processed inputs, reconstructs preprocessing from the saved raw or filtered input, and compares logits,
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
