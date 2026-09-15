# Run a Go2 shadow experiment with our baseline controller

Use three terminals on the robot: **A for the shared camera**, **B for the fixed
baseline controller**, and **C for shadow collection**. Drive with the gamepad;
press **B** on the gamepad to annotate transitions while driving solo.

The locomotion policy stays fixed to **rough/base (`-1`)** throughout the trial.
The six shadow selectors record proposed skills without changing the active
policy, publishing motor commands, or changing control modes.

See [the experiment reference](README.md) for storage, preprocessing, metrics,
clock conventions, and recovery details.

## 1. Configure one physical setup

Start with [rough_to_gap.yaml](../configs/shadow/rough_to_gap.yaml). Replace the
example terrain description, obstacle dimensions, difficulty, and robot identity
with the actual measured setup. Give each physical setup its own configuration
name.

Update these entries within the existing YAML sections, keeping their other
settings:

```yaml
initial_class: rough
target_class: gap

transition:
  type: operator
  marker: transition
  gamepad: {enabled: true, modifier: null, button: B}

camera:
  source: publisher_tap
  socket_path: /tmp/go2_shadow_camera.sock
  rgb: false

runtime:
  duration_s: 120
```

Define the annotation boundary before collecting—for example, “the front feet
reach the marked beginning of the gap.” Use the same definition across repeated
trials. Do not trigger the annotation based on classifier predictions.

In [single_policy.yaml](../configs/single_policy.yaml), set:

```yaml
fixed_policy_index: -1
```

The launch command below also explicitly selects `-1`, overriding the YAML value.

## 2. Prepare the three terminals

In each robot terminal:

```bash
cd ~/go2_deploy_python
conda activate depthwaqdeploy
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Replace `eth0` in the following commands if the robot uses a different DDS network
interface. All three processes must use the same interface and run on the same
host for the local camera socket and host timestamps.

Ensure these assets exist on the robot:

```text
models/deploy_multi_model/policy.pt
models/classifiers_latest_offline/terrain_selector_feature_nn_best_seed_0.pt
models/classifiers_latest_offline/terrain_selector_feature_nn_best_seed_0.pt.json
models/classifiers_latest_offline/terrain_selector_raw_depth_nn_best_seed_1.pt
models/classifiers_latest_offline/terrain_selector_raw_depth_nn_best_seed_1.pt.json
models/classifiers_latest_offline/best_terrain_selectors.json
```

Model assets are excluded from Git. Pulling the code alone does not transfer them.

## 3. Terminal A: start the shared camera

```bash
python -u rough_depth_image.py \
  --interface eth0 \
  --config configs/single_policy.yaml \
  --shadow-socket /tmp/go2_shadow_camera.sock
```

This publishes normal depth input for the baseline and offers original raw frames
to the shadow collector. It can run before collection starts: the optional tap
drops frames while no collector is connected and retries on subsequent frames.
Those earlier losses appear in its cumulative counters.

Keep this process running between trials. The publisher and shadow configuration
must agree on raw camera resolution; the example uses 640×480. Their subsequent
tensor preprocessing is separately configured and recorded, as explained in the
experiment reference.

## 4. Terminal B: start the baseline controller

```bash
python -u deploy.py \
  --type single_policy \
  --interface eth0 \
  --config single_policy.yaml \
  --policy-index -1
```

Confirm this startup message:

```text
Single-policy debugging: fixed policy index -1
```

Prepare the robot on the rough/flat starting area:

| Buttons | Action |
| --- | --- |
| L1 + Y | Enter damping |
| L1 + R1 | Damping → sit |
| L1 + R2 | Sit → stand |

Allow each pose transition to settle. Leave the robot standing with neutral
joystick commands while starting collection.

Use the direct Python command here. `deploy_single_policy.sh` would start another
camera publisher and conflict with Terminal A.

## 5. Terminal C: start trial 003

```bash
python -u -m shadow_experiment.collect \
  --config configs/shadow/rough_to_gap.yaml \
  --trial-id 003 \
  --interface eth0
```

The collector prints its trial directory. Save that path.

Collection starts immediately. The 120-second limit includes time spent standing
before movement. Let it record a short, consistent rough/flat period before
approaching the obstacle. All six selectors initialize from fresh state for this
trial.

## 6. Drive the approach and annotate

Press **L1 + A** to enter baseline control, then use the joysticks to approach the
obstacle.

At the predefined annotation boundary, press **B alone** on the gamepad. The
collector prints confirmation and records the robot-state receipt timestamp and
button source. Release B before pressing it; a button already held when collection
starts is ignored until released and pressed again. Other held buttons suppress
the marker, so do not combine B with a state-control shortcut.

Typing this in Terminal C remains an alternative:

```text
mark transition
```

Ground truth is `rough` before the marker's recorded receipt time and `gap` at and
after that time. The transition latches: another marker cannot move it or reverse
it. The configured upcoming terrain is recorded separately from current ground
truth.

If someone independently observes a physical crossing, record evidence separately:

```text
crossing front feet crossed the marked boundary; confirmed by side-view observer
```

This records operator verification time and evidence. It does not change the
annotation timestamp or establish an exposure-time measurement of the crossing.

The shared-camera tap currently supplies no RGB. Use an external video recording
if visual crossing evidence is needed, and identify that recording in the evidence
text. Direct RealSense collection supports optional RGB, but cannot own the same
camera simultaneously with the baseline depth publisher.

## 7. Finish the trial

After the planned observation period, bring the robot to rest. Reverse controls:

| Buttons | Action |
| --- | --- |
| L1 + R2 | Control → stand |
| L1 + R1 | Stand → sit |
| L1 + Y | Any state → damping |

In Terminal C, enter:

```text
stop
```

Wait for the collector to exit so its final chunk and manifest are written.

**Stopping collection does not stop the robot or its controller.** Terminals A
and B are independent.

If the configured transition never occurred, leave it unmarked: ground truth
stays rough and the trial is approach-only. If the annotation fired but the robot
did not cross, it remains an annotated, unverified trial; do not enter crossing
verification. The collector also exits at its configured duration limit.

## 8. Save the baseline configuration alongside the trial

The shadow manifest records classifier provenance. Document the independently
operated baseline as well, using the trial directory printed in Terminal C:

```bash
TRIAL="/full/path/printed/by/the/collector"

cp configs/single_policy.yaml "$TRIAL/baseline_controller.yaml"

sha256sum models/deploy_multi_model/policy.pt \
  > "$TRIAL/baseline_policy.sha256"

printf '%s\n' \
  'python deploy.py --type single_policy --interface eth0 --config single_policy.yaml --policy-index -1' \
  > "$TRIAL/baseline_command.txt"
```

Record the actual command and model path if you used different ones. These are
supplemental records; do not edit the collector's resolved YAML or manifest.

## 9. Repeat independently

Return the robot to the same starting position and orientation. Keep obstacle
geometry, annotation rule, and baseline selection unchanged.

To reset baseline observation history as well as the shadow selectors, stop and
restart Terminal B between trials after placing the robot in the appropriate
resting state. Repeat the preparation sequence.

Start a new collector with a fresh ID:

```bash
python -u -m shadow_experiment.collect \
  --config configs/shadow/rough_to_gap.yaml \
  --trial-id 004 \
  --interface eth0
```

The camera can stay running; the tap reconnects to each new collector. Each
collector process resets all shadow filter and trial state. Reusing an existing
trial ID is rejected to prevent overwriting data.

Use a different configuration name whenever obstacle dimensions or other
experimental settings change. For other terrain types, configure and use
[rough_to_stairs.yaml](../configs/shadow/rough_to_stairs.yaml) or
[rough_to_climb.yaml](../configs/shadow/rough_to_climb.yaml). Set `publisher_tap`
and the other shared-camera settings in each file you use. The reporting label
`climb` explicitly maps to the classifier's trained `pit` class.

## 10. Replay and generate the report

Copy the trial tree to the analysis machine, or analyze on the robot. Set `TRIAL`
to its path on that machine. Replay additionally needs the classifier exports:

```bash
python -m shadow_experiment.analyze replay "$TRIAL" \
  --model-root models/classifiers_latest_offline \
  --output replay_trial003.json
```

Replay checks model hashes, reconstructs preprocessing, and compares classifier
and filter outputs with the online records.

Generate the combined report from the recorded outputs:

```bash
python -m shadow_experiment.analyze report shadow_trials \
  --output shadow_report \
  --appendix-failure
```

Inspect `shadow_report/validation.json` first. Outputs include:

- Per-trial and aggregate CSVs and a LaTeX table.
- Confusion matrices, ground-truth/proposed-skill timelines, and figure data.
- Transition delays, explicit misses, and exclusion reasons.
- Separate approach-only, annotated-transition, and verified-crossing counts.

Reporting uses saved online outputs; it does not rerun hardware or replace those
outputs with replay predictions.

## 11. Measure isolated selector timing

Collection already records, for every accepted frame:

- Preprocessing time.
- Separate feature-classifier and raw-depth-classifier inference times.
- Six filter-update timings.
- Total six-pipeline cycle time, update intervals, input ages, and deadline misses.

Each classifier executes once and shares its output across its three filters.
A pipeline's component time is preprocessing + its classifier + its filter.
These are concurrent shadow-mode measurements, not six isolated measurements.

For isolated timing, finish robot operation and stop the other inference
processes. Then run on the robot using saved inputs:

```bash
python -m shadow_experiment.benchmark "$TRIAL" \
  --output isolated_benchmark \
  --warmup 10 \
  --repeats 3 \
  --device cpu
```

This runs each classifier/filter combination separately and writes sample timings
and summary statistics. It does not start locomotion or camera acquisition.
Neither timing mode includes locomotion-policy execution.

## Validation status

The implementation was tested using real classifiers and synthetic sensor
records, including replay, filter equivalence, transition boundaries, and
interruption recovery. The complete shared-camera workflow, physical trials,
and isolated on-robot timings still require hardware validation.
