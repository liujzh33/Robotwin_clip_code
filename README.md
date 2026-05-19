# Robotwin Clip Code

Rule-based **phase segmentation** and **language enrichment** for [RoboTwin](https://github.com/TianxingChen/RoboTwin) demonstration data. This repo covers all **50** standard manipulation tasks: each task has checkpoint rules (`task_def/`), template enrichment (`enrich_language/`), and optional trajectory plots (`plot_code/`).

> **Note:** The `plot/` folder (generated PNGs) is **not** tracked in git. Run the plotting step locally to create it.

## What this repo does

For each episode in processed data:

1. **Segment** — detect phase checkpoints from gripper / joint velocity / end-effector signals (`process_data_generic.py` → writes `phase_info` into `instructions.json`).
2. **Enrich** — attach per-phase language variants to each instruction (`enrich_language/enrich_<task>.py`).
3. **Visualize** (optional) — batch plot trajectories with checkpoint lines (`plot_code/batch_analyze_trajectories_<task>.py`).

## Requirements

```bash
cd /path/to/Robotwin_clip_code
pip install -r requirements.txt
```

- Python 3.8+
- Processed episodes: `episode_*/episode_*.hdf5` + `instructions.json`
- Optional raw RoboTwin HDF5 for richer endpose features and plots:  
  `{raw_root}/{task_name}/aloha-agilex_clean_50/data/episode{N}.hdf5`

## Repository layout

```text
Robotwin_clip_code/
├── process_data_generic.py    # Step 1: annotate checkpoints
├── task_def/                    # Per-task checkpoint rules (50 tasks)
├── enrich_language/             # Step 2: language templates (50 tasks)
├── plot_code/                   # Step 3: analysis & batch plotting
└── plot/                        # Local output only (gitignored)
```

## Data paths (how to change them)

All paths are passed on the **command line**. Replace the examples below with your own directories.

| Argument | Used in | Meaning |
|----------|---------|---------|
| `--data_dir` | All 3 steps | Root of **processed** data: contains `episode_0/`, `episode_1/`, … |
| `--raw_root` | Step 1 (optional), Step 3 (recommended) | Root of **raw** RoboTwin dataset: `{raw_root}/{task}/aloha-agilex_clean_50/data/episode{N}.hdf5` |
| `--output_dir` | Step 3 only | Where PNG plots are saved (create any folder you like) |
| `--episode_range` | Step 1 only | Subset, e.g. `0,3,4` or `0-10` |

**Processed data directory name** is usually:

```text
{task_name}-aloha-agilex_clean_50-50
```

Example on this machine:

```text
/data2/liujingzhi/robotwin_processed_pi0/turn_switch-aloha-agilex_clean_50-50/
├── episode_0/
│   ├── episode_0.hdf5
│   └── instructions.json
├── episode_1/
...
```

**Raw dataset** layout:

```text
{raw_root}/turn_switch/aloha-agilex_clean_50/data/episode0.hdf5
```

If your data lives elsewhere, only change the flags — **no code edit required**:

```bash
export PROCESSED=/your/path/turn_switch-aloha-agilex_clean_50-50
export RAW_ROOT=/your/path/robotwin2_dataset
export REPO=/your/path/Robotwin_clip_code
export PLOT_OUT=$REPO/plot/turn_switch

python $REPO/process_data_generic.py --task_name turn_switch --data_dir $PROCESSED --raw_root $RAW_ROOT
python $REPO/enrich_language/enrich_turn_switch.py --data_dir $PROCESSED
python $REPO/plot_code/batch_analyze_trajectories_turn_switch.py \
  --data_dir $PROCESSED --output_dir $PLOT_OUT --raw_root $RAW_ROOT
```

Default raw search paths (when `--raw_root` is omitted) are defined in `process_data_generic.py` and batch plot scripts; override with `--raw_root` on any machine.

## Standard 3-step pipeline (example: `turn_switch`)

```bash
REPO=/data2/liujingzhi/Robotwin_clip_code
PROCESSED=/data2/liujingzhi/robotwin_processed_pi0/turn_switch-aloha-agilex_clean_50-50
RAW=/data2/liujingzhi/robotwin/robotwin2_dataset

cd "$REPO"

# Step 1: rule-based phase checkpoints → instructions.json
python process_data_generic.py \
  --task_name turn_switch \
  --data_dir "$PROCESSED" \
  --raw_root "$RAW"

# Step 2: enrich per-phase language templates
python enrich_language/enrich_turn_switch.py \
  --data_dir "$PROCESSED"

# Step 3: optional QA plots (output not in git)
python plot_code/batch_analyze_trajectories_turn_switch.py \
  --data_dir "$PROCESSED" \
  --output_dir "$REPO/plot/turn_switch" \
  --raw_root "$RAW"
```

Process only some episodes:

```bash
python process_data_generic.py \
  --task_name turn_switch \
  --data_dir "$PROCESSED" \
  --episode_range 0,3,12 \
  --raw_root "$RAW"
```

## Another task (`adjust_bottle`)

Replace `turn_switch` with any task name below; script names follow the same pattern:

```bash
TASK=adjust_bottle
PROCESSED=/data2/liujingzhi/robotwin_processed_pi0/${TASK}-aloha-agilex_clean_50-50

python process_data_generic.py --task_name "$TASK" --data_dir "$PROCESSED"
python enrich_language/enrich_${TASK}.py --data_dir "$PROCESSED"
python plot_code/batch_analyze_trajectories_${TASK}.py \
  --data_dir "$PROCESSED" \
  --output_dir plot/${TASK} \
  --raw_root /data2/liujingzhi/robotwin/robotwin2_dataset
```

## Supported tasks (50)

| | | | |
|---|---|---|---|
| adjust_bottle | beat_block_hammer | blocks_ranking_rgb | blocks_ranking_size |
| click_alarmclock | click_bell | dump_bin_bigbin | grab_roller |
| handover_block | handover_mic | hanging_mug | lift_pot |
| move_can_pot | move_pillbottle_pad | move_playingcard_away | move_stapler_pad |
| open_laptop | open_microwave | pick_diverse_bottles | pick_dual_bottles |
| place_a2b_left | place_a2b_right | place_bread_basket | place_bread_skillet |
| place_burger_fries | place_can_basket | place_cans_plasticbox | place_container_plate |
| place_dual_shoes | place_empty_cup | place_fan | place_mouse_pad |
| place_object_basket | place_object_scale | place_object_stand | place_phone_stand |
| place_shoe | press_stapler | put_bottles_dustbin | put_object_cabinet |
| rotate_qrcode | scan_object | shake_bottle | shake_bottle_horizontally |
| stack_blocks_three | stack_blocks_two | stack_bowls_three | stack_bowls_two |
| stamp_seal | turn_switch | | |

`task_name` must match the module name under `task_def/` (snake_case).

## Output format

After step 1, each `instructions.json` contains:

```json
{
  "instructions": ["..."],
  "subtasks": [["phase0 text", "phase1 text", ...], ...],
  "phase_info": {
    "task_name": "turn_switch",
    "checkpoints": [25, 59],
    "total_steps": 120,
    "num_phases": 3
  }
}
```

Phases are half-open intervals: `[0, c0)`, `[c0, c1)`, …, `[c_{n-1}, end)`.

## Extending a new task

1. Add `task_def/<task_name>.py` implementing `BaseTaskProcessor`.
2. Add `enrich_language/enrich_<task_name>.py`.
3. Add `plot_code/analyze_trajectory_<task_name>.py` and `batch_analyze_trajectories_<task_name>.py` (copy from a similar task).

## License

Add your license here if publishing publicly.
