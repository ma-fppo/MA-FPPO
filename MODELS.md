# Trained models

The named model release provides pretraining and online checkpoints for 48 main-experiment configurations. Each configuration has pretraining and 50M-or-recorded-budget online checkpoints. Two additional 100M checkpoints reproduce the 2halfcheetah Good/Medium main-table endpoints. The SMACv2 Terran Random training recipe is included in the code, but no corresponding completed model was located and none is claimed in the model release.

These are the historical training-seed-0 models. The code's default seeds 0/1/2 are for new runs. The manifest records actual training steps, including the 15M 3m/2s3z runs and the recorded 8m early endpoints. No model is relabeled as a different training seed or budget.

The weights retain all model tensors exactly. Optimizer and random-number-generator states are omitted, and machine paths in metadata are replaced with portable placeholders. These exports support evaluation and initialization, not exact training resumption. Original checkpoint hashes and export hashes are recorded in MANIFEST.json.

## Evaluation

After downloading and extracting the model archive, verify it with `shasum -a 256 -c SHA256SUMS`. From this repository, evaluate a continuous policy:

```bash
python run.py main --phase evaluate \
  --config main_experiments/configs/mpe__simple_spread__medium.json \
  --data-root /path/to/data --output-root /path/to/runs \
  --checkpoint /path/to/models/main/mpe__simple_spread__medium/online/checkpoint_final.pt \
  --training-seed 0 --seed 10000 --episodes 20
```

For discrete tasks, place the matching downloaded pretraining checkpoint at `<output-root>/seed_0/main/<recipe>/pretrain/checkpoint_final.pt`. Run the recipe with `--phase audit` to generate environment information, then `--phase evaluate --checkpoint <downloaded-online-checkpoint>`. Set `--sc2path` to the matching StarCraft II installation. The evaluation loader validates policy settings and restores model tensors without requiring optimizer state or the original server paths.

For 2halfcheetah Good and Medium, `online/checkpoint_final.pt` is the 50M endpoint used in matched-budget comparisons, while `online/checkpoint_100000000.pt` is the later endpoint used in the main table.
