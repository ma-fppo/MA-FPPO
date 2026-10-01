# Benchmark data and environments

The shared [CoFlow dataset repository](https://huggingface.co/datasets/coflow-project/CoFlow-datasets) currently provides MPE Tag and World arrays (verified September 26, 2026). It is not a complete mirror of the MA-FPPO benchmarks. Obtain Spread, local-observation MA-MuJoCo, SMAC, OMIGA, and SMACv2 from the sources and conversion instructions below. Dataset file layouts and observation adapters must match the selected recipe.

Set `--data-root` to a directory with the following subdirectories. Dataset names are case sensitive.

```text
data/
  mpe/<simple_spread|simple_tag|simple_world>/<quality>/seed_0_data/...
  mamujoco/<2ant|4ant|2halfcheetah>/<Good|Medium|Poor>/...
  omiga/<3hopper|6halfcheetah|2ant>/<Expert|Medium|Medium-Replay|Medium-Expert>/...
  smac/<3m|2s3z|5m_vs_6m|8m>/<Good|Medium|Poor>/...
  smacv2/<terran_5_vs_5|zerg_5_vs_5|terran_10_vs_10>/<quality>/...
```

**MPE.** Obtain the original data and pretrained prey policies through [OMAR](https://github.com/ling-pan/OMAR), or the public data links documented by [MADiff](https://github.com/zbzhu99/madiff). Each quality directory contains `seed_0_data` through `seed_4_data`; each holds `obs_i.npy`, `next_obs_i.npy`, `acs_i.npy`, `rews_i.npy`, and `dones_i.npy` for controlled agents `i=0,1,2`. Tag and World also require the upstream `pretrained_adv_model.pt` at `mpe/<task>/pretrained_adv_model.pt`. These are fixed benchmark opponents, not learned submission policies. Qualities are `expert`, `medium`, and `random`; Spread additionally has `medium-replay`.

**Local-observation MA-MuJoCo and SMAC.** Use the converted arrays described by [MADiff](https://github.com/zbzhu99/madiff), originating from [OG-MARL](https://github.com/instadeepai/og-marl). Their transformation script is `scripts/transform_og_marl_dataset.py`. MA-MuJoCo requires `obs.npy`, `actions.npy`, `rewards.npy`, `discounts.npy`, and `path_lengths.npy`. SMAC requires `obs.npy`, `actions.npy`, `rewards.npy`, `legals.npy`, `states.npy`, `path_lengths.npy`, and `discounts.npy`. SMAC observations have leading one-hot IDs in the stored arrays; the supplied loader moves these IDs to the end. The MA-MuJoCo adapters use exactly their stored local observation layouts and action partitions. These are different from the full-observation OMIGA adapter even when a task is called `2ant` in both.

**OMIGA.** Download the published vault archives for `3hopper`, `6halfcheetah`, and `2ant` from the [OG-MARL dataset repository](https://huggingface.co/datasets/InstaDeepAI/og-marl/tree/main/prior_work/omiga/mamujoco). The release includes the actual conversion procedure, which excludes verified reset delimiter records and preserves real episode boundaries:

```bash
python -m pip install tensorstore==0.1.45
python tools/convert_omiga_vault.py --source-root /path/to/downloaded-vault-zips \
  --output-root /path/to/data/omiga
```

The output contains `obs`, `actions`, `rewards`, `terminals`, `truncations`, and `path_lengths` arrays plus `audit.json`. Each per-agent observation is the global observation followed by an agent ID and standardized as in the benchmark. The data card describes historical collection with MuJoCo 2.0; the experiments use the legacy Gym v2 tasks with MuJoCo 2.1. These simulator binaries are not asserted to be identical.

**SMACv2.** Obtain the Replay/Random vaults from [OG-MARL](https://huggingface.co/datasets/InstaDeepAI/og-marl). The exact converter used for these experiments is included. Use a separate Python 3.10 environment for the Flashbax/JAX reader, whose dependency versions differ from the training environment:

```bash
python3.10 -m venv .venv-conversion
source .venv-conversion/bin/activate
python -m pip install -r requirements-conversion.txt
CUDA_VISIBLE_DEVICES= JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python tools/convert_smacv2_vault.py \
  --vault-dir /path/to/terran_5_vs_5.vlt --vault-uid Replay \
  --scenario terran_5_vs_5 --output-dir /path/to/data/smacv2/terran_5_vs_5/Replay \
  --drop-incomplete-tail
```

Use the corresponding scenario and `--vault-uid Random` for the other supported case. Return to the training environment after conversion. Required fields are `obs`, `actions`, `rewards`, `legals`, `states`, and `path_lengths`; IDs are already appended to observations. Do not append a second set of IDs. The Good datasets are the paper's generated datasets: collect four times the number of Replay episodes using a fixed 15M online checkpoint, then select the top quarter by win and return. They are not official OG-MARL Good datasets.

The original collection and selection code is in `tools/smacv2_good/`. First produce the Replay policy at 15M steps by using a copy of the Replay JSON with `online.total_env_steps=15000000`. Generate and run a 32-episode collector preflight:

```bash
python tools/smacv2_good/make_config.py --task terran_5_vs_5 \
  --replay-dir /path/to/data/smacv2/terran_5_vs_5/Replay \
  --pretrained-checkpoint /path/to/pretrain/checkpoint_final.pt \
  --online-checkpoint /path/to/online/checkpoint_final.pt \
  --preflight --output-dir /path/to/collection-preflight --config-out /path/to/preflight.json
SC2PATH=/path/to/StarCraftII python tools/smacv2_good/collect.py \
  --config /path/to/preflight.json --gpu 0
```

Then run `make_config.py` without `--preflight`, with a fresh output directory and `--preflight-report /path/to/collection-preflight/selection_report.json`. Run `collect.py` with that configuration. It exports the selected `Good/` arrays under the collection output. Place that directory at `data/smacv2/<task>/Good/`. Repeat for Zerg. The collector records hashes, validates full episodes, and does not train the supplied policy. Exact recreation of an archived generated dataset requires its original collector checkpoint; these weights are not included. A newly trained checkpoint produces a new realization of the same collection protocol.

No large dataset is downloaded automatically by training or testing commands. Dataset access conditions and simulator installation remain those of the public upstream projects.
