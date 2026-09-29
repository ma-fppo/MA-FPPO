# MA-FPPO: Multi-Agent Flow-Pretrained Policy Optimization

Guowei Zou, Haonan Chen, Haitao Wang, Beiwen Zhang, Na Yan, Hejun Wu.

Sun Yat-sen University · National University of Singapore

[Project page](https://ma-fppo.github.io/) · [Paper (arXiv:2609.32594)](https://arxiv.org/abs/2609.32594) · [PDF](https://arxiv.org/pdf/2609.32594)

This source release contains MA-FPPO's offline pretraining and online fine-tuning implementation, with 49 main-experiment recipes.

- `main_experiments/runtime/`: MPE, MA-MuJoCo with local observations, OMIGA MuJoCo with full observations, SMAC, and SMACv2. The three local-observation MuJoCo task adapters are kept separate because their observation and action layouts differ.
- `main_experiments/configs/`: paper task and dataset-quality settings.
- `tools/`: data validation, OMIGA vault conversion, and the SMACv2 Good-data collector.
- `tests/`: tests that do not require benchmark datasets or simulators.

No datasets, trained checkpoints, logs, Git history, machine-specific launch scripts, or simulator binaries are included. See [DATA.md](DATA.md) for benchmark sources and array formats, and [NOTICE.md](NOTICE.md) for upstream licenses.

## Installation

Benchmark training uses Linux, Python 3.8.20, an NVIDIA CUDA GPU, legacy Gym, and the original simulator interfaces. Use a new environment. The core dependency versions and SMAC/SMACv2 Git commits were verified against the installed experiment environment; this release has not been installed and retrained end to end on a clean GPU machine.

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install pip==23.0.1 setuptools==65.5.0 wheel==0.38.4
python -m pip install -r requirements.txt
python -m pip install -r requirements-environments.txt
```

The requirements select the experiment's PyTorch 1.12.1 CUDA 11.3 build. A compatible NVIDIA driver is required. For MuJoCo, install **MuJoCo 2.1.0** and configure the dynamic library path before starting Python:

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
```

Install StarCraft II and the SMAC/SMACv2 maps using the [SMAC](https://github.com/oxwhirl/smac) and [SMACv2](https://github.com/oxwhirl/smacv2) instructions. Pass `--sc2path` for each run; using separate SC2 installations for the two benchmark families avoids map/version ambiguity. SMAC and SMACv2 are pinned to the installed experiment Git revisions in `requirements-environments.txt`. Both experiment installations use StarCraft II build `Base75689`. Do not substitute the current Gymnasium MuJoCo tasks for the supplied Gym v2 adapters when comparing with the paper.

For CPU tests only, an isolated modern Python/PyTorch environment can use `requirements-test.txt`; neither Gym nor SC2 is required:

```bash
python -m pip install -r requirements-test.txt
python tests/run_checks.py
```

## Main experiments

Run commands from the extracted package root. The launcher expands `${PACKAGE_ROOT}`, `${DATA_ROOT}`, and `${OUTPUT_ROOT}` and writes the resolved JSON next to your outputs. `--dry-run` only displays commands. Training defaults to three independent seeds, `0 1 2`, executed sequentially. Each seed has its own pretraining, online training, resolved configurations, and outputs under `<output-root>/seed_<seed>/`. Use `--training-seeds 0 1 2` to specify seeds or `--training-seeds 0` for one run. Evaluation uses `--training-seed 0` to select the training configuration and `--seed` for the separate evaluation seed. These defaults enable new three-seed runs; they do not change the paper’s existing reported results.

```bash
# Inspect a complete offline-to-online recipe.
python run.py main --config main_experiments/configs/mpe__simple_spread__medium.json \
  --data-root /path/to/data --output-root /path/to/runs --dry-run

# Run offline pretraining and fresh online fine-tuning, in sequence.
CUDA_VISIBLE_DEVICES=0 python run.py main \
  --config main_experiments/configs/mpe__simple_spread__medium.json \
  --data-root /path/to/data --output-root /path/to/runs --phase all

# Discrete runs first validate the data against the real environment.
CUDA_VISIBLE_DEVICES=0 python run.py main \
  --config main_experiments/configs/smac__5m_vs_6m__Medium.json \
  --data-root /path/to/data --output-root /path/to/runs \
  --sc2path /path/to/StarCraftII --phase all
```

Use `--phase pretrain` or `--phase online` to execute a single stage. For SMAC/SMACv2, run `--phase audit` before the first pretraining run. The discrete `--phase all` command does this automatically. Main online fine-tuning reads the pretrained checkpoint from the corresponding output directory and starts new actor/critic optimizers. Existing nonempty training outputs are not silently restarted.

Most recipes use one million offline updates and 50 million joint environment steps online. The supplied SMAC 3m/2s3z recipes retain their 15 million-step online budget. Read the selected JSON for exact values. Environment counts and minibatches affect the optimization schedule; reducing them changes the experiment. Resource-only settings exposed by this release are `MA_FPPO_MIN_FREE_RAM_GIB` (default 2) and `MA_FPPO_GPU_MEMORY_FRACTION` (default .95).

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python run.py main \
  --config main_experiments/configs/mpe__simple_spread__medium.json \
  --data-root /path/to/data --output-root /path/to/runs --phase evaluate \
  --checkpoint /path/to/runs/seed_0/main/mpe__simple_spread__medium/ppo/checkpoint_50000000.pt \
  --seed 10000 --episodes 20

```

Repeat with evaluation seeds `80000` and `90000`. Use `python tools/select_checkpoints.py /path/to/checkpoint-directory` to list the selected curve checkpoints (load only trusted local checkpoints). Learning curves use 20 existing checkpoints, selected approximately uniformly by actual training steps and including both endpoints. The reported bands are the sample standard deviation of the three evaluation-seed means for the same trained model. They do not measure variation across independent training seeds. The standalone evaluator accepts `--episodes 200` for table endpoints whose protocol uses 200 episodes; follow the paper's experimental setup for exceptions. The release's in-training endpoint diagnostics default to 20 episodes.

## Verification and scope

The release is assembled from preserved experiment sources and recorded configurations. Packaging changes are limited to portable paths/entrypoints, configurable host resource reserves, a data audit that handles every supplied discrete recipe, and a default launcher for three independent training seeds. The Spread Medium-Replay sampler uses its recorded shorter-data adaptation. Model equations, optimization objectives, and the supplied training budgets are retained.

The included checks cover syntax, all recipe paths, finite pretraining updates in seven runtimes, Gaussian mean inheritance and updates, raw-action likelihoods, GAE boundaries, discrete masks, and portable evaluation checkpoint loading. Full benchmark retraining and simulator installation were not repeated for this release. Benchmark data and trained weights must be obtained or produced separately.

## Citation

```bibtex
@misc{zou2026mafppo,
  title={MA-FPPO: Multi-Agent Flow-Pretrained Policy Optimization},
  author={Zou, Guowei and Chen, Haonan and Wang, Haitao and Zhang, Beiwen and Yan, Na and Wu, Hejun},
  year={2026},
  eprint={2609.32594},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2609.32594}
}
```
