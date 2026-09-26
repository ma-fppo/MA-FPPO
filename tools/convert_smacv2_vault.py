#!/usr/bin/env python3
"""Convert an OG-MARL SMACv2 Flashbax Vault into flat NumPy trajectories."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


SCENARIO_AGENT_COUNTS = {
    "terran_5_vs_5": 5,
    "zerg_5_vs_5": 5,
    "terran_10_vs_10": 10,
}


def _get_field(tree: Any, path: Sequence[str]) -> Any:
    current = tree
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                raise ValueError("missing required Vault field: {}".format(".".join(path)))
            current = current[key]
        else:
            if not hasattr(current, key):
                raise ValueError("missing required Vault field: {}".format(".".join(path)))
            current = getattr(current, key)
    return np.asarray(current)


def _require_shape(name: str, value: np.ndarray, expected_prefix: tuple) -> None:
    if value.shape[: len(expected_prefix)] != expected_prefix:
        raise ValueError(
            "{} has shape {}; expected prefix {}".format(name, value.shape, expected_prefix)
        )


def validate_experience(experience: Any, n_agents: int) -> Dict[str, np.ndarray]:
    """Return required Vault arrays after validating their shared time axes."""
    observations = _get_field(experience, ("observations",))
    if observations.ndim != 4:
        raise ValueError("observations must have shape [B, T, N, O]")
    batch, steps, agents, _ = observations.shape
    if agents != n_agents:
        raise ValueError("agent-count mismatch: expected {}, found {}".format(n_agents, agents))

    expected_time_agent = (batch, steps, agents)
    actions = _get_field(experience, ("actions",))
    rewards = _get_field(experience, ("rewards",))
    legals = _get_field(experience, ("infos", "legals"))
    states = _get_field(experience, ("infos", "state"))
    terminals = _get_field(experience, ("terminals",))
    truncations = _get_field(experience, ("truncations",))

    for name, value in (
        ("actions", actions),
        ("rewards", rewards),
        ("terminals", terminals),
        ("truncations", truncations),
    ):
        if value.ndim != 3:
            raise ValueError("{} must have shape [B, T, N]".format(name))
        _require_shape(name, value, expected_time_agent)
    if legals.ndim != 4:
        raise ValueError("infos.legals must have shape [B, T, N, A]")
    _require_shape("infos.legals", legals, expected_time_agent)
    if states.ndim != 3:
        raise ValueError("infos.state must have shape [B, T, S]")
    _require_shape("infos.state", states, (batch, steps))

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "legals": legals,
        "states": states,
        "terminals": terminals.astype(bool, copy=False),
        "truncations": truncations.astype(bool, copy=False),
    }


def episode_lengths(boundaries: np.ndarray, drop_incomplete_tail: bool = False) -> np.ndarray:
    """Turn an episode-end mask into positive lengths, optionally dropping its final tail."""
    if boundaries.ndim != 1:
        raise ValueError("episode boundaries must be one-dimensional")
    if len(boundaries) == 0:
        raise ValueError("unterminated final trajectory in Vault experience")

    ends = np.flatnonzero(boundaries)
    if len(ends) == 0:
        raise ValueError("unterminated final trajectory in Vault experience")
    if not bool(boundaries[-1]) and not drop_incomplete_tail:
        raise ValueError("unterminated final trajectory in Vault experience")

    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts + 1
    if np.any(lengths <= 0):
        raise ValueError("non-positive episode length")
    return lengths.astype(np.int64, copy=False)


def convert_experience(
    experience: Any, n_agents: int, drop_incomplete_tail: bool = False
) -> Dict[str, np.ndarray]:
    """Convert one Vault experience tree to the concatenated episode layout."""
    arrays = validate_experience(experience, n_agents=n_agents)
    batch, steps, agents, obs_dim = arrays["observations"].shape
    source_steps = batch * steps
    terminals = arrays["terminals"].reshape(source_steps, agents)
    truncations = arrays["truncations"].reshape(source_steps, agents)
    path_lengths = episode_lengths(
        np.any(terminals | truncations, axis=1),
        drop_incomplete_tail=drop_incomplete_tail,
    )
    retained_steps = int(path_lengths.sum())
    if retained_steps <= 0 or retained_steps > source_steps:
        raise ValueError("invalid retained trajectory length")

    observations = arrays["observations"].reshape(source_steps, agents, obs_dim)[:retained_steps]
    agent_ids = np.broadcast_to(
        np.eye(agents, dtype=np.float32), (retained_steps, agents, agents)
    )
    converted = {
        "obs": np.concatenate((observations.astype(np.float32, copy=False), agent_ids), axis=-1),
        "actions": arrays["actions"].reshape(source_steps, agents)[:retained_steps],
        "rewards": arrays["rewards"].reshape(source_steps, agents)[:retained_steps].astype(
            np.float32, copy=False
        ),
        "legals": arrays["legals"].reshape(
            source_steps, agents, arrays["legals"].shape[-1]
        )[:retained_steps].astype(np.float32, copy=False),
        "states": arrays["states"].reshape(source_steps, arrays["states"].shape[-1])[:retained_steps].astype(
            np.float32, copy=False
        ),
        "path_lengths": path_lengths,
    }
    if int(path_lengths.sum()) != len(converted["obs"]):
        raise ValueError("episode lengths do not cover every retained Vault timestep")
    return converted


def save_converted_dataset(
    converted: Dict[str, np.ndarray],
    output_dir: Path,
    overwrite: bool = False,
    metadata: Dict[str, Any] = None,
) -> None:
    """Persist conversion output without silently overwriting a previous conversion."""
    output_dir = Path(output_dir)
    expected = tuple("{}.npy".format(key) for key in converted)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError("output directory is non-empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in converted.items():
        np.save(output_dir / "{}.npy".format(key), value)
    if metadata is not None:
        with (output_dir / "conversion_metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
    missing = [name for name in expected if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("failed to write converted files: {}".format(", ".join(missing)))


def load_vault_experience(vault_dir: Path, vault_uid: str) -> Any:
    """Load a Flashbax Vault lazily so unit tests do not require Flashbax."""
    try:
        import jax
        from flashbax.vault import Vault
    except ImportError as exc:
        raise RuntimeError(
            "Flashbax and JAX are required only for Vault reading; install requirements-conversion.txt in a separate environment"
        ) from exc

    vault = Vault(str(vault_dir), vault_uid=vault_uid)
    experience = vault.read().experience
    return jax.tree_util.tree_map(np.asarray, experience)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-dir", type=Path, required=True)
    parser.add_argument("--vault-uid", default="Replay")
    parser.add_argument("--scenario", choices=sorted(SCENARIO_AGENT_COUNTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--drop-incomplete-tail",
        action="store_true",
        help="discard only the final timesteps after the last terminal/truncation boundary",
    )
    args = parser.parse_args()

    experience = load_vault_experience(args.vault_dir, args.vault_uid)
    source_steps = int(np.asarray(_get_field(experience, ("observations",))).shape[0])
    source_steps *= int(np.asarray(_get_field(experience, ("observations",))).shape[1])
    converted = convert_experience(
        experience,
        SCENARIO_AGENT_COUNTS[args.scenario],
        drop_incomplete_tail=args.drop_incomplete_tail,
    )
    retained_steps = int(len(converted["obs"]))
    metadata = {
        "scenario": args.scenario,
        "vault_dir": str(args.vault_dir),
        "vault_uid": args.vault_uid,
        "source_steps": source_steps,
        "retained_steps": retained_steps,
        "dropped_incomplete_tail_steps": source_steps - retained_steps,
        "drop_incomplete_tail_requested": bool(args.drop_incomplete_tail),
    }
    save_converted_dataset(
        converted, args.output_dir, overwrite=args.overwrite, metadata=metadata
    )
    shapes = {name: list(value.shape) for name, value in converted.items()}
    print("metadata={}".format(metadata))
    print("episodes={}".format(len(converted["path_lengths"])))
    print("shapes={}".format(shapes))


if __name__ == "__main__":
    main()
