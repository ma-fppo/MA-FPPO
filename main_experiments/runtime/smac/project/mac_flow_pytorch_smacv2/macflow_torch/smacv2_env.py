from __future__ import annotations

import os
from typing import Dict

import numpy as np


DISTRIBUTION_CONFIGS = {
    "terran_5_vs_5": {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {"dist_type": "surrounded_and_reflect", "p": 0.5, "n_enemies": 5, "map_x": 32, "map_y": 32},
    },
    "zerg_5_vs_5": {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        },
        "start_positions": {"dist_type": "surrounded_and_reflect", "p": 0.5, "n_enemies": 5, "map_x": 32, "map_y": 32},
    },
}

MAP_NAMES = {
    "terran_5_vs_5": "10gen_terran",
    "zerg_5_vs_5": "10gen_zerg",
}


def make_smacv2_env(scenario: str, seed: int = 0):
    os.environ.setdefault(
        "SC2PATH",
        os.path.expanduser("~/StarCraftII"),
    )
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

    return StarCraftCapabilityEnvWrapper(
        capability_config=DISTRIBUTION_CONFIGS[scenario],
        map_name=MAP_NAMES[scenario],
        debug=False,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
        seed=seed,
    )


def get_legal_actions(env) -> np.ndarray:
    return np.asarray(
        [np.asarray(env.get_avail_agent_actions(i), dtype=np.float32) for i in range(env.n_agents)],
        dtype=np.float32,
    )


def extract_win(info: Dict, stats: Dict | None = None) -> float:
    for payload in (info or {}, stats or {}):
        for key in ("battle_won", "won", "win", "win_rate"):
            if key in payload:
                return float(payload[key])
    return 0.0
