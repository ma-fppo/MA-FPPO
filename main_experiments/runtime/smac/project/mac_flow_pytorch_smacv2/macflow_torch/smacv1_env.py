"""Real, fixed-map SMAC v1; no mock fallback and no SMACv2 wrapper."""
import os


def make_smacv1_env(task, seed=0):
    if task not in ('3m', '2s3z', '5m_vs_6m', '8m'):
        raise ValueError('This experiment supports 3m, 2s3z, 5m_vs_6m and 8m')
    os.environ.setdefault('SC2PATH', str(Path.home() / 'StarCraftII'))
    from smac.env import StarCraft2Env
    env = StarCraft2Env(map_name=task, seed=int(seed), obs_last_action=False)
    env._ma_fppo_environment_kind = 'smac_v1'
    return env
