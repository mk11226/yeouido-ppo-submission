"""Shared fixed-population environment for collection and model evaluation."""
import json

from env.fixed_population import OUTPUT, LEVELS, make_population, config_path, snapshot_path
from env.yeouido_env import YeouidoEnv, traci


class FixedPopulationEnv(YeouidoEnv):
    def _ensure_snapshots(self):
        if not all(snapshot_path(level).exists() for level in LEVELS):
            raise FileNotFoundError('Prepare fixed-population snapshots before starting workers')
        self.population_manifest = json.loads((OUTPUT / 'manifest.json').read_text())
        self.population_targets = self.population_manifest['signature']['targets']

    def _cfg(self, level):
        return str(config_path(level))

    def _snapshot(self, level, t):
        return str(snapshot_path(level))

    def reset(self, seed=None, options=None):
        self.population_manager = None
        obs, info = super().reset(seed=seed, options=options)
        self.population_manager = make_population(self.population_targets[self.level],
                                                   (options or {}).get('sumo_seed', 7))
        self.population_manager.attach(traci)
        self.population_max_excess = 0
        info.update(self.population_info())
        return obs, info

    def population_info(self):
        manager = getattr(self, 'population_manager', None)
        return dict(traffic_mode='fixed_population',
                    fixed_population_target=self.population_targets[self.level],
                    fixed_population_created=manager.created if manager else 0,
                    fixed_population_pending=len(manager.pending) if manager else 0,
                    fixed_population_max_excess=getattr(self, 'population_max_excess', 0))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        manager = self.population_manager
        if not (terminated or truncated):
            manager.tick(traci)
            excess = manager.expected_population(traci) - manager.target
            self.population_max_excess = max(self.population_max_excess, excess)
            if excess > 0:
                raise RuntimeError(f'Fixed background population exceeded target by {excess}')
        info.update(self.population_info())
        return obs, reward, terminated, truncated, info
