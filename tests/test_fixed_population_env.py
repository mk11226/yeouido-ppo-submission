import unittest
from unittest.mock import Mock, patch

from env.fixed_population_env import FixedPopulationEnv


class FixedPopulationEnvironmentTests(unittest.TestCase):
    def environment(self):
        env = object.__new__(FixedPopulationEnv)
        env.level = 'night'
        env.population_targets = {'night': 462}
        env.population_max_excess = 0
        env.population_manager = Mock(target=462, created=3, pending={})
        env.population_manager.expected_population.return_value = 462
        return env

    def test_replenishment_preserves_policy_transition(self):
        env = self.environment()
        action, obs = object(), object()
        with patch('env.yeouido_env.YeouidoEnv.step', return_value=(obs, 2., False, False, {})) as step:
            result = env.step(action)
        step.assert_called_once_with(action)
        env.population_manager.tick.assert_called_once()
        self.assertIs(result[0], obs)
        self.assertEqual(result[1:4], (2., False, False))
        self.assertEqual(result[4]['fixed_population_target'], 462)

    def test_terminal_transition_does_not_add_more_traffic(self):
        env = self.environment()
        with patch('env.yeouido_env.YeouidoEnv.step', return_value=(None, 0., True, False, {})):
            env.step([0, 0])
        env.population_manager.tick.assert_not_called()

    def test_population_excess_fails_instead_of_changing_policy_action(self):
        env = self.environment()
        env.population_manager.expected_population.return_value = 463
        with patch('env.yeouido_env.YeouidoEnv.step', return_value=(None, 0., False, False, {})):
            with self.assertRaisesRegex(RuntimeError, 'exceeded target'):
                env.step([0, 0])


if __name__ == '__main__':
    unittest.main()
