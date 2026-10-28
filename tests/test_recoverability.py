"""Pure model/array fixtures; no optimization, Gymnasium interaction, or training."""

import unittest
import numpy as np

from src.recoverability import pack, unpack, shooting_problem, symbolic_dynamics, candidate_diagnostics
from src.safety import safety_margin


class RecoverabilityTests(unittest.TestCase):
    def test_transcription_dimensions_initial_state_and_indexing(self):
        horizon = 200
        nlp, lower, upper, _ = shooting_problem(horizon)
        self.assertEqual(nlp['x'].numel(), 600)
        self.assertEqual(nlp['g'].numel(), 400)
        self.assertEqual(nlp['p'].numel(), 2)
        states = np.arange(402, dtype=float).reshape(201, 2)
        actions = np.arange(200, dtype=float)
        packed = pack(states, actions)
        restored, torques = unpack(packed, states[0], horizon)
        np.testing.assert_array_equal(restored, states)
        np.testing.assert_array_equal(torques, actions)
        np.testing.assert_array_equal(lower[:4], [-1., -3., -1., -3.])
        np.testing.assert_array_equal(upper[400:], np.full(200, 2.))

    def test_dynamics_constraints_and_stage_cost_use_correct_time_indices(self):
        step = symbolic_dynamics()
        states = [np.array([.1, -.2])]
        torques = np.array([.2, -.3, .4, -.1])
        for torque in torques:
            states.append(np.array(step(states[-1], torque)).reshape(2))
        states = np.array(states)
        vector = pack(states, torques)
        _, _, _, check = shooting_problem(4)
        cost, residual = check(vector, states[0])
        np.testing.assert_allclose(np.array(residual), 0, rtol=0, atol=1e-15)
        expected = np.sum(states[:-1, 0]**2 + .1*states[:-1, 1]**2 + .001*torques**2)
        self.assertAlmostEqual(float(cost), expected, places=14)
        changed = vector.copy(); changed[6] += .2  # theta at terminal x4
        changed_cost, changed_residual = check(changed, states[0])
        self.assertEqual(float(changed_cost), float(cost))  # no terminal cost
        np.testing.assert_allclose(np.array(changed_residual).reshape(-1), [0,0,0,0,0,0,.2,0], atol=1e-15)
        _, wrong_initial = check(vector, states[0] + [.1, 0.])
        self.assertGreater(np.max(np.abs(np.array(wrong_initial))), .01)

    def test_smooth_and_clipped_models_are_not_confused(self):
        smooth, clipped = symbolic_dynamics(), symbolic_dynamics(True)
        for state in [[-1., -3.], [0., 0.], [1., 3.]]:
            for torque in [-2., 0., 2.]:
                np.testing.assert_array_equal(np.array(smooth(state, torque)), np.array(clipped(state, torque)))
        self.assertGreater(float(smooth([np.pi/2, 8.], 2.)[1]), 8.)
        self.assertEqual(float(clipped([np.pi/2, 8.], 2.)[1]), 8.)

    def test_solver_feasible_array_is_not_an_execution_witness(self):
        states = np.zeros((201, 2)); torques = np.zeros(200)
        diagnostics = candidate_diagnostics(pack(states, torques), states[0])
        self.assertEqual(diagnostics['maximum_dynamics_residual'], 0.)
        self.assertNotIn('witness', diagnostics)
        self.assertLess(safety_margin(2*np.pi, 0.), 0.)


if __name__ == '__main__':
    unittest.main()
