"""Offline regression checks for the paper's frozen Bayes configuration."""

import unittest

import torch

from terrain_selector import TerrainSelector


class TerrainSelectorBayesTests(unittest.TestCase):
    def test_matches_paper_posteriors_before_and_after_reset(self):
        selector = TerrainSelector.__new__(TerrainSelector)
        selector.class_ids = ["pit", "stairs", "gap", "random_uniform"]
        selector.stable_stay = 0.9
        # Captured from Legged_Gym_EX's BayesianTerrainFilter with the paper's
        # uniform prior, identity observation, evidence_power=1, and no adaptation.
        # Extreme logits expose probability floors and observation-matrix mixing.
        cases = [
            ([2, 0, -1, 1], "pit",
             [0.6439142823219299, 0.08714433014392853,
              0.03205861523747444, 0.23688282072544098]),
            ([0, 1, 2, -1], "pit",
             [0.4145210385322571, 0.20740890502929688,
              0.3165375292301178, 0.06153252720832825]),
            ([-1000, -1000, 1000, -1000], "gap",
             [2.5520209945284478e-08, 1.385189207780968e-08,
              1.0, 5.633483457501143e-09]),
            ([0, 0, 0, 0], "gap",
             [0.03333335742354393, 0.03333334997296333,
              0.8999999761581421, 0.03333333879709244]),
            ([1000, -1000, -1000, -1000], "pit",
             [0.9999997615814209, 1.9999992773023223e-08,
              2.6142839715248556e-07, 1.9999990996666384e-08]),
            ([1, 1, 1, 1], "pit",
             [0.899999737739563, 0.03333335369825363,
              0.033333562314510345, 0.03333335369825363]),
        ]
        for replay in range(2):
            selector.reset()
            for step, (logits, label, posterior) in enumerate(cases):
                with self.subTest(replay=replay, step=step):
                    index = selector._bayes(torch.tensor(logits, dtype=torch.float32))
                    self.assertEqual(selector.class_ids[index], label)
                    torch.testing.assert_close(
                        selector.belief, torch.tensor(posterior, dtype=torch.float32),
                        rtol=1e-6, atol=1e-12,
                    )


if __name__ == "__main__":
    unittest.main()
