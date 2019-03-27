import unittest

import numpy as np

from drlhp.drlhp_utils import LimitedRunningStat

class TestLimitedRunningStat(unittest.TestCase):
    def test(self):
        s = LimitedRunningStat(len=3)
        s.push(2)
        expected = [2]
        self.assertEqual(s.mean, np.mean(expected))
        self.assertEqual(s.var, np.var(expected))
        self.assertEqual(s.std, np.std(expected))
        self.assertEqual(s.n, len(expected))
        s.push(4)
        expected = [2, 4]
        self.assertEqual(s.mean, np.mean(expected))
        self.assertEqual(s.var, np.var(expected))
        self.assertEqual(s.std, np.std(expected))
        self.assertEqual(s.n, len(expected))
        s.push(9)
        expected = [2, 4, 9]
        self.assertEqual(s.mean, np.mean(expected))
        self.assertEqual(s.var, np.var(expected))
        self.assertEqual(s.std, np.std(expected))
        self.assertEqual(s.n, len(expected))
        s.push(5)
        expected = [5, 4, 9]
        self.assertEqual(s.mean, np.mean(expected))
        self.assertEqual(s.var, np.var(expected))
        self.assertEqual(s.std, np.std(expected))
        self.assertEqual(s.n, len(expected))


if __name__ == '__main__':
    unittest.main()
