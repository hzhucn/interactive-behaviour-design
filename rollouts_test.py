import unittest

from rollouts import RolloutsByHash


class TestRollouts(unittest.TestCase):
    def test_dict_no_max_len(self):
        d = RolloutsByHash(maxlen=None)
        d[1] = '1'
        self.assertEqual(len(d), 1)
        self.assertEqual(list(d.keys()), [1])
        self.assertEqual(list(d.values()), ['1'])
        d[2] = '2'
        self.assertEqual(len(d), 2)
        self.assertEqual(list(d.keys()), [1, 2])
        self.assertEqual(list(d.values()), ['1', '2'])
        d[3] = '3'
        self.assertEqual(len(d), 3)
        self.assertEqual(list(d.keys()), [1, 2, 3])
        self.assertEqual(list(d.values()), ['1', '2', '3'])

    def test_dict_max_len(self):
        d = RolloutsByHash(maxlen=2)
        d[1] = '1'
        self.assertEqual(len(d), 1)
        self.assertEqual(list(d.keys()), [1])
        self.assertEqual(list(d.values()), ['1'])
        d[2] = '2'
        self.assertEqual(len(d), 2)
        self.assertEqual(list(d.keys()), [1, 2])
        self.assertEqual(list(d.values()), ['1', '2'])
        d[3] = '3'
        self.assertEqual(len(d), 2)
        self.assertEqual(list(d.keys()), [2, 3])
        self.assertEqual(list(d.values()), ['2', '3'])


if __name__ == '__main__':
    unittest.main()
