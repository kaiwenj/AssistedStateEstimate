import sys

sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import sampling as targetCode

@ddt
class TestSampling(unittest.TestCase):

    @data((np.array([.8,.2]), 10000,
           [2000-2.81*math.sqrt((.9*.1)*10000), 2000+2.81*math.sqrt((.9*.1)*10000)]))
    @unpack
    def test_sample_initial(self, init_dist, size, expectedResult):
        calculatedResult = sum([sample_initial(init_dist) for i in range(size)])
        self.assertTrue(expectedResult[0] <= calculatedResult and calculatedResult <= expectedResult[1])
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
