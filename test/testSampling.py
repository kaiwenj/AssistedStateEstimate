import sys

sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import sampling as targetCode

@ddt
class TestSampling(unittest.TestCase):

    @data((np.array([.8, .2]), 10000, 1000,
           [0.2 - 1.96 * math.sqrt((.8 * .2) / 1000), 0.2 + 1.96 * math.sqrt((.8 * .2) / 1000)], .95),
          (np.array([.6, .4]), 10000, 1000,
           [0.4 - .38 * math.sqrt((.6 * .4) / 1000), 0.4 + .38 * math.sqrt((.6 * .4) / 1000)], .3)
          )
    @unpack
    def testSampleInitial(self, initDist, samples, size, confidenceInterval, expectedResult):
        proportions = np.array([np.mean([sampleDistribution(initDist) for _ in range(size)]) for _ in range(samples)])
        calculatedResult = np.mean(np.array(
            [confidenceInterval[0] <= proportions[i] and proportions[i] <= confidenceInterval[1] for i in
             range(samples)]))
        print("test samp: ", calculatedResult)
        self.assertTrue(expectedResult <= calculatedResult + 0.05 and expectedResult >= calculatedResult - 0.05)
    
    
    @data((np.array([[.8, .2], [.6, .4]]), 10000, 1000,
           [0.2 - 1.96 * math.sqrt((.8 * .2) / 1000), 0.2 + 1.96 * math.sqrt((.8 * .2) / 1000)], .95)
          )
    @unpack
    def testSampleConditional(self, Distribution, samples, size, confidenceInterval, expectedResult):
        proportions = np.array([np.mean([sampleConditional(Distribution, 0) for _ in range(size)]) for _ in range(samples)])
        calculatedResult = np.mean(np.array(
            [confidenceInterval[0] <= proportions[i] and proportions[i] <= confidenceInterval[1] for i in
             range(samples)]))
        print("test cond: ", calculatedResult)
        self.assertTrue(expectedResult <= calculatedResult + 0.05 and expectedResult >= calculatedResult - 0.05)
    
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
