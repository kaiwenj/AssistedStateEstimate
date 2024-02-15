import numpy as np
import unittest
from ddt import ddt, data, unpack


@ddt
class TestKLDivergence(unittest.TestCase):
    @data((np.array([.8, .2]), np.array([[.5, .5],[.3, .7],[.2, .8],[.8, .2],[.75, .25]]), range(5),3),
          (np.array([0.07756876, 0.06144809, 0.27270788, 0.39929912, 0.18897616]),
           np.array([[0.08074545, 0.39513818, 0.00243568, 0.12340891, 0.39827177],
                     [0.20359456, 0.17993053, 0.17661841, 0.23769238, 0.20216412],
                     [0.29404398, 0.2590098 , 0.24255292, 0.1482787 , 0.0561146 ],
                     [0.33724191, 0.32925206, 0.11140309, 0.01848469, 0.20361825],
                     [0.3021671 , 0.13648671, 0.12395892, 0.26563366, 0.17175361],
                     [0.11366903, 0.16063681, 0.17047124, 0.28559803, 0.26962488],
                     [0.07756876, 0.06144809, 0.27270788, 0.39929912, 0.18897616],
                     [0.25313291, 0.0392906 , 0.15406403, 0.40112362, 0.15238885],
                     [0.24673666, 0.01292078, 0.33049194, 0.05556696, 0.35428365],
                     [0.2296438 , 0.16883889, 0.29080619, 0.1584568 , 0.15225432]]),
           range(10),6))

    @unpack
    def test_KLDiv_obs(self, trueDist, condDist, obsSpace, expectedResult):
        calculatedResult = sythesizeObs(trueDist, condDist, obsSpace)
        print("Generated Obs: ", calculatedResult)
        self.assertTrue(expectedResult == calculatedResult)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
