import numpy as np
import unittest
from ddt import ddt, data, unpack


@ddt
class TestPomdpObservationFunction(unittest.TestCase):

    @data(((np.array([[1,2,1],[2,3,1],[4,4,4]]),np.array([1])),
          np.array([0,1]),
    (np.array([[[1,2,1],[2,3,1],[3,3,3]],[[0,0,0],[2,3,1],[4,4,4]],[[0,0,0],[1,1,1],[2,2,2]]]),
     np.array([1,1,1])),1/3))
    @unpack
    def testPomdpObservationFunctionTrue(self, current_img,patterns,dataset, expectedResult):
        calculatedResult = generateObservationProb(current_img,patterns,dataset)
        print(calculatedResult)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)






