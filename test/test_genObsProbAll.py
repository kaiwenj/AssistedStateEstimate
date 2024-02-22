import numpy as np
import unittest
from ddt import ddt, data, unpack

a=(np.array([[[1,2,1],
             [2,2,2],
             [7,9,3]],
            [[1,2,1],
            [4,4,4],
            [7,9,3]],
            [[7,7,7],
             [2,2,2],
             [7,9,3]]]),np.array([1,2,3]))


@ddt
class TestPomdpObservationFunction(unittest.TestCase):

    @data((np.array([[1,2,1]]),np.array([0]),
           a,np.array([1.,1.,0.])))
    @unpack
    def testPomdpObservationFunctionTrue(self,patterns,rowIndex,dataset,expectedResult):
        calculatedResult = genObservationProbAll(patterns,rowIndex,dataset)
        print(calculatedResult)
        np.array_equal(calculatedResult, expectedResult)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)







