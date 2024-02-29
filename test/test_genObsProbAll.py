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
             [7,9,3]]],dtype=np.float32),np.array([1,1,2]))

b=np.full((3,3), np.nan)
b[0,:]=np.array([1,2,1],dtype=np.float32)

c=np.array([[1,2,1],
             [2,2,2],
             [7,9,3]],dtype=np.float32)
c[2,:]=np.nan

@ddt
class TestPomdpObservationFunction(unittest.TestCase):

    def setUp(self):
        self.b=np.full((3,3), np.nan)
        self.b[0,:]=np.array([1,2,1],dtype=np.float32)

    @data((b,a,1,1),
          (c,a,1,1/2))
    @unpack
    def testPomdpObservationFunctionTrue(self,patterns,dataset,state,expectedResult):
        calculatedResult = genObservationProbAll(patterns,dataset,state)
        print(calculatedResult)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)












