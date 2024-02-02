import sys

sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import sampling as targetCode

@ddt
class TestCreatePointingPomdpSpace(unittest.TestCase):
        
    @data(([1, 2, 3], [(1, 2), (3, 4)], [2, 4], ([(2, 1), (4, 1), (2, 2), (4, 2), (2, 3), (4, 3)], [(1, 2), (3, 4), 'Silence'], [1, 2, 3])))
    @unpack
    def testCreatePointingPomdpSpace(self, stateSpace,  utteranceSpace, goalSpace, expectedResult):
        calculatedResult=targetCode.createPointingPomdpSpace(stateSpace, utteranceSpace, goalSpace)
        self.assertListEqual(calculatedResult[0], expectedResult[0])
        self.assertListEqual(calculatedResult[1], expectedResult[1])
        self.assertListEqual(calculatedResult[2], expectedResult[2])
        
    def tearDown(self):
        pass


if __name__ == '__main__':
	unittest.main(verbosity=2)
