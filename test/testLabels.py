import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import createPointingPomdpSpace as targetCode


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


@ddt
class TestPomdpTransitionFunction(unittest.TestCase):
    
    def setUp(self):
        self.observedIntegratedTransition=lambda receivergoal, s, sPrime: {0: {0:{0:1/3, 1:0, 2:2/3},
                                                                                1:{0:1/2, 1:1/4, 2:1/4},
                                                                                2:{0:1, 1:0, 2:0}}, 
                                                                            1: {0:{0:1/4, 1:1/4, 2:1/2},
                                                                                1:{0:1/3, 1:1/3, 2:1/3},
                                                                                2:{0:0, 1:1, 2:0}},
                                                                            2: {0:{0:1/5, 1:1/5, 2:3/5},
                                                                                1:{0:1/4, 1:3/4, 2:0},
                                                                                2:{0:0, 1:0, 2:1}}}[receivergoal][s][sPrime]
        
    @data(((1, 0), 2, (1, 0), [1, 2], 0))
    @unpack
    def testPomdpTransitionFunctionTellButNotEnd(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((1, 0), 2, (1, 1), [1, 2], 1))
    @unpack
    def testPomdpTransitionFunctionTellButEnd(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((1, 0), 2, (2, 2), [1, 2], 0))
    @unpack
    def testPomdpTransitionFunctionTellButChangeGoal(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 2), 2, (2, 1), [1, 2], 0))
    @unpack
    def testPomdpTransitionFunctionInGoalButChangeState(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 2), 2, (1, 1), [1, 2], 0))
    @unpack
    def testPomdpTransitionFunctionInGoalButChangeGoal(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)
    
    @data(((1, 2), 2, (1, 2), [1, 2], 1))
    @unpack
    def testPomdpTransitionFunctionInOtherGoal(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)
    
    @data(((2, 1), 2, (2, 2), [1, 2], 0))
    @unpack
    def testPomdpTransitionFunctionInOtherGoalButNotStay(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)
    
    @data(((2, 2), 2, (2, 2), [1, 2], 1))
    @unpack
    def testPomdpTransitionFunctionInGoalButStay(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 2), 'Silence', (2, 2), [1, 2], 1))
    @unpack
    def testPomdpTransitionFunctionInGoalButStaySilence(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 2), 'Silence', (1, 2), [1, 2], 0))
    @unpack
    def testPomdpTransitionFunctionInGoalButChangeSilence(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 1), 'Silence', (2, 1), [1, 2], 1))
    @unpack
    def testPomdpTransitionFunctionInOtherGoalSilence(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 1), 'Silence', (2, 1), [2], 3/4))
    @unpack
    def testPomdpTransitionFunctionNotInGoalSilence(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 0), 'Silence', (2, 2), [2], 3/5))
    @unpack
    def testPomdpTransitionFunctionGoToGoalSilence(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpTransitionFunction=targetCode.PomdpTransitionFunction(goalSpace, self.observedIntegratedTransition)
        calculatedResult=pomdpTransitionFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    def tearDown(self):
        pass


@ddt
class TestPomdpRewardFunction(unittest.TestCase):
    
    def setUp(self):
        self.observedIntegratedReward=lambda receivergoal, s, sPrime: {0: {0: {0:1, 1:2, 2:3},
                                                                            1: {0:4, 1:5, 2:6},
                                                                            2: {0:7, 1:8, 2:9}},
                                                                        1: {0: {0:0, 1:1, 2:10},
                                                                            1: {0:100, 1:10, 2:0},
                                                                            2: {0:10, 1:20, 2:50}},
                                                                        2: {0: {0:40, 1:20, 2:10},
                                                                            1: {0:100, 1:-50, 2:20},
                                                                            2: {0:0, 1:-100, 2:10}}}[receivergoal][s][sPrime]
        self.relevanceIterationQ=lambda receivergoal, s, u: {1: {0:{'a':10, 'b':100, 'c':-10},
                                                                1:{'a':0, 'b':20, 'c':100},
                                                                2:{'a':20, 'b':-20, 'c':0}},
                                                            2: {0:{'a':-10, 'b':-100, 'c':10},
                                                                1:{'a':0, 'b':-20, 'c':-100},
                                                                2:{'a':-20, 'b':20, 'c':0}}}[receivergoal][s][u]
        
    @data(((1, 0), 'a', (1, 0), [1, 2], 10))
    @unpack
    def testPomdpRewardFunctionTellButNotEnd(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpRewardFunction=targetCode.PomdpRewardFunction(goalSpace, self.observedIntegratedReward, self.relevanceIterationQ)
        calculatedResult=pomdpRewardFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 1), 'c', (2, 2), [2], -100))
    @unpack
    def testPomdpRewardFunctionTellButEnd(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpRewardFunction=targetCode.PomdpRewardFunction(goalSpace, self.observedIntegratedReward, self.relevanceIterationQ)
        calculatedResult=pomdpRewardFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 2), 'a', (2, 2), [1, 2], 0))
    @unpack
    def testPomdpRewardnFunctionAtGoal(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpRewardFunction=targetCode.PomdpRewardFunction(goalSpace, self.observedIntegratedReward, self.relevanceIterationQ)
        calculatedResult=pomdpRewardFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 2), 'Silence', (2, 2), [2], 0))
    @unpack
    def testPomdpRewardFunctionSilenceAtGoal(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpRewardFunction=targetCode.PomdpRewardFunction(goalSpace, self.observedIntegratedReward, self.relevanceIterationQ)
        calculatedResult=pomdpRewardFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)
    
    @data(((2, 1), 'Silence', (1, 1), [1, 2], 0))
    @unpack
    def testPomdpRewardFunctionSilenceAtGoalImpossibleTransition(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpRewardFunction=targetCode.PomdpRewardFunction(goalSpace, self.observedIntegratedReward, self.relevanceIterationQ)
        calculatedResult=pomdpRewardFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 0), 'Silence', (2, 2), [2], 10))
    @unpack
    def testPomdpRewardFunctionSilenceNotAtGoal(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpRewardFunction=targetCode.PomdpRewardFunction(goalSpace, self.observedIntegratedReward, self.relevanceIterationQ)
        calculatedResult=pomdpRewardFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((2, 0), 'Silence', (2, 0), [1, 2], 40))
    @unpack
    def testPomdpRewardFunctionSilenceNotAtGoalStay(self, s, a, sPrime, goalSpace, expectedResult):
        pomdpRewardFunction=targetCode.PomdpRewardFunction(goalSpace, self.observedIntegratedReward, self.relevanceIterationQ)
        calculatedResult=pomdpRewardFunction(s, a, sPrime)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    def tearDown(self):
        pass


@ddt
class TestPomdpObservationFunction(unittest.TestCase):
        
    @data(((1, 0), 'a', 0, 1))
    @unpack
    def testPomdpObservationFunctionTrue(self, sPrime, a, o, expectedResult):
        calculatedResult=targetCode.pomdpObservationFunction(sPrime, a, o)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    @data(((1, 2), 'a', 1, 0))
    @unpack
    def testPomdpObservationFunctionFalse(self, sPrime, a, o, expectedResult):
        calculatedResult=targetCode.pomdpObservationFunction(sPrime, a, o)
        self.assertAlmostEqual(calculatedResult, expectedResult)

    

    def tearDown(self):
        pass

 
if __name__ == '__main__':
	unittest.main(verbosity=2)
