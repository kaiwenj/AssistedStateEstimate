
import numpy as np



class ObtainOneTrainingPoint(object):

    def __init__(self, sampleObservations, provideAssistObservation, policy, transition):
        self.sampleObservations = sampleObservations
        self.provideAssistObservation = provideAssistObservation
        self.policy=policy
        self.transition=transition

    def __call__(self, s):
        oTrue=self.sampleObservation(s)
        oAssist=self.provideAssistObservation(oTrue)
        a=self.policy(oAssist)
        s=self.transition(s, a)
        return (s, oAssist, a)


class ObtainOneTrainingTrajectory(object):
        
    def __init__(self, sampleInitialState, obtainOneTrainingPoint, T):
        self.sampleInitialState=sampleInitialState
        self.obtainOneTrainingTrajectory=obtainOneTrainingPoint
        self.T = T

    def __call__(self, pInit):
        s = self.sampleInitialState(pInit)
        oList=[]
        aList=[]
        for t in range(self.T):
            s, oAssist, a = self.obtainOneTrainingTrajectory(s)
            oList.append(oAssist)
            aList.append(a)
        return oList, aList

















































