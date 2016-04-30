''' This module contains functinos/classes for probability models and betting strategies. '''
import logging
import time
from multiprocessing import Pool

import pandas as pd
import scipy.stats as ss

from x8313.BettingStrategy import *

logger = logging.getLogger(__name__)

class FilteredRaces(object):
    def __init__(self):
        self.reasons = {}
        
    def add(self, reason):
        existing = self.reasons.setdefault(reason, 0)
        self.reasons[reason] = existing + 1
        
    def getSummary(self):
        return self.reasons
    
    def merge(self, filtered):
        for (key, value) in filtered.getSummary().iteritems():
            if key in self.reasons:
                self.reasons[key] += value
            else:
                self.reasons[key] = value

class runStrategyOnRace(object):
    def __init__(self, strategy, simParam):
        self.strategy = strategy
        self.simParam = simParam
    
    def __call__(self, race):
        return self.strategy.generateWagers(race, self.simParam)
    
def selectRaces(selector, races):
    '''Actually handles the selection of races. '''
    filterCounts = FilteredRaces()
    goodRaces = []
    for race in races.itervalues():
        filterReason = selector(race)
        if(filterReason is None):
            goodRaces.append(race)
        else:
            filterCounts.add(filterReason)
    return (goodRaces, filterCounts)


class Strategy(object):
    def __init__(self, name, selector, probModel, betFunction, parallel =False):
        self.name = name
        self.selector = selector
        self.probModel = probModel
        self.betFunction = betFunction
        self.parallel = parallel

    def generateProbs(self, races):
        return {key : self.probModel(race) for (key, race) in races.iteritems()}
    def getPayoutProbability(self,wagerType,race,payout,useRank=False):
        '''returns the probability for the winning combination for the probModel'''
        try:
            probSeries=getOrderedWagerProbs(race,self.probModel,wagerType)
        except:
            print("Couldnt get probSeries for", race.id, wagerType)
            return 0.0
        try:
            winComb=payout.getPayouts(wagerType)[0]
        except:
            #print("no payouts",race.id,wagerType)
            return 0.0
        if useRank:
            try:
                return probSeries.rank(ascending=False)[winComb]
            except:
                #print ("No rank for :", winComb)
                return 9999
        else:
            try:
                return probSeries[winComb]
            except:
                #print ("No prob for :", winComb)
                return 0.0
        return 0.0

    def entropyTop(self,race,numContenders=4):
        '''Computes the entropy of the top N contenders based on the probModel using base N=numContenders'''
        probSeries=pd.Series(self.probModel(race))
        if len(probSeries)<4:
            print("Less than:" ,numContenders,"probs for race", race.id, "with probModel:", self.probModel)
            return 0.0
        orderProbs=probSeries.order(ascending=False)
        topNProbs=orderProbs[0:numContenders]
        try:
            entropyTop=ss.entropy(topNProbs,base=numContenders)
        except:
            print("could not compute entropy:", race.id)
            return 0.0
        return entropyTop
           
    def generateWagersForRaces(self, races, simParam, allowParallel = True):
        ''' Returns a list of wagers.  Each wager is a tuple of WagerKey and amount '''
        t1 = time.time()
        (goodRaces, filterCounts) = selectRaces(self.selector, races) 

        if allowParallel and self.parallel:
            print "Running parallel sim"
            resultList = Pool(4).map(runStrategyOnRace(self, simParam), goodRaces)
            print "Finished parallel sim"
        else:
            f = runStrategyOnRace(self, simParam)
            resultList = [f(race) for race in goodRaces]
        
        wagers = []
        betOn = 0
        for wagerList in resultList:
            #print "Processing {}".format(wagerList)
            if len(wagerList) == 0:
                filterCounts.add("No wagers")
            else:
                betOn += 1
                wagers.extend(wagerList)
        t2 = time.time()
        print "Generated wagers for {} out of {} races in {} secs.  Filters: {}".format(betOn, len(races), (t2-t1), filterCounts.getSummary())
        return wagers
        
    def generateWagers(self, race, simParam):
        #print "Generating wagers for {} for {}".format(self.getName(), race.id) 
      
        try:
            probs = self.probModel(race)
            wagers = self.betFunction(race, probs, simParam)
        except:
            return []
        #print wagers
        return wagers

    def getName(self):
        return self.name
    def dumpWagerFile(self,races,simParam,allowParallel=True):
        wagers=self.generateWagersForRaces(races, simParam, allowParallel)
        wagersList=[]
        badWagersList=[]
        columnNames=['raceId','wagerType','wagerAmount','runners']
        for w in wagers:
            wk=w[0]
            amount=w[1]
            try:
                tempWager=dict(zip(columnNames,[wk.raceId,wk.getWagerName(),amount,"/".join(list(wk.getCombinations()[0]))]))
                wagersList.append(tempWager)
            except:
                badWagersList.append((wk,amount))
        dfWagers=pd.DataFrame.from_records(pd.Series(wagersList),columns=['raceId','wagerType','wagerAmount','runners'])
        dfWagers['pos1']=dfWagers.runners.apply(lambda x:x.split("/")[0])
        dfWagers['pos2']=dfWagers.runners.apply(lambda x:x.split("/")[1])
        dfWagers['pos3']=dfWagers.runners.apply(lambda x:x.split("/")[2])
        print ("badwagersList:",badWagersList)
        return (dfWagers,badWagersList)

                                             
    def __repr__(self):
        return "Strategy('{}', {!r}, {!r}, {!r})".format(self.getName(), self.selector, self.probModel, self.betFunction)
