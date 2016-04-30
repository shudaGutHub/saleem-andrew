import math

import sqlite3

from x8313 import checkProbDict, normProbDict
def updateSpeed(sim,seriesMaury,attrName='speedMaury'):
    for race in sim.raceDict.keys():
        runners=sim.raceDict[race].bettableRunners()
        for runner in runners:
            setattr(runner,attrName,seriesMaury.get(runner.id))
    return True





class PastPerformanceScore(object):
    ''' Computes a score from past performances by looking up a field and then calling an aggregation function on the vector of values. '''
    def __init__(self, factorName, aggregateFunc, numRaces =None,distance=None,track=None):
        self.factorName = factorName
        self.aggregateFunc = aggregateFunc
        self.numRaces = numRaces
        self.distance=distance
        self.track=track
    def __call__(self, race):
        def agg(runner):
            pps = runner.pastPerformances.values()
            if self.track is not None:
                pps=[pp for pp in pps if getattr(pp,'trackId')==self.track]
            if self.distance is not None:
                pps=[pp for pp in pps if getattr(pp,'distance')>=self.distance]
            if self.numRaces is not None:
                pps = pps[:self.numRaces]
            return self.aggregateFunc([getattr(pp, self.factorName) for pp in pps if hasattr(pp, self.factorName)])
        return {runner.id : agg(runner) for runner in race.runners }
    def __repr__(self):
        return "PastPerformanceScore('{}', {!r})".format(self.factorName, self.aggregateFunc)


class TrainerScore(object):
    ''' Computes a score from past performances by looking up a field and then calling an aggregation function on the vector of values. '''
    def __init__(self, trainerDict):
        self.trainerDict=trainerDict
    def __call__(self, race):
        def agg(runner):
            trainer = runner.trainer
            return self.trainerDict.get(trainer,1.0)
        return {runner.id : agg(runner) for runner in race.runners }
    def __repr__(self):
        return "TrainerScore('{}', {!r})".format(self.factorName, self.aggregateFunc)
class PastPerformanceRatio(object):
    ''' Computes a score from past performances by looking up a field and then calling an aggregation function on the vector of values. '''
    def __init__(self, factorNameTop, factorNameBottom, aggregateFunc, numRaces = None):
        self.factorNameTop = factorNameTop
        self.factorNameBottom = factorNameBottom
        self.aggregateFunc = aggregateFunc
        self.numRaces = numRaces
    def __call__(self, race,track=None):
        def agg(runner):
            pps = runner.pastPerformances.values()
            dfpp = runner.getDataFramePP()
            if track is not None:
                pps=[pp for pp in pps if getattr(pp,'track')==track]
            if self.numRaces is not None:
                pps = pps[:self.numRaces]
            return self.aggregateFunc([getattr(pp, self.factorName) for pp in pps if hasattr(pp, self.factorName)])
        return {runner.id : agg(runner) for runner in race.runners }
    def __repr__(self):
        return "PastPerformanceRatio('{}', {!r})".format(self.factorName, self.aggregateFunc)

class NormProbModel(object):
    def __call__(self, race,addIndex=False):
        rawProbs = self.rawProbs(race)
        normProbs = normProbDict(rawProbs)
        #checkProbDict(normProbs)
        #print "{} probs for {}: {}".format(self.fieldName, race.id, normProbs)
        if addIndex==True:
            probs=pd.Series(normProbs)
            probsOrder=probs.order(ascending=False)
            idxABC=["A","B","C","D","E","F","G","H","I","J","K","L","M"]
            idxRunners=probsOrder.index.values
            idxZipABC=pd.MultiIndex.from_tuples(zip(idxABC,idxRunners))
            probsOrder.index=idxZipABC
            return probsOrder
        else:
            return normProbs

    def rawProbs(self, race):
        rawProbs = {}
        for runner in race.runners:
            prob = self.getProb(runner)
            if not hasattr(runner, 'coupledTo') or runner.coupledTo is None:
                assignTo = runner.id
            else:
                assignTo = runner.coupledTo
            if not self.sumCoupledHorses() and assignTo != runner.id:
                continue
            #print("{} {}".format(prob, runner.__dict__))
            if race.runnerIndex[assignTo].isBettable():
                raw = rawProbs.setdefault(assignTo, 0)
                rawProbs[assignTo] = raw + prob
        return rawProbs

    def sumCoupledHorses(self):
        '''Indicates how coupled horses should be treated.  If true, the the probabilties will be summed.  If not, only the main horse probability will be used'''
        return True


class NamedProbModel(NormProbModel):
    ''' Uses a probability model from a named field on each runner.  Normalizes it to a probability (sumProbs to 1 across all horses)'''
    def __init__(self, fieldName):
        self.fieldName = fieldName

    def getProb(self, runner):
        return runner.__dict__.get(self.fieldName,0.0)

    def __repr__(self):
        return "NamedProbModel({!r})".format(self.fieldName)

class NamedRankToPowerLawProbModel(NormProbModel):
    ''' Uses a probability model from a named field on each runner.  Normalizes it to a probability (sumProbs to 1 across all horses)'''
    def __init__(self, fieldName, defaultValue = 0):
        self.fieldName = fieldName
        self.defaultValue = defaultValue

    def getProb(self, runner):
        if not hasattr(runner, self.fieldName):
            print("Runner {} has no attribute {}.  Defaulting to {}.  {}".format(runner.id, self.fieldName, self.defaultValue, runner.__dict__))
        rating = getattr(runner, self.fieldName, self.defaultValue)
        rankPower=ss.rankdata(-rating)
        return math.log((rankPower+1)/rankPower)

    def __repr__(self):
        return "NamedRankToPowerLawProbModel({!r})".format(self.fieldName)


class NamedOddsModel(NormProbModel):
    ''' Uses a probability model based on fractional odds from a named field on each runner.  Normalizes it to a probability (sumProbs to 1 across all horses)'''
    def __init__(self, fieldName):
        self.fieldName = fieldName

    def getProb(self, runner):
        odds = getattr(runner, self.fieldName, None)
        if odds is None:
            return 0
        return 1/(1+odds)
    def getOdds(self,runner):
        odds = getattr(runner, self.fieldName, None)
        if odds is None:
            return 0
        return odds
    def getOddsRace(self,race):
        bettableRunners=race.bettableRunners()
        oddsDict={}
        for r in bettableRunners:
            oddsDict[r.id]= self.getOdds(r)
        return oddsDict

    def getTrackTake(self, race):
        oddsProbs = self.rawProbs(race)
        summedProb = sum(oddsProbs.values())
        tt = 1 - 1/summedProb
        assert tt < 0.33, "Computed track take for {} to be {}.  Odds were {}".format(tt, race, oddsProbs)
        return tt
    def getOddsNorm(self,runner,useLog=False):
        p=self.getProb(runner)
        try:
            odds=1.0/p-1.0
        except:
            print "zero prob"
            return np.NaN
        if useLog:
            return math.log(odds)
        else:
            return odds
    def getOddsNormRace(self,race,useLog=False):
        bettableRunners=race.bettableRunners()
        oddsDict={}
        for r in bettableRunners:
            oddsDict[r.id]= self.getOddsNorm(r,useLog)
        return oddsDict
    def logOddsRatio(self,runner,oddsDenom):
        odds=math.log(self.getOddsNorm(runner))
        oddsD=math.log(oddsDenom.getOddsNorm(runner))
        try:
            return odds/oddsD
        except ZeroDivisionError:
            print"Zero division finalOdds"
            print oddsD
            return np.NaN

    def logOddsRatioRace(self,race,oddsDenom):
        bettableRunners=race.bettableRunners()
        oddsDict={}
        for r in bettableRunners:
            oddsDict[r.id]= self.logOddsRatio(r,oddsDenom)
        return pd.Series(oddsDict).order(ascending=False)

    def sumCoupledHorses(self):
        '''Indicates how coupled horses should be treated.  If true, the the probabilties will be summed.  If not, only the main horse probability will be used'''
        return False

    def __repr__(self):
        return "NamedOddsModel({!r})".format(self.fieldName)










def runnerDistancePassFail(race):
    distance=race.distance
    runners=race.bettableRunners()
    try:
        seriesDist=pd.Series({r.id:r.getDistanceVector()[0:4] for r in runners})
    except:
        return None
    distRaces=seriesDist.apply(lambda x:len(x[x>0.8*distance]))
    return distRaces>3
def runnerSpeedPassFail(race):
    speedPar=race.__dict__.get('HDWSpeedParForClassLevel')
    if speedPar is None:
        return None
    runners=race.bettableRunners()
    try:
        seriesSpeed=pd.Series({r.id:r.getSpeedVector()[0:4] for r in runners})
    except:
        return None
    speedRaces=seriesSpeed.apply(lambda x:len(x[x>=speedPar-3]))
    return speedRaces>1
def runnerPurseClaimScore(race,conditional=False):
    racePurse=race.purse
    claimPrice=race.__dict__.get('claimingPrice',racePurse)
    runners=race.bettableRunners()
    try:
        seriesPurse=pd.Series({r.id:r.avePurseLastK(1) for r in runners})
    except:
        return None
    try:
        seriesLastFinish=pd.Series({r.id:r.getFinishPos()[0] for r in runners})
    except:
        return None
    seriesLastPurseWin=seriesPurse*1.0/seriesLastFinish
    seriesClaimValue=np.log((seriesLastPurseWin)/(claimPrice))
    if conditional==True:
        return ss.zscore(seriesClaimValue)
    else:
        return seriesClaimValue


def getRaceInputs(race,pmSpeedAC_5,MEAN_PURSECLASS_IN,STD_PURSECLASS_IN):
    seriesProbSpeed=pmSpeedAC_5(race)
    purseClass=runnerPurseClaimScore(race,conditional=False)
    seriesPurseClass=purseClass
    zPurseClass=(seriesPurseClass-MEAN_PURSECLASS_IN)/STD_PURSECLASS_IN
    purseClassConditional=runnerPurseClaimScore(race,conditional=True)
    seriesPurseClassCond=purseClassConditional
    distanceFail=runnerDistancePassFail(race)
    seriesDistFail=distanceFail
    dfFit=pd.DataFrame(seriesPurseClass,columns=['purseClass'])
    dfFit['probSpeedMedian']=seriesProbSpeed
    dfFit['zPurseClass']=zPurseClass
    dfFit['zPurseCondRace']=seriesPurseClassCond
    dfFit['distPass']=seriesDistFail
    dfFit['binDist']=dfFit['distPass'].apply(lambda x:int(x))
    return dfFit

class ZProbs():
    def __init__(self,func,fieldName,defaultValue=0.0):
        self.func=func
        self.fieldName=fieldName
        self.defaultValue = defaultValue
    def __call__(self,race):
        runners=race.bettableRunners()
        scores=pd.Series({runner.id:getattr(runner,self.fieldName,self.defaultValue) for runner in runners})
        zscores=ss.zscore(scores)
        cdfScores=zscores.apply(lambda x:ss.norm.cdf(x))
        normProbs=cdfScores/cdfScores.sum()
        return normProbs.to_dict()
    def __repr__(self):
        return "ZProbs({!r})".format(self.probModel)

templateFile_4 = 'C:/AnimalCrackers/github/AnimalCrackers/Python/JCapperModule/src/dev/saleem/dfTemplate4.csv'
dfTemplate4_20 = pd.read_csv(templateFile_4, index_col=['indexEntropy'])

class EvenOdds(object):
    def __call__(self, race):
        bettable = [runner for runner in race.runners if runner.isBettable() or runner.coupledTo is not None]
        prob = 1/float(len(bettable))
        ret = {runner.id : prob for runner in race.bettableRunners()}
        for runner in bettable:
            if not runner.isBettable():
                ret[runner.getBetHorseId()] += prob
        return ret
# class TemplateProbs(object):
#     '''Generates probs according to number of contenders from a choice of templates'''
#     def __call__(self, race):
#         bettable = race.bettableRunners()
#
#         scores = 1/float(len(bettable))
#         ret = {runner.id : PastPerformanceScore("HDWSpeedRating", for runner in race.bettableRunners()}
#         for runner in bettable:
#             if not runner.isBettable():
#                 ret[runner.getBetHorseId()] += prob
#         return re

class EdgeProbModel(object):
    ''' Combines a NormProbModel with FinalOdds and rebate to compute an edge adjusted set of probs'''
    def __init__(self, probModel):
        self.probModel = probModel

    def __call__(self, race):
        # Compute probabilities for each model
        normalProbs = self.probModel(race)
        oddsProbs = NamedOddsModel('finaltoteodds')
        return self.edgeProbs(race, normalProbs, oddsProbs)

    def edgeProbs(self, race,normalProbs,oddsProbs,rebate=0.0):
        retProbs = {}
        for runner in [r.id for r in race.bettableRunners()]:
            sprob=normalProbs[runner]
            fprob=max(oddsProbs[runner],.004)
            sodd=min(99,1/sprob-1)
            fodd=min(99,1/fprob-1)
            effectiveOddRebate=(fodd-rebate)/(1-rebate)
            effOddsRatio=effectiveOddRebate/sodd
            retProbs[runner] = sprob*effOddsRatio
        sumProbs = sum(retProbs.values())
        return { runnerId : prob/sumProbs for (runnerId, prob) in retProbs}

    def __repr__(self):
        return "EdgeProbModel({!r})".format(self.probModels)


class CombinedMeanProbModel(object):
    ''' Combines multiple probability models by taking the highest probability for each horse and then renormalizing back to a probability distribution.'''
    def __init__(self, *probModels):
        self.probModels = probModels
    def __call__(self, race):
        # Compute probabilities for each model
        probs = [probModel(race) for probModel in self.probModels]

        return self.mean(race, probs)

    def mean(self, race, probs):
        #         ix = 0
        #         for prob in probs:
        #             print "Prob: "+str(ix)+" "+str(prob)
        #             ix += 1
        # Compute probabilities for each model
        if hasattr(race, "bettableRunners"):
            runnerList = [r.id for r in race.bettableRunners()]
        else:
            runnerList = race.HorseName_JCP
        def runnerProbs(runnerId):
            return [prob[runnerId] for prob in probs if not math.isnan(prob[runnerId])]
        combinedProbs = { r : sum(runnerProbs(r)) for r in runnerList }
        try:
            sumProbs = sum(combinedProbs.itervalues())
        except:
            sumprobs=1.0
            print "sumProbs issue:"
        try:
            normProbs = { runner : combinedProbs[runner]/sumProbs for runner in runnerList }
        except:
           # print ("prob mormalizing", runnerList)
            return None
            #normProbs = {runner : 1.0/len(runnerList) for runner in runnerList}

        #print(normProbs)
        checkProbDict(normProbs)
        #print(normProbs)
        return normProbs

    def __repr__(self):
        return "CombinedMeanProbModel({!r})".format(self.probModels)

class CombinedMaxProbModel(object):
    ''' Combines multiple probability models by taking the highest probability for each horse and then renormalizing back to a probability distribution.'''
    def __init__(self, *probModels):
        self.probModels = probModels

    def __call__(self, race):
        # Compute probabilities for each model
        probs = [probModel(race) for probModel in self.probModels]
        return self.maxAndNormalize(race, probs)

    def maxAndNormalize(self, race, probs):
        # Compute probabilities for each model
        def runnerProbs(runnerId):
            return [prob[runnerId] for prob in probs]
        combinedProbs = { r.id : max(runnerProbs(r.id)) for r in race.bettableRunners() }
        sumProbs = sum(combinedProbs.itervalues())
        normProbs = { runner.id : combinedProbs[runner.id]/sumProbs for runner in race.bettableRunners() }
        #print(normProbs)
        checkProbDict(normProbs)
        #print(normProbs)
        return normProbs

    def __repr__(self):
        return "CombinedMaxProbModel({!r})".format(self.probModels)

class DiffProbModel(object):
    '''Takes the difference in two prob models for each runner and uses this to compute a probability for each runner'''
    '''e.g. MorningLine FinalOdds'''
    def __init__(self,probModelA,probModelB):
        self.probsA=probModelA
        self.probsB=probModelB

    def __call__(self,race):
        logpA=np.log(pd.Series(self.probsA(race)))
        logpB=np.log(pd.Series(self.probsB(race)))
        diff=logpB-logpA
        diff[diff<0.0]=0.0
        probNorm=diff/diff.sum()
        print probNorm
        return probNorm
def __repr__(self):
    return "DiffProbModel({!r})".format(self.probModels)


def matchProbability(probA,probB,alpha):
    if probA>= probB:
        return (1+(probA-probB)**alpha)/2.0
    elif probA<=probB:
        return (1-(probB-probA)**alpha)/2.0
