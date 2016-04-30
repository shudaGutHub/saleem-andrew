import itertools
import math
from collections import namedtuple
from itertools import product
import numpy as np
import pandas as pd
from x8313 import tuplize, roundTo, checkProbDict, normProbDict
from Models import WagerKey
from Probabilities import NamedOddsModel, PastPerformanceScore, EvenOdds
from ScoreToProbViaIntegral import ScoreToProbViaIntegral
import scipy.stats as ss
from dateutil.parser import parse

from datetime import date,timedelta
from Models import WagerMatrix
from TicketBoxer import *

from collections import defaultdict

#from dev.saleem.BreakevenFunctions import getOrderedWagerProbs,getOrderedWins,getOrderedExactas,getOrderedTrifectas,getOrderedSuperfectas
#import dev.saleem.HorseMath.Entropy
Point2D=namedtuple('Point2D',['x','y'])
Point3D=namedtuple('Point3D',['x','y','z'])

def breakdownNumber(seriesReturn,base=10000):
    '''takes any pnl stream orders from highest to lowest and determines how many values can be removed while the pnl remains positive
    this is a measure of robustness'''
    orderedReturns=seriesReturn.order(ascending=False)
    runningPL=orderedReturns.sum()+base
    orderedReturns.index=range(len(orderedReturns))
    cumsumOrdered=orderedReturns.cumsum()
    breakdown=0
    while runningPL >0.0:
        breakdown=breakdown+1
        runningPL=runningPL-cumsumOrdered[breakdown]
    return breakdown
def trifectaScore(bettableRunners, probs,dLambda=.9,dRho=0.9,dTau=1.0):
    def _trifectaProb(p1, p2, p3,probs,dLambda,dRho,dTau):
        probs = pd.Series(probs)


        p2_lambda = math.pow(p2,dLambda)
           

        p3_rho = math.pow(p3,dLambda)
            
        denom_1 = np.sum(np.power(probs,dLambda))
        denom_2 = np.sum(np.power(probs,dRho))
        dResult = p1 * (p2_lambda/denom_1) * (p3_rho / denom_2)
        return dResult
        #return p1 * p2/(1-p1) * p3/(1-p1-p2)
    return {(r1.id, r2.id, r3.id) : _trifectaProb(probs[r1.id], probs[r2.id], probs[r3.id],probs,dLambda,dRho,dTau) for (r1, r2, r3) in itertools.permutations(bettableRunners, 3) }  
def safeLog(x):
    if x is None or x<=0:
        # print("Log of {}".format(x))
        return 0
    else:
        return math.log(x)   
def weightedL1Dist(rankSeriesA,rankSeriesB,race):
    wtdL1={}
    br=race.bettableRunners()
    runnerIds=[r.id for r in br]
    indexB=rankSeriesB.index.values
    indexA=rankSeriesA.values
    for i in runnerIds:
        rankA=rankSeriesA.get(i,99)
        rankB=rankSeriesB.get(i,999)
        distL1=abs(rankA-rankB)
        minAB=min(rankA,rankB)
        wtdL1[i]=distL1/minAB
    return pd.Series(wtdL1)

def indexWagers(wagers,wagerType='Exacta'):
    return wagers
    
def impliedProbabilitiesRace(wagerDictRace,probModel=None):
    '''Takes multihorse wagers and computes the implied 1st,2nd,3rd,4th probs'''
    
    wagerSeriesRace=pd.Series(wagerDictRace)
    totalWagerDollars=wagerSeriesRace.sum()
    wagerKeysRace=wagerSeriesRace.index.values
    dictPos1={}
    dictPos2={}
    dictPos3={}
    jointPDF={} 
    for wk in wagerKeysRace:
        ticketRunners=wk.getCombinations()[0]
        ticketValue=wagerSeriesRace[wk]
        dictPos1[ticketRunners[0]]+=ticketValue
        dictPos2[ticketRunners[1]]+=ticketValue
        dictPos3[ticketRunners[2]]+=ticketValue
    seriesPos1=pd.Series(dictPos1)
    seriesPos2=pd.Series(dictPos2)
    seriesPos3=pd.Series(dictPos3)
    jointPDF['Pos1']=seriesPos1/seriesPos1.sum()
    jointPDF['Pos2']=seriesPos2/seriesPos2.sum()
    jointPDF['Pos3']=seriesPos3/seriesPos3.sum()
    dfJointPDF=pd.DataFrame(jointPDF)
    dfClean=dfJointPDF.fillna(0.0)
    return dfClean

def makeWagerDict(strategy,racedict,sims):
    '''generates a dict of DataFrames of wagers indexed on race for a strategy'''
    wagersList=strategy.generateWagersForRaces(racedict,sims)
    goodRaceKeys=[wager[0].raceId for wager in wagersList]
    logOddsDict={r:strategy.probModel.getOddsNormRace(racedict[r],useLog=True) for r in goodRaceKeys}

    wagerDictList={r:strategy.generateWagers(racedict[r], sims) for r in goodRaceKeys}
    wagerDictDict={}
  #  logOddsTupleDict={}
    for r in wagerDictList.keys():
        wagerList=wagerDictList[r]
        logOddsRace=logOddsDict[r]
        wagerDictDict[r]={i[0]:i[1] for i in wagerList}
        
    return wagerDictDict    
     


    
def chopExacta(wagerTemplate,wagers,probs):
    '''use wagerTemplate to constrain wagers'''
   
    probseries=pd.Series(probs)
    numRunners=np.float(len(probs))
    op=probseries.order(ascending=False)
    cumProb=op.cumsum()
    contenderIDs=op.index
    
    pos1Runners=[p[0][0][0] for p in wagers]
    pos2Runners=[p[0][1][0] for p in wagers] 
    pos1Probs=op[pos1Runners]
    pos2Probs=op[pos2Runners]
    maxPos1=wagerTemplate['pos1']
    maxPos2=wagerTemplate['pos2']
    filterPos1=pos1Runners[0:maxPos1]
    filterPos2=pos2Runners[0:maxPos2]
    dropWagers=[]
    for w in wagers:
        if w[0][0][0] not in filterPos1:
            wagers.remove(w)
        if w[0][1][0] not in filterPos2:
            wagers.remove(w)

    return wagers
       
def chopTrifecta(wagerTemplate,wagers,probs,wagersExa):

    '''eliminate overbet combinations'''
 
    probseries=pd.Series(probs)
    numRunners=np.float(len(probs))
    op=probseries.order(ascending=False)
    cumProb=op.cumsum()
    contenderIDs=op.index
    
    pos1Runners=[p[0][0][0] for p in wagers]
    pos2Runners=[p[0][1][0] for p in wagers] 
    pos3Runners=[p[0][2][0] for p in wagers]
    pos1Probs=op[pos1Runners]
    pos2Probs=op[pos2Runners]
    pos3Probs=op[pos3Runners]
    maxPos1=wagerTemplate['pos1']
    maxPos2=wagerTemplate['pos2']
    maxPos3=wagerTemplate['pos3']
    filterPos1=pos1Runners[0:maxPos1]
    filterPos2=pos2Runners[0:maxPos2]
    filterPos2=pos2Runners[0:maxPos3]
    dropWagers=[]
    for w in wagers:
        if w[0][0][0] not in filterPos1:
            wagers.remove(w)
        if w[0][1][0] not in filterPos2:
            wagers.remove(w)
        if w[0][2][0] not in filterPos2:
            wagers.remove(w)

    return wagers
def chopSuperfecta(wagerTemplate,wagers,probs,wagersExa,wagersTri):
    '''eliminate overbet combinations'''
 
    probseries=pd.Series(probs)
    numRunners=np.float(len(probs))
    op=probseries.order(ascending=False)
    cumProb=op.cumsum()
    contenderIDs=op.index

    pos1Runners=[p[0][0][0] for p in wagers]
    pos2Runners=[p[0][1][0] for p in wagers] 
    pos3Runners=[p[0][2][0] for p in wagers]
    pos4Runners=[p[0][3][0] for p in wagers]
    pos1Probs=op[pos1Runners]
    pos2Probs=op[pos2Runners]
    pos3Probs=op[pos3Runners]
    pos4Probs=op[pos4Runners]
    maxPos1=wagerTemplate['pos1']
    maxPos2=wagerTemplate['pos2']
    maxPos3=wagerTemplate['pos3']
    maxPos4=wagerTemplate['pos4']
    filterPos1=pos1Runners[0:maxPos1]
    filterPos2=pos2Runners[0:maxPos2]
    filterPos3=pos3Runners[0:maxPos3]
    filterPos4=pos4Runners[0:maxPos4]
    dropWagers=[]
    for w in wagers:
        if w[0][0][0] not in filterPos1:
            wagers.remove(w)
        if w[0][1][0] not in filterPos2:
            wagers.remove(w)
        if w[0][2][0] not in filterPos3:
            wagers.remove(w)
        if w[0][3][0] not in filterPos4:
            wagers.remove(w)
  
    return wagers
    
def winProbs(bettableRunners, probs,useExoticProbs='Trifecta',dLambda=1.0,dRho=1.0,dTau=1.0):
    return probs
def exactaProbsMLE(bettableRunners,probs,dLambda=.75):
    def _exactaProb(p1, p2,probs,dLambda=.75,dRho=1.0,dTau=1.0):
        dDenom = np.power(probs,dLambda).sum()
        p1_adj = math.pow(p1,dLambda)/dDenom
        p2_adj = math.pow(p2,dLambda)/dDenom
        dDenom = dDenom - p1_adj
        dResult = p1* (p2_adj / 1.0-p1_adj)
        return dResult
    ''' Converts individual horse win probabilities into exacta probabilties '''
    return {(r1.id, r2.id) : _exactaProb(probs[r1.id], probs[r2.id],probs,dLambda=.75) for (r1, r2) in itertools.permutations(bettableRunners, 2) }     
         
def exactaProbs(bettableRunners, probs,dLambda=1.0,dRho=1.0,dTau=1.0):
    def _exactaProb(p1, p2,probs,dLambda=1.0,dRho=1.0,dTau=1.0):
#         # The rationale here is that the probability of horse 1 coming in first is p1 
#         # Then the probability of horse 2 coming in second given that horse 1 wins is p2/(1-p1).
#         dDenom = np.power(probs,dLambda).sum()
#         p1_adj = np.power(p1,dLambda)
#         p2_adj = np.power(p2,dLambda)
#         dDenom = dDenom - p1_adj
#         dResult = p1_adj * (p2_adj / dDenom)
        # return dResult       
        return p1 * p2/(1-p1)
    ''' Converts individual horse win probabilities into exacta probabilties '''
    return {(r1.id, r2.id) : _exactaProb(probs[r1.id], probs[r2.id],probs,dLambda=1.0) for (r1, r2) in itertools.permutations(bettableRunners, 2) }

def trifectaProbs(bettableRunners, probs,dLambda=1.0,dRho=1.0,dTau=1.0):
    def _trifectaProb(p1, p2, p3,probs,dLambda,dRho,dTau):
#         probs = pd.Series(probs)
#         dDenom=np.power(probs,dLambda).sum()    
#         p1_lambda = math.pow(p1,dLambda)
#         p2_lambda = math.pow(p2,dLambda)
#           
#         p1_rho = math.pow(p3,dLambda)
#         p2_rho = math.pow(p3,dLambda)
#         p3_rho = math.pow(p3,dLambda)
#            
#         denom_1 = np.sum(np.power(probs,dLambda))
#         denom_2 = np.sum(np.power(probs,dRho))
#         dResult = p1 * (p2_lambda/denom_1) * (p3_rho / denom_2)
#         return dResult
        return p1 * p2/(1-p1) * p3/(1-p1-p2)
    return {(r1.id, r2.id, r3.id) : _trifectaProb(probs[r1.id], probs[r2.id], probs[r3.id],probs,dLambda,dRho,dTau) for (r1, r2, r3) in itertools.permutations(bettableRunners, 3) }
def trifectaProbsMLE(bettableRunners, probs,dLambda=.9,dRho=0.9,dTau=1.0):
    def _trifectaProb(p1, p2, p3,probs,dLambda,dRho,dTau):
        probs = pd.Series(probs)
        dDenom=np.power(probs,dLambda).sum()    
        p1_lambda = math.pow(p1,dLambda)
        p2_lambda = math.pow(p2,dLambda)
           
        p1_rho = math.pow(p3,dLambda)
        p2_rho = math.pow(p3,dLambda)
        p3_rho = math.pow(p3,dLambda)
            
        denom_1 = np.sum(np.power(probs,dLambda))
        denom_2 = np.sum(np.power(probs,dRho))
        dResult = p1 * (p2_lambda/denom_1) * (p3_rho / denom_2)
        return dResult
        #return p1 * p2/(1-p1) * p3/(1-p1-p2)
    return {(r1.id, r2.id, r3.id) : _trifectaProb(probs[r1.id], probs[r2.id], probs[r3.id],probs,dLambda,dRho,dTau) for (r1, r2, r3) in itertools.permutations(bettableRunners, 3) }
  
def superfectaProbs(bettableRunners, probs,dLambda=1.0,dRho=1.0,dTau=1.0):
    def _superfectaProb(p1, p2, p3, p4,probs,dLambda,dRho,dTau):
   
#         dDenom=np.power(probs,dLambda).sum()
#          
#         p1_lambda = math.pow(p1,dLambda)
#         p2_lambda = math.pow(p2,dLambda)
#          
#         p1_rho = math.pow(p1,dLambda)
#         p2_rho = math.pow(p2,dLambda)
#         p3_rho = math.pow(p3,dLambda)
#          
#         p1_tau = math.pow(p1,dTau)
#         p2_tau = math.pow(p2,dTau)
#         p3_tau = math.pow(p3,dTau)
#         p4_tau = math.pow(p4,dTau)
#                
#         denom_1 = np.sum(np.power(probs,dLambda))-p1_lambda
#         denom_2 = np.sum(np.power(probs,dRho))-(p1_rho+p2_rho)
#         denom_3 = np.sum(np.power(probs.values,dTau))-(p1_tau+p2_tau+p3_tau)      
        #dResult = p1 * (p2_lambda / denom_1) * (p3_rho/denom_2) * (p4_tau / denom_3)
        #return dResult
        return p1 * p2/(1-p1) * p3/(1-p1-p2) * p4/(1-p1-p2-p3)
    return {(r1.id, r2.id, r3.id, r4.id) : _superfectaProb(probs[r1.id], probs[r2.id], probs[r3.id], probs[r4.id],probs,dLambda,dRho,dTau) for (r1, r2, r3, r4) in itertools.permutations(bettableRunners, 4) }
#          
# def exactaProbs(bettableRunners, probs):
#     def _exactaProb(p1, p2):
#         # The rationale here is that the probability of horse 1 coming in first is p1 
#         # Then the probability of horse 2 coming in second given that horse 1 wins is p2/(1-p1).
#         return p1 * p2/(1-p1)
#     ''' Converts individual horse win probabilities into exacta probabilties '''
#     return {(r1.id, r2.id) : _exactaProb(probs[r1.id], probs[r2.id]) for (r1, r2) in itertools.permutations(bettableRunners, 2) }
#  
# def trifectaProbs(bettableRunners, probs):
#     def _trifectaProb(p1, p2, p3):
#         return p1 * p2/(1-p1) * p3/(1-p1-p2)
#     return {(r1.id, r2.id, r3.id) : _trifectaProb(probs[r1.id], probs[r2.id], probs[r3.id]) for (r1, r2, r3) in itertools.permutations(bettableRunners, 3) }
#  
# def superfectaProbs(bettableRunners, probs):
#     def _superfectaProb(p1, p2, p3, p4):
#         return p1 * p2/(1-p1) * p3/(1-p1-p2) * p4/(1-p1-p2-p3)
#     return {(r1.id, r2.id, r3.id, r4.id) : _superfectaProb(probs[r1.id], probs[r2.id], probs[r3.id], probs[r4.id]) for (r1, r2, r3, r4) in itertools.permutations(bettableRunners, 4) }


# def testProbModel(stoppingPointSeries):
#     '''       
# 
#     samp = rayleigh.rvs(loc=5,scale=2,size=150) # samples generation
#     
#     param = rayleigh.fit(samp) # distribution fitting
#     
#     x = linspace(5,13,100)
#     # fitted distribution
#     pdf_fitted = rayleigh.pdf(x,loc=param[0],scale=param[1])
#     # original distribution
#     pdf = rayleigh.pdf(x,loc=5,scale=2)
#     
#     title('Geometric distribution')
#     plot(x,pdf_fitted,'r-',x,pdf,'b-')
#     hist(samp,normed=1,alpha
#     show()
#     
    
def multiBoxer(*seqs):
    return (x for x in product(*seqs) if len(x) == len(set(x)))        
def trifectaBox(bettableRunners, probs, trifectaTemplate={1:3,2:4,3:6},dLambda=1.0,dRho=1.0,dTau=1.0):
    '''generates a probList that corresponds to a box / wheel bet'''
    prob_series = pd.Series(probs)
    rank_probs=prob_series.rank(method="first",ascending=False)
    numRunners=len(prob_series)
    A=rank_probs[rank_probs<trifectaTemplate.get(1,numRunners)+1].index.values
    B=rank_probs[rank_probs<trifectaTemplate.get(2,numRunners)+1].index.values
    C=rank_probs[rank_probs<trifectaTemplate.get(3,numRunners)+1].index.values
   # D=rank_probs[rank_probs<trifectaTemplate.get(4,numRunners)+1].index.values
    idxTri=list(multiBoxer(A,B,C))
    trifectaBoxProbs=pd.Series({t:1.0/len(idxTri) for t in idxTri})
    return trifectaBoxProbs.to_dict()        
def getLast(x):
    try:
        return x[0]
    except:
        return None
def posMean(x):
    xSeries=pd.Series(x)
    return xSeries.mean
def meanBestN(x,N=3):
    '''takes top N and returns mean'''
    numX=len(x)
    xSeries=pd.Series(x)
    xOrder=xSeries.order(ascending=False)
    return xOrder.head(min(numX,N)).mean()
                                
    
    
class FixedPoolSizes(object):
    def __init__(self, WPSRisk=0,exactaRisk=0, trifectaRisk = 0, superfectaRisk = 0,riskDict={}):
        self.riskDict=riskDict
        self.poolSizes = {"WPS":WPSRisk,"Exacta":exactaRisk, "Trifecta":trifectaRisk, "Superfecta":superfectaRisk}

    def __call__(self, race, wagerType):
#         if race.estimatedPoolSizes.get(wagerType,0.0) < self.minPoolSize:
#             #print ("PoolSize less than minpoolsize:", self.minPoolSize)
#             return 0.0
        return self.poolSizes[wagerType]*self.riskDict.get(len(race.bettableRunners()),1.0)
    def __repr__(self):
        return "FixedPoolSizes({}, {}, {}, {})".format(self.poolSizes["WPS"],self.poolSizes["Exacta"], self.poolSizes["Trifecta"], self.poolSizes["Superfecta"])
    def weightedL1Dist(self,rankSeriesA,rankSeriesB):
        wtdL1={}
        for i in rankSeriesA.index.values:
            distL1=abs(rankSeriesA[i]-rankSeriesB[i])
            minAB=min(rankSeriesA[i],rankSeriesB[i])
            wtdL1[i]=distL1/minAB
        return pd.Series(wtdL1)
    
class RaceClassifier(object):
    '''Generates a feature vector for a race based on race details and runners in the race'''
    def __init__(self,race):
        self.race =race
        self.officialFinish=self.getFinishPos()
        self.dfProbs=self.getDataFrameProbs()      
    def parseClass(self,ppClass):
        ppClass=ppClass.upper()
        racePrefixType=['FMD','FCLM','FSTR','FALW','FMCL','FAOC','FMSW','MD','CLM','STR','ALW','MCL','AOC','MSW']
        try:
            rtFind=[ppClass.find(t) for t in racePrefixType].index(0)
            raceType=racePrefixType[rtFind]
            return raceType
        except:
            print("couldnt parse:",ppClass)
            return None
    def getClassShiftScore(self,rcLast):
        '''Scores a class shift'''
        try:  
            typeCurrent=self.parseClass(self.race.raceClassification)
            typeLast=self.parseClass(rcLast)
            classShift="-".join([typeCurrent,typeLast])
            classShiftScoreDict={'MCL-MD':0.5,'MD-MD':0.0,'CLM-STR':1.0,'STR-CLM':0.0,'ALW-CLM':0.0,'MCL-MSW':1.0,'CLM-CLM':.25,'CLM-ALW':1.0}
            return classShiftScoreDict.setdefault(classShift,.15) 
        except:
            print("no class shift",self.race.id)
            return 0.25  
    def getTrainers(self):
        dictTrainers={r.id:r.trainer for r in self.race.bettableRunners()}
        return pd.Series(dictTrainers)
    def runnerPurseClaimScore(self):
        self.racePurse=self.race.purse
        claimPrice=self.race.__dict__.get('claimingPrice',self.racePurse)
        runners=self.race.bettableRunners()
        try:
            seriesPurse=pd.Series({r.id:r.avePurseLastK(2) for r in runners})
        except:
            return None
        seriesClaimValue=np.log(seriesPurse)/np.log(claimPrice) 
        return seriesClaimValue  
    def getPPDataRunners(self):
        '''returns a Panel of ppdata where index_0 = runners, index_1=ppNum,
        index_2 = field'''
        dictPP={r.id:r.getDataFramePP() for r in self.race.bettableRunners()}
        return dictPP
    def getEntropyMorningLine(self):
        pmOdds=NamedOddsModel('morningLine')
        probs=pd.Series(pmOdds(self.race))
        entropy=ss.entropy(probs,base=len(probs))
        return entropy
    def getEntropySpeedAC(self):
        probs=pd.Series(self.pmSpeedAC(self.race))
        entropy=ss.entropy(probs,base=len(probs))
        return entropy
    def getModelProbs(self,probModel):
        try:
            probs=probModel(self.race)
            return pd.Series(probs)
        except:
            return None
    def getFinishPos(self):
        ofp=pd.Series({r.id:r.__dict__.get('officialFinishPosition',99) for r in self.race.bettableRunners()})
        return ofp
    def getDataFrameRanks(self):
        probsDict={}
        probsDict['pmOtherMorningLine']=self.getModelProbs(NamedOddsModel('morningLine'))
        df= pd.DataFrame(probsDict)
        dfRank=df.rank(method='min',ascending=False)
        probSeries=pd.Series(self.pmSpeedAC(self.race))
        df=df.join(dfRank,lsuffix='_PROB',rsuffix='_RANK')  
#         df['trainerName']=self.getTrainers()
#         df['trainerScores']=self.getTrainerScores()
#         df['trainerRatio']=self.getTrainerRatio()
        
#         
#         df['bettable']=self.bettableRace
        df['raceId']=self.race.id
        df['raceType']=self.race.raceType
        df['maiden']=self.race.maiden
        df['ageRestriction']=self.race.ageRestriction
        df['ageLimit']=self.race.ageLimit
        df['stateBred']=self.race.stateBred
        df['purse']=self.race.purse
        df['distance']=self.race.distance
        df['numStarters']=len(self.race.bettableRunners())
        df['raceClassification']=self.race.raceClassification
        
        df['entropyMorningLine']=self.getEntropyMorningLine()
        df['entropyFinalOdds']=self.getEntropyFinalOdds()
        df['entropySpeedAC']=self.getEntropySpeedAC()
        rankSpeed=probSeries.rank(ascending=False)
        rankMorningLine=self.getModelProbs(NamedOddsModel('morningLine')).rank(ascending=False)
        df['L1DistWtd_MorningLine_Speed']=weightedL1Dist(rankMorningLine,rankSpeed,self.race)
        df['L1DistWtd_Finish_Speed']=weightedL1Dist(self.officialFinish,rankSpeed,self.race)
        df['L1DistWtd_Finish_MorningLine']=weightedL1Dist(self.officialFinish,rankMorningLine,self.race)

    

        df['OUTPUT_officialFinish']=self.officialFinish
        df['OUTPUT_pctBtn']=-1.0*(self.officialFinish-self.officialFinish.max())/(self.officialFinish.max())        

        return df

    def getSpeedRanks(self):
        dfRank=self.dfProbs.rank(method='min',ascending=False)
        return dfRank
   
    def scoreModelRank(self,ranks,finishPos,method='top4'):
        '''7 points for predicting Winner'''
        '''5 for second'''
        '''3 for third'''
        '''2 for fourth'''
        scores=pd.Series({1:7,2:5,3:3,4:2})
        ranks=ranks.rank(method='first')
        ofp=finishPos.order()
        ofp4=ofp[ofp<5]
        ranks4=ranks[ranks<5]
        diffs=ranks4-ofp4
        if method=='exact':
            exact=ofp4[diffs==0]
            return scores[exact].cumprod()
        if method=='top4':
            numInTop4=len(diffs.dropna())
            return numInTop4
                                   
    def __repr__(self):
        return "RaceClassifier({})".format( self.race.id)
    
                
class SmartPoolSizes(object):
    '''pctCover: percentage of combinations to cover'''
    def __init__(self,pctCover,wpsMax,exactaMax,trifectaMax,superfectaMax):
        self.pctCover = pctCover
        self.poolSizes={'WPS':wpsMax,'Exacta':exactaMax,'Trifecta':trifectaMax,'Superfecta':superfectaMax}
        self.baseRisk={'WPS':wpsMax,'Exacta':exactaMax,'Trifecta':trifectaMax,'Superfecta':superfectaMax}


    def scoreRace(self,race):
        '''Computes various metrics to characterize race'''
        return [race.date,race.stateBred,race.sexRestriction,race.distance,race.surface,race.ageRestriction,race.raceClassification,race.purse,race.track_id]
    def scoreRunners(self,race):
        dictPP=pd.Series({r.id:r.getDataFramePP() for r in race.bettableRunners})
        return dictPP
    def __call__(self,race,wagerType):
        #print "{} {} {}".format(race.id, wagerType, race.estimatedPoolSizes)
#         raceDate=race.date
#         runners=race.runners
#         raceNumStarters=len(runners)
#         raceDay=raceDate.weekday() #Monday==0 Sunday==6
#         raceMonth=raceDate.month
#         raceNum=race.raceNumber
        minBetInfo=race.minBet.get(wagerType, (1.0,2.0))
        numStarters=len(race.bettableRunners())
        numCombinations={'WPS':numStarters,'Exacta':numStarters*(numStarters-1),'Trifecta':numStarters*(numStarters-1)*(numStarters-2),'Superfecta':numStarters*(numStarters-1)*(numStarters-2)*(numStarters-3)}
        minNumToBet=math.ceil(numCombinations[wagerType]*self.pctCover)
        maxDollarsRequired=minNumToBet*minBetInfo[0]
        #estimatedPoolSize=race.estimatedPoolSizes[wagerType]#self.trifectaPoolDict.get(str(numStarters),'8').get(track,5000.0)
        riskConstrain=min(maxDollarsRequired,self.baseRisk[wagerType]*numStarters)
        #print(track,estimatedPoolSize,riskConstrain)
        return riskConstrain       
    def __repr__(self):
        return "SmartPoolSizes({!r})".format(self.pctCover)


class PercentagePoolSizes(object):
    def __init__(self, percent, maxTotalBet):
        assert percent < 1.0
        self.percent = percent
        self.maxTotalBet = maxTotalBet
        
    def __call__(self, race, wagerType):
        #print "{} {} {}".format(race.id, wagerType, race.estimatedPoolSizes)
        bet = self.percent * race.estimatedPoolSizes.get(wagerType, 0)
        return min(self.maxTotalBet[wagerType], bet)

    def __repr__(self):
        return "PercentagePoolSizes({}, {!r})".format(self.percent, self.maxTotalBet)

class IdealTicketBoxer(object):
    ''' Creates tickets ignoring all bet restrictions.  Bets everything exactly based on probabilities. ''' 
    def __call__(self, idealWagers, minTicketSize, minBet):
        return [(tuplize(key), value) for (key, value) in idealWagers.iteritems()]

class RoundingTicketBoxer(object):
    ''' Creates tickets by rounding bets to the minimum bet size, but ignoring the minimum ticket size (no boxing). ''' 
    def __call__(self, idealWagers, minTicketSize, minBet):
        return [(tuplize(key), roundTo(value, minBet)) for (key, value) in idealWagers.iteritems()]

    
    
class WPSWagers(object):
    '''Converts win probabilities to WPS bets'''
    def __init__(self, poolSizer):
        self.poolSizer = poolSizer

                
    def __call__(self, race, probs, defaultRebates = None):
        checkProbDict(probs.to_dict())
        wpsDict=[{'Win':(2.0,2.0)}]
        #print race.minBet
        wagers = self.wagers(race, probs, self.poolSizer(race, "WPS"),(2.0,2.0))
        #wagers = wagers + self.wagers(race, probs, trifectaProbs, self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', None))
        #wagers = wagers + self.wagers(race, probs, superfectaProbs, self.poolSizer(race, "Superfecta"), race.minBet.get('Superfecta', None)) 
        listwagers=[]
        for w in wagers:
            #temp=WagerKey(race.id,'win',((w,))),wagers[w]
            #temp=(WagerKey(race.id,WagerKey.Win,w),wagers[w])
            tempWK=WagerKey(race.id,WagerKey.Win,(w,)),wagers[w]
           
            listwagers.append(tempWK)
            
        return  listwagers#[(WagerKey(race.id, WagerKey.Win, ids), amt) for (ids, amt) in wagers if amt > 0]
    def wagers(self, race, probs, risk, minBet):
        ''' Convert probabilities into wagers based on a given risk size. '''
        if(risk == 0):
            return []
        #assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
            #print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
            return []
        idModelFav=probs.idxmax()
        contenderProbsDict={idModelFav:probs.max()}
        #checkProbDict(probs.to_dict())
        combProbs=pd.Series(probs).order(ascending=False)
        
    
        uniformProb=1.0/float(len(probs))
   
        idealWagers = {key : risk for (key, value) in contenderProbsDict.iteritems()}
        

        return idealWagers
        
#Code shared across examples
import pylab, random, string, copy

class EuclideanProbPoint(object):
    def __init__(self, name, originalAttrs, normalizedAttrs = None):
        """normalizedAttrs and originalAttrs are both arrays"""
        self.name = name
        self.unNormalized = originalAttrs
        if normalizedAttrs == None:
            self.attrs = originalAttrs
        else:
            self.attrs = normalizedAttrs
    def dimensionality(self):
        return len(self.attrs)
    def getAttrs(self):
        return self.attrs
    def getOriginalAttrs(self,nDimensions=4):
        attrs=self.unNormalized[0:nDimensions]
        return self.unNormalized
    def distance(self, other,nDimensions=4):
        values=other.getOriginalAttrs()[0:4]
        #Euclidean distance metric
        result = 0.0
        for i in range(self.dimensionality()):
            result += (self.attrs[i] - values[i])**2
        return result**0.5
    def getName(self):
        return self.name
    def toStr(self):
        return self.name + str(self.attrs)
    def __str__(self):
        return self.name        
class ContenderTemplate(Point):
    def __init__(self, name,probs):
        Point.__init__(self, name,probs)
        self.probs=probs
        self.labels=['A','B','C','D']
    def getProbs(self):
        return self.probs
    def distanceRunnerId(self,other,nDim):
        values=other.getProbs()[0:nDim]
        #Euclidean distance metric
        result = 0.0
        for i in range(self.dimensionality()):
            result += (self.attrs[i] - values[i])**2
        return result**0.5        
    def getTopN(self,nDim):
        vals=self.getOriginalAttrs()[0:nDim]
        labels=self.labels
        return pd.Series(dict(zip(labels,vals)))
class TemplateWagers(object):
    '''Takes Alpha Wagers and makes sure we cover the top trainers'''
    def __init__(self,riskMax,poolSizer, ticketBoxer,templateSeries):
        self.riskMax=riskMax
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer  
              
        self.templateSeries=templateSeries
        print templateSeries
    def matchToTemplate(self,x,templateList):
        distances=pd.Series({t.getName(): t.distance(x) for t in templateList})
        return distances.idxmin()  
    
    def __call__(self,race,probs,defaultRebates=None):    
        probsTemplate=self.forceTemplateProbs(race,probs)    
        wagers=self.overlayExactaTrainers(race,probsTemplate,exactaProbs,self.poolSizer(race, "Exacta"),race.minBet.get('Exacta', None)) 
        try:
            wagers = wagers+ self.overlayTrifectaTrainers(race,probsTemplate,trifectaProbs,self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', None))
        except:
            print('No TrifectaWagers',race)
            wagers=wagers
        temp= [(WagerKey(race.id, WagerKey.Multihorse, ids), amt) for (ids, amt) in wagers if amt > 0]
        return temp   
    def forceTemplateProbs(self,race,probs):
        templateContenderProbs=self.templateSeries[race.id]
        sumTemplateProbs=templateContenderProbs.sum()
        sumResidualProbs=1.0-sumTemplateProbs
        
        numRunners=len(probs)
        numRunnersInTemplate=len(templateContenderProbs)
        numResidualRunners=numRunners-numRunnersInTemplate
        residualProbs=sumResidualProbs/numResidualRunners
        
        probsOrder=probs.order(ascending=False)
        probsOrder.ix[0:numRunnersInTemplate]=templateContenderProbs
        probsOrder.ix[numRunnersInTemplate:numRunners]=residualProbs
        
        forcedProbs=probsOrder
       # print forcedProbs
        return forcedProbs

        
    def overlayExactaTrainers(self,race,probs,permuteFunc,risk,minBet): 
        speedSeries=pd.Series(probs)
        

        highestSpeed=speedSeries[speedSeries.rank(ascending=False)<2].index.values
        topSpeed=speedSeries[speedSeries.rank(ascending=False)<3].index.values   
        trainerSeries=pd.Series({r.id:r.trainer for r in race.bettableRunners() if r.trainer})
        trainerScores=trainerSeries.apply(lambda x: self.trainerScoreDict.get(x,1.0))
        trainerScoresTop=trainerScores[trainerScores>1.0]
        topTrainers=trainerScoresTop.index.values      
        preferredSpeedTrainer=pd.Series(list(set(topSpeed).union(set(topTrainers)))).values

    
        defaultMinBet=(1.0,1.0)
        if(risk == 0):
          #  print ("RISK is zero",race.id,permuteFunc.func_name)
            return []
        # assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
            print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}' setting minBet to defaultMinBet ".format(permuteFunc, race=race)
            print defaultMinBet
            minBet=defaultMinBet
        combProbs = permuteFunc(race.bettableRunners(), probs)
#         for c in combProbs.keys():
#             #Eliminate the highest speed runner from position 1
#             if c[0] in highestSpeed:
#                 combProbs[c]=0.0 
#             Insist that the second position is either a topSpeed or topTrainer or topJockey
#             if c[1] not in preferredSpeedTrainer:
#                 combProbs[c]=0.0

        combProbSeries=pd.Series(combProbs)
        
        contenderProbSeries=combProbSeries[combProbSeries>0.0]
        
#         #If we dont have enogh probability after the constraints then dont wager 
#         if contenderProbSeries.sum()<self.alpha:
#             return []

        #Renormalize probs if riskMax == True ensures equal betting
        
        if self.riskMax==True:
            contenderProbSeries=contenderProbSeries/contenderProbSeries.sum()
        
        idealWagers = {key : value*risk for (key, value) in contenderProbSeries.iteritems()}
                
        return self.ticketBoxer(idealWagers, minBet[0], minBet[1])    
                         
    def overlayTrifectaTrainers(self,race,probs,permuteFunc,risk,minBet):
        speedSeries=pd.Series(probs)

        highestSpeed=speedSeries[speedSeries.rank(ascending=False)<2].index.values
        topSpeed=speedSeries[speedSeries.rank(ascending=False)<3].index.values
        
        trainerSeries=pd.Series({r.id:r.trainer for r in race.bettableRunners() if r.trainer})
        print trainerSeries
        trainerScores=trainerSeries.apply(lambda x: self.trainerScoreDict.get(x,1.0))
        trainerScoresTop=trainerScores[trainerScores>1.0]
        topTrainers=trainerScoresTop.index.values
        preferredSpeedTrainer=pd.Series(list(set(topSpeed).union(set(topTrainers)))).values


        
        defaultMinBet=(1.0,1.0)

        if(risk == 0):
            print ("RISK is zero",race.id,permuteFunc.func_name)
            return []
        # assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
            print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}' setting minBet to defaultMinBet ".format(permuteFunc, race=race)
            print defaultMinBet
            minBet=defaultMinBet
            
            #return []
        combProbs = permuteFunc(race.bettableRunners(), probs)
#         for c in combProbs.keys():
#             #Eliminate the highest speed runner from the first position
#             if c[0] in highestSpeed:
#                 combProbs[c]=0.0 
#             #Insist on the 2nd position coming from either topSpeed or topTrainer
#             if c[1] not in preferredSpeedTrainer:
#                 combProbs[c]=0.0
            #In Trifecta we will allow 3rd choice to be any runner
        combProbSeries=pd.Series(combProbs)
        contenderProbSeries=combProbSeries[combProbSeries>0.0]
        if self.riskMax==True:
            contenderProbSeries=contenderProbSeries/contenderProbSeries.sum()
    
        idealWagers = {key : value*risk for (key, value) in contenderProbSeries.iteritems()}                
        return self.ticketBoxer(idealWagers, minBet[0], minBet[1])    
                   
    def __repr__(self):
        return "TemplateWagers({!r})".format(self.poolSizer)       
   
class SpeedTrainerWagers(object):
    '''Takes Alpha Wagers and makes sure we cover the top trainers'''
    '''If riskMax==False we do not bet equal amounts per race'''
    def __init__(self,riskMax,poolSizer, ticketBoxer):
        self.riskMax=riskMax
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer  
        self.trainerScoreDict={'DUTROW ANTHONY W':4.0,'GUERRERO J GUADALUPE':4.0,'ASMUSSEN STEVEN M':4.0,'SERVIS JOHN C':4.0,'KLESARIS STEVE':4.0,'PRECIADO GUADALUPE': 4.0,'ARISTONE PHILIP T':4.0,'GUERRERO JUAN CARLOS':4.0,'PRECIADO RAMON':4.0,'AUWARTER EDWARD K':4.0,'DANDY RONALD J':4.0,'LYNCH CATHAL A':4.0,'LAKE SCOTT A':4.0,'DEMASI KATHLEEN A':4.0,'DAY COREY':4.0,'MOSCO ROBERT':4.0,'LEVINE BRUCE N':4.0,'LEBARRON KEITH W':4.0,'REID JR ROBERT E':4.0}        
    
        
    def __call__(self,race,probs,defaultRebates=None): 
        wagers=self.overlayExactaTrainers(race,probs,exactaProbs,self.poolSizer(race, "Exacta"),race.minBet.get('Exacta', None)) 
        try:
            wagers = wagers+ self.overlayTrifectaTrainers(race,probs,trifectaProbs,self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', None))
        except:
       #     print('No TrifectaWagers',race)
            wagers=wagers
        temp= [(WagerKey(race.id, WagerKey.Multihorse, ids), amt) for (ids, amt) in wagers if amt > 0]
        return temp   
    def overlayExactaTrainers(self,race,probs,permuteFunc,risk,minBet): 
        speedSeries=pd.Series(probs)
        highestSpeed=speedSeries[speedSeries.rank(ascending=False)<2]        
        highestSpeedIDs=highestSpeed.index.values
        topSpeed=speedSeries[speedSeries.rank(ascending=False)<3]            
        topSpeedIDs=topSpeed.index.values
        trainerSeries=pd.Series({r.id:r.trainer for r in race.bettableRunners() if r.trainer})
        trainerScores=trainerSeries.apply(lambda x: self.trainerScoreDict.get(x,1.0))
        trainerScoresTop=trainerScores[trainerScores>1.0]
        topTrainers=trainerScoresTop.index.values  

        preferredSpeedTrainer=pd.Series(list(set(topSpeedIDs).union(set(topTrainers)))).values
        if len(topTrainers) < 2:
            print("No top trainers")
            return []
    
        defaultMinBet=(1.0,1.0)
        if(risk == 0):
         #   print ("RISK is zero",race.id,permuteFunc.func_name)
            return []
        # assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
  #          print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}' setting minBet to defaultMinBet ".format(permuteFunc, race=race)
   #         print defaultMinBet
            minBet=defaultMinBet
        combProbs = permuteFunc(race.bettableRunners(), probs)
        for c in combProbs.keys():
            #Eliminate the highest speed runner from position 1
            if c[0] in highestSpeedIDs:
                combProbs[c]=0.0 
            #Insist that the second position is either a topSpeed or topTrainer or topJockey
            if c[1] not in preferredSpeedTrainer:
                combProbs[c]=0.0

        combProbSeries=pd.Series(combProbs)
        contenderProbSeries=combProbSeries[combProbSeries>0.0]
        
#         #If we dont have enogh probability after the constraints then dont wager 
#         if contenderProbSeries.sum()<self.alpha:
#             return []

        #Renormalize probs if riskMax == True ensures equal betting
        
        if self.riskMax==True:
            contenderProbSeries=contenderProbSeries/contenderProbSeries.sum()
        
        idealWagers = {key : value*risk for (key, value) in contenderProbSeries.iteritems()}
       # idealWagerSeries=pd.Series(idealWagers)
       # top2=idealWagerSeries[idealWagerSeries.rank(ascending=False)<3]
       # topWagers=top2.to_dict()                
       # return self.ticketBoxer(topWagers, minBet[0], minBet[1])  
                
        return self.ticketBoxer(idealWagers, minBet[0], minBet[1])    
                         
    def overlayTrifectaTrainers(self,race,probs,permuteFunc,risk,minBet):
        speedSeries=pd.Series(probs)
        highestSpeed=speedSeries[speedSeries.rank(ascending=False)<2]        
        highestSpeedIDs=highestSpeed.index.values
        topSpeed=speedSeries[speedSeries.rank(ascending=False)<3]   
        topSpeedIDs=topSpeed.index.values
        trainerSeries=pd.Series({r.id:r.trainer for r in race.bettableRunners() if r.trainer})
        trainerScores=trainerSeries.apply(lambda x: self.trainerScoreDict.get(x,1.0))
        trainerScoresTop=trainerScores[trainerScores>1.0]
        topTrainers=trainerScoresTop.index.values      
        preferredSpeedTrainer=pd.Series(list(set(topSpeedIDs).union(set(topTrainers)))).values

        
        defaultMinBet=(1.0,1.0)

        if(risk == 0):
            #         print ("RISK is zero",race.id,permuteFunc.func_name)
            return []
        # assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
            #     print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}' setting minBet to defaultMinBet ".format(permuteFunc, race=race)
            #     print defaultMinBet
            minBet=defaultMinBet
            
            #return []
        combProbs = permuteFunc(race.bettableRunners(), probs)
        for c in combProbs.keys():
            #Eliminate the highest speed runner from the first position
            if c[0] not in set(preferredSpeedTrainer - highestSpeedIDs):
                combProbs[c]=0.0 
            #Insist on the 2nd position coming from either topSpeed or topTrainer
            if c[1] not in preferredSpeedTrainer:
                combProbs[c]=0.0
            #In Trifecta we will allow 3rd choice to be any runner
        combProbSeries=pd.Series(combProbs)
        contenderProbSeries=combProbSeries[combProbSeries>0.0]
        if self.riskMax==True:
            contenderProbSeries=contenderProbSeries/contenderProbSeries.sum()
    
        idealWagers = {key : value*risk for (key, value) in contenderProbSeries.iteritems()}
        #   idealWagerSeries=pd.Series(idealWagers)
     #   top8=idealWagerSeries[idealWagerSeries.rank(ascending=False)<9]
     #   topWagers=top8.to_dict()                
        return self.ticketBoxer(idealWagers, minBet[0], minBet[1])    
                   
    def __repr__(self):
        return "SpeedTrainerWagers({!r})".format(self.poolSizer)   
def chopFirst(wagers,excludeRunners):
    for comb in wagers:
        if comb[0] in excludeRunners:
            wagers[comb]=0.0
    return wagers


                        
class MultihorseWagers(object):
    ''' Takes win probabilities by horse and converts them to 2, 3, and 4 way exotic bets '''
    def __init__(self, poolSizer, ticketBoxer,exactaProbs=exactaProbs,trifectaProbs=trifectaProbs,superfectaProbs=superfectaProbs,filterSpeed=False,addWagers=False,riskMultiplier=1.0,scaleByPoolSize=False,filterTopSpeedPos1=False,filterBadRunners=False):
        '''poolSizer: PoolSizer object e.g. FixedPoolSizes returns an amount to risk per pool'''
        '''ticketBoxer: TicketBoxer object decides on how to construct actual tickets / wagers'''
        '''exactaProbs: defaults to standard exactaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''trifectaProbs: defaults to standard trifectaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''superfectaProbs: defaults to standard trifectaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''filterEntropy: default True, returns no wagers for races where entropy of top4 would fall into the lowest entropy bucket using TemplateWagers'''
        '''riskMultiplier: default 1.0 , applied after ticketBoxer has constructed wagers and allows multiplying risk on the set of chosen wagers'''
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer
        self.exactaProbs=exactaProbs
        self.trifectaProbs=trifectaProbs
        self.superfectaProbs=superfectaProbs
        self.filterSpeed=filterSpeed
        self.addWagers=addWagers
        self.riskMultiplier=riskMultiplier
        self.scaleByPoolSize=scaleByPoolSize
        self.filterTopSpeedPos1=filterTopSpeedPos1
        self.filterBadRunners=filterBadRunners
        self.keyHorsesMap={6:1,7:1,8:1,9:1,10:1,11:1,12:1}
        self.ppScoreSpeed=PastPerformanceScore('HDWSpeedRating', meanBestN)
        self.hardMax=3000.00
    def __call__(self, race, probs, defaultRebates = None):
        speedScores=pd.Series(self.ppScoreSpeed(race))
        mauryScores=pd.Series({r.id:r.__dict__.get('score',0.0) for r in race.bettableRunners()})
        mauryNormProbs=pd.Series({r.id:r.__dict__.get('probNormMaury',0.0) for r in race.bettableRunners()})
        maxSpeed=speedScores.idxmax()
        maxMaury=mauryScores.idxmax()
        if (self.filterSpeed==True) and (maxSpeed==maxMaury):
            return []   
        baseExacta=self.poolSizer(race, "Exacta")
        baseTrifecta=self.poolSizer(race, "Trifecta")
        baseSuperfecta=self.poolSizer(race, "Superfecta")
        if self.riskMultiplier*baseTrifecta > self.hardMax:
            return []
        probSeries=pd.Series(probs).order(ascending=False)
        if len(probSeries)<4:
            return []
        top4=probSeries[0:4]
        entropyTop4=ss.entropy(top4,base=4)
        wagers = self.wagers(race, probs, self.exactaProbs, self.poolSizer(race, "Exacta"), race.minBet.get('Exacta', (2.0,2.0)))
        wagers = wagers+self.wagers(race, probs, self.trifectaProbs, self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', (2.0,2.0)))
        wagers = wagers + self.wagers(race, probs, self.superfectaProbs, self.poolSizer(race, "Superfecta"), race.minBet.get('Superfecta', (.50,2.0)))
        poolSize=race.estimatedPoolSizes['Trifecta']
        poolSizeUnits=min(math.floor(poolSize/5000),10)      
        probsDropTop=probSeries.drop(maxMaury)
        probsDrop=probsDropTop.to_dict()
        wagersAdd =self.wagers(race, probsDrop, self.trifectaProbs, self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', (2.0,2.0))) 
        if self.scaleByPoolSize==True:
            #print("scaling by", poolSizeUnits)
            temp= [(WagerKey(race.id, WagerKey.Multihorse, ids), amt*poolSizeUnits) for (ids, amt) in wagers if amt > 0]
            tempAdd = [(WagerKey(race.id, WagerKey.Multihorse, ids), amt*poolSizeUnits/2.0) for (ids, amt) in wagersAdd if amt > 0]
        else:
            temp= [(WagerKey(race.id, WagerKey.Multihorse, ids), amt*self.riskMultiplier) for (ids, amt) in wagers if amt > 0]
            tempAdd = [(WagerKey(race.id, WagerKey.Multihorse, ids), amt*self.riskMultiplier) for (ids, amt) in wagersAdd if amt > 0]
        if  self.addWagers:
            return temp+tempAdd
        else:
            return temp
    
    def wagers(self, race, probs, permuteFunc, risk, minBet):
        ''' Convert probabilities into wagers based on a given risk size. '''
        '''minBet[0] : minimumTicketSize'''
        '''minbet[1]: minimumBetSize'''
        if(risk == 0):
            return []
        #assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
            #print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
            return []     
        minTicketSize=minBet[0]
        minBetSize=minBet[1]
        probSeries=pd.Series(probs).order(ascending=False)
        numPPScaling=np.sqrt(pd.Series({runner.id:len(runner.__dict__.get('pastPerformances').values())+1 for runner in race.runners}))
        probs=probSeries*numPPScaling
        probs=probs/probs.sum()
        probs=probs.to_dict()
        combProbs = permuteFunc(race.bettableRunners(), probs)
        topSpeed=[]
        speedScores=pd.Series(self.ppScoreSpeed(race))
        maxSpeed=speedScores.idxmax()
        topSpeed.append(maxSpeed)
        mauryScores=pd.Series({r.id:r.__dict__.get('score',0.0) for r in race.bettableRunners()}).order(ascending=False)
        if (self.filterBadRunners) & (race.getBadRunnersCount()>0):
            print ("No Wagers. Num badrunners is :",race.getBadRunnersCount())
            print ("Min score;",mauryScores.min())
            return []
        numStarters=len(race.bettableRunners())
        pos1Runners=probSeries[0:self.keyHorsesMap[len(probs.values())]].index.values
        pos2Runners=probSeries[0:numStarters-3].index.values
        pos3Runners=mauryScores[0:numStarters-2].index.values
#        if self.filterTopSpeedPos1==True:
#            idealWagers = {key : value*risk for (key, value) in combProbs.iteritems() if (key[0] in mauryScores[0:self.keyHorsesMap[numStarters]]) and (key[0] not in topSpeed)}
#        else:
        #print(permuteFunc.__name__)
        if permuteFunc.__name__=="exactaProbs":
            idealWagers = {key : value*risk for (key, value) in combProbs.iteritems() if ((key[0] in pos1Runners) and (key[1] in pos2Runners))} # mauryScores[0:self.keyHorsesMap[numStarters]])}
        elif permuteFunc.__name__=="trifectaProbs":
            idealWagers = {key : value*risk for (key, value) in combProbs.iteritems() if (key[0] in pos1Runners) and ((key[1] in pos2Runners) and (key[2] in pos3Runners))} # mauryScores[0:self.keyHorsesMap[numStarters]])}
        elif permuteFunc.__name__=="superfectaProbs":
            idealWagers = {key : value*risk for (key, value) in combProbs.iteritems() if ((key[0] in pos1Runners) and (key[1] in pos2Runners) and (key[2] in pos3Runners))} # mauryScores[0:self.keyHorsesMap[numStarters]])}
        tbWagers= self.ticketBoxer(idealWagers, minTicketSize, minBetSize)
        return tbWagers        
    def __repr__(self):
        return "(baseRisk_{!r},scaleByPoolSize_{!r})".format(self.poolSizer,str(self.scaleByPoolSize))
    def __str__(self):
        return "(baseRisk_{!r},scaleByPoolSize_{!r})".format(self.poolSizer,str(self.scaleByPoolSize))
    
class KeyhorseWagers(object):
    ''' Takes win probabilities by horse and converts them to 2, 3, and 4 way exotic bets '''
    def __init__(self, poolSizer, ticketBoxer,keyHorseProbs):
        '''poolSizer: PoolSizer object e.g. FixedPoolSizes returns an amount to risk per pool'''
        '''ticketBoxer: TicketBoxer object decides on how to construct actual tickets / wagers'''
        '''exactaProbs: defaults to standard exactaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''trifectaProbs: defaults to standard trifectaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''superfectaProbs: defaults to standard trifectaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''filterEntropy: default True, returns no wagers for races where entropy of top4 would fall into the lowest entropy bucket using TemplateWagers'''
        '''riskMultiplier: default 1.0 , applied after ticketBoxer has constructed wagers and allows multiplying risk on the set of chosen wagers'''
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer
        self.keyHorseProbs = keyHorseProbs
        self.hardMax=3000.00
    def __call__(self, race, probs, defaultRebates = None):
        probSeries=pd.Series(probs)
        if len(probSeries)<4:
            return []
        wagers = self.wagers(race, probs, self.exactaProbs, self.poolSizer(race, "Exacta"), race.minBet.get('Exacta', (2.0,2.0)))
        wagers = wagers+self.wagers(race, probs, self.trifectaProbs, self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', (2.0,2.0)))
        wagers = wagers + self.wagers(race, probs, self.superfectaProbs, self.poolSizer(race, "Superfecta"), race.minBet.get('Superfecta', (.50,2.0)))
        wagers = self.adjustWagers(wagers)
        
        temp= [(WagerKey(race.id, WagerKey.Multihorse, ids), amt*self.riskMultiplier) for (ids, amt) in wagers if amt > 0]
    def wagers(self, race, probs, permuteFunc, risk, minBet):
        ''' Convert probabilities into wagers based on a given risk size. '''
        '''minBet[0] : minimumTicketSize'''
        '''minbet[1]: minimumBetSize'''
        if(risk == 0):
            return []
        #assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
            #print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
            return []     
        minTicketSize=minBet[0]
        minBetSize=minBet[1]
        probSeries=pd.Series(probs).order(ascending=False)
        numPPScaling=np.sqrt(pd.Series({runner.id:len(runner.__dict__.get('pastPerformances').values())+1 for runner in race.runners}))
        probs=probSeries*numPPScaling
        probs=probs/probs.sum()
        probs=probs.to_dict()
        combProbs = self.keyHorseProbs(race)
        topSpeed=[]
        speedScores=pd.Series(self.ppScoreSpeed(race))
        maxSpeed=speedScores.idxmax()
        topSpeed.append(maxSpeed)
        mauryScores=pd.Series({r.id:r.__dict__.get('score',0.0) for r in race.bettableRunners()}).order(ascending=False)
        if (self.filterBadRunners) & (race.getBadRunnersCount()>0):
            print ("No Wagers. Num badrunners is :",race.getBadRunnersCount())
            print ("Min score;",mauryScores.min())
            return []
        numStarters=len(race.bettableRunners())
        pos1Runners=probSeries[0:self.keyHorsesMap[len(probs.values())]].index.values
        pos2Runners=probSeries[0:numStarters-3].index.values
        pos3Runners=mauryScores[0:numStarters-2].index.values
        if permuteFunc.__name__=="exactaProbs":
            idealWagers = {key : value*risk for (key, value) in combProbs.iteritems() if ((key[0] in pos1Runners) and (key[1] in pos2Runners))} # mauryScores[0:self.keyHorsesMap[numStarters]])}
        elif permuteFunc.__name__=="trifectaProbs":
            idealWagers = {key : value*risk for (key, value) in combProbs.iteritems() if (key[0] in pos1Runners) and ((key[1] in pos2Runners) and (key[2] in pos3Runners))} # mauryScores[0:self.keyHorsesMap[numStarters]])}
        elif permuteFunc.__name__=="superfectaProbs":
            idealWagers = {key : value*risk for (key, value) in combProbs.iteritems() if ((key[0] in pos1Runners) and (key[1] in pos2Runners) and (key[2] in pos3Runners))} # mauryScores[0:self.keyHorsesMap[numStarters]])}
        tbWagers= self.ticketBoxer(idealWagers, minTicketSize, minBetSize)
        return tbWagers        
    def __repr__(self):
        return "(baseRisk_{!r},scaleByPoolSize_{!r})".format(self.poolSizer,str(self.scaleByPoolSize))
    def __str__(self):
        return "(baseRisk_{!r},scaleByPoolSize_{!r})".format(self.poolSizer,str(self.scaleByPoolSize))
    
class BayesClassProbs(object):
    '''classDict,testDict,primaryAttr='HDWPSRRating',seconday='mauryScore'''
    def __init__(self,classDict,primary='HDWPSRRating',secondary='score'):
        self.classDict = classDict
        self.bins_purse=[0,10000,20000,30000,50000,100000]
        self.bins_distance=[0,1760,10000]     
    def getClassIndex(self,race):
        track=race.track
        maiden=race.maiden
        purse_bin=pd.cut([race.purse],self.bins_purse,right=False).__array__()[0]
        distance_bin=pd.cut([race.distance],self.bins_distance,right=False).__array__()[0]
        index_class=(track,maiden,purse_bin,distance_bin)
        return index_class
    def getClassDataFrame(self,race):
        idx=self.getClassIndex(race)
        df=self.classDict.get(idx)
        return df
    def getClassDataFrameZMap(self,race,attr,drop=True):
        '''returns class dataframe excluding race'''
        df=self.getClassDataFrame(race)
        df_drop=df[df.raceId!=race.id]
        seriesAttr=df_drop[attr]
        df_drop['zmap_'+attr]=ss.zmap(seriesAttr,compare=seriesAttr)
        qcutcolname='qcutzmap_'+attr
        df_drop[qcutcolname]=pd.qcut(df_drop['zmap_'+attr],4,labels=False,retbins=True)[0]
        return df_drop
    def getCdf(self,race,attr,pos=1):
        df=self.getClassDataFrame(race)
        dfpos=df[df.officialFinishPosition<(pos+1)]
        scores=dfpos[attr].values
        cdf=MakeCdfFromList(scores)
        return cdf
    def getClassPmfZmap(self,race,attr,compare=None):
        '''returns the Cdf for finish as a function of zmap(attr) from the race class based on attr'''
        dfzmapclass=self.getClassDataFrameZMap(race,attr)
        qcutcolname='qcutzmap_'+attr        
        dfzmapclass[qcutcolname]=pd.qcut(dfzmapclass['zmap_'+attr],4,labels=False,retbins=True)[0]
        qbins=pd.qcut(dfzmapclass[attr],4,labels=False,retbins=True)[1]
        quantileCDFs=dfzmapclass['officialFinishPosition'].groupby(dfzmapclass[qcutcolname]).apply(lambda x:MakeCdfFromList(x))
        quantilePMFs=quantileCDFs.apply(lambda x:x.MakePmf())
        return (quantilePMFs,qbins)    
    def getRaceZScore(self,race,attr,compare=None):
        '''computes the zscore of a runner relative to the runners in THIS race
        race=race, attr='HDWPSRRating',compare=None. This tells us how 
        strong the score is relative to other horses in this race historically'''
        race_series_primary=race.getRunnerAttrs(attr)
        race_series_median=race_series_primary.dropna().median()
        race_series_primary[race_series_primary<-99]=race_series_median
        zscore_race_primary=pd.Series(ss.zscore(race_series_primary))                               
        return (zscore_race_primary)
    def getRaceZMap(self,race,attr,compare=None):
        '''computes the zscore of a runner relative to the class of race
        race=race, attr='HDWPSRRating',compare=None. This tells us how 
        string the score is relative to other horses in the class historicallyd'''
        dftest=self.getClassDataFrameZMap(race,attr)
        index_class=self.getClassIndex(race)
        df_class=self.classDict[index_class]
        class_median_primary=df_class[attr].median()
        race_series_primary=race.getRunnerAttrs(attr)
        race_series_primary[race_series_primary<-99]=class_median_primary
        (classPmfZmap,qbins)=self.getClassPmfZmap(race,attr)
        runner_quantiles_class=pd.Series(pd.cut(race_series_primary,qbins,labels=False),index=race_series_primary.index)       
        return pd.Series(ss.zmap(race_series_primary,compare(dftest[attr])))
    def getRaceQuantiles(self,race,attr,compare=None):
        dftest=self.getClassDataFrameZMap(race,attr)
        index_class=self.getClassIndex(race)
        df_class=self.classDict[index_class]
        class_median_primary=df_class[attr].median()
        race_series_primary=race.getRunnerAttrs(attr)
        race_series_primary[race_series_primary<-99]=class_median_primary
        (classPmfZmap,qbins)=self.getClassPmfZmap(race,attr)
        runner_quantiles_class=pd.Series(pd.cut(race_series_primary,qbins,labels=False),index=race_series_primary.index)
        return runner_quantiles_class
    def getRacePmf(self,race,attr,compare=None):
        '''computes the zscore of a runner relative to the class of race
        race=race, attr='HDWPSRRating',compare=None. This tells us how 
        string the score is relative to other horses in the class historically'''
        dftest=self.getClassDataFrameZMap(race,attr)
        try:
            index_class=self.getClassIndex(race)
        except:
            print ('no classindex for',race.id)
        df_class=self.classDict[index_class]
        class_median_primary=df_class[attr].median()
        race_series_primary=race.getRunnerAttrs(attr)
        race_series_primary[race_series_primary<-99]=class_median_primary
        (classPmfZmap,qbins)=self.getClassPmfZmap(race,attr)
        runner_quantiles_class=pd.Series(pd.cut(race_series_primary,qbins,labels=False),index=race_series_primary.index)       
        return runner_quantiles_class.map(classPmfZmap)
    def getRaceJointPmf(self,race,attrlist,compare=None):
        '''computes the zscore of a runner relative to the class of race
        race=race, attr='HDWPSRRating',compare=None. This tells us how 
        string the score is relative to other horses in the class historically'''
        pmfseries=pd.Series({a:self.getRacePmf(race,a) for a in attrlist})
        return pmfseries
    def getTrifectaScores(self,race,attr,weights={'pos1':3,'pos2':2,'pos3':1}):
        '''returns a score for the trifecta combination based on quantiles for the attr''' 
        scores=self.getRaceQuantiles(race, attr)+1
        series_weights=pd.Series(weights)
        triCombList=[TriComb(c[0],c[1],c[2]) for c in itertools.permutations(scores.keys(),3)]
        dd=defaultdict()
        for c in triCombList:
            dd[c]=scores.get(c.pos1,0.0)*series_weights.get('pos1')+scores.get(c.pos2,0.0)*series_weights.get('pos2')+scores.get(c.pos3)*series_weights.get('pos3')      
        return pd.Series(dd)
    def getTrifectaOverlays(self,race,attr_proprietary,attr_public):
        '''(race,attr_proprietary='mauryScore',attr_public='HDWPSRRating'''
        '''returns the combinations where attr_proprirtary score is higher than attr_public'''
        scores_proprietary = self.getTrifectaScores(race,attr_proprietary)
        scores_public = self.getTrifectaScores(race,attr_public)
        df=pd.DataFrame(scores_proprietary,columns=[attr_proprietary])
        df[attr_public]=scores_public
        df['diff_proprietary_public']=df[attr_proprietary]-df[attr_public]
        return df[(df.diff_proprietary_public>-1) & (df[attr_public]<24)]
    def scoreTrifecta(self,race,attr,comb,weights={'pos1':3,'pos2':2,'pos3':1}):
        '''returns the score for a combination in a race'''
        scores=self.getTrifectaScores(race,attr,weights)
        return scores.get(comb)    
    def __call__(self,race):
        pmfs1=self.getRacePmf(race,'HDWPSRRating')
        pmfs2=self.getRacePmf(race,'mauryScore')
        winprobs1=pmfs1.apply(lambda x:x.Prob(1))
        winprobs2=pmfs2.apply(lambda x:x.Prob(1))
        summodels= (winprobs1+winprobs2)
        sumprobs=summodels/summodels.sum()
        return sumprobs
        
class Wagers(MultihorseWagers):
    ''' Takes win probabilities by horse and converts them to 2, 3, and 4 way exotic bets '''
    def __init__(self, poolSizer, ticketBoxer,exactaProbs=exactaProbs,trifectaProbs=trifectaProbs,superfectaProbs=superfectaProbs,filterEntropy=True,riskMultiplier=1.0):
        '''poolSizer: PoolSizer object e.g. FixedPoolSizes returns an amount to risk per pool'''
        '''ticketBoxer: TicketBoxer object decides on how to construct actual tickets / wagers'''
        '''exactaProbs: defaults to standard exactaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''trifectaProbs: defaults to standard trifectaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''superfectaProbs: defaults to standard trifectaProbs calc but allows override e.g. if we want to assign scores rather than probs'''
        '''filterEntropy: default True, returns no wagers for races where entropy of top4 would fall into the lowest entropy bucket using TemplateWagers'''
        '''riskMultiplier: default 1.0 , applied after ticketBoxer has constructed wagers and allows multiplying risk on the set of chosen wagers'''
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer
        self.exactaProbs=exactaProbs
        self.trifectaProbs=trifectaProbs
        self.superfectaProbs=superfectaProbs
        self.filterEntropy=filterEntropy
        self.riskMultiplier=riskMultiplier         
    def __repr__(self):
        return "({!r},{!r},{!r},{!r})".format(self.poolSizer,self.trifectaProbs.__name__,str(self.filterEntropy),str(self.riskMultiplier))
    def __str__(self):
        return "({!r},{!r},{!r},{!r})".format(self.poolSizer,self.trifectaProbs.__name__,str(self.filterEntropy),str(self.riskMultiplier))

def underlayProbs(race,probs,pmUnderlay=NamedOddsModel('morningLine')): 
    '''avoids combinations with matching morningLineRanks'''
    probsUnderlay=pd.Series(pmUnderlay(race))
    probsModel=pd.Series(probs)
    favUnderlay=probsUnderlay[probsUnderlay.rank(ascending=False)<2]
    favModel=probsModel[probsModel.rank(ascending=False)<2]
    favUnderlayId=favUnderlay.index.values
    favModelId=favModel.index.values
    if favUnderlayId==favModelId:
        return True
    else:
        return False
    
def underlayConditions(race,probs):
    probsModel=pd.Series(probs)
    probFavorite=probsModel[probsModel.rank(ascending=False)<2]
    idFavorite=probFavorite.index.values[0]
    
    return None
        
    
def rankMaidenClaiming(race):
    '''Uses ranked runner factors to create a rankDifferentialMatrix'''
    runnerSeries=pd.Series({r.id:r for r in race.bettableRunners()})
    classThisRace=race.raceClassification
    classLast=runnerSeries.apply(lambda x: x.getClassLast())
    classPath=runnerSeries.apply(lambda x: x.getRaceClassPath())
    racesAtDist=runnerSeries.apply(lambda x: x.getRacesAtMinDistance(race.distance))
    return racesAtDist
    
    
#     earlySpeedRank
#     raceSimilarityRank
#     layoffRank
#     lastRaceBetter
#     lastRacePurse
#     tripSentiment
#     
#     avePurseLast3=runnerSeries.apply(lambda x: x.avePurseLastN(3)).rank(ascending=False)
#     numRacesAtDistLast3=runnerSeries.apply(lambda x: x.racesAtDist(3)).rank(ascending=False)
#     

  
# class RankDifferentialMatrix():
#     def __init__(self,race,startDate):
#         self.race=race
#         self.startDate=startDate
#         
#     def getZDiffMatrixLife(self,race,lookbackWindow=None,ppScore):
#         runners=pd.Series({r.id:r for r in self.race.bettableRunners()})
#         idxRunner=runners.index
#         pairs=list(itertools.permutations(idxRunner,2))
#         diffMatrix=pd.DataFrame(index=idxRunner,columns=idxRunner)
#         for p in pairs:
#             runner1=p[0]
#             runner2=p[1]
#             diffMatrix.ix[runner1,runner2]=self.getRankDiffLife(race, runner1, runner2,lookbackWindow,ppScore)
#             diffMatrix=diffMatrix.fillna(0.0)
#         return diffMatrix
#     def getRankDiffLife(self,race,runner1,runner2,lookbackWindow,ppScore):
#         starts1=dfRunner.ix[runner1,'startsLife']
#         starts2=dfRunner.ix[runner2,'startsLife']
#         finish1=dfRunner.ix[runner1,'finishAtPosLife']
#         finish2=dfRunner.ix[runner2,'finishAtPosLife']
#         loss1=starts1-finish1
#         loss2=starts2-finish2
#         oddsratio, pvalue = ss.fisher_exact([[finish1, finish2], [loss1, loss2]])

            
class OverlayTrainer(object):
    def __init__(self,dfTrainer,startDate=date(2010,12,31)):
        self.startDate=startDate 
        self.dfTrainer=dfTrainer[dfTrainer.date>startDate]
        self.pmEvenOdds=EvenOdds()
        self.pmFinal=NamedOddsModel('finalToteOdds')
        self.pmMorningLine=NamedOddsModel('morningLine')
        self.posWeightsDict={1:1.2,2:1.1,3:1.0}
        
    def getRunnerTrainerIndex(self,race):
        date=race.date
        runners=race.bettableRunners()
        IDXRUNNERS=pd.Series({runner.id:runner.trainer for runner in runners})
        IDXTRAINERS=pd.Series({runner.trainer:runner.id for runner in runners})
        idxRunnerTrainer=pd.MultiIndex.from_arrays([IDXRUNNERS.index.values,IDXTRAINERS.index.values])
        idxRunnerTrainer.names=(['runnerId','trainer_name'])
        return idxRunnerTrainer
    def getStartsOverWindow(self,race,lookbackWindow=None):
        idx=self.getRunnerTrainerIndex(race)
        ofpCount=self.getFinishPosOverWindow(race,lookbackWindow)
        starts=ofpCount.groupby(level=0).apply(sum)
        return starts  
    def getFinishPosOverWindow(self,race,lookbackWindow=None):
        date=race.date
        if lookbackWindow == None:
            dateWindowStart=self.startDate
        else:
            dateWindowStart=date-timedelta(days=lookbackWindow)
        runners=race.bettableRunners()
        TRAINERS=pd.Series({runner.id:runner.trainer for runner in runners})
        idx=self.getRunnerTrainerIndex(race)#pd.Series({runner.trainer:runner.id for runner in runners})
        dfRaceTrainers=self.dfTrainer[self.dfTrainer['trainer_name'].isin(TRAINERS)]
        dfRaceTrainersPrior=dfRaceTrainers[(dfRaceTrainers['date']<date) & (dfRaceTrainers['date']>dateWindowStart)]
        by_trainerOfficalFinish=dfRaceTrainersPrior.groupby('trainer_name')['official_finish_position']
        ofpFreq=by_trainerOfficalFinish.apply(lambda x:x.value_counts())
        return ofpFreq
    def getWeightedFinishScoreOverWindow(self,race,lookbackWindow=None):
        win=self.getFinishPosOverWindow(race,lookbackWindow).ix[:,1]*self.posWeightsDict[1]
        place=self.getFinishPosOverWindow(race,lookbackWindow).ix[:,2]*self.posWeightsDict[2]
        show=self.getFinishPosOverWindow(race,lookbackWindow).ix[:,3]*self.posWeightsDict[3]
        wtdSum=win+place+show
        return wtdSum
#         dfWins=dfRaceTrainers[dfRaceTrainers['official_finish_position']==1]
#         dfWinsPrior=dfWins[dfWins['date']<date]
#         winSeries=dfWinsPrior.groupby('trainer_name').official_finish_position.count()
#         runnersByTrainer=IDXTRAINERS[winSeries]
#         print runnersByTrainer
#         idxRunnerTrainer=pd.MultiIndex.from_arrays([runnersByTrainer.index.values,winSeries.index.values])
#         winSeries.index=idxRunnerTrainer
    def getZDiffLife(self,race,runner1,runner2,lookbackWindow=None,finishPos=1):
        df=self.getTrainerDF(race,lookbackWindow,finishPos)
        dfRunner=df.reset_index(level=1,drop=True)
        
        starts1=dfRunner.ix[runner1,'startsLife']
        starts2=dfRunner.ix[runner2,'startsLife']
        finish1=dfRunner.ix[runner1,'finishAtPosLife']
        finish2=dfRunner.ix[runner2,'finishAtPosLife']
        loss1=starts1-finish1
        loss2=starts2-finish2
        oddsratio, pvalue = ss.fisher_exact([[finish1, finish2], [loss1, loss2]])

#         pct1=dfRunner.ix[runner1,'pctFinishAtPosLife']
#         pct2=dfRunner.ix[runner2,'pctFinishAtPosLife']
#         pooledWins=finish1+finish2
#         pooledStarts=starts1+starts2
#         pooledNonWin=pooledStarts-pooledWins
#         sumN1N2=(1.0/starts1+1.0/starts2)
#         pctP=float(pooledWins)/float(pooledStarts)
#         denom=math.sqrt(pctP*(1-pctP)*sumN1N2)
#         TS=(pct1-pct2)/denom
        return pvalue
    def getZDiffWindow(self,race,runner1,runner2,lookbackWindow=None,finishPos=1):
        df=self.getTrainerDF(race,lookbackWindow,finishPos)
        dfRunner=df.reset_index(level=1,drop=True)
        
        starts1=dfRunner.ix[runner1,'startsWindow']
        starts2=dfRunner.ix[runner2,'startsWindow']
        finish1=dfRunner.ix[runner1,'finishAtPosWindow']
        finish2=dfRunner.ix[runner2,'finishAtPosWindow']
        loss1=starts1-finish1
        loss2=starts2-finish2
        oddsratio, pvalue = ss.fisher_exact([[finish1, finish2], [loss1, loss2]])
        
#         pct1=dfRunner.ix[runner1,'pctFinishAtPosWindow']
#         pct2=dfRunner.ix[runner2,'pctFinishAtPosWindow']
#         pooledWins=finish1+finish2
#         pooledStarts=starts1+starts2
#         pooledNonWin=pooledStarts-pooledWins
#         sumN1N2=(1.0/starts1+1.0/starts2)
#         pctP=float(pooledWins)/float(pooledStarts)
#         denom=math.sqrt(pctP*(1-pctP)*sumN1N2)
#         TS=(pct1-pct2)/denom
        return pvalue        
    def getZDiffMatrixLife(self,race,lookbackWindow=None,finishPosition=1):
        idxRunnerTrainer=self.getRunnerTrainerIndex(race)
        idxRunner=idxRunnerTrainer.get_level_values(level=0)
        pairs=list(itertools.permutations(idxRunner,2))
        diffMatrix=pd.DataFrame(index=idxRunner,columns=idxRunner)
        for p in pairs:
            runner1=p[0]
            runner2=p[1]
            diffMatrix.ix[runner1,runner2]=self.getZDiffLife(race, runner1, runner2,lookbackWindow,finishPosition)
        diffMatrix=diffMatrix.fillna(0.0)
        return diffMatrix
    def getZScorePValsLife(self,race,lookbackWindow=None,finishPosition=1):
        dm=self.getZDiffMatrixLife(race,lookbackWindow,finishPosition)
        sumPVals=dm.apply(sum,1)
        zScores=-1.0*ss.zscore(sumPVals)
        return zScores
    def getZDiffMatrixWindow(self,race,lookbackWindow,finishPosition):
        idxRunnerTrainer=self.getRunnerTrainerIndex(race)
        idxRunner=idxRunnerTrainer.get_level_values(level=0)
        pairs=list(itertools.permutations(idxRunner,2))
        diffMatrix=pd.DataFrame(index=idxRunner,columns=idxRunner)
        for p in pairs:
            runner1=p[0]
            runner2=p[1]
            diffMatrix.ix[runner1,runner2]=self.getZDiffWindow(race, runner1, runner2,lookbackWindow,finishPosition)
        return diffMatrix.fillna(0.0)
    def getZScorePValsWindow(self,race,lookbackWindow=None,finishPosition=1):
        dm=self.getZDiffMatrixWindow(race,lookbackWindow,finishPosition)
        sumPVals=dm.apply(sum,1)
        zScores=-1.0*ss.zscore(sumPVals)
        return zScores
    def getPctOverWindow(self,race,lookbackWindow=None,finishPos=1):
        starts=self.getStartsOverWindow(race, lookbackWindow)
        finishPosCount=self.getFinishPosOverWindow(race,lookbackWindow)
        pctFinish=finishPosCount.ix[:,finishPos]/starts
        return pctFinish.fillna(0.0)
    def getWeightedPctOverWindow(self,race,lookbackWindow):
        starts=self.getStartsOverWindow(race, lookbackWindow)
        weightedScoreOverWindow=self.getWeightedFinishScoreOverWindow(race,lookbackWindow)
        
        
    def fisherOverWindow(self,race,lookbackWindow=None,finishPos=1):
        starts=self.getStartsOverWindow(race,lookbackWindow)
        finishPct=self.getPctOverWindow(race,lookbackWindow,finishPos)
        fisher=starts/(finishPct*(1-finishPct))
        fisher[fisher==np.inf]=1.0
        logFisher=np.log(fisher)
        return logFisher        
    def getTrainerDF(self,race,lookbackWindow,finishPos):
        startsLife=self.getStartsOverWindow(race, lookbackWindow=None)
        startsWindow=self.getStartsOverWindow(race,lookbackWindow=lookbackWindow)
        df=pd.DataFrame(startsLife,columns=['startsLife'])
        df['startsWindow']=startsWindow        
        df['finishAtPosLife']=self.getFinishPosOverWindow(race).ix[:,finishPos]
        df['finishAtPosWindow']=self.getFinishPosOverWindow(race,lookbackWindow).ix[:,finishPos]
        df['pctFinishAtPosLife']=self.getPctOverWindow(race,lookbackWindow=None,finishPos=finishPos)
        df['pctFinishAtPosWindow']=self.getPctOverWindow(race,lookbackWindow,finishPos)
        df['wtdFinishLife']=self.getWeightedFinishScoreOverWindow(race,lookbackWindow=None)
        df['wtdFinishWindow']=self.getWeightedFinishScoreOverWindow(race,lookbackWindow)
        df['wtdPctLife']=df['wtdFinishLife']/df['startsLife']
        df['wtdPctWindow']=df['wtdFinishWindow']/df['startsWindow']
        df['informationLife']=self.fisherOverWindow(race,lookbackWindow=None,finishPos=finishPos)
        df['informationWindow']=self.fisherOverWindow(race,lookbackWindow,finishPos)
        df=df.fillna(0)
        probsFinal=pd.Series(self.pmFinal(race))
        probsMorningLine=pd.Series(self.pmMorningLine(race))
        df.index=self.getRunnerTrainerIndex(race)
        probsFinal.index=df.index
        probsMorningLine.index=df.index
        df['probsMorningLine']=probsMorningLine
        df['probsFinal']=probsFinal
        
        return df
        
    
#         getFinishPosOverWindow(testRace2011).ix[:,1]/starts
        
def runnerPurseClaimScore(race):
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
    return seriesClaimValue  
def HDWProjectedSpeedScore(race):
    runners=race.bettableRunners()
    try:
        seriesSpeed=pd.Series({r.id:r.__dict__.get('HDWPSRRating',0.0)for r in runners})
    except:
        return None
#     try:
#         seriesLastFinish=pd.Series({r.id:r.getFinishPos()[0] for r in runners})
#     except:
#         return None
    return seriesSpeed  
def PositionSpeedScore(race):
    runners=race.bettableRunners()
    try:
        seriesSpeed=pd.Series({r.id:r.__dict__.get('HDWPSRRating',0.0)for r in runners})
    except:
        return None
    
class SpeedClassWagers(object):
    def __init__(self,poolSizer,ticketBoxer):
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer
        self.ppSpeed=PastPerformanceScore('HDWSpeedRating',meanBestN,numRaces=5)  
    def __call__(self,race,probs,defaultRebates=None): 
        wagers= self.wagers(race, probs, exactaProbs, self.poolSizer(race, "Exacta"), race.minBet.get('Exacta', None))
        temp= [(WagerKey(race.id, WagerKey.Multihorse, ids), amt) for (ids, amt) in wagers if amt > 0]
        return temp    
    def wagers(self, race, probs, permuteFunc, risk, minBet):      
        ''' Convert probabilities into wagers based on a given risk size. '''
        defaultMinBet=(1.0,1.0)
        #HDWSpeedParForClassLevel
        SPEEDPAR_FOR_RACE=race.__dict__.get('HDWSpeedParForClassLevel',0.0)        
        
        speedACScore=pd.Series(self.ppSpeed(race))
        speedACParAdjusted=speedACScore-SPEEDPAR_FOR_RACE
        
        seriesSpeedHDW=HDWProjectedSpeedScore(race)
        speedHDWParAdjusted=seriesSpeedHDW-SPEEDPAR_FOR_RACE
        
        q3SpeedACParAdjusted=pd.Series(pd.qcut(speedACParAdjusted,3,labels=False),index=speedACScore.index)
        q3SpeedHDWParAdjusted=pd.Series(pd.qcut(speedHDWParAdjusted,3,labels=False),index=seriesSpeedHDW.index)
        
        topSpeedAC=q3SpeedACParAdjusted[q3SpeedACParAdjusted==2].index.values
        midSpeedAC=q3SpeedACParAdjusted[q3SpeedACParAdjusted==1].index.values
        lowSpeedAC=q3SpeedACParAdjusted[q3SpeedACParAdjusted==0].index.values

        topSpeedHDW=q3SpeedHDWParAdjusted[q3SpeedHDWParAdjusted==2].index.values
        midSpeedHDW=q3SpeedHDWParAdjusted[q3SpeedHDWParAdjusted==1].index.values
        lowSpeedHDW=q3SpeedHDWParAdjusted[q3SpeedHDWParAdjusted==0].index.values
        
        topSpeed=pd.Series(list(set(topSpeedAC).union(set(topSpeedHDW)))).values
        midSpeed=pd.Series(list(set(midSpeedAC).union(set(midSpeedHDW)))).values
        lowSpeed=pd.Series(list(set(lowSpeedAC).union(set(lowSpeedHDW)))).values
        
        seriesPurseClaim=runnerPurseClaimScore(race)
        
        classPosition1=seriesPurseClaim[seriesPurseClaim>0.0].index.values
        classPosition2=seriesPurseClaim[seriesPurseClaim>0.0].index.values
        
        preferredPos1=pd.Series(list(set(midSpeed).union(set(classPosition1)))).values
        preferredPos2=pd.Series(list(set(midSpeed+topSpeed).union(set(classPosition1)))).values
        if(risk == 0):
            #print ("RISK is zero",race.id,permuteFunc.func_name)
            #risk=defaultRiskExacta
            return []
        # assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
            #print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}' setting minBet to defaultMinBet ".format(permuteFunc, race=race)
            # print defaultMinBet
            minBet=defaultMinBet

        combProbs = permuteFunc(race.bettableRunners(), probs)
        combProbSeries=pd.Series(combProbs).order(ascending=False)
#         if self.overlayTrainer != None:
#             trainers=self.overlayTrainer(race,1)
#             keepTrainer=trainers
        for c in combProbSeries.keys():
            if c[0] not in preferredPos1:
                combProbSeries[c]=0.0
            if c[1] not in preferredPos2:
                combProbSeries[c]=0.0  
           
        idealWagers = {key : value*risk for (key, value) in combProbSeries.iteritems()}        
        return self.ticketBoxer(idealWagers, minBet[0], minBet[1])    
    def __repr__(self):
        return "SpeedClassWagers({!r})".format(self.poolSizer)        
        

class AlphaWagers(object):
    '''Takes the wagers with cumProb < alpha of wagers for a given combination'''
    def __init__(self, poolSizer, ticketBoxer,alpha,maxNumBets,minNumBets,overlayTrainer=None,maxRisk=False,underlayTop=True):
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer  
        self.alpha=alpha
        self.maxNumBets=maxNumBets
        self.minNumBets=minNumBets
        self.maxRisk=maxRisk
        self.underlayTop=underlayTop
        self.overlayTrainer=overlayTrainer
        
 
    def __call__(self,race,probs,defaultRebates=None): 
        wagers= self.wagers(race, probs, exactaProbs, self.poolSizer(race, "Exacta"), race.minBet.get('Exacta', None),self.alpha,self.maxNumBets,self.minNumBets)
#         wagersExacta=chopExacta(self.wagerTemplate['Exacta'],wagersTemp,probs)
        wagers = wagers+self.wagers(race, probs, trifectaProbs, self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', None),self.alpha,self.maxNumBets,self.minNumBets)
#         wagersTrifecta=chopTrifecta(self.wagerTemplate['Trifecta'],wagersTemp,probs,wagersExacta)
        wagers =wagers+self.wagers(race, probs, superfectaProbs, self.poolSizer(race, "Superfecta"), race.minBet.get('Superfecta', None),self.alpha,self.maxNumBets,self.minNumBets) 
#         wagersSuperfecta=chopSuperfecta(self.wagerTemplate['Superfecta'],wagersTemp,probs,wagersExacta,wagersTrifecta)
#         wagers=wagersExacta+wagersTrifecta+wagersSuperfecta
        temp= [(WagerKey(race.id, WagerKey.Multihorse, ids), amt) for (ids, amt) in wagers if amt > 0]
        return temp 
    
    def wagers(self, race, probs, permuteFunc, risk, minBet,alpha,maxNumBets,minNumBets):
        ''' Convert probabilities into wagers based on a given risk size. '''
        defaultMinBet=(1.0,1.0)
        pseries=pd.Series(probs).order(ascending=False)
        topProbRunner=pseries.idxmax()
        numStarters=len(pseries)
        if(risk == 0):
            return []
        if minBet is None:
            minBet=defaultMinBet
        combProbs = permuteFunc(race.bettableRunners(), probs)
        combProbSeries=pd.Series(combProbs).order(ascending=False)          
        cumProbSeries=combProbSeries.cumsum()
        contenderProbs=combProbSeries[cumProbSeries<alpha]
        if len(contenderProbs)>maxNumBets:
            return []
        if len(contenderProbs)<minNumBets:
            #print("numBets less than",minNumBets)
            contenderProbs=combProbSeries.iget(range(minNumBets))
       
        if self.maxRisk==True:
            contenderProbs=contenderProbs/contenderProbs.sum()
            #print("Using max Risk:" ,risk)
#         for cp in contenderProbs.keys():
#             if cp[0] in tossPosition1:
#                 print("toss runnner",cp[0])
#                 contenderProbs[cp]=0.0
        idealWagers = {key : value*risk for (key, value) in contenderProbs.iteritems()}        
        return self.ticketBoxer(idealWagers, minBet[0], minBet[1])    
    def __repr__(self):
        return "AlphaWagers({!r})".format(self.poolSizer)   
    
class BetaWagers(object):
    '''Constructs multihorse wagers by allocating a portion of risk to wagers that didn't make it into Alpha'''
    '''Takes the top alpha% of wagers for'''
    def __init__(self, poolSizer, ticketBoxer,alpha):
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer  
        self.alpha=alpha

        
    def __call__(self,race,probs,defaultRebates=None):
        wagers= self.wagers(race, probs, self.poolSizer(race, "Exacta"),self.poolSizer(race,"Trifecta"), race.minBet.get('Exacta', None),race.minBet.get('Trifecta'),self.alpha)
#         wagersExacta=chopExacta(self.wagerTemplate['Exacta'],wagersTemp,probs)
#        wagers = wagers+self.wagers(race, probs, trifectaProbs, self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', None),self.alpha,self.minDrop)
#         wagersTrifecta=chopTrifecta(self.wagerTemplate['Trifecta'],wagersTemp,probs,wagersExacta)
#        wagers =wagers+self.wagers(race, probs, superfectaProbs, self.poolSizer(race, "Superfecta"), race.minBet.get('Superfecta', None),self.alpha,self.minDrop) 
#         wagersSuperfecta=chopSuperfecta(self.wagerTemplate['Superfecta'],wagersTemp,probs,wagersExacta,wagersTrifecta)
#         wagers=wagersExacta+wagersTrifecta+wagersSuperfecta
        temp= [(WagerKey(race.id, WagerKey.Multihorse, ids), amt) for (ids, amt) in wagers if amt > 0]
        return temp    
    
    def wagers(self, race, probs, riskExacta,riskTrifecta, minBetExacta,minBetTrifecta,alpha):
        ''' Convert probabilities into wagers based on a given risk size. '''
        if(riskExacta == 0):
            return []
        
        #assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBetExacta is None:
            #print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
            return []
        if minBetTrifecta is None:
            #print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
            return []     
        '''Compute alpha and beta probs for Exacta
        alpha probs are the high prob events beta are the remainder'''
        combProbsExacta = exactaProbs(race.bettableRunners(), probs)
        combProbSeriesExacta=pd.Series(combProbsExacta).order(ascending=False)
        cumProbSeriesExacta=combProbSeriesExacta.cumsum()
        alphaProbsExacta=combProbSeriesExacta[cumProbSeriesExacta<=alpha]
        betaProbsExacta=combProbSeriesExacta[cumProbSeriesExacta>alpha]
        
        
        '''Compute ideal wagers for exactaAlpha and exactaBeta'''
        idealWagersExactaAlpha = {key : value*riskExacta for (key, value) in alphaProbsExacta.iteritems()}
        exactaWagersAlpha= self.ticketBoxer(idealWagersExactaAlpha, minBetExacta[0], minBetExacta[1])
        
        idealWagersExactaBeta= {key : value*riskExacta for (key, value) in betaProbsExacta.iteritems()}
        exactaWagersBeta= self.ticketBoxer(idealWagersExactaBeta, minBetExacta[0], minBetExacta[1])
        
        '''Compute alpha and beta probs for Trifecta'''
        combProbsTrifecta = trifectaProbs(race.bettableRunners(), probs)
        combProbSeriesTrifecta=pd.Series(combProbsTrifecta).order(ascending=False)
        cumProbSeriesTrifecta=combProbSeriesTrifecta.cumsum()
        alphaProbsTrifecta=combProbSeriesTrifecta[cumProbSeriesTrifecta<=alpha]
        betaProbsTrifecta=combProbSeriesTrifecta[cumProbSeriesTrifecta>alpha]
        
        
        
        '''Compute ideal wagers for trifectaAlpha and trifectaBeta'''
        idealWagersTrifectaAlpha = {key : value*riskTrifecta for (key, value) in alphaProbsTrifecta.iteritems()}
        trifectaWagersAlpha= self.ticketBoxer(idealWagersTrifectaAlpha, minBetTrifecta[0], minBetTrifecta[1])
        idealWagersTrifectaBeta={key : value*riskTrifecta for (key, value) in betaProbsTrifecta.iteritems()}
        trifectaWagersBeta= self.ticketBoxer(idealWagersTrifectaBeta, minBetTrifecta[0], minBetTrifecta[1])
        
        
        return trifectaWagersBeta + exactaWagersBeta
            
    def __repr__(self):
        return "BetaWagers({!r})".format(self.poolSizer)   
class PriceHedged(object):
    ''' Takes probs from one using another pool'''        
    def __init__(self, probsTrifecta,probsExacta,probsWin):
        self.pT=probsTrifecta
        self.pE=probsExacta
        self.pW=probsWin
    def hedgeWin(self, r1):
        return None
    def hedgeExacta(self,r1,r2):
        return None
    def hedgeTrifecta(self,r1,r2,r3):
        return None
    
def optimalAllocation(probs, oddsProbs, takeout, rebate):
    d = 1- takeout
    er = [(d * prob / oddsProbs[comb] + rebate, comb) for (comb, prob) in probs.iteritems()]
    erSorted = sorted(er, reverse=True)
    includedBets = []
    probSum = 0
    oddsSum = 0
    rs = 1
    for er in erSorted:
        if er[0] >= rs:
            includedBets.append(er[1])
            probSum += probs[er[1]]
            oddsSum += oddsProbs[er[1]]
            rs = (1 - (1-rebate)*probSum)/(1 - (1-rebate)/d*oddsSum)
        else:
            break
    return { bet : probs[bet] - (rs-rebate)/(d/oddsProbs[bet]) for bet in includedBets }
            
def equalRiskPerRace(risk, allocations):
    '''Uses all the risk available for each race.  The relative allocations among bets are kept as passed in.'''
    weightedAllocations = normProbDict(allocations)
    return {key : value*risk for (key, value) in weightedAllocations.iteritems()}

def kellyRiskPerRace(risk, allocations):
    '''Bets only a fraction of the available risk per race.  Amount bet will vary depending on the strength of signal.'''
    return {key : value*risk for (key, value) in allocations.iteritems()}

def getPayoutsByType(pd):
    '''takes a payout dictionary or RacePayout and returns payouts by WagerName'''
    assert(type(pd) is dict)
    pdTypeDict={}
    for pk in pd.keys():
        pts=pd[pk].__dict__['payouts']
        payKeys=pts.keys()
        pdTypeDict[pk]={payKeys[pk].getWagerName() : (payKeys[pk].getCombinationsWithIndex(),pts[payKeys[pk]]) for pk in range(0,len(payKeys))}
    return pdTypeDict
def getPayoutsByTypeForRace(racePayout):
    '''takes a RacePayout and returns payouts by type'''
    pdTypeDict={}
    pts=racePayout.__dict__['payouts']
    payKeys=pts.keys()
    pdTypeDict={payKeys[pk].getWagerName() : (payKeys[pk],pts[payKeys[pk]]) for pk in range(0,len(payKeys))}
    return pdTypeDict
def getOrderedWagerProbs(race,pm,wagerType,dLambda=1.0,dRho=1.0,dTau=1.0):
    if wagerType=='Win':
        return getOrderedWins(race,pm)
    elif wagerType=='Exacta':
        return getOrderedExactas(race,pm)
    elif wagerType == 'Trifecta':
        return getOrderedTrifectas(race,pm)
    elif wagerType =='Superfecta':
        return getOrderedSuperfectas(race,pm)
    else:
        return "No such wager"
def getEntropyVector(probs):
    base=len(probs)
    return -1.0*np.log(probs)/np.log(base)*probs        
def getOrderedWins(race,pm):
    '''takes a payoutDict and probModel and returns an ordered set of combinations'''
    win=pm(race)
    return pd.Series(win).order(ascending=False)
def getOrderedExactas(race,pm):
    '''takes a payoutDict and probModel and returns an ordered set of combinations'''
    exa=exactaProbs(race.bettableRunners(),pm(race))
    return pd.Series(exa).order(ascending=False)
def getOrderedTrifectas(race,pm):
    '''takes a payoutDict and probModel and returns an ordered set of combinations'''
    try:
        tri=trifectaProbs(race.bettableRunners(),pm(race))
        return pd.Series(tri).order(ascending=False)
    except:
        print "no triprobs: ",race.id
        return ("no tri")
    
def getOrderedSuperfectas(race,pm):
    '''takes a payoutDict and probModel and returns an ordered set of combinations'''
    sup=superfectaProbs(race.bettableRunners(),pm(race))
    return pd.Series(sup).order(ascending=False)
def getBreakevensForWagerType(rd,pdict,pm,wagerType):
    '''Takes a payoutDict and probModel and returns multipool hedged payoffs'''
    paysByType=getPayoutsByType(pdict)  # Returns payouts by type 
    breakevenDict={}
    paysByType=getPayoutsByType(pdict)  # Returns payouts by type 
    breakevenDict={}
    for payRace in pdict.keys():
        payouts=pdict[payRace].__dict__
        race=rd[payRace]      
        numStarters=payouts['numStarters']
        try:
            paysWagerType=paysByType[payRace][wagerType][1]
            #print paysWagerType
        except:
            #print("no wager for ", wagerType, race)
            return {}
        ticketWagerType=paysByType[payRace][wagerType][0].getCombinations()[0]
        orderedProbs=getOrderedWagerProbs(race, pm, wagerType)
        ixTicket=orderedProbs.index     
        breakevenDataDict={}
        try:
            stopPos=ixTicket.get_loc(ticketWagerType)
        except:
            print "cant find stopPos"
            stopPos=len(orderedProbs)-1
        stopPosRunner=int(stopPos+1)         
        minBets=rd[payRace].__dict__['minBet']
        minBets['Win']=(2.0,2.0)
        minBetWager=2.0
        minTicketWager=2.0
        try:
            minBetWager=minBets[wagerType][0]
            minTicketWager=minBets[wagerType][1]
            
        except:
            minBetWager=2.0
            minTicketWager=2.0
        effectiveMinBet = max(minTicketWager,minBetWager)
        numCombinations =len(orderedProbs)
        orderedEntropy=ss.entropy(orderedProbs,base=len(orderedProbs))
        effectiveCombinations=math.ceil(orderedEntropy*numCombinations)
        costToCover=(stopPos+1)*minTicketWager
        cumProbVector=orderedProbs.cumsum()
        cs=cumProbVector[cumProbVector<.5]
        stop50=len(cs)
        cumProb=orderedProbs.cumsum()[stopPos]
        prob=orderedProbs[stopPos]       
        netReturn=paysWagerType-costToCover  
        indexWagers=pd.Index(['Win','Exacta','Trifecta','Superfecta'])
        dataDict={}
        dataDict['numPossible']=numCombinations
        dataDict['pm_stopPos']=stopPosRunner
        dataDict['pm_stop50']=stop50
        dataDict['payout']=paysWagerType
        dataDict['pm_probRunner']=prob
        dataDict['pm_cumProb']=cumProb
        dataDict['pm_entropy']=orderedEntropy
        dataDict['minBet']=effectiveMinBet
        dataDict['pm_costToCover']=costToCover
        dataDict['pm_netReturn']=netReturn
        dataDict['pm_coverRatio']=float(stopPosRunner)/float(numCombinations)
        
        breakevenDict[payRace] = pd.Series(dataDict)
    return pd.DataFrame(breakevenDict).transpose()  
            
class OptimalEdgeMultihorseWagers(object):
    ''' Takes win probabilities by horse and converts them to 2, 3, and 4 way exotic bets '''
    def __init__(self, oddsModel, risker, poolSizer, ticketBoxer,exactaProbs=exactaProbs,trifectaProbs=trifectaProbs,defaultTakeout = None):
        self.oddsModel= oddsModel
        self.poolSizer = poolSizer
        self.ticketBoxer = ticketBoxer
        self.risker = risker
        self.defaultTakeout = defaultTakeout
        self.exactaProbs=exactaProbs
        self.trifectaProbs=trifectaProbs
    def __call__(self, race, probs, simParam):
        #checkProbDict(probs)
        # Compute probabilities of winning based on a particular odds model 
        oddsProbs = self.oddsModel(race)
        if self.defaultTakeout is None:
            tt = self.oddsModel.getTrackTake(race)
        else:
            tt = self.defaultTakeout
        defaultRebates = simParam.defaultRebates
        wagers = self.wagers(race, probs, oddsProbs, exactaProbs, self.poolSizer(race, "Exacta"), race.minBet.get('Exacta', None), self.getRebate(race, defaultRebates, 'exactaRebatePct', 'Exacta'), tt)
        wagers = wagers + self.wagers(race, probs, oddsProbs, trifectaProbs, self.poolSizer(race, "Trifecta"), race.minBet.get('Trifecta', None), self.getRebate(race, defaultRebates, 'trifectaRebatePct', 'Trifecta'), tt)
        wagers = wagers + self.wagers(race, probs, oddsProbs, superfectaProbs, self.poolSizer(race, "Superfecta"), race.minBet.get('Superfecta', None), self.getRebate(race, defaultRebates, 'superfectaRebatePct', 'Superfecta'), tt) 
        return [(WagerKey(race.id, WagerKey.Multihorse, ids), amt) for (ids, amt) in wagers if amt > 0]
    def wagers(self, race, probs, oddsProbs, permuteFunc, risk, minBet, rebate, trackTake):
        '''Convert probabilities into wagers based on a given risk size.'''
        if(risk == 0):
            return []
        #assert minBet != 0, "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
        if minBet is None:
            #print "Min bet wasn't set for {.__name__} in  {race.id}.  Bets were: '{race.betSummary}'".format(permuteFunc, race=race)
            return []
        combProbs = permuteFunc(race.bettableRunners(), probs)
        checkProbDict(combProbs)
        #print(combProbs)
        combOddsProbs = permuteFunc(race.bettableRunners(), oddsProbs)
       # print(combOddsProbs)
        checkProbDict(combOddsProbs)
        idealAllocations = optimalAllocation(combProbs, combOddsProbs, rebate, trackTake)
        idealWagers = self.risker(risk, idealAllocations)
        #self.printWagers(idealWagers, {key : value*risk for (key, value) in combProbs.iteritems()})
        return self.ticketBoxer(idealWagers, minBet[0], minBet[1])    
   

    def getRebate(self, race, defaultRebates, attr, name):
        if hasattr(race.trackObj, attr):
            return getattr(race.trackObj, attr)
        else:
            return defaultRebates[name]
        
    
    
    def printWagers(self, wagers1, wagers2):
        keys = set(wagers1.keys()).union(set(wagers2.keys()))
        sum1 = 0
        sum2 = 0
        for key in keys:
            sum1 += wagers1.get(key, 0)
            sum2 += wagers2.get(key, 0)
            #print("{}\t{}\t{}".format(key, wagers1.get(key, "-"), wagers2.get(key, "-")))
        #print("{} {}".format(sum1, sum2))
    def __repr__(self):
        return "OptimalEdgeMultihorseWagers({!r},{!r},{!r})".format(self.oddsModel,self.poolSizer,self.risker.__name__)
    
def stoppingPointForWagerType(probs,racePayout,wagerType):
    
    try:
        if wagerType=='Win':
            wk=getPayoutsByTypeForRace(racePayout)[wagerType][0].getCombinations()[0][0]
        else:
            wk=getPayoutsByTypeForRace(racePayout)[wagerType][0].getCombinations()[0]
    except:
        return None
    probs=pd.Series(probs).order(ascending=False)
    cumProbs=probs.cumsum()
    try:
        ixStop=cumProbs.index.get_loc(wk)
    except:
        print('ixStop',wk,racePayout)
        ixStop=None
    return ixStop     
def stoppingPoint(probs,wkWin):
    '''Takes a prob series and orders it highest to lowest finding the stopping point for a combination'''
    '''assumes receiving a tuple of runnerIds ('PHA_20110101_10_1','PHA_20110101_2') e.g.'''
    comb=wkWin
    probs=pd.Series(probs).order(ascending=False)
    cumProbs=probs.cumsum()
    ixStop=cumProbs.index.get_loc(comb)
    return ixStop
def profileProbModels(rm,racePayout,df,wagerType='Exacta'):
    race =rm.race
    
    wk=getPayoutsByTypeForRace(racePayout)[wagerType]
    winningComb=getPayoutsByTypeForRace(racePayout)[wagerType][0].getCombinations()[0]
   # orderedExaDict={s:getOrderedExactas(srd[race],pmDict[s]) for s in stratNames}     
    df['combs']=df.runnerIds.apply(lambda x: (x.split('/')[0],x.split('/')[1]))
    df['payouts']=df['combs'].apply(lambda x: wk[1] if wk[0].matches(x) else 0) 
    df['rebatePct']=race.__dict__['rebatePcts']['Exacta']
    df['return']=df['wagerAmount']*df['payouts']-df['payouts']+df['rebatePct']*df['wagerAmount']
    df.index=df.combs
      
    #probsDict={s:df['combs'].apply(lambda x:orderedExaDict[s][x]) for s in stratNames}
 
    probsExactaML=getOrderedExactas(race, rm.pmMorningLine)
    probsExactaFinal=getOrderedExactas(race, rm.pmFinal)
    probsExactaPace=getOrderedExactas(race, rm.pmPace)
    probsExactaPower=getOrderedExactas(race, rm.pmPower)
    df['probsML']=df['combs'].apply(lambda x: probsExactaML[x])
    df['probsFinal']=df['combs'].apply(lambda x: probsExactaFinal[x])
    df['probsPace']=df['combs'].apply(lambda x: probsExactaPace[x])
    df['probsPower']=df['combs'].apply(lambda x: probsExactaPower[x])
     
    df['entropyML']=ss.entropy(probsExactaML,base=len(probsExactaML))
    df['entropyFinal']=ss.entropy(probsExactaFinal,base=len(probsExactaFinal)) 
    df['entropyPace']=ss.entropy(probsExactaPace,base=len(probsExactaPace)) 
    df['entropyPower']= ss.entropy(probsExactaPower,base=len(probsExactaPower))    
      
    df['ixML']=stoppingPoint(probsExactaML,winningComb)
    df['ixFinal']=stoppingPoint(probsExactaFinal,winningComb)
    df['ixPace']=stoppingPoint(probsExactaPace,winningComb)
    df['ixPower']=stoppingPoint(probsExactaPower,winningComb)  
    return df