import datetime
from dateutil.parser import parse
import itertools
import logging
import re
import string

from psycopg2 import extras

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psycopg2 as pg
from x8313 import *

import scipy.stats as ss


logger = logging.getLogger(__name__)

def exactaProbs(bettableRunners, probs):
    def _exactaProb(p1, p2):
        # The rationale here is that the probability of horse 1 coming in first is p1
        # Then the probability of horse 2 coming in second given that horse 1 wins is p2/(1-p1).
        return p1 * p2/(1-p1)
    ''' Converts individual horse win probabilities into exacta probabilties '''
    return {(r1.id, r2.id) : _exactaProb(probs[r1.id], probs[r2.id]) for (r1, r2) in itertools.permutations(bettableRunners, 2) }

def trifectaProbs(bettableRunners, probs):
    def _trifectaProb(p1, p2, p3):
        return p1 * p2/(1-p1) * p3/(1-p1-p2)
    return {(r1.id, r2.id, r3.id) : _trifectaProb(probs[r1.id], probs[r2.id], probs[r3.id]) for (r1, r2, r3) in itertools.permutations(bettableRunners, 3) }

def superfectaProbs(bettableRunners, probs):
    def _superfectaProb(p1, p2, p3, p4):
        return p1 * p2/(1-p1) * p3/(1-p1-p2) * p4/(1-p1-p2-p3)
    return {(r1.id, r2.id, r3.id, r4.id) : _superfectaProb(probs[r1.id], probs[r2.id], probs[r3.id], probs[r4.id]) for (r1, r2, r3, r4) in itertools.permutations(bettableRunners, 4) }


def getRaceId(trackId, date, raceNumber):
    return trackId+"_"+date.strftime("%m-%d-%Y")+"_"+raceNumber

def lookupRace(raceDict, trackId, date, raceNumber):
    try:
        return raceDict[getRaceId(trackId, date, raceNumber)]
    except KeyError, e:
        #print raceDict.keys()
        raise e

def parseDashedDate(dateStr):
    try:
        return datetime.datetime.strptime(dateStr, "%m-%d-%Y").date()
    except ValueError:
        try:
            return datetime.datetime.strptime(dateStr, "%Y%m%d").date()
        except ValueError:
            try:
                return datetime.datetime.strptime(dateStr, "%m/%d/%Y").date()
            except ValueError:
                return datetime.datetime.strptime(dateStr, "%m/%d/%Y 0:00:00").date()

def getDataFrameRace(race, cols=[]):
    dict_race = {r.id : pd.Series(r.__dict__) for r in race.bettableRunners()}
    dfr=pd.DataFrame.from_dict(dict_race, orient='index')
    return dfr
def getWagerName(wk):
    if wk.wagerType == WagerKey.Multihorse:
        n = len(wk.runners)
        if (n==1):
            return "Win"
        elif(n == 2):
            return "Exacta"
        elif(n == 3):
            return "Trifecta"
        elif(n == 4):
            return "Superfecta"
        elif(n == 5):
            return "Pentefecta"
        else:
            raise "No exotic bet name for "+str(n)+" runners."
    return wk.wagerType
def getDataFramePayout(simulator):
    spd = simulator.payoutDict
    df = pd.DataFrame.from_dict({raceid: pd.Series({k : v for k,v in racepayout.__dict__.iteritems()}) for raceid,racepayout in spd.iteritems()}, orient='index')
    return df
class BaseSetFromDict():
    '''A base class that allows attributes to be set from a map with specified overrides to field names. '''
    def setattrWithMapping(self, specifiedFields, definedFields, fieldMap):
        for (field, parse) in definedFields:
            mappedFieldName = field
            if(field in fieldMap):
                mappedFieldName = fieldMap[field]
            if(mappedFieldName in specifiedFields):
                val = specifiedFields[mappedFieldName]
                if val is not None:
                    setattr(self, field, parse(val))

class Track(BaseSetFromDict):
    trackFields = [('id', str), ('resultFileCode', str), ('name', str), ('ebetMeetID', int), ('ebetSym', str), ('winRebatePct', float), ('placeRebatePct', float), ('showRebatePct', float), ('exactaRebatePct', float), ('trifectaRebatePct', float), ('superfectaRebatePct', float), ('pentafectaRebatePct', float), ('minTicketExacta', float), ('minBetExacta', float), ('minTicketTrifecta', float), ('minBetTrifecta', float), ('minTicketSuperfecta', float), ('minBetSuperfecta', float), ('minTicketPentafecta', float), ('minBetPentafecta', float)]

    def __init__(self, fields, fieldMap = {}):
        self.setattrWithMapping(fields,Track.trackFields, fieldMap)
def validateWagers(races, wagers):
    for (wagerKey, amt) in wagers:
        (minTicket, minBet) = races[wagerKey.raceId].minBet[wagerKey.getWagerName()]
        assert amt >= minBet, "Invalid bet: Bet amount {} was less then minimum amount of {} for wager {}".format(amt, minBet, wagerKey)
        nCombs = len(wagerKey.getCombinations())
        assert nCombs*amt >= minTicket, "Invalid bet: Ticket total {} ({} combinations) was less then minimum ticket of {} for wager {}".format(nCombs*amt, nCombs, minTicket, wagerKey)

class Race(BaseSetFromDict):
    ''' Basic abstraction for a race.  Metadata about the race along with a list of runners.  (runner list and a runnerIndex dictionary which goes by key)'''
    raceFields = [('id', str), ('track', str), ('date', parseDashedDate), ('allWeatherSurface', bool), ('ageRestriction', str), ('sexRestriction', str), ('ageLimit', str), ('raceConditions', str), ('raceClassification', str), ('raceNumber', int), ('raceType', str), ('purse', int), ('claimingPrice', int), ('minBet', lambda x: x), ('maiden', bool), ('betSummary', str), ('startTime', str), ('distance', int), ('surface', str),('stateBred',str)]
    def __init__(self, fields, runners = [], raceFieldMap = {}, runnerFieldMap = {}):
        self.runners = []
        self.runnerIndex = {}
        self.setattrWithMapping(fields,Race.raceFields, raceFieldMap)
        for runner in runners:
            self.addRunner(runner, runnerFieldMap)
    def check(self, raceFields):
        for (key, value) in raceFields.iteritems():
            assert getattr(self, key) == value, 'Field '+key+' mismatched in race '+self.id
    def addRunner(self, fields, fieldMap):
        runner = Runner(fields, fieldMap)
        #assert not self.runnerIndex.has_key(runner.id), "Duplicate runner "+str(runner.id)
        if not self.runnerIndex.has_key(runner.id):
            self.runners.append(runner)
            self.runnerIndex[runner.id] = runner
        return runner
    def runnerByHorseName(self, name):
        for runner in self.runners:
            if runner.isSameName(name):
                return runner
        raise KeyError("No runner named '{}' for race {} {}".format(name, self.id, [x.name for x in self.runners]))
    def runnerByPgmPos(self, pgmPos):
        pgmPos = pgmPos.upper()
        runner = [x for x in self.runners if x.pgmPos == pgmPos]
        assert len(runner) == 1, "{} not in {!s}".format(pgmPos, self.runners)
        return runner[0]
    def runnerByResultPos(self, resultPos):
        ''' Result position is the program position with the A or B stripped off. '''
        runner = [x for x in self.runners if x.resultPos == resultPos and x.isBettable()]
        #for x in runner:
        #    print x.__dict__
        assert len(runner) == 1, "Result {}, found {} in {!s}".format(resultPos, len(runner), self.runners)
        return runner[0]
    def bettableRunners(self):
        return [x for x in self.runners if x.isBettable()]
    def getCombinations(self, wagerType, baseValue=0.0):
        if wagerType == 'Win':
            return { r1.id : baseValue  for r1 in self.bettableRunners() }
        elif wagerType == 'Exacta':
            return {(r1.id, r2.id) : baseValue  for (r1,r2) in itertools.permutations(self.bettableRunners(), 2) }
        elif wagerType == 'Trifecta':
            return {(r1.id, r2.id, r3.id) : baseValue for (r1, r2, r3) in itertools.permutations(self.bettableRunners(), 3) }
        elif wagerType == 'Superfecta':
            return {(r1.id, r2.id, r3.id, r4.id) : baseValue for (r1, r2, r3, r4) in itertools.permutations(self.bettableRunners(), 4)}
    def getWagerKeys(self, wagerType):
        if wagerType == 'Win':
            return { r1.id :  0  for r1 in self.bettableRunners() }
        elif wagerType == 'Exacta':
            return {(r1.id, r2.id) : 0  for (r1,r2) in itertools.permutations(self.bettableRunners(), 2) }
        elif wagerType == 'Trifecta':
            return {(r1.id, r2.id, r3.id) : 0 for (r1, r2, r3) in itertools.permutations(self.bettableRunners(), 3) }
        elif wagerType == 'Superfecta':
            return {(r1.id, r2.id, r3.id, r4.id) : 0 for (r1, r2, r3, r4) in itertools.permutations(self.bettableRunners(), 4)}

    def getAttrStdev(self, attr='score'):
        br=self.bettableRunners()
        seriesAttr=pd.Series({r.id:r.__dict__.get(attr) for r in br})
        return seriesAttr.std()
    def getMonth(self):
        month=str(self.date).split("-")[1]
        return int(month)
    def getBadRunnersCount(self,attr='score',badThresh=-99):
        '''attr='score',badThresh=-99'''
        br=self.bettableRunners()
        seriesAttr=pd.Series({r.id:getattr(r,attr) for r in br if hasattr(r,attr)})
        return len(seriesAttr[seriesAttr<badThresh].index.values)
    def getRankBadRunners(self,attrBad,attrGood='HDWPSRRating',badThresh=-99,method='average',ascending=False):
        '''attrBad='score',attrGood='HDWPSRRating',badThresh=-99,method='average',ascending='False'''
        badRunners=self.getBadRunners(attrBad,badThresh)
        rankRunnersGood=self.getRunnerRankAttrs(attr=attrGood,method=method,ascending=ascending)
        return rankRunnersGood[badRunners]
    def getRunnerAttrs(self,attr):
        br=self.bettableRunners()
        seriesAttr=pd.Series({r.id:getattr(r,attr) for r in br if hasattr(r,attr)})
        return seriesAttr
    def getRunnerFunc(self,func):
        br=self.bettableRunners()
        seriesFunc=pd.Series({r.id:func(r) for r in br})
        return seriesFunc
    def getRunnerRankAttrs(self,attr):
        seriesAttr=self.getRunnerAttrs(attr)
        rankSeries=seriesAttr.rank(method='average',ascending=False)
        return rankSeries
    def getNumAboveThresh(self,thresh=10.0,attr='score'):
        br=self.bettableRunners()
        seriesAttr=pd.Series({r.id:r.__dict__.get(attr) for r in br})
        return len(seriesAttr[seriesAttr>=thresh])
    def matchTopSpeed(self):
        br=self.bettableRunners()
        speedSeries=pd.Series({r.id:r.aveSpeedBestNLastK(3,3) for r in br})
        scoreSeries=pd.Series({r.id:r.__dict__.get('score') for r in br})
        topSpeed=speedSeries.idxmax()
        topScore=scoreSeries.idxmax()
        if topSpeed==topScore:
            return True
        else:
            return False
    def getClassVector(self):
        attrs=['date','track','stateBred','raceType','raceClassification','purse','claimingPrice','surface','distance','ageRestriction','ageLimit','sexRestriction']
        setattr(self, 'month', self.getMonth())
        setattr(self, 'weekday', self.getWeekday())
        setattr(self, 'numRunners', len(self.bettableRunners()))
        setattr(self, 'medianHDWPSRRating', self.getMedian('HDWPSRRating'))
        attrs = attrs + ['month', 'weekday', 'numRunners', 'medianHDWPSRRating']
        attrSeries=pd.Series([getattr(self,a) for a in attrs])
        attrSeries.index=attrs
        return attrSeries[attrs]
    def getJockeyByRunner(self):
        '''returns the Jockey  by runner'''
        return self.getRunnerAttrs('jockey')
    def getTrainerByRunner(self):
        '''returns the trainer name by runner'''
        return self.getRunnerAttrs('trainer')
    def getTrainerJockeyStatsByRunner(self):
        '''returns tuple(trainerWins,jockeyWins)'''
        jockeyStats=self.getJockeyStatsByRunner()
        trainerStats=self.getTrainerStatsByRunner()
        return (jockeyStats,trainerStats)
    def getAttrByFinishPos(self,attr='score'):
        scores=self.getRunnerAttrs(attr)
        finish=self.getRunnerAttrs('officialFinishPosition').order()
        return pd.Series({x:scores[x] for x in finish.index.values})
    def getClassVariables(self,cuts,classVariables=['track','maiden','purse','distance']):
        return [getattr(self,attr) for attr in classVariables]
    def __repr__(self):
        return "Race: "+self.id

def noneStr(s):
    if s is None or len(s) == 0:
        return None
    else:
        return str(s)
class Runner(BaseSetFromDict):
    ''' Models a horse running in a race. 'coupledTo' indicates that this horse is not bettable individually, but is coupled to another horse.'''
    '''ppfields=
     4: {'ClassRatingFromPace': 42.0,
  'DRFSpeedRating': 76.0,
  'DRFTrackVariant': 14.0,
  'HDW2fPaceFig': 52.0,
  'HDW4fPaceFig': 63.0,
  'HDWLatePaceFig': 33.0,
  'HDWSpeedRating': 37.0,
  'approxDist': False,
  'bute': False,
  'claimingPrice': 7500,
  'comment': 'No threat',
  'date': '20131016',
  'distance': 1210.0,
  'entryFlag': False,
  'equipment': 'b',
  'extraComment': 'Claimed from Lozano Martin Trainer',
  'finishCallBtnLengthsLdrMargin': 8.25,
  'finishPos': 8,
  'firstCallBtnLengthsLdrMargin': 11.5,
  'firstCallPos': 9,
  'gateCallPos': 9,
  'lasix': True,
  'moneyPos': 8,
  'numEntrants': 11,
  'odds': 24.0,
  'placeMargin': 0.5,
  'placeName': 'ROMEO ROMEO',
  'placeWeight': 120,
  'postPosition': 5,
  'ppNum': 4,
  'purse': 9000,
  'raceClassification': 'MCL7500',
  'raceNumber': 6,
  'runner': Runner: OPX_20140302_4_6 (6),
  'secondCallBtnLengthsLdrMargin': 9.5,
  'secondCallPos': 9,
  'showMargin': 1.75,
  'showName': 'ROMEO ROMEO',
  'showWeight': 120,
  'specialChute': False,
  'startCallPos': 9,
  'stretchCallBtnLengthsLdrMargin': 7.5,
  'stretchPos': 7,
  'surface': 'D',
  'trackCondition': 'FT',
  'trackId': 'RP',
  'weight': 120,
  'winnersMargin': 1.75,
  'winnersName': 'ANTEROS',
  'winnersWeight': 120}}'''
    runnerFields = [('rider',str),('jockey',str),('id', str), ('name', str), ('pgmPos', str),('trainer',str), ('horseClaimingPrice', int), ('probUPRML', float), ('probBC', float), ('scratched', bool), ('nonBetting', bool), ('coupledTo', noneStr), ('morningLine', float), ('finalToteOdds', float), ('sex', str), ('birthYear', int)]

    def __init__(self, fields, fieldMap = {}):
        '''Initialized fields the object based on the given map.  A fieldMap can be passed which converts names in the passed map to standard names in the object. '''

        self.setattrWithMapping(fields,Runner.runnerFields, fieldMap)
        self.resultPos = re.search("^([0-9.]+).*$", self.pgmPos).group(1)
        if hasattr(self, 'name'):
            self.canonicalName = self.name.upper().replace("'", "").strip()
            ending = self.canonicalName.find("(")
            if ending != -1:
                self.canonicalName = self.canonicalName[0:ending].strip()
        self.pastPerformances = {}

    def __repr__(self):
        return "Runner: {} ({})".format(self.id, self.pgmPos)

    def isSameName(self, name):
        #return name.upper().replace("'", "").find(self.canonicalName) != -1
        ending = name.find("(")
        if ending != -1:
            name = name[0:ending]
        ret = name.upper().replace("'", "").strip() == self.canonicalName
        return ret
    def isBettable(self):
        #print "{} {} {}".format(self.scratched, self.nonBetting, self.coupledTo)
        return not self.scratched and not self.nonBetting and self.coupledTo is None
    def getBetHorseId(self):
        if self.coupledTo is None:
            return self.id
        else:
            return self.coupledTo
    def getRaceId(self):
        runnerId=self.id
        splitRunner=runnerId.split("_")
        track=splitRunner[0]
        date=splitRunner[1]
        race=splitRunner[2]
        raceId="_".join([track,date,race])
        return raceId
    def addPastPerformance(self, pp):
        assert pp.runner == self
        assert pp.ppNum not in self.pastPerformances
        self.pastPerformances[pp.ppNum] = pp
    def get_pp(self, ppNum, attr):
        '''produce a time series for attr in pastPerformances'''
        pp = getattr(self, 'pastPerformances')
        if pp:
            return pp.get(ppNum).__dict__.get(attr)
        else:
            return PastPerformance()
    def get_ratio_purseclaim(self,current_claiming_price):
        last_purse = float(self.get_pp(0, 'purse'))
        last_finish = float(self.get_pp(0,'finishPos'))
        this_claim = float(current_claiming_price)
        return np.log((last_purse / this_claim) * 1.0/last_finish)

class PastPerformance(BaseSetFromDict):
    pass
class WorkoutProfile(BaseSetFromDict):
    pass
class WagerKey(object):
    Multihorse = 'multihorse'
    Win = 'Win'
    Place = 'Place'
    Show = 'Show'

    ''' Models a type of bet and the specific horses bet on.  In the case of exacta+, runners is a tuple of tuples of runners'''
    def __init__(self, raceId, wagerType, runners):
        '''(raceId, wagerType, runners)'''
        ''' exacta+ is an exacta, trifecta, superfecta, etc depending on the number of ids. '''
        self.raceId = raceId
        self.wagerType = wagerType
        if self.wagerType == WagerKey.Multihorse:
            # For multihorse bets the runners is a tuple of tuples of ids.
            for runner in runners:
                for entry in runner:
                    assert type(entry) == type("aa"), type(entry)
        else:
            assert wagerType == WagerKey.Win or wagerType == WagerKey.Place or wagerType == WagerKey.Show

            # For single horse bets the runners is just an id
            #assert type(runners) == type("aa"), runners
        self.runners = runners
        self.numRunners = len(runners)
    def getWagerName(self):
        if self.wagerType == WagerKey.Multihorse:
            n = len(self.runners)
            if (n==1):
                return "Win"
            elif(n == 2):
                return "Exacta"
            elif(n == 3):
                return "Trifecta"
            elif(n == 4):
                return "Superfecta"
            elif(n == 5):
                return "Pentefecta"
            else:
                raise "No exotic bet name for "+str(n)+" runners."
        return self.wagerType
    def matches(self, runners):
        ''' Matches a tuple of runners with this wager key to see if it matches. '''
        assert (len(runners) == len(self.runners) , "Matching bets of different types {} and {}", self, runners)
        if type(runners) == str:
            print ("Single runner {}".format(runners))
            runners = (runners,)
        for i in range(len(runners)):
            if not runners[i] in self.runners[i]:
                return False
        return True
    def getCombinations(self):
        ''' Given a ticket which can have multiple ids at a given position, expand that out into all the possible combinations it represents. '''
        if self.wagerType=='Win':
            return (self.runners,)
        return expandCombinations(self.runners)
    def getCombinationsWithIndex(self):
        indexPos=["pos1","pos2","pos3","pos4"]
        try:
            combinations=list(self.getCombinations()[0])
        except:
            print ("could not get combinations for", self.raceId)
            return
        numRunners=len(combinations)
        return pd.Series(combinations,index=indexPos[0:numRunners])
    def __str__(self):
        return self.wagerType+str(self.runners)

    def __repr__(self):
        return "WagerKey('"+self.raceId+"','"+self.wagerType+"',"+str(self.runners)+")"

    def __eq__(self, other):
        return self.raceId == other.raceId and self.wagerType == other.wagerType and self.runners == other.runners

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        PRIME = 31;
        result = PRIME + hash(self.raceId)
        result = PRIME * result + hash(self.wagerType)
        result = PRIME * result + hash(self.runners)
        return result


class WagerMatrix(object):
    '''Creates a runner_id X pos matrix from a race object'''
    '''can aggregate wagers to use for implied probs'''
    def __init__(self,race):
        ''' exacta+ is an exacta, trifecta, superfecta, etc depending on the number of ids. '''
        self.raceId=race.id
        self.idxRunners=pd.Index([r.id for r in race.bettableRunners()])
        self.idxPos=pd.Index(["pos1","pos2","pos3","pos4"])
    def getMatrix(self,wagerTuple):
        wk=wagerTuple[0]
        runnerPos=wk.getCombinationsWithIndex()
        wagerPositions=runnerPos.index.values
        #r1=runnerPos['pos1']
        #r2=runnerPos['pos2']
        wv=wagerTuple[1]
        df=pd.DataFrame(index=self.idxRunners,columns=self.idxPos).fillna(0.0)
        #df.ix[r1,'pos1']=wv
        #df.ix[r2,'pos2']=wv
        for p in wagerPositions:
            df.ix[runnerPos[p],p]=wv
        return df
    def aggregateWagers(self,wagerList):
        wagerMatrixList=[]
        for wl in wagerList:
            wagerMatrixList.append(self.getMatrix(wl))
        wagerMatrixSeries=pd.Series(wagerMatrixList)
        wmAggregated=wagerMatrixSeries.sum()
        #print wmAggregated
        return wmAggregated
    def impliedProbs(self,wagerList,pos='pos1'):
        aggWagers=self.aggregateWagers(wagerList)
        pSeries=aggWagers[pos]
        try:
            pSeries=pSeries/pSeries.sum()
        except:
            print ()
        return pSeries
    def getImpliedProbMatrix(self,wagerList,wagerType='Trifecta'):
        df=self.aggregateWagers(wagerList)
        totalsByPosition=df.sum(axis=0)
        positions=df.columns
        dfip = df.copy()
        for p in positions:
            if df[p].sum()<.00001:
                dfip[p]=df[p]
            else:
                dfip[p]=df[p]/df[p].sum()
        return dfip
    def totalWagers(self,wagerList):
        awm=self.aggregateWagers(wagerList)
        return awm.sum()['pos1']


class RaceResultRunner(object):
    def __init__(self, name, pgmPos, scratched, finalToteOdds, officialFinishPos):
        self.name = name
        self.scratched = scratched
        if pgmPos is not None:
            self.pgmPos = pgmPos
        if finalToteOdds is not None:
            self.finalToteOdds = finalToteOdds
        if officialFinishPos is not None:
            self.officialFinishPos = officialFinishPos


    def __str__(self):
        if self.scratched:
            return "Scratched runner {}".format(self.name)
        else:
            return "RaceResultRunner('{}', '{}', {})".format(self.name, self.pgmPos, self.scratched)

class RacePayout(object):
    def __init__(self, raceId, numStarters, payouts, poolSizes, rebatePcts, runners):
        self.raceId = raceId
        self.numStarters = numStarters
        self.payouts = payouts
        self.poolSizes = poolSizes
        self.rebatePcts = rebatePcts
        self.runners = runners
    def getPayouts(self,wagerType):
        payoutDict={wk.getWagerName():(wk.getCombinations()[0],v) for (wk,v) in self.payouts.iteritems()}
        try:
            return payoutDict[wagerType]
        except:
            return 0.0
    def getPoolSizes(self,wagerType):
        return self.poolSizes.get(wagerType,0.0)
    def getDataFrame(self):
        df=pd.DataFrame(index=pd.Index(self.raceId), columns=['runners', 'numStarters', 'payout_Win', 'payout_Exacta', 'payout_Trifecta', 'payout_Superfecta'])
        payoutSeries=pd.Series({"payout_"+k.getWagerName():v for (k,v) in self.iteritems()})
        return df
    #pd.Series({getWagerName(k) : v for k,v in tpd['payouts'].iteritems()})
    def __str__(self):
        return "Payout on {} ({} starters, Rebates: {}): {} {}".format(self.raceId, self.numStarters, self.rebatePcts, self.poolSizes, self.payouts)

class RaceMetrics(object):
    '''Computed metrics from race,racePayout,probModel,strategy. '''
    def __init__(self,race,racePayout,strategy=None):
        self.race=race
        self.raceId=self.race.id
        self.track=self.race.track
        self.date=self.race.date
        self.payouts=racePayout.__dict__['payouts']
        self.bettableRunners=self.race.bettableRunners()
        self.numStarters=racePayout.numStarters
        self.date=self.race.date
        self.raceNumber=self.race.raceNumber
        self.minBet=self.race.minBet
        self.pmFinal=NamedOddsModel('finalToteOdds')
        self.morningLine=pd.Series({r.id:r.morningLine for r in self.bettableRunners})
        self.pmMorningLine=NamedOddsModel('morningLine')
        try:
            self.probsFinal=pd.Series(self.pmFinal(self.race))
        except:
            self.probsFinal=None
        self.probsMorningLine=pd.Series(self.pmMorningLine(self.race))
        self.entropyFinal=ss.entropy(self.probsFinal,base=len(self.probsFinal))
        self.entropyMorningLine=ss.entropy(self.probsMorningLine,base=len(self.probsMorningLine))
        self.effectiveNumFinal=math.floor(self.entropyFinal*self.numStarters)
        self.effectiveNumMorningLine=math.floor(self.entropyMorningLine*self.numStarters)
        self.entropyDropFinal=self.numStarters-self.effectiveNumFinal
        self.entropyDropMorningLine=self.numStarters-self.effectiveNumMorningLine
        self.trainers=pd.Series({r.id:r.trainer for r in self.bettableRunners})
        self.odds=pd.Series({r.id:r.__dict__.setdefault('finalToteOdds',None) for r in self.bettableRunners})
        self.officialFinishPosition=pd.Series({r.id:r.officialFinishPosition for r in self.bettableRunners}).order()
        self.pgmPos=pd.Series({r.id:int(r.pgmPos) for r in self.bettableRunners}).order()
        self.resultPos=pd.Series({r.id:int(r.resultPos) for r in self.bettableRunners}).order()
        self.speedRating=pd.Series({r.id:r.__dict__.get('HDWPSRRating',0.0) for r in self.bettableRunners}).order(ascending=False)
        self.rankSpeed=self.speedRating.rank(ascending=False)
        self.bestSpeed=pd.Series({r.id:r.__dict__.get('bestCramerSpeedFigDirt',0.0) for r in self.bettableRunners})
        self.rankBestSpeed=self.bestSpeed.rank(ascending=False)
        self.speedDiff=self.bestSpeed-self.speedRating
        self.birthYear=pd.Series({r.id:r.__dict__.get('birthYear',0) for r in self.bettableRunners})
        self.ageLimit=self.race.ageLimit
        self.ageRestriction=self.race.ageRestriction
        self.raceClassification=self.race.raceClassification
        self.raceConditions=self.race.raceConditions
        self.raceType=self.race.raceType
        try:
            self.claimingPrice=self.race.claimingPrice
        except:
            self.claimimngPrice=self.race.purse

        self.maiden=self.race.maiden
        self.sexRestriction=self.race.sexRestriction
        self.stateBred=self.race.stateBred
        self.sexResDict={'N': 'No Sex Restrictions',
        'M': 'Mares and Fillies Only',
        'C':'Colts and/or Geldings Only',
        'F': 'Fillies Only' }
        self.sexResMap=self.sexResDict[self.sexRestriction]
        self.purse=self.race.purse

        try:
            self.claimingPrice=self.race.claimingPrice
        except:
            self.claimingPrice=None

        if self.claimingPrice:
            self.purseClaimRatio=float(self.purse)/float(self.claimingPrice)
        else:
            self.purseClaimRatio=None

        self.estimatedPoolSizes=self.race.estimatedPoolSizes

        #Race data PaceSpeedClass
        try:
            self.speedPar=self.race.HDWSpeedParForClassLevel
        except:
            self.speedPar=None
        try:
            self.pace2f=self.race.__dict__['2fHDWPaceParforLevel']
        except:
            self.pace2f=None
        try:
            self.latePacePar=self.race.__dict__['HDWSLatePaceParForLevel']
        except:
            self.latePacePar=None
        self.numStarters=racePayout.__dict__['numStarters']
        self.poolSizes=racePayout.__dict__['poolSizes']
        self.rebatePcts=racePayout.__dict__['rebatePcts']
        assert racePayout.raceId == self.raceId
        self.runners=racePayout.__dict__['runners']
    def getRaceEntropy(self):
        return self.entropy
    def getLogOdds(self):
        return self.probModel.getNormOddsRace(self.race)
    def getProbs(self,pm=None):
        if pm:
            return pm.getProbs(self.race)
        else:
            return self.probs
    def getTrainers(self,trainerList=None):
        '''gets trainers that match list'''
        br=self.bettableRunners
        trainerDict={r.id:r.trainer for r in br}
        if not trainerList:
            return trainerDict
        else:
            filterDict= {r:trainerDict[r] for r in trainerDict.keys() if trainerDict[r] in trainerList}
            return filterDict
    def ppSummary(self):
        numPP={}
        ppData={}
        for r in self.bettableRunners:
            ppData[r.id]=r.pastPerformances
            numPP[r.id]=len(r.pastPerformances.keys())
        print numPP[r.id]
        return numPP
    def runnerPurseClaimScore(self):
        racePurse=self.race.purse
        claimPrice=self.race.__dict__.get('claimingPrice',racePurse)
        runners=self.race.bettableRunners()
        try:
            seriesPurse=pd.Series({r.id:r.avePurseLastK(2) for r in runners})
        except:
            return None
        seriesClaimValue=np.log(seriesPurse)/np.log(claimPrice)
        return seriesClaimValue
    def runnerDistancePassFail(self):
        distance=self.race.distance
        runners=self.race.bettableRunners()
        try:
            seriesDist=pd.Series({r.id:r.getDistanceVector()[0:4] for r in runners})
        except:
            return None
        distRaces=seriesDist.apply(lambda x:len(x[x>0.8*distance]))
        return distRaces>3
    def runnerSpeedPassFail(self):
        speedPar=self.race.HDWSpeedParForClassLevel
        runners=self.race.bettableRunners()
        try:
            seriesSpeed=pd.Series({r.id:r.getSpeedVector()[0:4] for r in runners})
        except:
            return None
        speedRaces=seriesSpeed.apply(lambda x:len(x[x>=speedPar-3]))
        return speedRaces>1



    def getSpeedMaury(self):
        runners=self.race.bettableRunners()
        try:
            seriesMaury=pd.Series({r.id:r.__dict__.get('speedMaury') for r in runners})
        except:
            return None
        return seriesMaury
    def setupSpeedModel(self):
        dfSpeed=pd.DataFrame([self.speedRating,self.bestSpeed]).transpose()
        dfSpeed.columns=['speed','bestSpeed']#,np.log(self.bestSpeed)-np.log(self.speedRating),np.log(self.speedPar)]).transpose()
        dfSpeed['gapSpeed']=dfSpeed['speed'].order(ascending=False).diff()
        dfSpeed['gapBest']=dfSpeed['bestSpeed'].order(ascending=False).diff()
        dfSpeed['logSpeed']=np.log(self.speedRating)
        dfSpeed['logBest']=np.log(dfSpeed['bestSpeed'])
        dfSpeed['logGapSpeed']=dfSpeed['logSpeed'].order(ascending=False).diff()
        dfSpeed['logGapBest']=dfSpeed['logBest'].order(ascending=False).diff()
        dfSpeed['raceId']=self.raceId
        dfSpeed['numStarters']=self.numStarters
        dfSpeed['zSpeed']=ss.zscore(dfSpeed['logSpeed'])
        dfSpeed['zBest']=ss.zscore(dfSpeed['logBest'])
        dfSpeed['zSpeedGap']=ss.zscore(dfSpeed['zSpeed'].order().diff())
        dfSpeed['speedPar']=self.speedPar
        try:
            dfSpeed['logSpeedPar']=np.log(dfSpeed['speedPar'])
        except:
            dfSpeed['logSpeedPar']=0.0
            return None
        dfSpeed['zDiffBestSpeed']=ss.zscore(dfSpeed['logBest']-dfSpeed['logSpeed'])
        dfSpeed['zDiffParBest']=ss.zscore(dfSpeed['logSpeedPar']-dfSpeed['logBest'])
        dfSpeed['zDiffParSpeed']=ss.zscore(dfSpeed['logSpeedPar']-dfSpeed['logSpeed'])
        dfSpeed['officialFinishPosition']=self.officialFinishPosition
        dfSpeed['rankSpeed']=dfSpeed['logSpeed'].rank(ascending=False)
        dfSpeed['rankBest']=dfSpeed['logBest'].rank(ascending=False)
        dfSpeed['outperfSpeed']=dfSpeed['rankSpeed']-dfSpeed['officialFinishPosition']
        dfSpeed['outperfBest']=dfSpeed['rankBest']-dfSpeed['officialFinishPosition']
        dfSpeed['cutFinish']=pd.cut(dfSpeed['officialFinishPosition'],2,labels=False)
        dfSpeed['binaryWin']=dfSpeed['officialFinishPosition']==1

        return dfSpeed.fillna(0)
    def computeScratchRatio(self):
        return 0.0
    def getEntriesTrainerCount(self):
        return 0
    def setupBrokenRaceModel(self):
        '''Training model for broken race prediction'''
        '''Inputs: entropyMorningLine,entropyModel,L1Dist_SpeedRank_ML'''
        '''Outputs:L1Dist_Final_Finish'''

        df={}
        '''Inputs'''
        df['input_entropyMorningLine']=self.entropyMorningLine
        df['input_entropyModel']=self.entropyModel


        '''Outputs'''
        df['out_L1Dist_Final_Finish']=self.weightedL1Dist(self.odds.rank(),self.officialFinishPosition).sum()
        return df
    def weightedL1Dist(self,rankSeriesA,rankSeriesB):
        wtdL1={}
        for i in rankSeriesA.index.values:
            distL1=abs(rankSeriesA[i]-rankSeriesB[i])
            minAB=min(rankSeriesA[i],rankSeriesB[i])
            wtdL1[i]=distL1/minAB
        return pd.Series(wtdL1)
    def setupOddsModel(self):
        '''Generates a dataframe of metrics computed from odds/probs to be used to characterize race'''
        dfOdds=pd.DataFrame([self.odds,self.morningLine,self.probsFinal,self.probsMorningLine,self.probs]).transpose()
        dfOdds.columns=['finalToteOdds','morningLineOdds','probsFinal','probsMorningLine','probsModel']
        logOddsFinal=np.log(self.odds)
        logOddsMorningLine=np.log(self.morningLine)
        dfOdds['raceId']=self.raceId
        dfOdds['numStarters']=self.numStarters
        dfOdds['logFinal']=logOddsFinal
        dfOdds['logML']=logOddsMorningLine
        dfOdds['diff_Probs_final_ml']=np.log(self.probsFinal)-np.log(self.probsMorningLine)
        dfOdds['diff_Odds_final_ml']=logOddsFinal-logOddsMorningLine
        dfOdds['zDiffProbs']=ss.zscore(dfOdds.diff_Probs_final_ml).rank(ascending=False)
        dfOdds['gapRunners_logOddsFinal']=dfOdds['logFinal']
        dfOdds['gapRunners_logOddsML']=dfOdds['logML'].diff()
        dfOdds['rankProbsFinal']=dfOdds['probsFinal'].rank(ascending=False)
        dfOdds['rankProbsML']=dfOdds['probsMorningLine'].rank(ascending=False)
        dfOdds['rankProbsModel']=dfOdds['probsModel'].rank(ascending=False)
        dfOdds['officialFinishPosition']=self.officialFinishPosition
        dfOdds['gapRunners_logOddsFinal']=dfOdds['logFinal']
        dfOdds['gapRunners_logOddsML']=dfOdds['logML'].diff()
        dfOdds['speedRating']=self.speedRating
        dfOdds['rankSpeed']=self.rankSpeed
        dfOdds['bestSpeed']=self.bestSpeed
        dfOdds['resultPos']=self.resultPos
        dfOdds['rankBestSpeed']=self.rankBestSpeed
        dfOdds['birthYear']=self.birthYear
        dfOdds['entropyDropML']=float(self.entropyDropMorningLine)
        dfOdds['entropyDropFinal']=float(self.entropyDropFinal)
        dfOdds['entropyDropModel']=float(self.entropyDropModel)
        dfOdds['rank_delta_probFinalMinusprobML']=dfOdds['diff_Probs_final_ml']
        dfOdds['L1_wtd_speed_post']=self.weightedL1Dist(dfOdds['rankSpeed'],dfOdds['resultPos'])
        dfOdds['L1_wtd_final_finish']=self.weightedL1Dist(dfOdds['rankProbsFinal'],dfOdds['officialFinishPosition'])
        dfOdds['L1_wtd_ml_final']=self.weightedL1Dist(dfOdds['rankProbsML'],dfOdds['rankProbsFinal'])
        dfOdds['L1_wtd_ml_finish']=self.weightedL1Dist(dfOdds['rankProbsML'],dfOdds['officialFinishPosition'])
        return dfOdds
    def getRunnersResults(self,wagerType,runners):
        '''takes a wagerKey and returns result by type'''
        payRunners=0.0
        payouts=self.payouts
        for k in payouts.keys():
            if k.getWagerName()==wagerType:
                if k.matches(runners):
                    payRunners=payouts[k]
                else:
                    payRunners=0.0
        return payRunners

    def scoreRunners(self,runnerTuple,scoreMap={1:4,2:3,3:2,4:1,5:.25,6:.125,7:.075,8:0.075,9:.075,10:.075,11:.075,12:.075,13:.025,14:.0125}):
        '''Scores a tuple of runners combination based on the racePayout data
        a runner in a combination gets 4 points for 1st , 3 points for 2nd , 2 points for 3rd, 1 point for 4th
        Exacta with the 1st place runner is worth 4 points, Trifecta with the 2nd place is 3 etc.'''
        runnerIndex=self.race.runnerIndex
        scores={}
        for r in runnerTuple:
            runner=runnerIndex[r]
            try:
                ofp=runner.officialFinishPosition
            except:
                ofp=0
            scores[r]=scoreMap[ofp]
        return pd.Series(scores).sum()

#     def quantileVector(self,attr,nbins=4,useLog=True,defaultScore=0.0):
#         featureVector=self.featureVectorNorm(attr,useLog,defaultScore)
#         qVect=pd.qcut(featureVector,nbins,labels=False,retbins=True)
#         qv=pd.Series(qVect[0],index=featureVector.index)
#         return qv
    def featureVectorNorm(self,attr,useLog=True,defaultScore=0.0):
        fv={}
        for r in self.race.bettableRunners():
            fv[r.id] = r.__dict__.get(attr,defaultScore)
        if useLog:
            fvs=np.log(pd.Series(fv))
        else:
            fvs=pd.Series(fv)
        zVector=ss.zscore(fvs)
        return zVector


    def __str__(self):
        "RaceMetrics"





