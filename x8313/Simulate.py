import cPickle
import csv
import datetime
import math
import os
import string
import pandas as pd
from x8313 import FileLoader, dateRange
from x8313.DBLoader import getConn, loadTracks, \
    whereClauseForRacesByTrackAndDate, loadRacesWithWhere, loadPayouts, \
    whereClauseForRacesByDate
from x8313.FileLoader import loadRacesFromDailyFile, loadDefaultTrackFile, \
    loadPayoutsFromDir, updateRacesFromResults, updateRacesWithTrack, \
    loadRacesFromStarterHistory
#from x8313.LiveBettor import writeEBetFile
from x8313.Models import validateWagers, WagerKey
#from x8313.Strategies import selectRaces
#from x8313.test import projectFile
#from x8313.Probabilities import NamedOddsModel
import pickle
import pandas.io.sql as psql
import pandas.io.sql as sqlio

import psycopg2
from psycopg2 import extras
import scipy.stats as ss
import sys


sqlRaceAll="select * from race"
ONE_FURLONG_IN_YARDS = 220.0
PAR_SPEED_WORKOUT_MAJOR = 12.0
fields_RaceQualityBase = ['race_type', 'age_restriction', 'age_limit', 'surface']
fields_RaceClass = ['race_classification', 'track_id', 'distance']

trackMappings = {"resultFileCode" : 'result_file_code', 'ebetMeetID' : 'e_bet_meet_id', "eBetSym" : 'e_bet_sym'}
poolMappings = {0 : 'WPS', 1 : 'WPS', 2 : 'WPS', 3: 'Exacta', 4: 'Trifecta', 5: 'Superfecta', 6:'Hi5' }
betMappings = {0 : 'Win', 1 : 'Place', 2 : 'Show', 3: 'Exacta', 4: 'Trifecta', 5: 'Superfecta', 6:'Hi5' }
watchAndWagerMappings = {'THOROUGHBRED' : 'TrackName', 'WPS' : 'WPS', 'EX' : 'Exacta', 'QN' : 'Quinella', 'TRI' : 'Trifecta', 'SF' : 'Superfecta'}


def makesqllist(id_list):
    idList=",".join(["'"+x+"'" for x in id_list])
    return idList
    
def extract_pp_raceid(pprow):
    track = pprow['pp_trackId']
    date = str(pprow['date'])
    racenum = str(pprow['raceNumber'])
    return "_".join([track, date, racenum])

defaultPoolSizes={'Superfecta': 10000.0, 'Exacta': 10000.0, 'Trifecta': 10000.0, 'WPS': 10000.0}
def getPayoutsByType(spd):
    '''takes a payout dictionary or RacePayout and returns payouts by WagerName'''
    assert(type(spd) is dict)
    wagertypes =  ['Win', 'Exacta', 'Trifecta', 'Superfecta']
    dfp=pd.DataFrame(index=spd.keys(), columns = wagertypes)
    payseries={}
    for raceid in spd.keys():
        racepayout = spd.get(raceid)
        this_payout = pd.Series({wk.getWagerName():wv for wk,wv in racepayout.__dict__.get('payouts').iteritems()})
        payseries[raceid] = this_payout
    payseries = pd.Series(payseries)
    dfp['payWin'] = payseries.apply(lambda x : x.get('Win', 0))
    dfp['payExacta'] = payseries.apply(lambda x : x.get('Exacta', 0))
    dfp['payTrifecta'] = payseries.apply(lambda x : x.get('Trifecta', 0))
    dfp['paySuperfecta'] = payseries.apply(lambda x : x.get('Superfecta', 0))
    dfp['raceId'] = dfp.index.values
    return dfp
    
import numpy as np
def getDataFrameSimulator(simulator, cols = []):
    '''Summarizes a simulator in a dataframe'''
    srd = simulator.raceDict
    spd = simulator.payoutDict
    dfr = pd.DataFrame.from_dict({tr.id : pd.Series(tr.__dict__) for tr in simulator.raceDict.itervalues()}, orient='index')
    trainers = pd.Series({raceid:race.getTrainerByRunner() for raceid,race in srd.iteritems()})
    dfr['trainers']=trainers
    dfp = pd.DataFrame.from_dict({raceid: pd.Series({k : v for k,v in racepayout.__dict__.iteritems()}) for raceid,racepayout in spd.iteritems()}, orient='index')
    dfpayouts = getPayoutsByType(simulator.payoutDict)
    df = pd.merge(dfr,dfp, left_on=['id'], right_on=['raceId'])
    df = pd.merge(df,dfpayouts, left_on=['raceId'], right_on=['raceId'])
    df.index=df['id']
    df['poolWin'] = df.estimatedPoolSizes.apply(lambda x:x.get('WPS',0))
    df['poolExacta']=df.estimatedPoolSizes.apply(lambda x:x.get('Exacta',0))
    df['poolTrifecta']=df.estimatedPoolSizes.apply(lambda x:x.get('Trifecta',0))
    df['poolSuperfecta']=df.estimatedPoolSizes.apply(lambda x:x.get('Superfecta',0))
    df.loc[:, 'log_poolWin'] = df['poolWin'].apply(lambda x : np.log(x))
    df.loc[:, 'log_poolExacta'] = df['poolExacta'].apply(lambda x : np.log(x))
    df.loc[:, 'log_poolTrifecta'] = df['poolTrifecta'].apply(lambda x : np.log(x))
    df.loc[:, 'log_poolSuperfecta'] = df['poolSuperfecta'].apply(lambda x : np.log(x)) 
    return df   


def getDataFrameRace(race, cols=[]):
    dict_race = {r.id : pd.Series(r.__dict__) for r in race.bettableRunners()}
    dfr=pd.DataFrame.from_dict(dict_race, orient='index')
    return dfr
def getRunnerNames(simulator):
    allRunners = pd.Series(simulator.getAllRunners())
    return allRunners.apply(lambda x:x.canonicalName)
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
def writeSimulator(simulator, path='D:/data/datapickle/'):
    '''Writes a simulator
       writeSimulator(simulator,path)'''
    fid=open(path + simulator.simulatorName + '.pickle','wb')
    pickle.dump(simulator,fid)
    fid.close()
    print ("wrote pickle:",path+simulator.simulatorName)
    return True
def getTestSimulator(simulator, racelist, name):
    '''writes a subset of simulator to a pickle file and returns a simulator'''
    '''getTestSimulator(simulator,racelist,name'''
    assert name!=None
    srd_sub = {raceid : simulator.raceDict[raceid] for raceid in racelist}
    spd_sub = {raceid : simulator.payoutDict[raceid] for raceid in racelist}
    testSimulator = Simulator(simulatorName=name, trackDict=simulator.trackDict, raceDict=srd_sub, payoutDict=spd_sub, validateWagers=simulator.validateWagers)
    return testSimulator 
def processOutput(dfbt,srd,output): 
    dfbt['rebatePctAdj']=.10
    dfbt['netReturnAdj']=dfbt.apply(lambda x:x['payout']-x['cost']+x['cost']*x['rebatePctAdj'],axis=1)
    dfbt['poolSize']=dfbt.apply(lambda x:srd[x['raceId']].estimatedPoolSizes.get(x['wagerType']),axis=1)
    dfbt['actualPctPool']=dfbt.apply(lambda x:x['cost']/x['poolSize'],axis=1)
    dfbt['claimingPrice']=dfbt.raceId.apply(lambda x:srd[x].__dict__.get('claimingPrice'))
    dfbt.to_csv(projectFile(output),index_label=None)
    return dfbt
def getAllRunners(raceDict):
    '''getAllRunners(raceDict: simulator.raceDict)'''
    '''returns a dict of all runners'''
    runnerDict={}
    for race in raceDict.keys():
        runners = raceDict[race].bettableRunners()
        for runner in runners:
            runnerDict[runner.id]=runner
    return runnerDict
def getAllDates(raceDict):
    '''returns all dates in the simulator'''
    return [race.date for race in raceDict.values()]
def getAllTrainers(raceDict):
    '''getAllTrainers(raceDict: simulator.raceDict)'''
    '''returns a dict of trainers by raceid'''
    return pd.Series({raceid:race.getTrainerByRunner() for raceid,race in raceDict.iteritems()})
def getAllJockeys(raceDict):
    '''getAllJockeys
    returns a dict of trainers by raceid'''
    return pd.Series({raceid:race.getJockeyByRunner() for raceid,race in raceDict.iteritems()})
def getRaceRunnerDataFrame(races,attrList,raceDict):
    attrDict={attr: getRaceRunnerScores(races,attr, raceDict) for attr in attrList}
    return attrDict
def getRaceRunnerScores(races,attr,raceDict):
    '''returns a series of series race,runner
        races: list of race_id
       attr = string
       raceDict = simulator.raceDict '''
    attrDict={}
    for raceId in races:
        runners=raceDict[raceId].bettableRunners()
        attrDict[raceId]=pd.Series({r.id:getattr(r, attr) for r in runners if hasattr(r,attr)}).order(ascending=False)
    attrSeries=pd.Series(attrDict)
    return attrSeries.fillna(0.0)
def getRaceClassVector(raceId,attrList,raceDict):
    #df=pd.DataFrame(index=races)
    return pd.Series({a:getattr(raceDict[raceId],a) for a in attrList})   
def getRaceRunnerProbs(races,pm,raceDict):
    probDict={}
    for raceId in races:
        probDict[raceId]=pd.Series(pm(raceDict[raceId]))
    probSeries=pd.Series(probDict)
    return probSeries.fillna(0.0)
def computeROINet(dfStrat):
        return (dfStrat.netReturn.sum()+dfStrat.cost.sum())/dfStrat.cost.sum()
def getReport(rrFile):
    df=pd.read_csv(projectFile(rrFile))
    dfReport=pd.DataFrame(columns=['numRaces','netReturn','cost','payout','rebate','costMean','costStd','costQ50','costMin','costMax','ROINet'],index=df.strategy.unique())
    dfReport['numRaces']=len(df.raceId.unique())
    dfReport['netReturn']=df.netReturn.sum()
    dfReport['cost']=df.cost.sum()
    dfReport['payout']=df.payout.sum()
    dfReport['rebate']=df.rebate.sum()
    dfReport['costMean']=df.cost.describe()['mean']
    dfReport['costStd']=df.cost.describe()['std']
    dfReport['costQ50']=df.cost.describe()['50%']
    dfReport['costMin']=df.cost.describe()['min']
    dfReport['costMax']=df.cost.describe()['max']
    dfReport['ROINet']=(df.cost.sum()+df.netReturn.sum())/df.cost.sum()
    #print dfReport
    return dfReport    
    
def getMultipleReport(rrFile,setGoodRaces):
    df=pd.read_csv(projectFile(rrFile))
    df=df[df.raceId.isin(setGoodRaces)]
    df.index=pd.MultiIndex.from_arrays([df['date'],df['raceId'],df['strategy']])  
    grpStrat=df.groupby(['strategy','track'])
    returnSummary=grpStrat.netReturn.sum()
    #print returnSummary
    by_track=df.groupby('track')
    by_rebate=df.groupby('rebatePct')
    by_strat=df.groupby('strategy')
    by_date=df.groupby('date')
    dfReport=by_strat[['netReturn','cost','payout','rebate']].sum()
    dfReport=dfReport.applymap(round)
    dfReport['numRaces']=by_strat.netReturn.count()
    dfReport['costMean']=by_strat.cost.apply(lambda x :x.describe()['mean'])
    dfReport['costStd']=by_strat.cost.apply(lambda x :x.describe()['std'])
    dfReport['costQ50']=by_strat.cost.apply(lambda x :x.describe()['50%'])
    dfReport['costMin']=by_strat.cost.apply(lambda x :x.describe()['min'])
    dfReport['costMax']=by_strat.cost.apply(lambda x :x.describe()['max'])
    dfReport['ROINet']=by_strat.apply(computeROINet,axis=1)
    #print dfReport
    return dfReport            
class RaceResult(object):
    def __init__(self, raceId, track, date, wagerType, poolSize, rebatePct):
        self.raceId = raceId
        self.track = track
        self.date = date
        self.wagerType = wagerType
        self.cost = 0
        self.payout = 0
        self.numBets = 0
        self.winningBetTotalAmount = 0
        self.numWinningBets = 0
        self.poolSize = poolSize
        self.rebatePct = rebatePct
        
    def update(self):
        self.rebate = self.rebatePct * self.cost
        self.netReturn = self.payout + self.rebate - self.cost 

    def __cmp__(self, other):
        idCmp = cmp(self.raceId, other.raceId)
        if idCmp == 0:
            return cmp(self.wagerType, other.wagerType)
        else:
            return idCmp

class SimParams(object):
    StandardRebates = { 'Win':0.02,'Place':0.02,'Show':0.02,'Exacta':0.06,'Trifecta':0.12,'Superfecta':0.12,'Pentafecta':0.12 }
    Max6Rebates = { 'Win':0.02,'Place':0.02,'Show':0.02,'Exacta':0.06,'Trifecta':0.06,'Superfecta':0.06,'Pentafecta':0.06 }
    NoRebates  = { 'Win':0.0,'Place':0.0,'Show':0.0,'Exacta':0.0,'Trifecta':0.0,'Superfecta':0.0,'Pentafecta':0.0 }
    HTFRebates  = { 'Win':0.08,'Place':0.08,'Show':0.08,'Exacta':0.10,'Trifecta':0.18,'Superfecta':0.18,'Pentafecta':0.18 }
    def __init__(self, name, alpha=.25,maxNumBets=25,defaultRebates = NoRebates, simulateWithImpact = True):
        self.name = name
        self.alpha=alpha
        self.maxNumBets=maxNumBets
        self.defaultRebates = defaultRebates
        self.simulateWithImpact = simulateWithImpact 

Max6WithImpact = SimParams("Impact", defaultRebates=SimParams.Max6Rebates, simulateWithImpact=True)
Max6NoImpact = SimParams("NoImpact", defaultRebates=SimParams.Max6Rebates, simulateWithImpact=False)
HTFRebatesWithImpact = SimParams("Impact", defaultRebates=SimParams.HTFRebates, simulateWithImpact=True)
HTFRebatesNoImpact = SimParams("NoImpact", defaultRebates=SimParams.HTFRebates, simulateWithImpact=False)
def loadFromDb(startDate, endDate, skip_dates=[]):
        simulator = Simulator.loadFromDbByDates(startDate, endDate)
        simulator.snoopPoolSizes()
        simulator.snoopFinalOdds()
        simulator.snoopScratches()
        return simulator
 
def loadFromFile(startDate, endDate, skip_dates=[]):
        dates = [d.strftime("%m%d") for d in dateRange(startDate, endDate)]
        userDir = os.path.expanduser('~')
        raceFiles = [userDir+'/Dropbox/DaveSaleem/MorningTransforms/mt{}.txt'.format(x) for x in dates if x not in skip_dates]
        #raceFiles = [userDir+'/Dropbox/DaveSaleem/MorningTransforms/mt{}.txt'.format(x) for x in ['0601']
        return Simulator.loadFromMtFilesAndResultDir(raceFiles, userDir+'/Dropbox/DaveSaleem/Results')

class Simulator(object):
    """ The rebate rates to use when no specific rebate amount is know/listed for the track.  """
    DefaultTakeout = { 'Win':0.19,'Place':0.19,'Show':0.19,'Exacta':0.26,'Trifecta':0.26,'Superfecta':0.26,'Pentafecta':0.26 }
    
    def __init__(self, trackDict, raceDict, payoutDict, validateWagers = False, simulatorName = 'simulator'):
        self.validateWagers = validateWagers
        self.trackDict = trackDict
        self.raceDict = raceDict
        self.payoutDict = payoutDict
        self.simulatorName = simulatorName
        self.runnerDict = self.getAllRunners()
    
    @staticmethod
    def load(startTuple, endTuple, reloadData = False, selector = None):
        return Simulator.getSimulatorWithCache(datetime.date(*startTuple), datetime.date(*endTuple), reloadData = reloadData, selector=selector)
        
    @staticmethod
    def getSimulatorWithCache(startDate, endDate, loadFunc = loadFromDb, skip_dates=[], reloadData = False, selector = None):
        print "Simulating {} to {}, skipping {}".format(startDate, endDate, skip_dates)
        pickleFile = projectFile("scratch/data.pickle-{}-{}".format(startDate.strftime("%Y%m%d"), endDate.strftime("%Y%m%d")))
        if reloadData or not os.path.exists(pickleFile):
            print "Loading data from DB"
            simulator = loadFunc(startDate, endDate, skip_dates)
            if selector is not None:
                (goodRaces, filterCounts) = selectRaces(selector, simulator.raceDict)
                print "Filtered {} races, kept {} out of {}.  Filters: {}".format(len(simulator.raceDict)-len(goodRaces), len(goodRaces), len(simulator.raceDict), filterCounts.getSummary())
                simulator.raceDict = {race.id : race for race in goodRaces}
                simulator.payoutDict = {race.id : simulator.payoutDict[race.id] for race in goodRaces if race.id in simulator.payoutDict }
            output = open(pickleFile,"wb")
            cPickle.dump(simulator, output)
            output.close()
        else:
            print "Loading pickled data file: {}".format(pickleFile)
            output = open(pickleFile,"r")
            simulator = cPickle.load(output)
            output.close()
            print "Loaded pickled data file.  {} races".format(len(simulator.raceDict))
        return simulator
        
    @staticmethod
    def _loadResults(raceDict, totalRejected, resultDir, validateWagers):
        trackDict = loadDefaultTrackFile()
        updateRacesWithTrack(raceDict, trackDict)
        initialRaces = len(raceDict)
        if resultDir != None:
            updateRacesFromResults(trackDict, raceDict, resultDir)
            payoutDict = loadPayoutsFromDir(trackDict, raceDict, resultDir)
            # Remove races we don't have payouts for and therefore can't simulate
            racesWithoutPayoff = set(raceDict.keys()) - set(payoutDict.keys())
            for key in racesWithoutPayoff:
                raceDict.pop(key)
        else:
            # If we aren't updating values from result files, then we need to default the fields that come from those files.
            FileLoader.defaultCouplingAndScratches(raceDict)
        print "Loaded {} out of {} races.  {} rejected during initial load and {} rejected because of payout data.".format(len(raceDict), initialRaces, totalRejected, initialRaces - len(raceDict))
        return Simulator(trackDict, raceDict, payoutDict, validateWagers)

    @staticmethod
    def loadFromStFiles(historyFile, resultDir, startDate = None, endDate = None, validateWagers = False):
        raceDict = loadRacesFromStarterHistory(historyFile, startDate, endDate)
        return Simulator._loadResults(raceDict, 0, resultDir, validateWagers)

    @staticmethod
    def loadFromMtFilesAndResultDir(raceFiles, resultDir, validateWagers = False):
        raceDict = {}
        totalRejected = 0
        for raceFile in raceFiles:
            (races, rejected) = loadRacesFromDailyFile(raceFile)
            totalRejected += rejected
            raceDict.update(races)
        return Simulator._loadResults(raceDict, totalRejected, resultDir, validateWagers)

    @staticmethod
    def loadFromDbByTracksAndDates(tracks, firstDate, lastDate, validateWagers = False, conn=None):
        whereClause = whereClauseForRacesByTrackAndDate(tracks, firstDate, lastDate)
        if conn is None:
            conn = getConn()
        return Simulator.loadFromDbWithWhere(conn, whereClause, validateWagers)

    @staticmethod
    def loadFromDbByDates(firstDate, lastDate, validateWagers = False):
        whereClause = whereClauseForRacesByDate(firstDate, lastDate)
        return Simulator.loadFromDbWithWhere(whereClause, validateWagers)

    @staticmethod
    def loadFromDbWithWhere(con, whereClause, validateWagers = False):
        trackDict = loadTracks(con)
        raceDict = loadRacesWithWhere(con, trackDict, whereClause)
        payoutDict = loadPayouts(con, trackDict, raceDict, whereClause)
        races = set(raceDict.keys())
        payouts = set(payoutDict.keys())
        racesWithoutPayouts = races - payouts
        payoutsWithoutRaces = payouts - races 
        for race in racesWithoutPayouts:
            del raceDict[race]
        for payout in payoutsWithoutRaces:
            del payoutDict[payout]
        assert len(raceDict) == len(payoutDict)
        con.close()
        print "Loaded {} races.  ({} races had no result info, and {} result infos were missing race definitions.)".format(len(raceDict), len(racesWithoutPayouts), len(payoutsWithoutRaces))
        return Simulator(trackDict, raceDict, payoutDict, validateWagers)

    def writeWagerResultsFile(self,trackDict, races,payouts, filename, wagerList,strategy):    
        with open(filename, 'wb') as f: 
            wagerResultDict={}
            wagerResultHeaders=["raceId","wagerType","wagerAmount","runnerIds","runnerFinalOddsRank","runnerFinalOdds"]
            f.write(",".join(wagerResultHeaders)+"\n")
            for (wagerKey, amount) in wagerList:
                race = races[wagerKey.raceId]
                
                bettableRunners=race.bettableRunners()
                runnerOdds=pd.Series({r.id:r.__dict__.get('finalToteOdds') for r in bettableRunners})
                runnerRanks=runnerOdds.rank()
                runners = [",".join([runnerId for runnerId in runnerIds]) for runnerIds in wagerKey.runners]
          
                ranks=[str(runnerRanks[rIds[0]]) for rIds in wagerKey.runners]
                odds=[str(runnerOdds[rIds[0]]) for rIds in wagerKey.runners]
                track = trackDict[race.track]
    
                fields = (str(wagerKey.raceId), wagerKey.getWagerName(), str(amount), '/'.join(runners),'/'.join(ranks),'/'.join(odds))
                #print fields
                f.write(','.join(fields)+"\n")
                wagerResultDict[wagerKey]=fields
        return wagerResultDict
    def snoopPoolSizes(self):
        for race in self.raceDict.itervalues():
            try:
                payouts = self.payoutDict[race.id]
                race.estimatedPoolSizes = { key[1] : value for (key, value) in payouts.poolSizes.iteritems() }
                print(race.estimatedPoolSizes)
            except:
                race.estimatedPoolSizes=defaultPoolSizes
                continue            
    def snoopFinalOdds(self):
        for race in self.raceDict.itervalues():
            payouts = self.payoutDict[race.id]
            #print payouts
            #print(("Set odds for {}").format(payouts.runners))
            for runnerResult in payouts.runners.itervalues():
                if hasattr(runnerResult, "pgmPos"):
                    runner = race.runnerByPgmPos(runnerResult.pgmPos)
                    if hasattr(runnerResult, 'finalToteOdds'):
                        runner.finalToteOdds = runnerResult.finalToteOdds
                    #print(("Set odds for {}").format(runner))
                    if hasattr(runnerResult, 'officialFinishPos'):
                        runner.officialFinishPosition = runnerResult.officialFinishPos
            
    def snoopScratches(self):
        for race in self.raceDict.itervalues():
            payouts = self.payoutDict[race.id]
            #print(("Set odds for {}").format(payouts.runners))
            for runnerResult in payouts.runners.itervalues():
                if runnerResult.scratched:
                    toScratch = [runner for runner in race.runners if runner.isSameName(runnerResult.name)]
                    assert len(toScratch) <= 1, "{} {} {} {}".format(race.id, runnerResult.name, toScratch, race.runners)
                    if(len(toScratch) == 0):
                        assert len(payouts.runners) > len(race.runners), "{} {} {}".format(race.id, toScratch, runnerResult)
                    else:
                        toScratch[0].scratched = True
    def updateRebatesFromFile(self, rebateFile):
        '''rebateFile is a csv dataframe 
        trackid,Win,Exacta,Trifecta,Superfecta
        PHA,.05,...'''
        df_rebates = pd.read_csv(rebateFile)
        df_rebates.index = df_rebates['track_id']
        dict_rebates = df_rebates.transpose().to_dict()
        for raceId, racePayout in self.payoutDict.iteritems():
            trackId = raceId.split("_")[0]
            setattr(racePayout, 'rebatePcts',dict_rebates.get(trackId))
        for raceId, racePayout in self.payoutDict.iteritems():
            print (raceId, getattr(racePayout, 'rebatePcts'))
        
    def simulateMultiple(self, strategies, simParams, races = [], betDir = None):
        numStrategies=float(len(strategies))
        secondsPerStrat=100.0
        totalTimeSeconds=secondsPerStrat*numStrategies
        totalTimeMinutes=float(totalTimeSeconds)/60.0
        print "Simulating {} strategies should take {} minutes at {} per strat".format(numStrategies,totalTimeMinutes,secondsPerStrat)    
        return {strategy.getName()+"/"+simParam.name : self.simulateResults(strategy, simParam, races, betDir) for strategy in strategies for simParam in simParams}

    def simulateResults(self, strategy, simParam, races = [], betDir = None):
        wagers = strategy.generateWagersForRaces({raceId:race for (raceId, race) in self.raceDict.iteritems() if raceId in races or len(races) == 0}, simParam=simParam)
        if self.validateWagers:
            validateWagers(self.raceDict, wagers)
        if betDir is not None:
            writeEBetFile(self.trackDict, self.raceDict, os.path.join(betDir, strategy.getName()+"Bets.txt"), wagers)     
        raceResults = self.computeReturn(wagers, simParam)
        if betDir is not None:
        #wr=self.writeWagerResultsFile(self.trackDict, self.raceDict,self.payoutDict, os.path.join(betDir, strategy.getName()+"BetsResults.txt"), wagers,strategy)
            print "notDumpingFile" 
        return raceResults
    def strategyMultiple(self,strategies, simParams, races = [], betDir = None):
        return {strategy.getName() : self.strategyResults(strategy, simParam, races, betDir) for strategy in strategies for simParam in simParams}
        
    def strategyResults(self,strategy,simParam,races=[],betDir=None):
        wagers = strategy.generateWagersForRaces({raceId:race for (raceId, race) in self.raceDict.iteritems() if raceId in races or len(races) == 0}, simParam=simParam)
        if self.validateWagers:
            validateWagers(self.raceDict, wagers)
        raceResults = self.computeReturn(wagers, simParam)
        if betDir is not None:
            wr=self.writeWagerResultsFile(self.trackDict, self.raceDict,self.payoutDict, os.path.join(betDir, strategy.getName()+"BetsResults.txt"), wagers,strategy)           
        return wr   
         
    def _initRaceResult(self, simParams, raceId, wagerName, droppedRaces):
        # First setup pool sizes for each race
        if raceId not in self.payoutDict:
            if raceId not in droppedRaces:
                print "No results for {}, dropping race".format(raceId)
                droppedRaces[raceId] = 1
            return
        racePayout = self.payoutDict[raceId]
        #print(racePayout.poolSizes)
        poolSize=0.0
        rebatePct=0.0
        if wagerName=='Win' or wagerName=='Place' or wagerName=='Show':
            poolSize = racePayout.poolSizes.get((raceId, 'WPS'), None)
           
        if poolSize is None:
            if raceId not in droppedRaces:
                #print "Race {} has a {} bet, but no payout was listed.  Skipping race.  Pools were: {}.  Bet summary was: {}".format(raceId, wagerName, racePayout.poolSizes, self.raceDict[raceId].betSummary)
                droppedRaces[raceId] = 1
            return      
        try:
            rebatePct = racePayout.rebatePcts[wagerName]
        except:
            rebatePct=0.0
        if math.isnan(rebatePct):
            rebatePct = simParams.defaultRebates[wagerName]
        return RaceResult(raceId, self.raceDict[raceId].track, self.raceDict[raceId].date, wagerName, poolSize, rebatePct)

    def computeReturn(self, wagers, simParam):
        wagersByRaceAndType = {}
        droppedRaces = {}
        resultDict = {}
        # Sort out wagers into race and bet type
        for wager in wagers:
            betKey = (wager[0].raceId, wager[0].getWagerName())
            wagersByRaceAndType.setdefault(betKey, []).append(wager)
            
        for (betKey, wagerList) in wagersByRaceAndType.iteritems():
            #print betKey
            (raceId, wagerName) = betKey
            raceResult = self._initRaceResult(simParam, raceId, wagerName, droppedRaces)
            if raceResult is None:
                continue
            resultDict[betKey] = raceResult 
            racePayout = self.payoutDict[raceId]

            # Now evaluate the actual bets
            totalWagered = sum([x[1] for x in wagerList])
            for (wagerKey, amount) in wagerList:
                assert wagerKey.raceId == raceId and wagerKey.getWagerName() == wagerName, "{} {} {} {}".format(wagerKey.raceId, raceId, wagerKey.getWagerName(), wagerName)
                combs = wagerKey.getCombinations()
                raceResult.cost += amount * len(combs)
                raceResult.numBets += len(combs)
                betPayouts = [p for p in racePayout.payouts.iterkeys() if p.raceId == wagerKey.raceId and p.getWagerName() == wagerKey.getWagerName()]
                #assert len(betPayouts) == 1, betPayouts
                #print len(betPayouts)
                for payout in betPayouts:
                    for comb in combs:
                        # Go through each combination of horses bet and check if it is a winner
                        if payout.matches(comb):
                            payoutAmount = racePayout.payouts[payout]
#                             print payoutAmount
                            if wagerKey.wagerType == WagerKey.Multihorse and simParam.simulateWithImpact == True:
                                takeout = Simulator.DefaultTakeout[wagerKey.getWagerName()]
                                payoutVal = racePayout.payouts[payout]
                                assert(payoutVal!=None,"payoutVal is: ", payoutVal)
                                poolSize = racePayout.poolSizes.get((wagerKey.raceId, wagerKey.getWagerName()),10000)
                                payoutPool = poolSize * (1 - takeout)
                                winningPoolBet = payoutPool / payoutVal
                                # Adjust our payout based on the values coming in
                                payoutAmount = (payoutPool + totalWagered* (1 - takeout)) /(winningPoolBet + amount)
                            raceResult.payout += amount * payoutAmount
                            raceResult.winningBetTotalAmount += amount
                            raceResult.numWinningBets += 1
        results = resultDict.values()
        for result in results:
            result.update()
        results.sort()
        return results
        
    def writeRaceResults(self, filename, raceResults):
        self.writeMultipleRaceResults(filename, {"default" : raceResults})
    def writeMultipleRaceResults(self, filename, raceResultsMap):
        resultHeaders = ['raceId', 'track', 'date', 'wagerType', 'netReturn', 'payout', 'cost', 'rebate', 'numBets', 'numWinningBets', 'winningBetTotalAmount', 'poolSize', 'rebatePct']
        raceHeaders = ['defaultRebates', 'day', 'bettableHorses', 'surface', 'distance', 'weekend', 'raceType']
        headers = ['strategy']+resultHeaders + raceHeaders
        with open(filename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for (strategy, raceResults) in raceResultsMap.iteritems():
                for result in raceResults:
                    race = self.raceDict[result.raceId]
                    track = self.trackDict[race.track]
                    defaultRebate = not hasattr(track, 'exactaRebatePct') or math.isnan(track.exactaRebatePct)
                    day = race.date.strftime("%a")
                    raceType = race.raceType
                    surface = getattr(race, "surface", "X")
                    writer.writerow([strategy]+[str(getattr(result, col)) for col in resultHeaders]+[str(defaultRebate), day, str(len(race.bettableRunners())), surface, str(race.distance), day == "Sat" or day == "Sun", raceType])
    def getDataFrame(self):
        '''returns a dataframe representation of simulator'''
        df=pd.DataFrame(index=self.getAllRunners().keys(),columns=['raceId','runnerId','finalToteOdds','morningLineOdds','mauryScore','HDWPSRRating','trainer','rider'])
        return df
    def compareSimulatorRaces(self,otherSimulator):
        this_set_races=set(self.raceDict.keys())
        other_set_races=set(otherSimulator.raceDict.keys())
        intersection_races=this_set_races.intersection(other_set_races)
        difference_races=this_set_races.difference(other_set_races)
        return (intersection_races,difference_races)
    
    def getAllRunners(self):
        '''getAllRunners(raceDict: simulator.raceDict)'''
        '''returns a dict of all runners'''
        runnerDict={}
        for race in self.raceDict.keys():
            runners = self.raceDict[race].bettableRunners()
            for runner in runners:
                runnerDict[runner.id]=runner    
        return runnerDict
    def extract_runner_data(self, runner, attrs):
        return {attr: runner.__dict__.get(attr, -100.0) for attr in attrs}
    def extract_pp_data(self,runner, pp_num=0):
        pp_dict = getattr(runner,('pastPerformances')).get(pp_num)
        return pp_dict    
        '''assumes receiving 'PHA_20160101_1'''
    def add_datetime_raceid(self, _df, attr):
        def extract_date(raceid):
            '''assumes receiving 'PHA_20160101_1'''
            return pd.to_datetime(raceid.split("_")[1]).date()
        '''Adds additional time index data and day of week for seasonality'''
        _df["date"] = _df[attr].apply(lambda x : self.extract_date(x))
        _df["month"] = _df['date'].apply(lambda x : x.month)
        _df["weekday"] = _df['date'].apply(lambda x : x.strftime('%A'))
        _df["year"] = _df['date'].apply(lambda x :x.year)
        _df["weeknum"] = _df['date'].apply(lambda x :x.strftime('%w'))
        return _df         


    @staticmethod
    def getDF(sqlQuery, index_col=None, add_datetime = False):
        '''getRaceResultPayout(sql="select * from race")'''
        conn_string = "host='dev.tennis-edge.com' dbname='ac' user='ac_2012' password='suspicious'"
        #conn_string = "host='localhost' dbname='AnimalCrackers' lenuser='postgres' password='password'"
        print "Connecting to database\n    ->%s" % (conn_string)
        conn = psycopg2.connect(conn_string)
        df = sqlio.read_sql(sqlQuery,conn,index_col=index_col)
        conn.close()
        if add_datetime:
            try:
                df['date']= df.date.apply(lambda x: str(pd.datetools.parse(x).date()))
                return df
            except:
                print('Could not parse date')
                return df
        else:
            return df
    
    def profile_target_track(self, races):
        dfrace = self.getDF("SELECT race.*, race_result.num_starters FROM  race, race_result  WHERE race.id IN ("+makesqllist(races)+") AND race_result.race_id = race.id")
        dfrace['date'] = dfrace.date.apply(lambda x:pd.to_datetime(x).date())
        dfrace['month'] =  dfrace.date.apply(lambda x:x.month)
        dfrace['year'] =  dfrace.date.apply(lambda x:x.year)
        dfrace['weeknum'] =  dfrace.date.apply(lambda x:x.strftime('%W'))
        dfrace['weekday'] =  dfrace.date.apply(lambda x:x.strftime('%A'))
        dfrace['racetype_agesex'] = dfrace['race_type'] + dfrace['age_restriction'] + dfrace['age_limit'] + dfrace['sex_restriction']
        dfrace.rename(columns={'id' : 'race_id'}, inplace=True)
        return dfrace



   