'''
Created on Mar 13, 2014

@author: User
'''
import csv
import itertools
import itertools
import math
from collections import namedtuple

from scipy.stats import norm, rayleigh

import numpy as np
import numpy as np
import pandas as pd
import pandas as pd
from x8313 import tuplize, roundTo, checkProbDict, normProbDict
from x8313.Models import WagerKey
from x8313.Probabilities import NamedOddsModel
import scipy.stats as ss
from dateutil.parser import parse

from collections import namedtuple
from datetime import date
from itertools import combinations, permutations
from multiprocessing import freeze_support, freeze_support
import pandas as pd
import psycopg2
from psycopg2 import extras
import scipy.stats as ss
import sys
import pickle
import os
import pandas.io.sql as sqlio
import pandas as pd
from dateutil.parser import parse
from matplotlib.widgets import Cursor, Button
import matplotlib.pyplot as plt
from psycopg2 import extras, extras
from scaffold.Models import *
from scaffold.RaceFilters import *
from scaffold.Simulate import Simulator, Max6WithImpact, Max6NoImpact
from scaffold.BettingStrategy import RaceClassifier,AlphaWagers,RoundingTicketBoxer,FixedPoolSizes
from scaffold.ScoreToProbViaIntegral import ScoreToProbViaIntegral
from scaffold.Strategies import Strategy,selectRaces
import os
import scaffold.DBLoader as db
from scaffold.Simulate import projectFile
import pylab 



userDir = os.path.expanduser('~')    
conn_string = "host='www.tennis-edge.com' dbname='ac2011' user='ac_2011' password='suspicious'"
# print "Connecting to database\n    ->%s" % (conn_string)
# conn = psycopg2.connect(conn_string)

'''TrackDict'''
# trackdict=db.loadTracks(conn)
# tracksTestLoad=trackdict.keys()
pickleFileCircuit='PHA_MTH_PIM_DEL_TAM_AQU_MNR_data.pickle-20100101-20101231'
TRACKS=['PHA','MTH','PIM','DEL','TAM','AQU','MNR','CTX','BEL','SAR','WOX']


TRACKSTEST=TRACKS
stringTRACKS="_".join(TRACKS)

'''Dates '''
startDate2010=datetime.date(2010,1,1)
endDate2010=datetime.date(2010,12,31)
dates2010=(startDate2010,endDate2010)
startDate2011=datetime.date(2011,1,1)
endDate2011=datetime.date(2011,12,31)
dates2011=(startDate2011,endDate2011)
DATES = [dates2010,dates2011]

pickleFilePHAMNR = projectFile("scratch/PHAMNR.pickle-{}-{}".format(startDate2010.strftime("%Y%m%d"), endDate2011.strftime("%Y%m%d")))
pickleFile2010 = projectFile("scratch/"+stringTRACKS+"_"+"data.pickle-{}-{}".format(startDate2010.strftime("%Y%m%d"), endDate2010.strftime("%Y%m%d")))
pickleFile2011 = projectFile("scratch/"+stringTRACKS+"_"+"data.pickle-{}-{}".format(startDate2011.strftime("%Y%m%d"), endDate2011.strftime("%Y%m%d")))

pickleFile = projectFile("scratch/circuitTestdata.pickle-{}-{}".format(startDate2010.strftime("%Y%m%d"), endDate2010.strftime("%Y%m%d")))


if os.path.exists(pickleFile):
    simulator=pickle.load(open(projectFile(pickleFile),'rb'))
else:
    print ("No pickle")
    print "Connecting to database\n    ->%s" % (conn_string)
    conn = psycopg2.connect(conn_string)
    simulator = Simulator.loadFromDbByTracksAndDates(TRACKS, startDate2011, endDate2011, False, conn)
 
simulator.snoopPoolSizes()
simulator.snoopFinalOdds()
simulator.snoopScratches() 

def getTrackTrainerDict():
    ttd={'ARISTONE PHILIP T':.26,'GUERRERO JUAN CARLOS':.25,'PRECIADO RAMON':.25,'AUWARTER EDWARD K':.15,'DANDY RONALD J':.17,'LYNCH CATHAL A':.23,'LAKE SCOTT A':.16,'DEMASI KATHLEEN A':.18,'DAY COREY':.18,'MOSCO ROBERT':.18,'LEVINE BRUCE N':.25,'LEBARRON KEITH W':.18,'REID JR ROBERT E':.22}
    return ttd

data=projectFile('scratch/StarterHistoryPHA.csv')
trackList=['PHA','MNR','PEN','MTH','TAM','DEL','AQU','BEU','FLX']
df=pd.read_csv(data)
idx_year_month_track_date_race_horsename=pd.MultiIndex.from_arrays([df.year,df.month,df.track,df.date,df.race,df.horsename],names=['year','month','track','date','race','horsename'])
df.index=idx_year_month_track_date_race_horsename
groupTrack=df.groupby('track',as_index=False)
trainerMeanOFP=groupTrack.get_group('MNR').groupby('trainer').officialfinishposition.apply(np.mean)
trainerCountDict={t:groupTrack.get_group(t).groupby('trainer').officialfinishposition.apply(lambda x:x.count()) for t in trackList}
trainerDict={track:trainerCountDict[track][trainerCountDict[track]>100] for track in trackList}
trainerMeanOFPDict={t:groupTrack.get_group(t).groupby('trainer').officialfinishposition.value_counts() for t in trackList}


def scoreFunc(ofp):
    scores=ofp.apply(lambda x: max(6-x,1))
    return scores.sum()
    
trainerCountDict={t:groupTrack.get_group(t).groupby('trainer').officialfinishposition.apply(lambda x:x.count()) for t in trackList}
trainerDict={track:trainerCountDict[track][trainerCountDict[track]>100] for track in trackList}
trainerScore={t:groupTrack.get_group(t).groupby('trainer').officialfinishposition.apply(lambda x: scoreFunc(x)) for t in trackList}




    
    
    #nonWinList=['NW1','NW1L','NW2','NW2L','NW3','NW3L','NW4','NW4L']
    #durationList=['1M','2M','3M','4M','5M','6M','7M','8M','9M','10M','1Y']
    #dollarList=np.arange(500,200000,500)
    
    
    
    

    


srd=simulator.raceDict
spd=simulator.payoutDict


BADRACES=['PID_20100508_8','PHA_20110313_8','PHA_20111213_8','SUF_20110716_8','OPX_20110319_10','HST_20111002_8','PHA_20110508_3','PHA_20111024_1','PHA_20111024_5','PHA_20111024_9','PHA_20110725_3','PHA_20110725_1','PHA_20111205_6','PHA_20111205_4','PHA_20111207_6','PHA_20111207_2','PHA_20110621_4']
BADRACES.extend([simulator.raceDict.pop(r).id for r in BADRACES if r in simulator.raceDict.keys()])




RACES=[r  for r in simulator.payoutDict.keys() if r not in BADRACES if simulator.raceDict[r].track in ['PHA','MNR']]


df=[pd.Series({r:RaceClassifier(srd[r]).getDataFrameProbs() for r in RACES})]
dfAll=pd.concat(df)
dfProbs=pd.concat(dfAll)

idx=pd.MultiIndex.from_arrays([dfProbs.raceId,dfProbs.index.values])
idx.names=['raceId','runnerId']
dfProbs.index=idx
grpRace=dfProbs.groupby(level='raceId')

dfProbs.to_csv(projectFile('scratch/dfProbsMNR.csv'),index_label=['raceId','runnerId'])

colSeries=pd.Series(dfProbs.columns)
colSeries.str.contains('RANK')
colRAW=colSeries[colSeries.str.contains('RAW')]
colRANK=colSeries[colSeries.str.contains('RANK')]
colPROB=colSeries[colSeries.str.contains('PROB')]
colOUTPUT=colSeries[colSeries.str.contains('OUTPUT')]
dfRAW=dfProbs[colRAW]
dfRANK=dfProbs[colRANK]
dfPROB=dfProbs[colPROB]
dfOUT=dfProbs[colOUTPUT]
tfP=dfProbs.reset_index(level=0,drop=True)

runnerList=[srd[race].runnerIndex[runner] for race in srd.keys() for runner in srd[race].runnerIndex]
fields=colPROB
for r in runnerList:
    for field in fields:
        try:
            setattr(r,field,float(tfP.ix[r.id][field]))
        except:
            print( 'could not set attr',field,"for",r.id)
            pass
    
        
WINPOOL=[0]
EXACTAPOOL=[400]
TRIFECTAPOOL=[0]
SUPERFECTAPOOL=[0]
SIMS=[Max6WithImpact]
MAXRISKS=[False]
NUMHORSES=[8]
ALPHAS=[.25]
MAXNUMBETS=[8]
MINNUMBETS=[2]
PERCENTMAX=[.01]
SELECTORDICT={n:CompositeSelector(SelectEBetTracks(),SelectNumRunners(n),SelectSurface('D')) for n in NUMHORSES}
probModels={pm:NamedProbModel(pm) for pm in colPROB}
pmCombined=CombinedMeanProbModel(probModels)
#probModels['pmCombined']=pmCombined
STRATEGIESDIST=[Strategy("pmCombined_"+('_').join([str(pm),str(mr),str(a),str(pct),str(maxBets),str(minBets),str(n),str(w),str(e),str(t),str(s)]),SELECTORDICT[n],probModels[pm],AlphaWagers(FixedPoolSizes(w,e,t,s),RoundingTicketBoxer(),a,maxNumBets=maxBets,minNumBets=minBets,maxRisk=mr) )for a in ALPHAS for maxBets in MAXNUMBETS for minBets in MINNUMBETS for pct in PERCENTMAX for w in WINPOOL for e in EXACTAPOOL for t in TRIFECTAPOOL for s in SUPERFECTAPOOL for n in NUMHORSES for mr in MAXRISKS for pm in probModels.keys()]
# STRATEGIESMAURY=[Strategy("speedMaury_"+('_').join([str(mr),str(a),str(pct),str(maxBets),str(minBets),str(n),str(w),str(e),str(t),str(s)]),SELECTORDICT[n],pmSpeedMaury,AlphaWagers(FixedPoolSizes(w,e,t,s),RoundingTicketBoxer(),a,maxNumBets=maxBets,minNumBets=minBets,maxRisk=mr) )for a in ALPHAS for maxBets in MAXNUMBETS for minBets in MINNUMBETS for pct in PERCENTMAX for w in WINPOOL for e in EXACTAPOOL for t in TRIFECTAPOOL for s in SUPERFECTAPOOL for n in NUMHORSES for mr in MAXRISKS]
# STRATEGIESTRACK=[Strategy("speedAC2_"+('_').join([str(mr),str(a),str(pct),str(maxBets),str(minBets),str(n),str(w),str(e),str(t),str(s)]),SELECTORDICT[n],pmSpeedTrackQ75,AlphaWagers(FixedPoolSizes(w,e,t,s),RoundingTicketBoxer(),a,maxNumBets=maxBets,minNumBets=minBets,maxRisk=mr) )for a in ALPHAS for maxBets in MAXNUMBETS for minBets in MINNUMBETS for pct in PERCENTMAX for w in WINPOOL for e in EXACTAPOOL for t in TRIFECTAPOOL for s in SUPERFECTAPOOL for n in NUMHORSES for mr in MAXRISKS]
# STRATEGIESML=[Strategy("speedAC2_"+('_').join([str(mr),str(a),str(pct),str(maxBets),str(minBets),str(n),str(w),str(e),str(t),str(s)]),SELECTORDICT[n],pmMorningLine,AlphaWagers(FixedPoolSizes(w,e,t,s),RoundingTicketBoxer(),a,maxNumBets=maxBets,minNumBets=minBets,maxRisk=mr) )for a in ALPHAS for maxBets in MAXNUMBETS for minBets in MINNUMBETS for pct in PERCENTMAX for w in WINPOOL for e in EXACTAPOOL for t in TRIFECTAPOOL for s in SUPERFECTAPOOL for n in NUMHORSES for mr in MAXRISKS]

RACES=[r  for r in simulator.payoutDict.keys() if r not in BADRACES if simulator.raceDict[r].track in TRACKSTEST]



TESTRACES=[RACES[0]]
output = 'scratch/backtestResult.txt'
# pickleFile = projectFile("scratch/"+stringTRACKS+"_"+"data.pickle-{}-{}".format(startDate2010.strftime("%Y%m%d"), endDate2010.strftime("%Y%m%d")))
outputLabel=projectFile("scratch/"+stringTRACKS+"_"+"backtest-{}-{}.csv".format(startDate2011.strftime("%Y%m%d"), endDate2011.strftime("%Y%m%d")))
# betDir=projectFile('scratch')

raceResults = simulator.simulateMultiple(strategies=STRATEGIESDIST, simParams=SIMS, races=RACES, betDir=projectFile('scratch'))
simulator.writeMultipleRaceResults(projectFile(output), raceResults)
df=pd.read_csv(projectFile(output))
df.to_csv(outputLabel)
grpTrackStrategy_NetReturn=df.netReturn.sum()
print grpTrackStrategy_NetReturn
def scoreModelRank(ranks,finishPos,method='exact'):
    '''7 points for predicting Winner'''
    '''5 for second'''
    '''3 for third'''
    '''2 for fourth'''
    scores=pd.Series({1:7,2:5,3:3,4:2})
    ranks=ranks.rank(method='first')
    ofp=finishPos.order()
    ofp4=ofp[ofp<5]
    ofp4ByFinish=dict(zip(ofp4.values,ofp4.index.values))
    ranks4=ranks[ranks<5]
    diffs=ranks4-ofp4
    if method=='exact':
        exact=ofp4[diffs==0]
        return scores[exact].cumprod()
    if method=='top4':
        numInTop4=len(diffs.dropna())
        return numInTop4
# ofp=testRC.getFinishPos()
# dfProbs=pd.DataFrame(probsDict)
# dfProbsRank=dfProbs.rank(ascending=False)
# dfProbsRank['ofp']=ofp
# print dfProbsRank
# L1=dfProbsRank.apply(lambda x: weightedL1Dist(ofp,x).sum())       
# Age/Sex Restriction Codes (3 character sting):
#   1st character
#   -------------
#   A - 2 year olds
#   B - 3 year olds
#   C - 4 year olds
#   D - 5 year olds
#   E - 3 & 4 year olds
#   F - 4 & 5 year olds
#   G - 3, 4, and 5 year olds
#   H - all ages
# 
#   2nd character
#   -------------
#   O - That age Only
#   U - That age and Up
# 
#   3rd character
#   -------------
#   N - No Sex Restrictions
#   M - Mares and Fillies Only
#   C - Colts and/or Geldings Only
#   F - Fillies Only