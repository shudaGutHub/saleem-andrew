from x8313.Models import Race, Track, lookupRace, WagerKey, RacePayout, \
    getRaceId, parseDashedDate
import csv
import datetime
import logging
import os
import re
import sys
import math

logger = logging.getLogger(__name__)

def loadScratches(fileName):
    scratches = []
    with open(fileName) as f:
        reader = csv.DictReader(f)
        for d in reader:
            raceId = getRaceId(d['track'], parseDashedDate(d['date']), d['raceNumber'])
            scratches.append((raceId, d['horseName']))
    return scratches

def loadPoolSizes(fileName):
    pools = {}
    with open(fileName) as f:
        reader = csv.DictReader(f)
        for d in reader:
            track = d['track']
            pools[track] = {'Exacta' : float(d['Exacta']), 'Trifecta' : float(d['Trifecta']), 'Superfecta' : float(d['Superfecta'])}
    return pools


def extractRaceId(runnerId):
    splitRunner=runnerId.split("_")
    track=splitRunner[0]
    date=splitRunner[1]
    race=splitRunner[2]
    raceId="_".join([track,date,race])
    return raceId

def updateRunnersFromFile(raceDict,fileName,fields) :
    '''Loads races and updates runners from a csv file'''
    '''fields = ['purseClaimScore']  '''
    '''file headers should be 'runnerId','purseClaimScore' '''
    missingRaces = {}
    foundRaces = {}
    with open(fileName) as f:
        reader = csv.DictReader(f)
        for d in reader:
            raceId=extractRaceId(d['runnerId'])
            race=raceDict.get(raceId)
            if race == None:
                missingRaces[raceId] = 1
            else:
                foundRaces[raceId] = 1
                runner = race.runnerIndex.get(d['runnerId'])
                for field in fields:
                    setattr(runner, field, float(d[field]))
        print("Updated {} races from StarterHistory.  {} races not in the existing race list.".format(len(foundRaces), len(missingRaces)))


def updateRunnersFromStarterHistory(raceDict, fileName, fields) :
    ''' Loads races from a starter_history CSV file. '''
    missingRaces = {}
    foundRaces = {}
    with open(fileName) as f:
        reader = csv.DictReader(f)
        for d in reader:
            date = parseDashedDate(d['date'])
            track = d['track']
            raceNumber = int(d['race'])
            raceId=track+"_"+date.strftime("%Y%m%d")+"_"+str(raceNumber)
            race = raceDict.get(raceId)
            #print("{} {}".format(raceId, raceDict.keys()[0]))
            if race == None:
                missingRaces[raceId] = 1
            else:
                foundRaces[raceId] = 1
                runner = race.runnerByHorseName(d['horsename'])
                for field in fields:
                    setattr(runner, field, float(d[field]))
                #print(runner.__dict__)
    print("Updated {} races from StarterHistory.  {} races not in the existing race list.".format(len(foundRaces), len(missingRaces)))

def loadRacesFromStarterHistory(fileName, startDate = None, endDate = None) :
    ''' Loads races from a starter_history CSV file. '''
    raceFieldMap = {'betSummary':'wagertypes'}
    raceDict= {}
    with open(fileName) as f:
        reader = csv.DictReader(f)
        for d in reader:
            date = parseDashedDate(d['date'])
            if (startDate is not None and startDate > date) or (endDate is not None and endDate <= date):
                continue
            track = d['track']
            raceNumber = int(d['race'])
            raceId=track+"_"+date.strftime("%Y%m%d")+"_"+str(raceNumber)

            df = float(d['dist'])
            distance = int(df)
            assert abs(df - distance) < 1e-9
            raceType = d['classdescriptor']
            maiden = maidenRaceClass(raceType)
            #betSummary = d['wagertypes']
            betSummary = ""
            #betDict = parseBetLine(d['betSummary'], raceId)
            betDict = {}
            surface = d['surface']
            raceFields = {'id':raceId, 'track':track,'date':d['date'], 'raceType': raceType, 'raceNumber':raceNumber, 'betSummary':betSummary, 'maiden':maiden, 'distance':distance, 'surface':surface, 'minBet':betDict}

            race = raceDict.get(raceId)
            if race == None:
                race = Race(raceFields)
                raceDict[race.id] = race
            else:
                pass
                #race.check(d)
            race.addRunner(d, {'id' : 'idRunner', 'name': 'horsename', 'probBC' : 'uprzscoreprob', 'pgmPos': 'saddlecloth'})
    return raceDict

def loadTracksFromFile(fileName):
    trackDict = {}
    with open(fileName) as f:
        reader = csv.DictReader(f)
        for d in reader:
            #print(str(d))
            track = Track(d, {'id':'JCapperSym', 'name':'ebetName'})
            trackDict[track.id] = track
    return trackDict

def loadDefaultTrackFile():
    '''Loads tracks from the track file stored in the same directory as the FileLoader module'''
    defaultFile = os.path.join(os.path.dirname(__file__), "trackMaster.csv")
    #print(defaultFile)
    return loadTracksFromFile(defaultFile)

resultFileHeaders = ['track', 'date', 'raceNumber','breed', 'distance', 'aboutDistanceIndicator', 'surface', 'offTurfIndicator', 'courseType', 'raceType', 'abbrevRaceConditions', 'raceName', 'sexRestriction', 'ageRestriction', 'statebredRestriction', 'purse', 'maxClaimPrice', 'minClaimPrice', 'grade', 'division','numStarters','trackCondition', 'weather', 'postTime', 'tempRailDistance', 'chuteStartIndicator', 'trackSealedIndicator', 'raceF1Time', 'raceF2Time', 'raceF3Time', 'raceF4Time', 'raceF5Time', 'raceWinningTime', 'WPSCombinedPool','unused_35', 'unused_36', 'unused_37', 'unused_38', 'unused_39', 'unused_40', 'unused_41', 'unused_42', 'unused_43', 'unused_44', 'unused_45', 'unused_46', 'unused_47', 'unused_48', 'unused_49', 'unused_50', 'horseName', 'horseAge', 'horseSex', 'medication', 'equipment', 'weightCarried', 'claimPrice', 'earnings', 'programNumber', 'coupledType', 'morningLineOdds', 'finalToteOdds', 'favoriteIndicator', 'nonBettingIndicator', 'winMutuelPaid', 'placeMutuelPaid', 'showMutuelPaid', 'postPosition', 'pos_Call_Start', 'pos_Call_1', 'pos_Call_2', 'pos_Call_3', 'pos_Call_4', 'pos_Call_Stretch', 'pos_Call_Finish', 'officialFinishPosition', 'call_1_LengthsBehind', 'call_2_LengthsBehind', 'Call_3_LengthsBehind', 'call_4_LengthsBehind', 'stretch_Call_Lengths_Behind', 'finish_Lengths_Behind', 'commentLine', 'trainerFirstName', 'trainerMiddleName', 'trainerLLastName', 'jockeyFirstName', 'jockeyMiddleName', 'jockeyLastName', 'apprenticeWeightAllowance', 'ownerFullName', 'claimedIndicator', 'claimTrainFirstName', 'claimTrainMidName', 'claimTrainLastName', 'claimOwnerFullName', 'scratchIndicator', 'reasonForScratch', 'lastRaceTrack', 'lastRaceDate', 'lastRaceRaceNumber', 'lastRaceFinishPosition', 'longCommentLine','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','ExoticWager1Type','ExoticWager1WagerUnit','ExoticWager1WagerNumRight','ExoticWager1WagerPayout','ExoticWager1WinningPGMNumbers','ExoticWager1TotalPool','ExoticWager2Type','ExoticWager2WagerUnit','ExoticWager2WagerNumRight','ExoticWager2WagerPayout','ExoticWager2WinningPGMNumbers','ExoticWager2TotalPool','ExoticWager3Type','ExoticWager3WagerUnit','ExoticWager3WagerNumRight','ExoticWager3WagerPayout','ExoticWager3WinningPGMNumbers','ExoticWager3TotalPool','ExoticWager4Type','ExoticWager4WagerUnit','ExoticWager4WagerNumRight','ExoticWager4WagerPayout','ExoticWager4WinningPGMNumbers','ExoticWager4TotalPool','ExoticWager5Type','ExoticWager5WagerUnit','ExoticWager5WagerNumRight','ExoticWager5WagerPayout','ExoticWager5WinningPGMNumbers','ExoticWager5TotalPool','ExoticWager6Type','ExoticWager6WagerUnit','ExoticWager6WagerNumRight','ExoticWager6WagerPayout','ExoticWager6WinningPGMNumbers','ExoticWager6TotalPool','ExoticWager7Type','ExoticWager7WagerUnit','ExoticWager7WagerNumRight','ExoticWager7WagerPayout','ExoticWager7WinningPGMNumbers','ExoticWager7TotalPool','ExoticWager8Type','ExoticWager8WagerUnit','ExoticWager8WagerNumRight','ExoticWager8WagerPayout','ExoticWager8WinningPGMNumbers','ExoticWager8TotalPool','ExoticWager9Type','ExoticWager9WagerUnit','ExoticWager9WagerNumRight','ExoticWager9WagerPayout','ExoticWager9WinningPGMNumbers','ExoticWager9TotalPool','ExoticWager10Type','ExoticWager10WagerUnit','ExoticWager10WagerNumRight','ExoticWager10WagerPayout','ExoticWager10WinningPGMNumbers','ExoticWager10TotalPool','ExoticWager11Type','ExoticWager11WagerUnit','ExoticWager11WagerNumRight','ExoticWager11WagerPayout','ExoticWager11WinningPGMNumbers','ExoticWager11TotalPool','ExoticWager12Type','ExoticWager12WagerUnit','ExoticWager12WagerNumRight','ExoticWager12WagerPayout','ExoticWager12WinningPGMNumbers','ExoticWager12TotalPool','ExoticWager13Type','ExoticWager13WagerUnit','ExoticWager13WagerNumRight','ExoticWager13WagerPayout','ExoticWager13WinningPGMNumbers','ExoticWager13TotalPool','ExoticWager14Type','ExoticWager14WagerUnit','ExoticWager14WagerNumRight','ExoticWager14WagerPayout','ExoticWager14WinningPGMNumbers','ExoticWager14TotalPool' ,'ExoticWager15Type','ExoticWager15WagerUnit','ExoticWager15WagerNumRight','ExoticWager15WagerPayout','ExoticWager15WinningPGMNumbers','ExoticWager15TotalPool','ExoticWager16Type','ExoticWager16WagerUnit','ExoticWager16WagerNumRight','ExoticWager16WagerPayout','ExoticWager16WinningPGMNumbers','ExoticWager16TotalPool','ExoticWager17Type','ExoticWager17WagerUnit','ExoticWager17WagerNumRight','ExoticWager17WagerPayout','ExoticWager17WinningPGMNumbers','ExoticWager17TotalPool','ExoticWager18Type','ExoticWager18WagerUnit','ExoticWager18WagerNumRight','ExoticWager18WagerPayout','ExoticWager18WinningPGMNumbers','ExoticWager18TotalPool','ExoticWager19Type','ExoticWager19WagerUnit','ExoticWager19WagerNumRight','ExoticWager19WagerPayout','ExoticWager19WinningPGMNumbers','ExoticWager19TotalPool']

def raceFromResultFileRow(trackDict, raceDict, d):
    matchingTracks = [x.id for x in trackDict.itervalues() if x.resultFileCode == d["track"]]
    assert len(matchingTracks) == 1, "Missing track: {} matched {}".format(d["track"], matchingTracks)
    return lookupRace(raceDict, matchingTracks[0], datetime.datetime.strptime(d["date"], "%m/%d/%Y"), d["raceNumber"])

def defaultCouplingAndScratches(raceDict):
    ''' If no results file is available, then defaults are selected for non-bettable, scratched, and coupling flags.'''
    for race in raceDict.itervalues():
        defaultCouplingAndScratchesForRace(race)

def defaultCouplingAndScratchesForRace(race):
    ''' If no results file is available, then defaults are selected for non-bettable, scratched, and coupling flags.'''
    for runner in race.runners:
        runner.scratched = False
        runner.nonBetting = False
        runner.coupledTo = None

exacta = ('Exacta', 'exactaRebatePct')
trifecta = ('Trifecta', 'trifectaRebatePct')
superfecta = ('Superfecta', 'superfectaRebatePct')
rebatePctLookup = {x[0] : x[1] for x in [exacta, trifecta, superfecta]}
wagerNames = {'E':exacta, 'F':exacta, 'G':exacta, 'J':exacta, 'T':trifecta, 'A':trifecta, 'S':superfecta}

def updateRacesFromResults(trackDict, raceDict, dirName):
    '''Snoops some values from the result files into the race. '''
    resultFiles = {}
    races = raceDict.values()
    for race in races:
        resultFile = resultFileName(trackDict, race)
        if resultFile not in resultFiles:
            resultFiles[resultFile] = updateRacesFromResultFile(trackDict, raceDict, os.path.join(dirName, resultFile))
        success = resultFiles[resultFile]
        if not success:
            # No result file, eliminate the race
            raceDict.pop(race.id)
    missingRaceFiles = [key for (key, val) in resultFiles.iteritems() if val == False]
    missingRaceFiles.sort()
    for file in missingRaceFiles:
        print "Missing result file: {}".format(file)

    # Check that all races and runners have been handled.
    races = list(raceDict.itervalues())
    for race in races:
        if not hasattr(race, "estimatedPoolSizes"):
            print "Race {} was in initial data, but missing from results file.  Dropping race.".format(race.id)
            raceDict.pop(race.id)
        else:
            for runner in race.runners:
                assert hasattr(runner, 'coupledTo') and hasattr(runner, 'scratched') and hasattr(runner, 'nonBetting'), runner.__dict__


def parseFloatOptional(odds):
    try:
        return float(odds)
    except:
        return None

def updateRacesFromResultFile(trackDict, raceDict, fileName):
    ''' Returns true if the file was loaded successfully.  False if it failed. '''
    #print(fileName)
    badRaces = {}
    couplings = {}
    try:
        with open(fileName) as f:
            reader = csv.DictReader(f, fieldnames=resultFileHeaders)
            for d in reader:
                race = None
                try:
                    race = raceFromResultFileRow(trackDict, raceDict, d)
                    exotics = _parseExotics(race.id, d)
                    poolSizes = {}
                    for exotic in exotics:
                        wagerName = wagerNames[exotic.wagerType][0]
                        assert wagerName not in poolSizes or poolSizes[wagerName] == exotic.poolSize
                        poolSizes[wagerName]= exotic.poolSize
                    poolSizes.setdefault("Exacta", 0)
                    poolSizes.setdefault("Trifecta", 0)
                    poolSizes.setdefault("Superfecta", 0)
                    poolSizes.setdefault("Pentafecta", 0)
                    assert (not hasattr(race, 'estimatedPoolSizes')) or race.estimatedPoolSizes == poolSizes
                    race.estimatedPoolSizes = poolSizes

                    # Program numbers are erased for horses that are scratched, so we set the scratched flag for every horse when we first see a race and then remove it for those that aren't.
                    if not hasattr(race.runners[0], 'scratched'):
                        for runner in race.runners:
                            runner.scratched = True
                            runner.nonBetting = False
                            runner.finalToteOdds = None
                            runner.coupledTo = None
                            runner.officialFinishPosition = None
                    if d['scratchIndicator'] == "Y":
                        assert len(d['programNumber']) == 0
                    else:
                        runner = race.runnerByPgmPos(d['programNumber'])
                        runner.scratched = False
                        runner.finalToteOdds = parseFloatOptional(d['finalToteOdds'])
                        runner.officialFinishPosition = int(d['officialFinishPosition'])
                        runner.nonBetting = d['nonBettingIndicator'] == "Y"
                        assert runner.isSameName(d['horseName']), "{} {}".format(runner.name, d['horseName'])

                        # Resolve coupling of horses
                        coupling = d['coupledType']
                        #print "{} coupling: '{}'".format(runner.name, coupling)
                        if len(coupling) > 0:
                            key = (race.id, coupling)
                            couplings.setdefault(key, []).append(runner)
                        else:
                            runner.coupledTo = None
                except NoPayoutError:
                    if not badRaces.has_key(race.id):
                        print "{} had a missing payout entry. Skipping race".format(race)
                        raceDict.pop(race.id)
                        badRaces[race.id] = 1
                except KeyError, k:
                    raceId = k[0]
                    if not badRaces.has_key(raceId):
                        print "Race {} was in result file but not in original data. Skipping race".format(raceId)
                        badRaces[raceId] = 1
                except BaseException, e:
                    type, value, traceback = sys.exc_info()
                    print "Error updating race {} from {}".format(race.id, fileName)
                    raise type, value, traceback
        for ((raceId, coupling), runners) in couplings.iteritems():
            assert len(runners) > 1
            # By convention, we treat the first program position alphabetically as the one to keep.  ('1' over '1A', '1A' over '1X')
            runners.sort(key=lambda runner: runner.pgmPos)
            mainRunner = runners[0]
            assert mainRunner.coupledTo is None, mainRunner
            mainRunner.coupledTo = None
            for coupledRunner in runners[1:]:
                assert coupledRunner.coupledTo is None, coupledRunner
                assert coupledRunner.resultPos == mainRunner.resultPos
                coupledRunner.coupledTo = mainRunner.id

        return True
    except IOError:
        #print "Missing result file {}".format(fileName)
        return False

class NoPayoutError(BaseException):
    pass

class MultiplePayoutError(BaseException):
    pass

class UnparseablePosition(BaseException):
    def __init__(self, msg):
        BaseException.__init__(self, msg)

def loadPayoutsFromDir(trackDict, raceDict, dirName):
    payouts = {}
    missingFiles = {}
    badRaces = {}
    for (raceId, race) in raceDict.iteritems():
        if raceId not in payouts:
            resultFile = resultFileName(trackDict, race)
            if resultFile not in missingFiles and race.id not in badRaces:
                try:
                    resultDict = loadPayoutsFromFile(trackDict, raceDict, os.path.join(dirName, resultFile), badRaces)
                    payouts.update(resultDict)
                except IOError:
                    print "{} not found, skipping payout computations".format(resultFile)
                    missingFiles[resultFile] = 1
    return payouts

def resultFileName(trackDict, race):
    return "{}{}F.TXT".format(trackDict[race.track].resultFileCode, race.date.strftime("%m%d"))

ties = ['EVD_07-25-2013_2', 'GPX_07-20-2013_8', 'EVD_07-03-2013_4', 'IND_08-13-2013_7']

def setBetPayout(d, payouts, raceId, runnerId, resultField, wagerType):
    betPayout = parseFloatOptional(d[resultField])
    if betPayout is not None:
        betPayout = betPayout/2.0
        key = WagerKey(raceId, wagerType, runnerId)
        if key in payouts:
            assert payouts[key] == betPayout, "{} {}".format(payouts[key], betPayout)
        else:
            #if wagerType == WagerKey.Win:
            #    odds = parseFloatOptional(d['finalToteOdds'])
            #    assert (odds+1) == betPayout, "Race: {} Odds of {} didn't match win payout of {}, {}".format(raceId, odds, betPayout, (odds+1)*2)
            payouts[key] = betPayout

def loadPayoutsFromFile(trackDict, raceDict, fileName, badRaces={}):
    resultDict = {}
    with open(fileName) as f:
        reader = csv.DictReader(f, fieldnames=resultFileHeaders)
        for d in reader:
            race = None
            try:
                race = raceFromResultFileRow(trackDict, raceDict, d)
                result = _buildResult(race, d, trackDict[race.track])
                if race.id in resultDict:
                    _checkResult(resultDict[race.id], result)
                else:
                    resultDict[race.id] = result
                # Add in win/place/show/payours
                payouts = resultDict[race.id].payouts
                pgmPos = d['programNumber']
                if len(pgmPos) > 0:
                    runnerId = race.runnerByPgmPos(pgmPos).getBetHorseId()
                    setBetPayout(d, payouts, race.id, runnerId, 'winMutuelPaid', WagerKey.Win)
                    setBetPayout(d, payouts, race.id, runnerId, 'placeMutuelPaid', WagerKey.Place)
                    setBetPayout(d, payouts, race.id, runnerId, 'showMutuelPaid', WagerKey.Show)
            except NoPayoutError:
                if race.id not in badRaces:
                    print "{} was missing payout numbers for 1 or more bets.  Skipping race.".format(race.id)
                badRaces[race.id] = 1
            except MultiplePayoutError:
                if race.id not in badRaces:
                    print "{} had multiple payout numbers for 1 or more bets.  Skipping race.".format(race.id)
                badRaces[race.id] = 1
            except UnparseablePosition, e:
                if race.id not in badRaces:
                    print "{}.  Skipping race.".format(e)
                badRaces[race.id] = 1
            except KeyError, k:
                if not badRaces.has_key(k[0]):
                    print "Race {} was in result file but not in original data. Skipping race".format(k[0])
                    badRaces[k[0]] = 1
    return resultDict

class Exotic(object):
    def __init__(self, wagerType, pgmNumbers, payoutPerUnit, poolSize):
        self.wagerType = wagerType
        self.pgmNumbers = pgmNumbers
        self.payoutPerUnit = payoutPerUnit
        self.poolSize = poolSize

def _parseExotics(raceId, d):
    exotics = []
    for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
        wagerType = d['ExoticWager'+str(i)+"Type"]
        if wagerType in wagerNames.keys():
            poolSize = float(d['ExoticWager'+str(i)+"TotalPool"])
            payoutStr = d['ExoticWager'+str(i)+'WagerPayout']
            if len(payoutStr.strip()) == 0:
                raise NoPayoutError
            payout=float(payoutStr)
            unitStr = d['ExoticWager'+str(i)+'WagerUnit']
            if len(unitStr) == 0:
                #print "In race {} results, missing unit string for exotic wager #{} ('{}').  Assuming $1".format(raceId, i, wagerType)
                unit = 1.0
            else:
                unit=float(unitStr)
            pgmNumbers=d['ExoticWager'+str(i)+'WinningPGMNumbers']
            exotics.append(Exotic(wagerType, pgmNumbers, payout/unit, poolSize))
    return exotics

def _buildResult(race, d, track):
    payouts = {}
    rebatePcts = {}
    poolSizes = {}
    exotics = _parseExotics(race.id, d)
    for exotic in exotics:
        poolSizes[(race.id, wagerNames[exotic.wagerType][0])] = exotic.poolSize
        rebatePcts[wagerNames[exotic.wagerType][0]] = getattr(track, wagerNames[exotic.wagerType][1])
        if exotic.pgmNumbers == "REBATE":
            # TDODO Need to deal with what happens with a "REBATE".  See race ARP_20130526_1
            continue
        multihorseKey = multihorseKeys(race, exotic.pgmNumbers)
        key = WagerKey(race.id, WagerKey.Multihorse, multihorseKey)
        if key in payouts:
            raise MultiplePayoutError
        payouts[key] = exotic.payoutPerUnit
    # If we have a return for a particular bet type, we make sure we had a minimum set.  Doesn't really hold up
    #assert ((race.id, 'Exacta') not in poolSizes) or hasattr(race, 'minBetExacta'), "{} {}".format(poolSizes, race.__dict__)
    #assert ((race.id, 'Trifecta') not in poolSizes) or hasattr(race, 'minBetTrifecta'), "{} {}".format(poolSizes, race.__dict__)
    #assert ((race.id, 'Superfecta') not in poolSizes) or hasattr(race, 'minBetSuperfecta'), "{} {}".format(poolSizes, race.__dict__)
    return RacePayout(race.id, int(d['numStarters']), payouts, poolSizes, rebatePcts, {})

def parsePos(race, pos):
    if pos == "ALL" or pos == "*" or pos == "A":
        return tuple([x.id for x in race.runners if x.isBettable()])
    else:
        if pos.endswith("A") or pos.endswith("B"):
            #print("{} truncated to  {}".format(pos, pos[0:-1]))
            pos = pos[0:-1]
        try:
            int(pos)
        except:
            msg = "Race {} results had unparseable program position {} in exotic bet payouts".format(race.id, pos)
            raise UnparseablePosition(msg)
        runner = race.runnerByResultPos(pos)
        assert runner.isBettable(), "Runner {} was in results but wasn't bettable.".format(runner.__dict__)
        return (runner.id,)

def multihorseKeys(race, pgmNumbers):
    if pgmNumbers.find('-') == -1:
        pgmNumbers = pgmNumbers.replace('/','-')
    return tuple([parsePos(race, pos) for pos in pgmNumbers.split('-')])

def _checkResult(existingResult, newResult):
    assert existingResult.raceId == newResult.raceId
    for payout in existingResult.payouts.iterkeys():
        if payout.wagerType == WagerKey.Multihorse:
            assert newResult.payouts[payout] == existingResult.payouts[payout]
    assert existingResult.poolSizes == newResult.poolSizes
    assert existingResult.numStarters == newResult.numStarters

class InvalidDistance(BaseException):
    def __init__(self, distance):
        self.distance = distance


def updateRacesWithTrack(raceDict, trackDict):
    ''' Puts the track object in the race and also overrides bets with general values from the track'''
    for race in raceDict.itervalues():
        #print(race)
        track = trackDict[race.track]
        race.trackObj = track
        if "Exacta" in race.minBet and not math.isnan(track.minBetExacta):
            race.minBet["Exacta"] = (track.minTicketExacta, track.minBetExacta)
        if "Trifecta" in race.minBet and not math.isnan(track.minBetTrifecta):
            race.minBet["Trifecta"] = (track.minTicketTrifecta, track.minBetTrifecta)
        if "Superfecta" in race.minBet and not math.isnan(track.minBetSuperfecta):
            race.minBet["Superfecta"] = (track.minTicketSuperfecta, track.minBetSuperfecta)
        #print race.minBet

def loadRacesFromDailyFile(fileName, delimiter=",") :
    ''' Loads races from a CSV file. '''
    raceDict= {}
    rejected = 0
    with open(fileName) as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Throw away header line
        reader.next()
        while 1:
            line = ""
            try:
                line = reader.next()
            except StopIteration:
                break
            header = line[0]
            headerFields = [x for x in header.split(' ') if len(x) > 0]
            description = reader.next()
            bets = reader.next()[0]
            track = reader.next()[0]
            raceNumber = int(reader.next()[0])
            date = reader.next()[0]
            junk = reader.next()

            raceId=track+"_"+date+"_"+str(raceNumber)
            try:
                skip = False
                maiden = parseMaiden(description)
                raceType = description[1]
                betDict = parseBetLine(bets, raceId)
                # Track is in 2 places, check that they match
                assert headerFields[0] == track, "Track isn't matching: {} != {} in {}".format(line, track, raceId)
                # RaceNumber is in 2 places, check that they match
                assert int(headerFields[1][1:]) == raceNumber
                # Distance and surface in the header line
                try:
                    distance = parseDistance(headerFields[2])
                except InvalidDistance, e:
                    print "Invalid distance {}.  Skipping race {}".format(e.distance, raceId)
                    skip = True
                surface = headerFields[3].upper()
                datetime = parseDateTime(headerFields[4:])
                raceFields = {'id':raceId, 'track':track,'date':date, 'raceType': raceType, 'raceNumber':raceNumber, 'betSummary':bets, 'maiden':maiden, 'distance':distance, 'surface':surface, 'minBet':betDict}
                if datetime != None:
                    raceFields['startTime'] = datetime
                race = Race(raceFields)
                headers = reader.next()
                while 1:
                    next = reader.next()
                    if(next[0] == 'end_of_race.'):
                        break
                    runnerFields = dict(zip(headers, next))
                    runnerFields['id'] = race.id+'_'+str(runnerFields['PP'])
                    race.addRunner(runnerFields, {'name' : 'Name', 'pgmPos':'PP', 'probUPRML':'UPRProb', 'probBC':'BCProb', 'surface':surface})
                if not skip:
                    raceDict[race.id] = race
                else:
                    rejected += 1
            except BaseException, e:
                type, value, traceback = sys.exc_info()
                print "Error parsing race {} in {}".format(fileName, raceId)
                raise type, value, traceback
    return (raceDict, rejected)

def parseMaiden(description):
    try:
        return maidenRaceClass(description[1])
    except IndexError, e:
        type, value, traceback = sys.exc_info()
        print "Error parsing maiden designation: {}".format(description)
        raise type, value, traceback

def maidenRaceClass(raceClass):
    return raceClass in ("M", "S")

def parseDistance(distance):
    '''Returns the distance in yards given the text description'''
    m = re.search("^([0-9.]+)(f)$", distance)
    if not hasattr(m, 'group'):
        raise InvalidDistance(distance)
    return float(m.group(1)) * 220

def parseDateTime(dateFields):
    '''Returns the timestamp for the race from the time in the description'''
    if len(dateFields) == 1:
        return None
    assert dateFields[1] == "Eastern"
    return datetime.datetime.strptime(dateFields[0], "%m-%d-%Y<br>%H:%M")


# Used to normalize some things in the bet lines so the matching is easier
betLineReplacements = []
betLineReplacements.append((("TRIFECTA / 50 CENT TRIFECTA",), "50 CENT TRIFECTA"))
betLineReplacements.append((("SUPERFECTA 10 CENT SUPERFECTA", "SUPERFECTA / 10 CENT SUPERFECTA",), "10 CENT SUPERFECTA"))
betLineReplacements.append((("*","$5;000 GUARENTEED TRIFECTA POOL"), "/"))
betLineReplacements.append((("$1.00","1.00"), "$1"))
betLineReplacements.append((("$0.50", "$.50", "50-CENT", ".50 CENT", "50 CENT"), ".50"))
betLineReplacements.append((("$0.10", "$.10", "TEN CENT", "10-CENT", ".10 CENT", "10 CENT", "10CENT"), ".10"))
betLineReplacements.append((("$0.20", "$.20", "20-CENT", ".20 CENT", "20 CENT"), ".20"))
betLineReplacements.append((("FIRST", ), "1ST"))
betLineReplacements.append((("SECOND", ), "2ND"))
betLineReplacements.append((("THREE", ), "3"))
betLineReplacements.append((("FIVE", ), "5"))
betLineReplacements.append((("SIX", ), "6"))
betLineReplacements.append((("HI-5", "HI 5"), "HIGH 5"))
betLineReplacements.append((('(RACES 4-5-6-7-8-9)', '(RACES 5-6-7-8-9)', '(4-5-6-7-8)', '(RACES 1-2)', '(RACES 1 & 2)', '(RACES 5-10)', '(RACES 8-9)', '(RACES 8 & 9)', '(RACES 7 & 9)', '(RACES 1-2-3)', '(RACES 2-3-4)', '(RACES 3-4-5)', '(RACES 5-7)', '(RACES 4-5-6)', '(4-5-6)', '(RACES 5-6-7)', '(5-6-7)', '(RACES 6-7-8)', '(RACES 6-7-8-9)', '(6-7-8)', '(RACES 7-8-9)', '(7-8-9)', '(RACES 8-9-10)', '(8-9-10)', '(RACES 7-8-9-10)', '(RACES 6-7-8-9-10)', '(7-8-9-10)', '(RACES 2-3-4-5)', '(RACES 1-10)'), "(RACES)"))

# This is used to parse bet lines apart.  The keys must be listed in a "greedy" order, meaning that if one entire key is the start of another key, the longer key must be listed first.  e.g. "SUPERFECTA (10 CENT MIN)" must be listed before "SUPERFECTA".  There is a test that confirms this.
betLines = []
betLines.append((("$1 TRIFECTA BOX", '$1 PERFECTA BOX', '1ST HALF TRI SUPERFECTA', '1ST HALF TWIN TRIFECTA', '2ND HALF TWIN TRIFECTA', '2ND HALF TWIN-TRIFECTA'), {}))
betLines.append((("$3 EXACTA",), {'Exacta':(3.0,1.0)}))
betLines.append((("$2 EXACTA","$2 PERFECTA"), {'Exacta':(2.0, 1.0)}))
betLines.append((("PLACE AND SHOW EXACTA", "EXACTA", "$1 EXACTA", "EXACTOR", "PERFECTA"), {'Exacta':(1.0,1.0)}))
betLines.append((("$2 TRIFECTA",), {'Trifecta':(2.0,1.0)}))
betLines.append(((".50 TRIFECTA.", ".50 TRIFECTA", 'TRIFECTA (MIN .50)', 'TRIFECTA (.50. MIN.)', 'TRIFECTA (.50 MIN.)', 'TRIFECTA (.50 MIN.'), {'Trifecta':(0.5, 0.5)}))
betLines.append(((".20 TRIACTOR",), {'Trifecta':(1.0, 0.2)}))
betLines.append((("TRIFECTA", "$1 TRIFECTA", "$1TRIFECTA", "TRIACTOR"), {'Trifecta':(1.0, 1.0)}))
betLines.append((("$3 TRIFECTA ($1 TRI BOX)", ), {'Trifecta':(3.0, 1.0)}))
betLines.append((("$1 SUPERFECTA HIGH 5 - MANDATORY PAYOUT", "$1 SUPERFECTA HIGH 5", "SUPERFECTA 5", "SUPERFECTA HIGH 5"), {'Pentefecta':(1.0, 1.0)}))
betLines.append(((".50 SUPERFECTA HIGH 5","(.50) SUPERFECTA 5"), {'Pentefecta':(1.0, 0.5)}))
betLines.append(((".20 SUPERFECTA HIGH 5",), {'Pentefecta':(1.0, 0.2)}))
betLines.append(((".10 SUPERFECTA HIGH 5",), {'Pentefecta':(1.0, 0.1)}))
betLines.append(((".50 SUPERFECTA PICK 5 (RACES)","2ND HALF TRI SUPERFECTA"), {}))
betLines.append((("$2 SUPERFECTA",), {'Superfecta':(2.0, 1.0)}))
betLines.append((("$1 SUPERFECTA (.10 MIN.)",), {'Superfecta':(1.0, 0.1)}))
betLines.append((('.10 SUPERFECTA',), {'Superfecta':(0.1, 0.1)}))
betLines.append(((".20 SUPERFECTA",), {'Superfecta':(1.0, 0.2)}))
betLines.append(((".50 SUPERFECTA","(.50) SUPERFECTA","(.50)SUPERFECTA"), {'Superfecta':(0.5, 0.5)}))
betLines.append((("$1 SUPERFECTA", "SUPERFECTA"), {'Superfecta':(1.0,1.0)}))
betLines.append((("$1 ROLLING PICK 3", "$2 ROLLING DOUBLE", "PICK 6 (RACES)", "PICK 3 (RACES)", "$1 PICK 3 (RACES)", "$1 PICK 3", ".50 PICK 3 (RACES)", ".50 PICK 3", ".20 PICK 3 (RACES)", ".20 PICK 3", "$2 PICK 3 (RACES)", "$2 PICK 3", "$1 BET 3 (RACES)", ".50 PLAYER'S PICK 5 (14% TAKEOUT - MANDATORY PAYOUT)", "$1 PLACE PICK ALL (RACES)", "$1 PLACE PICK ALL", ".50 PICK 4 (RACES)", ".20 PICK 4 (RACES)", "$2 PICK 6 (RACES)", "$2 PICK 6"), {}))
betLines.append((("$1 DAILY DOUBLE",), {}))
betLines.append((("$2 1ST HALF DAILY DOUBLE", "$2 2ND HALF DAILY DOUBLE", "$2 1ST HALF EARLY DOUBLE", "$2 1ST HALF LATE DOUBLE", "$2 2ND HALF EARLY DOUBLE", "$2 2ND HALF LATE DOUBLE", "$2 1ST HALF EARLY DAILY DOUBLE", "$2 1ST HALF LATE DAILY DOUBLE", "$2 2ND HALF EARLY DAILY DOUBLE", "$2 2ND HALF LATE DAILY DOUBLE", "2ND HALF EARLY DOUBLE", "2ND HALF LATE DOUBLE", "1ST HALF DAILY DOUBLE","2ND HALF DAILY DOUBLE","2ND HALF $2 LATE DAILY DOUBLE (RACES)", "$2 LATE DAILY DOUBLE (RACES)", "$2 DOUBLE","1ST HALF $2 DAILY DOUBLE (RACES)","2ND HALF $2 DAILY DOUBLE (RACES)","$2 DAILY DOUBLE"), {}))
betLines.append((("$2 QUINELLA","$3 QUINELLA"), {}))
betLines.append((("WPS","WIN", "PLACE", "SHOW", "AND", "&"), {}))

def verifyBetLines(data):
    '''Verifies that keys in betLines are never the start of a later key, which would prevent the later key from ever being detected.'''
    previousKeys = []
    for (keyList, val) in data:
        for key in keyList:
            for prevKey in previousKeys:
                assert len(key) > 1 or key == "&", "Problem with the betLines.  You probably have a tuple that has just one element without a comma, so it is interpreted as a string.  Single tuples need to look like ('string',)"
                assert not key.startswith(prevKey), "{} starts with {}".format(key, prevKey)
            previousKeys.append(key)
verifyBetLines(betLines)

def parseBetLine(line, raceId = ""):
    ret = {}
    editedLine = line
    for (toReplaceList, replaceWith) in betLineReplacements:
        for toReplace in toReplaceList:
            editedLine = editedLine.replace(toReplace, replaceWith)
    for firstSplit in editedLine.split("/"):
        for entry in firstSplit.split(";"):
            trimmed = entry.strip()
            while 1:
                # Loop through the terms left and keep pulling them off until you get all the bets
                if len(trimmed) == 0:
                    break
                x = None
                for (keyList, d) in betLines:
                    for key in keyList:
                        if(trimmed.startswith(key)):
                            x = d
                            trimmed = trimmed[len(key):len(trimmed)].strip()
                            break
                    if (x != None):
                        break
                if x == None:
                    #print("Skipping unknown bet "+trimmed)
                    assert trimmed.find("EXACTA") == -1 and trimmed.find("TRIFECTA") == -1 and trimmed.find("SUPERFECTA") == -1, "Unknown bet entry '"+trimmed+"' in " + line
                    break
                else:
                    keyCheck = set(ret.viewkeys()).intersection(set(x.viewkeys()))
                    assert len(keyCheck) == 0, "Race {} has repeated bet {} in line: {}".format(raceId, keyCheck, line)
                    ret.update(x)
    return ret
