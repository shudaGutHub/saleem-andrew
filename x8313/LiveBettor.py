import os

import pandas as pd
from x8313 import tuplize
from x8313.FileLoader import loadDefaultTrackFile, loadRacesFromDailyFile, \
    updateRacesWithTrack
from x8313.Models import validateWagers, WagerKey
from x8313.RaceFilters import SelectTrack
from x8313.TicketBoxer import HorseKey


def writeEBetFile(trackDict, races, filename, wagerList):
    with open(filename, 'wb') as f: 
        for (wagerKey, amount) in wagerList:
            race = races[wagerKey.raceId]
            wagerType=wagerKey.getWagerName()
            if wagerType=='Win':
                runners = [wagerKey.runners[0]]
            else:
                runners = [",".join([race.runnerIndex[runnerId].pgmPos for runnerId in runnerIds]) for runnerIds in wagerKey.runners]
            track = trackDict[race.track]
            fields = (str(race.raceNumber), wagerKey.getWagerName(), str(amount), '/'.join(runners))
            f.write('|'.join(fields)+"\n")


class LiveBettor(object):
    """ The rebate rates to use when no specific rebate amount is know/listed for the track.  """
    
    def __init__(self, tracks, raceFile, scratches, pools, outputDir):
        self.trackDict = loadDefaultTrackFile()
        
        (self.races, rejected) = loadRacesFromDailyFile(raceFile)
        #except:
            # Try tab-delimited if comman doesn't work
         #   print "Rerunning load of {} assuming tab-delimited format instead of CSV.".format(raceFile)
          #  (self.races, rejected) = loadRacesFromDailyFile(raceFile, delimiter='\t')
        initialSize = len(self.races)
        self.races = { key : value for (key, value) in self.races.iteritems() if SelectTrack(tracks)(value) is None}
        self._updateRacesWithScratches(scratches)
        self._updateCouplingAndNonbettable()
        self._updateRacesWithPoolSizes(pools)
        updateRacesWithTrack(self.races, self.trackDict)
        print "Loaded {} races for the selected tracks. (total {}, rejected {})".format(len(self.races), initialSize, rejected)
        self.outputDir = outputDir

    def generateBetFile(self, strategy):
        wagers = strategy.generateWagersForRaces(self.races, False)
        validateWagers(self.races, wagers)
        filename = self._generateFileName(strategy)
        writeEBetFile(self.trackDict, self.races, filename, wagers)

        testWagers = self.checkMinimums()
        badBets = self._generateFileName(strategy, "testBadBets.txt")
        writeEBetFile(self.trackDict, self.races, badBets, testWagers)
        
        return filename

    def _generateFileName(self, strategy, name="ebet.txt"):
        path = os.path.join(self.outputDir, strategy.getName(), self.races.values()[0].date.strftime("%Y-%m-%d"))
        try:
            os.makedirs(path)
        except:
            # If this fails, we will catch it later on when we try to create a file.  Problem is that it will always fail if the directory already exists.
            pass
        return os.path.join(path, name)

    def _updateRacesWithScratches(self, scratches):
        # Update scratches
        for (raceId, horseName) in scratches:
            self.races[raceId].runnerByHorseName(horseName).scratched = True
        
        # Anyone who isn't scratched gets marked as so.
        for race in self.races.values():
            for runner in race.runners:
                if not hasattr(runner, "scratched"):
                    runner.scratched = False
        
    def _updateCouplingAndNonbettable(self):
        for race in self.races.values():
            for runner in race.runners:
                runner.nonBetting = False
                if runner.resultPos != runner.pgmPos:
                    potentialCoupling = race.runnerByPgmPos(runner.resultPos)
                    if potentialCoupling is not None and potentialCoupling.isBettable():
                        runner.coupledTo = potentialCoupling.id
                if not hasattr(runner, "coupledTo"):
                    runner.coupledTo = None
        
        
    def _updateRacesWithPoolSizes(self, pools):
        for race in self.races.values():
            race.estimatedPoolSizes = pools[race.track]

    def checkMinimums(self):
        ''' Takes the first race from each track and generates a set of bets that should violate each of the rules. '''
        firstRaces = {}
        for race in self.races.itervalues():
            track = race.track
            if track not in firstRaces or firstRaces[track].raceNumber > race.raceNumber:
                firstRaces[track] = race
        illegalTickets = []
        for race in firstRaces.itervalues():
            for (wager, mins) in race.minBet.iteritems():
                nPlaces = self.num(wager)
                smallerBet = self.nextMin(mins[1])
                if mins[0] == mins[1]:
                    # MinBet is the same as min ticket.  Just create one bet. 
                    ticket = (WagerKey(race.id, WagerKey.Multihorse, self.createKey(race, nPlaces, 1, 1)), smallerBet)
                    illegalTickets.append(ticket)
                else:
                    combs = int(mins[0]/mins[1])
                    # Create a bet with 1 combination that is smaller than the ticket size.
                    ticket = (WagerKey(race.id, WagerKey.Multihorse, self.createKey(race, nPlaces, 1, 1)), self.nextMin(mins[0]))
                    illegalTickets.append(ticket)
                    # Now create that meets the minimum ticket size, but the individual bets are too small 
                    ticket = (WagerKey(race.id, WagerKey.Multihorse, self.createKey(race, nPlaces, combs, 10000)), smallerBet)
                    illegalTickets.append(ticket)
        return illegalTickets
                
    def createKey(self, race, n, min, max):
        ids = [runner.id for runner in race.bettableRunners()]
        seed = HorseKey(tuplize(tuple(ids[0:n])))
        while seed.nCombs() < min:
            it = seed.extIter(ids)
            seed = it.next()
        assert seed.nCombs() <= max, "Couldn't generate a test ticket with {}-{} combinations of {}.  Ended up with {} ({})".format(min, max, n, seed.nCombs(), seed)
        return seed

    def num(self, wagerType):
        if wagerType == "Exacta":
            return 2
        if wagerType == "Trifecta":
            return 3
        if wagerType == "Superfecta":
            return 4
        assert wagerType == "Pentafecta"
        return 5

    def nextMin(self, amount):
        ''' Returns the next smallest betting size '''
        if amount == 2.0:
            return 1.0
        if amount == 1.0:
            return 0.50
        if amount == 0.5:
            return 0.2
        if amount == 0.2:
            return 0.1
        assert amount == 0.1, "Invalid min bet {}".format(amount)
        return 0.05
                
    