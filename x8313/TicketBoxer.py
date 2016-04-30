from x8313 import tuplize, expandCombinations, floorTo, roundTo
from x8313.BettingStrategy import *
from x8313.Models import Runner
import cProfile
import math
import unittest

# The flat structure consists a dictionary where the key is tuples of horses and the value is a float.

class HorseKey(tuple):
    def extIter(self, ids):
        ''' An iterator over extensions of this tuple. '''
        n = len(self)
        for pos in range(n):
            for id in ids:
                if id not in self[pos]:
                    ret = []
                    for i in range(n):
                        if pos == i:
                            ret.append(self[i]+(id,))
                        else:
                            ret.append(self[i])
                    yield HorseKey(ret)

    def expandCombs(self):
        if not hasattr(self, 'combinations'):
            self._expandCombinations()
        return self.combinations

    def nCombs(self):
        return len(self.expandCombs())

    def _expandCombinations(self):
        ''' Given a ticket which can have multiple ids at a given position, expand that out into all the possible combinations it represents. '''
        ret = [[]]
        for runnersAtPos in self:
            # Go through each result in the current result list and extend each result if it makes sense to
            ix = 0
            size = len(ret)
            #print "{} {}".format(ix, size)
            while ix < size:
                currentResult = ret[ix]
                if len(runnersAtPos) == 1:
                    # One runner, just extend the current list
                    runner = runnersAtPos[0]
                    if runner in currentResult:
                        del ret[ix]
                        size -= 1
                    else:
                        currentResult.append(runner)
                        ix += 1
                else:
                    # Multiple runners, remove the existing and insert new
                    del ret[ix]
                    size -= 1
                    for runner in runnersAtPos:
                        if not runner in currentResult: 
                            newResult = list(currentResult)
                            newResult.append(runner)
                            ret.append(newResult)
        self.combinations = [tuple(x) for x in ret]
        self.combinations.sort()

class ProbList():
    def __init__(self, iter):
        self.list = list(iter)
        self._sortLists()
    
    def __len__(self):
        ret = len(self.list)
        assert ret == len(self.valueList)
        return ret

    def valueIx(self, ix):
        return self.valueList[-ix - 1]

    def _sortLists(self):
        self.list.sort()
        self.valueList = list(self.list)
        self.valueList.sort(cmp = lambda x1, x2: cmp(x1[1], x2[1]))
        self.total = sum([abs(x[1]) for x in self.list])

    def sumProbs(self):
        return self.total

    def find(self, comb, low = 0):
        ''' Does a binary search to find the given combination '''
        hi = len(self.list)
        mid = -1
        while low < hi:
            mid = (low+hi)/2
            midVal = self.list[mid]
            val = midVal[0] 
            if val < comb:
                low = mid+1
            elif val > comb:
                hi = mid
            else:
                return mid
        return -mid

    def sumProbsWithTicket(self, ticket):
        # Start with the full amount before accounting for this ticket
        total = self.total
        # Now adjust for each combination in the ticket
        low = 0
        (key, amount) = ticket
        for comb in key.expandCombs():
            ix = self.find(comb, low)
            if ix < 0:
                low = -ix
                total += amount
            else:
                low = ix
                val = self.list[ix][1]
                total =  total - abs(val) + abs(val - amount)
        #print "{} {} {}".format(self.total, ticket, self.list) 
        #assert abs(total-self.xsumProbsWithTicket(ticket))< 1e-5, "{} {}".format(total, self.xsumProbsWithTicket(ticket))
        return total

    def updateProbsWithTicket(self, ticket):
        (key, amount) = ticket
        combs = key.expandCombs()
        toAdd = []
        for (listIx, combIx) in self.iterProbAndCombs(combs):
            if listIx is None:
                toAdd.append((combs[combIx], -amount))
            elif combIx is not None:
                listVal = self.list[listIx]
                combVal = combs[combIx]
                assert listVal[0] == combVal 
                self.list[listIx] = (listVal[0], listVal[1] - amount)
        for x in toAdd:
            self.list.append(x)
        self._sortLists()

    def iterProbAndCombs(self, comb):
        listIx = 0
        ticketIx = 0
        while True:
            listVal = self.valOrNone(self.list, listIx)
            ticketVal = self.valOrNone(comb, ticketIx)
            if listVal is None and ticketVal is None:
                break
            if listVal is None:
                yield (None, ticketIx)
                ticketIx += 1
            elif ticketVal is None:
                yield (listIx, None)
                listIx += 1
            elif ticketVal == listVal[0]:
                yield (listIx, ticketIx)
                listIx += 1
                ticketIx += 1
            elif listVal[0] > ticketVal:
                yield (None, ticketIx)
                ticketIx += 1
            else:
                assert listVal[0] < ticketVal #, "{} {}".format(listVal[0], ticketVal)
                yield (listIx, None)
                listIx += 1
    
    def valOrNone(self, l, ix):
        return ix < len(l) and l[ix] or None

    def xsumProbsWithTicket(self, ticket):
        (key, amount) = ticket
        combs = key.expandCombs()
        s = 0
        for (listIx, combIx) in self.iterProbAndCombs(combs):
            if listIx is None:
                s += amount
            elif combIx is None:
                s += abs(self.list[listIx][1])
            else:
                s += abs(self.list[listIx][1] - amount)
            #print "{} {} {}".format(listIx, combIx, s)
        return s

    

class SmartTicketBoxer(object):
    # Basic greedy algorithm
    # Take the results and sort smallest to largest (and in horse order)
    # Start with the biggest and bet that, but less than the max
    # Take the next biggest and do it again
    def __call__(self, probs, minTicket, minBet):
        ''' Basic greedy algorithm.  Take the probabilities and sort largest to smallest.  Start with the largest and make a valid ticket, choosing the best extension.  Then work your way down.'''
        #print "Min Ticket: {} Min Bet: {}".format(minTicket, minBet)
        ids = list(set([id for prob in probs.iterkeys() for id in prob])) 
        currentProbs = ProbList(probs.iteritems())
        tickets = []
        skip = 0
        a=0
        b=0
        c=0
        d=0
        while True:
            #print "Current probs: {}, {}".format(currentProbs.sumProbs(), currentProbs.list)
            ticketSeed = self.takeFirst(currentProbs, minBet, skip)
            updatedTicket = ticketSeed
            a += 1
            while updatedTicket is not None and not self.meetsMinimum(updatedTicket, minTicket):
                b += 1
                #print "Ticket: {}".format(updatedTicket)
                updatedTicket = self.bestExpansion(ids, currentProbs, updatedTicket)
                #print "updated Ticket: {}".format(updatedTicket)
                if updatedTicket is None:
                    if roundTo(ticketSeed[1],minBet) > minBet:
                        c += 1
                        # Try again using a smaller bet size
                        ticketSeed = (ticketSeed[0], roundTo(ticketSeed[1]-minBet, minBet))
                        #print "Lowering bet size {}".format(ticketSeed)
                        updatedTicket = ticketSeed
                    else:
                        # Can't expand these any more.  See if we would do better by increasing bet size.
                        pass
                        #bestCombs = expandCombinations(ticketSeed[0])
                        #minCombs = int(minTicket/minBet)
                        #updatedAmount = (minCombs / len(bestCombs)) * minBet 
                        #baseCombs = {comb : updatedAmount for comb in bestCombs}
                        #minDiff = self.diffSumProbs(probs, baseCombs)
            if updatedTicket is None:
                d += 1
                skip += 1
                #print "Couldn't improve, trying next"
                if(abs(skip) >= len(currentProbs)):
                    break
                continue
            #print "Adding ticket: {}".format(updatedTicket)
            tickets.append(updatedTicket)
            currentProbs.updateProbsWithTicket(updatedTicket)
        #print "Final Prob: {} - counts {} {} {} {}".format(currentProbs.sumProbs(), a, b, c, d)
        return tickets
            
    def meetsMinimum(self, ticket, minTicket):
        ticketValue = ticket[0].nCombs()* ticket[1]
        return ticketValue >= minTicket

    def takeFirst(self, probs, minBet, skip):
        if len(probs) == 0:
            return None
        (key, value) = probs.valueIx(skip)
        if value >= minBet:
            bettableValue = floorTo(value, minBet)
        else:
            bettableValue = roundTo(value, minBet)
            if bettableValue <= 0:
                return None
        ret = (HorseKey(tuplize(key)), bettableValue)
        return ret

    def bestExpansion(self, ids, probs, ticket):
        startingSum = probs.sumProbsWithTicket(ticket)
        for ext in ticket[0].extIter(ids):
            extTicket = (ext, ticket[1])
            newSum = probs.sumProbsWithTicket(extTicket)
            if newSum < startingSum:
                return extTicket
        return None

if __name__ == "__main__":
    wagers = MultihorseWagers(None, None)
    probs = {'1':0.4, '2':0.4, '3': 0.05, '4': 0.05, '5': 0.05, '6': 0.03, '7': 0.02}
    risk = 100
    runners = [Runner({'id':key, 'pgmPos':key}) for key in probs.keys()]
    superfectaWagers = {key : value*risk for (key, value) in wagers.superfectaProbs(runners, probs).iteritems()}
    tickets_test = cProfile.run('SmartTicketBoxer()(superfectaWagers, 1.0, 0.1)')
    print tickets_test