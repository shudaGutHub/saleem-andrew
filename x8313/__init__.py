import datetime
import math

def dateRange(start_date, end_date):
    for x in range((end_date - start_date).days):
        yield start_date +datetime.timedelta(x)

def makesqllist(id_list):
    idList=",".join(["'"+x+"'" for x in id_list])
    return idList

def normProbDict(rawProbs):
    sumProbs = sum(rawProbs.values())
    assert sumProbs > 0, "Prob vector was 0"
    try:
        return { runnerId : rawProb/sumProbs for (runnerId, rawProb) in rawProbs.iteritems()}
    except:
        return None

def checkProbDict(probDict):
    ''' Verifies that a given dictionary contains a valid probability distribution (sums to 1) '''
    total = sum(probDict.itervalues())
    assert abs(total - 1) < 1e-5, "Probabilities summed to "+str(total)+" "+str(probDict)

def tuplize(key):
    ''' Convert a tuple of horses into a ticket '''
    return tuple([(x, ) for x in key])

def floorTo(prob, minBet):
    return math.floor(prob/minBet)*minBet

def floorAllTo(probs, minBet):
    return [(key, floorTo(value, minBet)) for (key, value) in probs]

def roundTo(prob, minBet):
    return round(prob/minBet)*minBet

def roundAllTo(probs, minBet):
    ''' Rounds all probs to the min bet size.  Used to determine the optimal betting '''
    [(key, roundTo(value, minBet)) for (key, value) in probs]


def expandCombinations(ticket):
    ''' Given a ticket which can have multiple ids at a given position, expand that out into all the possible combinations it represents. '''
    ret = [[]]
    for runnersAtPos in ticket:
        # Go through each result in the current result list and extend each result if it makes sense to
        ix = 0
        size = len(ret)
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
    return [tuple(x) for x in ret]
