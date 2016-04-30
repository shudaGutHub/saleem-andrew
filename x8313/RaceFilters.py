import numpy as np
import pandas as pd


class SelectAll(object):
    def __call__(self, race):
        return None

    def __repr__(self):
        return "SelectAll()"

def filterResult(test, reason):
    if test:
        return None
    else:
        return reason

class SelectRacesWithRunnerFactor(object):
    '''Selects only races where runners have a particular factor present'''
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, race):
        hasProb = len([1 for x in race.runners if not hasattr(x, self.factor)]) == 0
        return filterResult(hasProb, str(self))
    
    def __str__(self):
        return "Has {} factor".format(self.factor)

    def __repr__(self):
        return "SelectRacesWithRunnerFactor('{}')".format(self.factor)

class SelectMTCompatibleRaces(SelectRacesWithRunnerFactor):
    '''Selects only races that have the fields loaded from mt files'''
    def __init__(self):
        super(SelectMTCompatibleRaces, self).__init__("probBC")
    
    def __str__(self):
        return "Has MT file data"

    def __repr__(self):
        return "SelectMTCompatibleRaces()"

class SelectNumRunners(object):
    ''' Selects only races with a specified number of runners'''
    def __init__(self, numRunners):
        self.numRunners = numRunners
    
    def __call__(self, race):
        bettable = len([x for x in race.runners if x.isBettable()])
        return filterResult(bettable == self.numRunners, str(self))
    
    def __str__(self):
        return "numRunners {} runners".format(self.numRunners)

    def __repr__(self):
        return "SelectNumRunners({})".format(self.numRunners)
    
class SelectMinimumRunners(object):
    ''' Selects only races with a certain minimum number of runners'''
    def __init__(self, minRunners):
        self.minRunners = minRunners
    
    def __call__(self, race):
        bettable = len([x for x in race.runners if x.isBettable()])
        return filterResult(bettable >= self.minRunners, str(self))
    
    def __str__(self):
        return "At least {} runners".format(self.minRunners)

    def __repr__(self):
        return "SelectMinimumRunners({})".format(self.minRunners)
class SelectMaximumRunners(object):
    ''' Selects only races with a certain maximum number of runners'''
    def __init__(self, maxRunners):
        self.maxRunners = maxRunners
    
    def __call__(self, race):
        bettable = len([x for x in race.runners if x.isBettable()])
        return filterResult(bettable <= self.maxRunners, str(self))
    
    def __str__(self):
        return "At most {} runners".format(self.maxRunners)

    def __repr__(self):
        return "SelectMaximumRunners({})".format(self.maxRunners)

class SelectMinimumDistance(object):
    ''' Selects only races with a certain minimum distance in yards'''
    def __init__(self, minDistance):
        self.minDistance = minDistance
    
    def __call__(self, race):
        ret = race.distance >= self.minDistance
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "At least {} yards".format(self.minDistance)

    def __repr__(self):
        return "SelectMinimumDistance({})".format(self.minDistance)
class SelectMinimumPoolSize(object): 
    ''' Selects only races with a certain minimum pool size for a wagerType'''
    '''defaults : wagerType:'Trifecta', minPoolSize:25000'''
    def __init__(self, wagerType='Trifecta',minPoolSize=0):
        self.wagerType= wagerType
        self.minPoolSize=minPoolSize
    def __call__(self,race):
        ret = race.estimatedPoolSizes.get(self.wagerType,0.0) >= self.minPoolSize
        return filterResult(ret,str(self))
    def __str__(self):
        return "minPoolSize: {}".format(self.minPoolSize)
    def __repr__(self):
        return "SelectMinimumPoolSize({})".format(self.wagerType)
class SelectMaximumPoolSize(object):
    ''' Selects only races with a certainmaximum pool size for a wagerType '''
    '''defaults : wagerType:'Trifecta', minPoolSize:25000'''
    def __init__(self, wagerType='Trifecta',minPoolSize=0):
        self.wagerType= wagerType
        self.maxPoolSize=minPoolSize
    def __call__(self,race):
        ret = race.estimatedPoolSizes.get(self.wagerType,0.0) <= self.maxPoolSize
        return filterResult(ret,str(self))
    def __str__(self):
        return "maxPoolSize: {}".format(self.maxPoolSize)
    def __repr__(self):
        return "SelectMaximumPoolSize({})".format(self.wagerType)
        
class SelectMinimumClaimPrice(object):
    ''' Selects only races with a certain minimum distance in yards'''
    def __init__(self, minClaimPrice,inclusive=True):
        self.minClaimPrice = minClaimPrice
        self.inclusive = inclusive
    def __call__(self, race):
        claimingPrice=race.__dict__.get('claimingPrice',0.0)
        if self.inclusive:
            ret = claimingPrice >= self.minClaimPrice
        else:
            ret = claimingPrice > self.minClaimPrice
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "At least {} minClaimPrice".format(self.minClaimPrice)

    def __repr__(self):
        return "SelectMinimumClaimPrice({})".format(self.minClaimPrice)
class SelectMaximumClaimPrice(object):
    ''' Selects only races with a max claim price'''
    def __init__(self, maxClaimPrice,inclusive=True):
        self.maxClaimPrice = maxClaimPrice
        self.inclusive = inclusive
    def __call__(self, race):
        claimingPrice=race.__dict__.get('claimingPrice',0.0)
        if self.inclusive:
            ret = claimingPrice <= self.maxClaimPrice
        else:
            ret = claimingPrice < self.maxClaimPrice
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "At most {} maxClaimPrice".format(self.maxClaimPrice)

    def __repr__(self):
        return "SelectMaximumClaimPrice({})".format(self.maxClaimPrice)  
class SelectTrackConditions(object):
    ''' Selects only certain track conditions'''
    def __init__(self, *types):
        self.types = types

    def __call__(self, race):
        ret = race.trackCondition in self.types
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "trackCondition"

    def __repr__(self):
        return "TrackCondition({!r})".format(self.types)

class SelectSurface(object):
    ''' Selects only certain surfaces'''
    def __init__(self, *types):
        self.types = types

    def __call__(self, race):
        surf=race.__dict__.get('surface')
        ret = surf in self.types
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "Surface not "

    def __repr__(self):
        return "SelectSurface({!r})".format(self.types)
    
        
class SelectMaximumDistance(object):
    ''' Selects only races with a certain maximum distance in yards'''
    def __init__(self, maxDistance):
        self.maxDistance = maxDistance
    
    def __call__(self, race):
        ret = race.distance <= self.maxDistance
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "At most {} yards".format(self.maxDistance)

    def __repr__(self):
        return "SelectMaximumDistance({})".format(self.maxDistance)
    
class SelectTrack(object):
    ''' Selects only races run on specified tracks'''
    def __init__(self, tracks):
        self.tracks = tracks
    
    def __call__(self, race):
        ret = race.track in self.tracks
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "Not run on a selected track"

    def __repr__(self):
        return "SelectTrack({!r})".format(self.tracks)
class SelectCleanRunnerData(object):
    '''Selects only races where no runner has a zero'''
    def __init__(self,attr='score',badVal=-100):
        self.attr=attr
        self.badVal=badVal
    def __call__(self,race):
        runners=race.bettableRunners()
        scores=pd.Series({r.id:getattr(r,self.attr,0.0) for r in runners if hasattr(r,'attr') }).order(ascending=False)
        print ("min is:", scores.min())
        ret = scores.min()>self.badVal
        return filterResult(ret,str(self))
    def __str__(self):
        return "Has a {} in {} attribute".format(self.badVal,self.attr)

    def __repr__(self):
        return "SelectCleanSpeedData('{}')".format(self.attr)    
class SelectHasFinalOdds(object):
    '''Selects only races where no runner has a zero'''
    def __init__(self,attr='finalToteOdds'):
        self.attr=attr
    def __call__(self,race):
        runners=race.bettableRunners()
        finalOdds=pd.Series({r.id:r.__dict__.get('finalToteOdds',None) for r in runners}).order(ascending=False)
        ret = finalOdds.value_counts()[None]==0
        return filterResult(ret,str(self))
    def __str__(self):
        return "Has a zero in {} numeric attribute".format(self.attr)

    def __repr__(self):
        return "SelectHasFinalOdds('{}')".format(self.factor) 
    
class SelectRaceMinAttr(object):
    '''Selects only races with minimum level for attr default is HDWSpeedParForClassLevel''' 
    def __init__(self,minVal,attr='HDWSpeedParForClassLevel'):
        self.minVal=minVal
        self.attr=attr
    def __call__(self,race):
        ret=self.race.__dict__[self.attr]> self.minVal
        return filterResult(ret,str(self))
    def __str__(self):
        return "Has not greater than  {} SpeedPar".format(self.minVal)

    def __repr__(self):
        return "SelectRaceMinAttr('{}')".format(self.attr)    
                
            
class SelectSexRestriction(object):
    '''Selects by SexRestriction
     N - No Sex Restrictions
     M - Mares and Fillies Only
     C - Colts and/or Geldings Only
    F - Fillies Only'''
    def __init__(self, *types):
        self.types = types

    def __call__(self, race):
        sr=race.sexRestriction
        print ("sexRestriction:",sr)
        ret = race.sexRestriction in self.types
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "sexRestriction"

    def __repr__(self):
        return "sexRestriction({!r})".format(self.types)
            
class SelectAgeRestriction(object):
    ''' Selects only certain race types'''
    def __init__(self, *types):
        self.types = types

    def __call__(self, race):
        ar=race.ageRestriction
        print ("ageRestrictionr:",ar)
        ret = race.ageRestriction in self.types
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "ageRestriction"

    def __repr__(self):
        return "ageRestriction({!r})".format(self.types)
class SelectAgeLimit(object):
    ''' Selects only certain race types'''
    def __init__(self, *types):
        self.types = types

    def __call__(self, race):
        al=race.ageLimit
        print("al", al)
        ret = race.ageLimit in self.types
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "ageLimit"

    def __repr__(self):
        return "ageLimit({!r})".format(self.types)
class ExcludeRaceClass(object):
    '''Excludes race classifications i.e. SHP'''
    ''' Selects only certain race types'''
    def __init__(self, *types):
        self.types = types

    def __call__(self, race):
        ret = race.__dict__.get('raceClassification','X') not in self.types
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "raceClassification"

    def __repr__(self):
        return "raceClassification({!r})".format(self.types)
                   
class SelectRaceTypes(object):
    ''' Selects only certain race types'''
    def __init__(self, *types):
        self.types = types

    def __call__(self, race):
        ret = race.raceType in self.types
        return filterResult(ret, str(self))
    
    def __str__(self):
        return "Race type"

    def __repr__(self):
        return "RaceType({!r})".format(self.types)

class SelectEBetTracks(object):
    ''' Selects only races run on tracks with an EBet specifier'''
    def __call__(self, race):
        return filterResult(hasattr(race.trackObj, 'ebetMeetID') and race.trackObj.ebetMeetID >= 0, str(self))
    
    def __str__(self):
        return "Run on a track without an eBet id"

    def __repr__(self):
        return "SelectEBetTracks()"
class SelectDaysOfWeek(object):
    '''Selects races that occur on days of week where 0:Monday 6:Sunday'''
    def __init__(self, *daysOfWeek):
        self.daysOfWeek=daysOfWeek    
    def __call__(self, race):
       # ret = race.maiden == self.maiden
       
       return filterResult(race.date.weekday() in self.daysOfWeek,str(self))  
        #isWeekend=race.date.weekday() in self.daysOfWeek
        #ret = race.weekend == self.weekend
        #return filterResult(ret, str(self))    
    #filterResult(ret, str(self))
    
    def __str__(self):
        return "in {}".format(self.daysOfWeek)
    
    

    def __repr__(self):
        return "SelectDaysOfWeek({!r})".format(self.weekend)
    


class SelectByMaiden(object):
    ''' Selects Maiden'''
    def __init__(self, maiden):
        self.maiden = maiden
    
    def __call__(self, race):
        ret = race.maiden == self.maiden
        return filterResult(ret, str(self))
    
    def maidenStr(self):
        if self.maiden:
            return "maiden"
        else:
            return "non-maiden"
    
    def __str__(self):
        return "Only {} races".format(self.maidenStr())

    def __repr__(self):
        return "SelectByMaiden({!r})".format(self.maiden)

class SelectRacesById(object):
    '''Selects specific races by id'''
    def __init__(self, *ids):
        self.ids = ids
        
    def __call__(self, race):
        return filterResult(race.id in self.ids, str(self))
    
    def __str__(self):
        return "in {}".format(self.ids)

    def __repr__(self):
        return "SelectRacesById({!r})".format(self.ids)

class CompositeSelector(object):
    def __init__(self, *selectors):
        self.selectors = selectors
    
    def __call__(self, race):
        for selector in self.selectors:
            result = selector(race)
            if result is not None:
                return result
        return None

    def __repr__(self):
        return "CompositeSelector{!r}".format(self.selectors)
