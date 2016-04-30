from x8313.FileLoader import maidenRaceClass, multihorseKeys, parsePos,\
    MultiplePayoutError, rebatePctLookup
from x8313.Models import Track, Race, Runner, RacePayout, WagerKey, \
    RaceResultRunner, PastPerformance
import psycopg2

sqlRace="select * from  race"
sqlRaceResult="select * from race_result"
sqlRaceResultPool="select * from race_result_pool"
sqlRaceResultPayout="select * from race_result_payout"
sqlRaceResultRunner="select * from race_result_runner"
sqlRunner="select * from runner"
sqlRunnerNumeric="select * from runner_numeric_factor"
sqlTrackBetDetail="select * from track_bet_detail"

BADRACES = ['HST_20111002_8', 'WOX_20150621_5']
trackMappings = {"resultFileCode":'result_file_code', 'ebetMeetID':'e_bet_meet_id', "eBetSym":'e_bet_sym'}
poolMappings = {0 : 'WPS', 1 : 'WPS', 2 : 'WPS', 3: 'Exacta', 4: 'Trifecta', 5: 'Superfecta', 6:'Hi5' }
betMappings = {0 : 'win', 1 : 'place', 2 : 'show', 3: 'exacta', 4: 'trifecta', 5: 'superfecta', 6:'hi5' }

raceMappings = {'stateBred':'state_bred_flag','track':'track_id', 'raceNumber':'race_number', 'ageRestriction':'age_restriction', 'sexRestriction':'sex_restriction', 'ageLimit':'age_limit', 'raceConditions':'race_conditions', 'allWeatherSurface':'all_weather_surface', 'raceClassification':'race_classification', 'raceType':'race_type', 'claimingPrice':'claiming_price', 'betSummary':'bet_description'}
runnerMappings = {'rider':'jockey','jockey':'jockey','trainer':'trainer','pgmPos':'program_number', 'nonBetting':'nonbetting', 'morningLine':'morning_line', 'birthYear':'year_of_birth', 'horseClaimingPrice':'horse_claiming_price'}
pastPerformanceCallFields = [("pp_num", ("ppNum", int)),
    ("start_call_pos", ("startCallPos", int)),
    ("first_call_pos", ("firstCallPos", int)),
    ("second_call_pos", ("secondCallPos", int)),
    ("gate_call_pos", ("gateCallPos", int)),
    ("stretch_pos", ("stretchPos", int)),
    ("finish_pos", ("finishPos", int)),
    ("money_pos", ("moneyPos", int)),
    ("start_call_btn_lengths_ldr_margin", ("startCallBtnLengthsLdrMargin", float)),
    ("first_call_btn_lengths_ldr_margin", ("firstCallBtnLengthsLdrMargin", float)),
    ("second_call_btn_lengths_ldr_margin", ("secondCallBtnLengthsLdrMargin", float)),
    ("stretch_call_btn_lengths_ldr_margin", ("stretchCallBtnLengthsLdrMargin", float)),
    ("finish_call_btn_lengths_ldr_margin", ("finishCallBtnLengthsLdrMargin", float))]
pastPerformanceFields = [
    ("date", "date"),
    ("days_since_previous_race", "daysSincePreviousRace"),
    ("track_id", "trackId"),
    ("race_number", "raceNumber"),
    ("track_condition", "trackCondition"),
    ("distance", "distance"),
    ("approx_dist", "approxDist"),
    ("surface", "surface"),
    ("special_chute", "specialChute"),
    ("num_entrants", "numEntrants"),
    ("post_position", "postPosition"),
    ("equipment", "equipment"),
    ("lasix", "lasix"),
    ("bute", "bute"),
    ("weight", "weight"),
    ("odds", "odds"),
    ("entry_flag", "entryFlag"),
    ("claiming_price", "claimingPrice")
]
pastPerformanceDetailFields = [
    ("race_classification", "raceClassification"),
    ("comment", "comment"),
    ("extra_comment", "extraComment"),
    ("purse", "purse"),
    ("winners_name", "winnersName"),
    ("place_name", "placeName"),
    ("show_name", "showName"),
    ("winners_weight", "winnersWeight"),
    ("place_weight", "placeWeight"),
    ("show_weight", "showWeight"),
    ("winners_margin", "winnersMargin"),
    ("place_margin", "placeMargin"),
    ("show_margin", "showMargin")
]

def getConn():
    return psycopg2.connect(database='AnimalCrackers', user='postgres', password="user")
    #return psycopg2.connect(database='ac2011', user='ac_2011', password="suspicious", host='ec2-54-234-225-231.compute-1.amazonaws.com')

def loadTracks(con):
    trackDict = {}
    cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute('select * from track')
    rows = cur.fetchall()
    for row in rows:
        track = Track(row, trackMappings)
        trackDict[track.id] = track
    cur.close()

    cur = con.cursor()
    cur.execute('select * from track_pool_detail')
    for row in cur.fetchall():
        track = trackDict[row[0]]
        setattr(track, 'takeout'+poolMappings[row[1]], row[2])

    cur.execute('select * from track_bet_detail')
    for row in cur.fetchall():
        track = trackDict[row[0]]
        bet = betMappings[row[1]]
        rebate = row[2]
        minTicket = row[3]
        minBet = row[4]
        if rebate is not None:
            setattr(track, bet+"RebatePct", rebate)
        if minTicket is not None:
            setattr(track, "minTicket"+bet.capitalize(), minTicket)
        if minBet is not None:
            setattr(track, "minBet"+bet.capitalize(), minBet)
    return trackDict

def trainerQuery(con,whereClause):
    cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(whereClause)
    r = cur.fetchall()
    return r




def whereClauseForRacesByTrackAndDate(tracks, dateFrom, dateTo):
    trackList = ",".join(["'"+x+"'" for x in tracks])
    whereClause = "r.track_id in ("+trackList+") and r.date >= '{}' and r.date <= '{}'".format(dateFrom.strftime("%Y%m%d"), dateTo.strftime("%Y%m%d"))
    return whereClause

def whereClauseForRacesByDate(dateFrom, dateTo):
    return "r.date >= '{}' and r.date <= '{}'".format(dateFrom.strftime("%Y%m%d"), dateTo.strftime("%Y%m%d"))

def loadRacesWithWhere(con, trackDict, whereClause):
    raceDict = {}
    cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute('select * from race r where '+whereClause)
    for row in cur.fetchall():
        race = Race(row, raceFieldMap = raceMappings)
        race.maiden = maidenRaceClass(row['race_type'])
        race.minBet = {}
        race.trackObj = trackDict[race.track]
        raceDict[race.id] = race
        #print(race.__dict__)

    # Now load runners
    couplings = {}
    allRunnerMap = {}
    cur.execute('select u.*, ud.* from runner u join runner_detail ud using (id) join race r on (u.race_id = r.id) where '+whereClause)
    for row in cur.fetchall():
        race = raceDict[row['race_id']]
        runner = race.addRunner(row, runnerMappings)
        allRunnerMap[runner.id] = runner
        coupling = row["coupled_entry"]
        if coupling is not None:
            key = (race.id, coupling)
            couplings.setdefault(key, []).append(runner)
        else:
            runner.coupledTo = None
    for ((raceId, coupling), runners) in couplings.iteritems():
        if len(runners) == 1:
            runners[0].coupledTo = None
        else:
            # By convention, we treat the first program position alphabetically as the one to keep.  ('1' over '1A', '1A' over '1X')
            runners.sort(key=lambda runner: runner.pgmPos)
            mainRunner = runners[0]
            mainRunner.coupledTo = None
            for coupledRunner in runners[1:]:
                # Field horses have a coupling of F and they don't have the same program position.  (e.g. 11F and 12F).  They are running as the same betting entry
                # because there are too many horses in the race.
                assert  coupling == 'F' or coupledRunner.resultPos == mainRunner.resultPos, "{}\n{}\n{}".format(raceId, coupledRunner.__dict__, mainRunner.__dict__)
                coupledRunner.coupledTo = mainRunner.id

    # Add pastPerformance calls
    cur.execute('select f.* from past_performance_call f, runner u, race r where f.runner_id=u.id and u.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        runner = allRunnerMap[row['runner_id']]
        pp = PastPerformance()
        pp.runner = runner
        for field in pastPerformanceCallFields:
            v = row[field[0]]
            if v is not None:
                setattr(pp, field[1][0], field[1][1](v))
        runner.addPastPerformance(pp)

    cur.execute('select f.* from past_performance f, runner u, race r where f.runner_id=u.id and u.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        runner = allRunnerMap[row['runner_id']]
        pp = runner.pastPerformances[int(row['pp_num'])]
        for field in pastPerformanceFields:
            v = row[field[0]]
            if v is not None:
                setattr(pp, field[1], v)

    cur.execute('select f.* from past_performance_detail f, runner u, race r where f.runner_id=u.id and u.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        runner = allRunnerMap[row['runner_id']]
        pp = runner.pastPerformances[int(row['pp_num'])]
        for field in pastPerformanceDetailFields:
            v = row[field[0]]
            if v is not None:
                setattr(pp, field[1], v)

    cur.execute('select f.* from past_performance_numeric_factor f, runner u, race r where f.runner_id=u.id and u.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        runner = allRunnerMap[row['runner_id']]
        pp = runner.pastPerformances[int(row['pp_num'])]
        setattr(pp, row['factor_name'], float(row['factor_value']))
    cur.close()

    # Now load bet details
    cur = con.cursor()
    cur.execute('select b.* from race_bet_detail b, race r where b.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        race = raceDict[row[0]]
        minTicket = row[2]
        minBet = row[3]
        race.minBet[betMappings[row[1]].capitalize()] = (minTicket, minBet)

    # Add race numeric factors
    cur.execute('select f.* from race_numeric_factor f, race r where f.race_id=r.id and '+whereClause)
    for row in cur.fetchall():
        race = raceDict[row[0]]
        #print(runner.id+" "+row[1]+" "+str(row[2]))
        setattr(race, row[1], row[2])

    # Add runner numeric factors
    cur.execute('select f.* from runner_numeric_factor f, runner u, race r where f.runner_id=u.id and u.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        runner = allRunnerMap[row[0]]
        #print(runner.id+" "+row[1]+" "+str(row[2]))
        setattr(runner, row[1], row[2])
    cur.close()

    # Apply track overrides
    for race in raceDict.itervalues():
        track = trackDict[race.track]
        if "Exacta" in race.minBet and hasattr(track, 'minBetExacta') and hasattr(track, 'minTicketExacta'):
            race.minBet["Exacta"] = (track.minTicketExacta, track.minBetExacta)
        if "Trifecta" in race.minBet and hasattr(track, 'minBetTrifecta') and hasattr(track, 'minTicketExacta'):
            race.minBet["Trifecta"] = (track.minTicketTrifecta, track.minBetTrifecta)
        if "Superfecta" in race.minBet and hasattr(track, 'minBetSuperfecta') and hasattr(track, 'minTicketExacta'):
            race.minBet["Superfecta"] = (track.minTicketSuperfecta, track.minBetSuperfecta)

    return raceDict
#('maiden', bool), ('startTime', str),

class RefundError(BaseException):
    pass

def loadPayouts(con, trackDict, raceDict, whereClause):
    #RacePayout(race.id, int(d['numStarters']), payouts, poolSizes, rebatePcts)
    payoutDict = {}
    cur = con.cursor()
    cur.execute('select * from race_result x, race r where x.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        raceId = row[0]
        payoutDict[raceId] = RacePayout(raceId, row[1], {}, {}, {'Win':float('NaN'),'Place':float('NaN'),'Show':float('NaN'),'Exacta':float('NaN'), 'Trifecta':float('NaN'), 'Superfecta':float('NaN')}, {})
    cur.execute('select * from race_result_pool x, race r where x.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        raceId = row[0]
        payout = payoutDict[raceId]
        pool = poolMappings[row[1]]
        payout.poolSizes[(raceId, pool)] = row[2]
        if pool in rebatePctLookup:
            track = raceDict[raceId].trackObj
            attr = rebatePctLookup[pool]
            if hasattr(track, attr):
                payout.rebatePcts[pool] = getattr(track, attr)
    cur.execute('select * from race_result_payout x, race r where x.race_id = r.id and '+whereClause)
    badRaces = []
    for row in cur.fetchall():
        raceId = row[0]
        payout = payoutDict[raceId]
        betType = row[1]
        if betType == 3 or betType == 4 or betType == 5 or betType ==6:
            try:
                horses = row[3]
                payoutVal = row[4]
                if horses == "REFUND":
                    # TDODO Need to deal with what happens with a "REBATE".  See race ARP_20130526_1
                    raise RefundError
                multihorseKey = multihorseKeys(raceDict[raceId], horses)
                key = WagerKey(raceId, WagerKey.Multihorse, multihorseKey)
                if key in payout.payouts:
                    print "{} {} {}".format(raceId, key, payout.payouts)
                    raise MultiplePayoutError
                payout.payouts[key] = payoutVal
            except BaseException as inst:
                print("Base race {}".format(inst))
                badRaces.append(raceId)
        if betType == 0: #WIN #or betType == 1 or betType == 2: #WPS bets"
            try:
                horses = row[3]
                payoutVal = row[4]
                if horses == "REFUND":
                    # TDODO Need to deal with what happens with a "REBATE".  See race ARP_20130526_1
                    raise RefundError
             #   multihorseKey = multihorseKeys(raceDict[raceId], horses)
                singlehorseKey = (parsePos(raceDict[raceId], horses),)

                key = WagerKey(raceId, WagerKey.Win, singlehorseKey)
                if key in payout.payouts:
                    print "{} {} {}".format(raceId, key, payout.payouts)
                    raise MultiplePayoutError
                payout.payouts[key] = payoutVal
            except BaseException as inst:
                print("Base race {}".format(inst))
                badRaces.append(raceId)

    cur.execute('select * from race_result_runner x, race r where x.race_id = r.id and '+whereClause)
    for row in cur.fetchall():
        raceId = row[0]
        payout = payoutDict[raceId]
        payout.runners[row[1]] = RaceResultRunner(row[1], row[2], row[3], row[4], row[5])
    for race in badRaces:
        try:
            del payoutDict[race]
        except:
            print("couldnt delete:", race)
            pass
    return payoutDict

