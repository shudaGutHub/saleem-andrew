import pickle
import x8313.DBLoader as db
import x8313.Models
import x8313.Simulate
from x8313.GeneratePickle import getSimulator
from collections import defaultdict
from x8313.Simulate import *
import pandas as pd
path_data = 'D:/data/'
path_pandas = path_data + 'datapandas/'
path_pickle = path_data + 'datapickle/'
path_wagers = path_data + 'datawagers/'
path_simulation = path_data + 'datasimulation/'
path_eclipse_simulation = path_data + 'dataeclipse/simulation/'
path_GTS = path_data + 'datasystemGTS/'
#df_race = sm.getDF("SELECT * from race")
#df_race=df_race[(df_race.date>20150000) &(df_race.track_id.isin(['PHA','PEN','CTX','MNR']))]

df_sim_races = pd.read_csv(path_GTS+'df_gts.csv')
df_sim_races.index= df_sim_races.race_id
races=list(df_sim_races.race_id.unique())
races_test = races[0:5]
import psycopg2
def makesqllist(id_list):
    idList=",".join(["'"+x+"'" for x in id_list])
    return idList


from x8313.Simulate import Simulator
import dateutil.parser

path_pickle = "D:/data/datapickle/"
path_pandas = "D:/data/datapandas/"


def getSimulator(races):
    race_list = ",".join(["'" + x + "'" for x in races])
    sql_races = "r.id in (" + race_list + ") "
    conn_string = "host='dev.tennis-edge.com' dbname='ac' user='ac_2012' password='suspicious'"
    print "Connecting to database\n    ->%s" % (conn_string)
    conn = psycopg2.connect(conn_string)
    simulator = Simulator.loadFromDbWithWhere(conn, sql_races, validateWagers=False)
    conn.close()
    return simulator



simulator = getSimulator(races_test)
print simulator.__dict__