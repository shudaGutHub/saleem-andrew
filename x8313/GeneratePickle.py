'''
Created on Jun 12, 2014

@author: shuda
'''
from collections import namedtuple, namedtuple
from datetime import date
from dateutil.parser import parse, parse
from itertools import combinations, permutations
from matplotlib.widgets import Cursor, Button
from multiprocessing import freeze_support, freeze_support
from psycopg2 import extras, extras, extras
import csv
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas.io.sql as sqlio
import pickle
import psycopg2
import x8313
import x8313.DBLoader as db
import scipy.stats as ss
import sys
import csv
from itertools import product
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


def writePickle(s, experiment_name, extension='.pickle'):
    '''Writes a pickle file holding simulator
        (s=simulator,path_data='D:/data/datapickle',experiment_name='simulator_2014',extension='.pickle'(default))
    '''
    pickle_file = path_pickle + experiment_name + extension
    fid = open(pickle_file, 'wb')
    pickle.dump(s, fid)
    fid.close()
    print ("wrote pickle:", pickle_file)
    return True
