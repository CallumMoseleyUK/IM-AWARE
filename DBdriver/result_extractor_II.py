
import os 
import pandas as pd
from pathlib import Path
import numpy as np
import DBFunctions as dbf

basePath = Path(os.path.realpath(__file__)).parent

def query_result(table, criterion, fieldsToCollect):
    #fields to collect should be a list
    #note about criterion
    #Damping = 0.05 Example of numerical criterion
    #File_Address = '/aaa/bb/cc ' example of text criterion

    fields = ''
    for each in fieldsToCollect:
        fields += '{},'.format(each)
    fields = fields[0:-2] + ' ' 

    sql1 = "select {} from {} where {} ".format(fields,table,criterion)
    out = dbf.do_in_DB(sql1)
    return out[0][0]
#print(get_simul_from_ID('Xingu_VALESA-DAMBREAK-20210901-174150_9'))

print(query_result('Analysis_Results', 'Damping = 0.04', ['File_Address'] ))