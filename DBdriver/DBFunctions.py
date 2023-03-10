#Functions for Datawarehouse management 

# import sqlite3
from gc import collect
import shutil
import datetime 
from pathlib import Path  
import os 
import sys
from numpy.core import records
from numpy.lib.function_base import iterable
import pandas as pd 
import re
import DBdriver.Field_Extractor as fe
import directory_manager

import mysql.connector
from mysql.connector.constants import ClientFlag

home_addr = os.path.expanduser('~')
WorkFolder = directory_manager.get_work_dir()
WarehouseFolder = directory_manager.get_warehouse_dir()
KeyFolder = directory_manager.get_key_dir()
# dBname = str(directory_manager.get_database())
dbName = 'datawarehouseGCP'

# config = {
#     'user': 'root',
#     'password': 'imaware2021',
#     'host': '35.197.211.213',
#     'client_flags': [ClientFlag.SSL],
#     'ssl_ca': home_addr+'/cloudsql/server-ca.pem',
#     'ssl_cert': home_addr+'/cloudsql/client-cert.pem',
#     'ssl_key': home_addr+'/cloudsql/client-key.pem'
# }
config = {
    'user': 'root',
    'password': 'imaware2021',
    'host': '35.197.211.213',
    'port': '3306',
    'database': dbName}
# now we establish our connection

#Change workfolder to the place where the datawarehouse is being implemented
#WorkFolder = Path(str(Path.home())+'/im_aware_collab') #Path.cwd()

'''
## TODO: review the below suggestion for code portability.
    WorkFolder will be defined relative to the directory above SRC, so it will work
    if im_aware_collab is renamed or moved to a non-home directory.
'''
'''
WorkFolder = Path(str(Path.cwd()).split("SRC")[0])

WarehouseFolder = WorkFolder.joinpath('IMAWARE') 
KeyFolder = WarehouseFolder.joinpath('Keys') 
#>>>>>>> c28cf5444cb7cb6c5cf62401deead9638a36f66f
dBname = str(WarehouseFolder.joinpath('datawarehouse.db'))
print(dBname)
'''



def clean_ID_field(arg):
    #Function made by Dominic Calleja, 20-VIII-2021
    #It trasnforms any text into standard utf8 encoding
    return re.sub(r"[^a-zA-Z0-9]","", str(arg.encode('utf-8')))

def clean_escapes(arg):
    return re.sub(r"\\","",arg)


def do_in_DB(sqlc, dbName=dbName):
    # this function executes an sql command that inputs information in the database
    #input parameters
    #sqlc = sql Command that inputs information in the database
    # conn = sqlite3.connect (dBname)
    conn = mysql.connector.connect(**config)
    crs = conn.cursor()
    #crs.execute("USE {}".format(dbName))
    try:
        crs.execute(sqlc)
        conn.commit()
    except:
        print('UPLOAD ERROR: \n sql_error: \n\t {} \n\n Check Entry Consistency'.format(sqlc))
    conn.close()

def collect_from_DB(sqlc, dbName=dbName):
    #This function executes and sql command that extracts information from the database
    #input parameter
    #sqlc = sql command that extracts information from the database
    #try:
    # conn = sqlite3.connect(dBname)
    conn = mysql.connector.connect(**config)
    crs = conn.cursor()
    #crs.execute("USE {}".format(dbName))
    crs.execute(sqlc)
    output = crs.fetchall() 
    #conn.commit()
    conn.close()
    #except:
    #    return []
    return output

def check_key (table,keyFields,keyValues):
    #this function perform a sql search to check how many records share the same key. The result should be []
    #input parameters
    #table = table that is being queried
    #keyFields = list with the name of the fields that make up the key
    #keyValues = list of values from key fields
    sqlc = "select * from " + table + " where "
    for i in range(len(keyFields)):
        if type(keyValues[i]) is str :
            sqlc = sqlc + keyFields[i] + "  =  '" + keyValues[i] + "' AND "  
        else:
            sqlc = sqlc + keyFields[i] + "  =  " + keyValues[i] + " AND " 
    sqlc = sqlc[0:-4]    
    return collect_from_DB(sqlc)

def get_current_time():
    return str(datetime.datetime.now().strftime("%H:%M:%S"))


def get_current_time_forPath():
    return str(datetime.datetime.now().strftime("%H_%M_%S"))
    

def current_t():
    # a simple time stamp
    day = datetime.date.today().strftime("%Y-%m-%d")
    time = get_current_time()
    return '{}t{}'.format(day,time)

def insert_into_DB(table, keyFields, keyValues, record):
    #function that inserts a record into the database
    #input parameters
    #table : table where the record is going to be inputed
    #keyFields: list with the name of the fields that make up the key
    #keyValues = list of values from key fields
    #record = dictionary that contains the record. Its keys are the fields of the table
    
    repeated = len(check_key(table,keyFields,keyValues))

    if repeated == 0:
        fields = record.keys()
        fieldsS = ""
        valuesS = ""
        for x in fields:
            #fieldsS += x + " , "
            fieldsS += "`%s` , " % x
            if type(record[x]) is str : 
                valuesS += "'" + str(record[x]) + "' , "
            else:
                valuesS += str(record[x]) + " , "
        fieldsS = fieldsS[0:-2]
        valuesS = valuesS[0:-2]
        sql = "insert into {} ({}) {} ({})".format(table,fieldsS,"values",valuesS)

        do_in_DB(sql)

        return 'OK'
    else: 
        return 'record already exists' 

def insertFile(record,table,filepath):
    #function that inserts a record in the database and saves a file in the file management system
    #parameters:
    #record = dictionary that contains the record. Its keys are the fields of the table
    #table = table where the record is going to be inputed
    #filepath = path of the file that is going to be copied into the file system

    repeated = len(check_key(table,['ID'],[record['ID']]))
    record2 = record 
    if repeated == 0:
        ext = filepath.name.split('.')[1]
        endPath = WarehouseFolder.joinpath(table) 
        endPath = endPath.joinpath(record['ID'] + '.' + ext)
        record2['File_Address'] = str(endPath) 
        shutil.copyfile(filepath ,endPath)
        insert_into_DB(table,['ID'],[record2['ID']],record2)
        return 'Success'  

    else:
        return 'failure'  
    #note :
    #the file is going to be saved to the folder associated with the table, and its filename
    # will match the Id field. This will ensure the file is unique.  

def pandasToDb(pandaData, table, keyFields):
    # This function will insert a pandas dataframe into the database
    #parameters:
    #pandaData = Panda Dataframe to be inserted
    #Table = Table where the panda Dataframe will be inserted
    #keyFields = list containing the names of the key fields in the database
    pandaData = pandaData.fillna('')
    bigDic = pandaData.to_dict(orient='records')
    for record in bigDic:
        keyParams = []
        for param in keyFields:
            keyParams.append(str(record[param]))
        try:
            insert_into_DB(table,keyFields,keyParams,record)
        except:
            print('UPLOAD ERROR: \n Failed to add : \n\t {} \n\n Check Entry Consistency'.format(keyParams))

def get_valid_fields(table,fields=None):
    '''
    Returns a list of field names removing ones which conflict with MySql syntax (e.g. 'Repeat' and 'Long')
        Output 0: list of valid fields
        Output 1: list of invalid fields excluded
    '''
    if not fields: fields = get_fields(table)

    sqlCode = 'SELECT ID \n \
            FROM {} \n \
            LIMIT 1'.format(table)
    ID = collect_from_DB(sqlCode)[0][0]
    criterion = 'ID=\'%s\'' % ID

    validFields = fields.copy()
    invalidFields = []
    for f in fields:
        sqlCode = "select {} from {} where {} ".format(f,table,criterion)
        try:
            collect_from_DB(sqlCode)
        except:
            validFields.remove(f)
            invalidFields.append(f)
    return validFields,invalidFields

def update_record(table,ID,newRecord):
    '''
    Updates a record
    '''
    primaryKey = get_primary_key(table)
    
    fieldUpdates = ''
    keyList = list(newRecord.keys())
    for i,key in enumerate(keyList):
        value = newRecord[key]
        if isinstance(value,str):
            fieldUpdates += '`%s` = \'%s\'' % (key,value)
        else:
            fieldUpdates += '`%s` = %s' % (key,value)
        if i < (len(keyList)-1):
            fieldUpdates += ', '

    sqlCode = 'UPDATE %s SET %s WHERE %s = \'%s\';' % (table,fieldUpdates,primaryKey,ID)
    do_in_DB(sqlCode)

    return None
def delete_record(table,ID):
    '''
    Deletes a record with a given ID from the specified table.
    '''
    primaryKey = get_primary_key(table)
    sqlCode = 'DELETE FROM `%s` WHERE %s = \'%s\';' % (table,primaryKey,ID)
    do_in_DB(sqlCode)

def query_result(table, criterion, *fieldsToCollect):
    #fields to collect should be a list
    #note about criterion
    #Damping = 0.05 Example of numerical criterion
    #File_Address = '/aaa/bb/cc ' example of text criterion
    
    if fieldsToCollect:
        fieldsToCollect = fieldsToCollect[0]
        if isinstance(fieldsToCollect,str):
            fieldsToCollect = [fieldsToCollect]
    else:
        fieldsToCollect = get_fields(table)

    fields = ''
    # Put back-ticks around each field to prevent conflicts with MySql reserved words
    for each in fieldsToCollect:
        fields += '`{}`,'.format(each) 
    fields = fields[0:-1]
    sql1 = "select {} from {} where {} ".format(fields,table,criterion)

    out = collect_from_DB(sql1)
    lout = []
    for record in out:
        rout = {}
        for i in range(len(fieldsToCollect)):
            rout[fieldsToCollect[i]] = record[i]
        lout.append(rout)
    return lout


def query_by_dam(damID,table,*fieldsToCollect):
    '''
    Returns all table records for a given (unique) dam ID
    e.g. records = query_by_dam('Alexmc3_GERDAUA','Flooding_model_Description')
         records = query_by_dam(['Alexmc3_GERDAUA', 'B3B4_Minerax'],'Flooding_model_Description')
    '''
    records = []
    if not isinstance(damID,list):
        damID = [damID]
    for id in damID:
        records += query_result(table, 'Dam_ID = \'{}\''.format(id),*fieldsToCollect)

    return records


def get_primary_key(table, dbName=dbName):
    '''
    Returns the primary key of the given table
    '''
    sqlc = 'SELECT COLUMN_NAME \n \
            FROM INFORMATION_SCHEMA.COLUMNS \n \
            WHERE TABLE_SCHEMA = "{}" \n \
            AND TABLE_NAME = "{}" \n \
            AND COLUMN_KEY = "PRI"'.format(dbName, table)
    out = collect_from_DB(sqlc)
    return out[0][0]

def get_fields(table, dbName=dbName):
    '''
    Returns a list of all field names in a given table
    '''
    sqlc = 'SELECT COLUMN_NAME \n \
            FROM INFORMATION_SCHEMA.COLUMNS \n \
            WHERE TABLE_SCHEMA = "{}" \n \
            AND TABLE_NAME = "{}"'.format(dbName, table)
    out = collect_from_DB(sqlc)

    # TODO: removing MySql keyword Repeat from table schema should be done on the database level
    #return [f[0] for f in out if f[0]!='Repeat' and f[0]!='Long']
    return [f[0] for f in out]
    

def query_by_ID(ID,table,*fieldsToCollect):
    '''
    Returns the database record from the given table with primary key ID
    '''
    primaryKey = get_primary_key(table)
    records = query_result(table,'%s = \'%s\'' % (primaryKey,ID),*fieldsToCollect)
    if len(records)>1:
        print('warning: more than one entry with same unique ID')
    if len(records)==0:
        return None
    return records[0]

def get_all_dams(*tables):
    '''
    Retrieve all dams from from ANM table.
    If a list of tables are given, returns a list of dams common to all of them.
    NOTE: assumes all non-ANM tables use Dam_ID as their dam field.
    '''

    # Retrieve full ANM dam list
    # - This must be done separately, since dam ID is the primary key for the ANM table
    if not tables:
        sqlc = 'SELECT DISTINCT ID FROM ANM'
        queryOut = collect_from_DB(sqlc) #note: returns a list of 1-tuples
        damSet = set()
        for i in queryOut:
            damSet.add(i[0])
    else:
        tables = tables[0]
        if isinstance(tables,str):
            tables = [tables]
        damSet = set()
        for table in tables:
            sqlc = 'SELECT DISTINCT Dam_ID FROM %s' % table
            queryOut = collect_from_DB(sqlc)
            for i in queryOut:
                damSet.add(i[0])

    damList = list(damSet)
    damList.sort()
    return damList

'''
NOTE: The following functions are specific to dam break analyses, and should maybe be moved elsewhere.
'''
def query_by_analysis(analysisID,*fieldsToCollect):
    '''
    Returns all Analysis_Results records for a given (unique) analysis ID
    e.g. records = query_by_analysis('Alemxc3_GERDAUA-DAMBREAK-20210910-102550')
    '''
    records = []
    if not isinstance(analysisID,list):
        analysisID = [analysisID]
    for id in analysisID:
        records += query_result('Analysis_Results','Analysis_ID = \'{}\''.format(id),*fieldsToCollect)

    return records

def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

def query_all_analyses(damID, *fieldsToCollect):
    '''
    Returns all Analysis_Results records for a (unique) dam ID
    e.g. records = query_by_analyses('Alemxc3_GERDAUA-DAMBREAK-20210910-102550')
    '''
    floodRecs = query_by_dam(damID, 'Flooding_Model_Description', ['ID'])
    recList = []
    for r in floodRecs:
        query = query_by_analysis(r['ID'], *fieldsToCollect)
        if not isinstance(query,list): query = [query]
        recList += query
    
    return recList

def query_by_sim(simID):
    '''
    Returns the Analysis_Results record corresponding to unique ID simID
    '''
    return query_result('Analysis_Results','ID = \'{}\''.format(simID))

def query_by_type(queryType,damID):
    '''
    Returns a list of Flooding_Model_Description table records fitting the prescribed type
    Types:
    - Monte_Carlo
    - Particle_Number varied
    - Damping varied
    - Volume_Factor varied
    - Latitude_Offset varied
    - Tailings_Density varied

    Example use: records = query_by_type('Monte_Carlo')
                 records = query_by_type('Damping_Dist')
                 records = query_by_type(['Damping_dist','Volume_factor_Dist'])
    '''

    table = 'Flooding_Model_Description'
    bMonteCarlo = queryType=='' or queryType=='Monte_Carlo'
    fieldsToCollect = get_fields(table)

    ## NOTE: Should Particle_Num_Dist be considered its own simulation "type" based on resolution?
    #distTypes = ['Particle_Num_Dist', 'Damping_Dist', 'Volume_Factor_Dist', 'Latitude_Offset_Dist', 'Longitude_Offset_Dist','Tailings_Density_Dist']
    distTypes = ['Particle_Num_Dist', 'Damping_Dist', 'Volume_Factor_Dist', 'Latitude_Offset_Dist', 'Longitude_Offset_Dist']
    #distTypes = ['Damping_Dist', 'Volume_Factor_Dist', 'Latitude_Offset_Dist', 'Longitude_Offset_Dist','Tailings_Density_Dist']

    criterion = 'Dam_ID = \'%s\' AND ' % damID
    if not bMonteCarlo:
        if isinstance(queryType,str):
            queryType = [queryType]

        simTypeIn = queryType.copy()
        notTypes = [i for i in distTypes if not i in simTypeIn or simTypeIn.remove(i)]
        for i in notTypes:
            criterion += i + " LIKE \'(constant%\'"
            if i!=notTypes[-1]:
                criterion += ' AND '
    else:
        notTypes = []
        queryType = distTypes

        for i in queryType:
            criterion += i + " LIKE \'(rand_%\'"
            if i!=queryType[-1]:
                criterion += ' AND '

    records = query_result(table,criterion,fieldsToCollect)
    return records
'''
if __name__ == '__main__':


    ## All distribution types:
    ['Particle_Num_Dist', 'Damping_Dist', 'Volume_Factor_Dist', 'Latitude_Offset_Dist', 'Longitude_Offset_Dist','Tailings_Density_Dist']
    Use ['Damping_Dist', 'Volume_Factor_Dist', 'Latitude_Offset_Dist', 'Longitude_Offset_Dist'] for the last run of "Monte Carlo" type simulations

    r1 = query_result('Flooding_Model_Description','Dam_ID = \'Alemxc3_GERDAUA\'')

    r2 = query_by_type('Damping_Dist')

    r3 = query_by_type('Monte_Carlo')

    r4 = query_by_type(['Damping_Dist', 'Volume_Factor_Dist', 'Latitude_Offset_Dist', 'Longitude_Offset_Dist'])
'''
