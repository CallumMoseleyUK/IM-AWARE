#Datawarehouse installer, 
#Andrés Alonso Rodriguez PhD. March 22, 2021


__author__ = "Andres"
__copyright__ = "Copyright 2021"
__credits__ = ["Dominic Calleja", "Andres ..."]
__license__ = "MIT"
__version__ = "0.1"
__date__='20/08/2021'
__maintainer__ = "Andres"
__status__ = "Working"


from datetime import datetime
import sqlite3
import os
from pathlib import Path 
import sys
import pandas as pd 

db_gen_dir = os.path.dirname(__file__)  # can be removed after setup.py installer complete
sys.path.append(db_gen_dir)
from Field_Extractor import *

"""
Functions for the generation of the database
"""

def do_in_DB (sqlc,dbfile):
    print(sqlc)
    conn = sqlite3.connect(dbfile) 
    cr = conn.cursor()
    cr.execute(sqlc)
    conn.commit
    conn.close()

def folder_creator(path): 
    if os.path.exists(path) == False:
        os.mkdir(path)
        return "FolderCreated"
    else:
        print("Folder Already Exists")  # sys.exit("Folder Already Exists")

def key_file_creator(path) :
    f = open(path,'w')
    f.write(str(key))
    f.close()


def gen_ANM_DB_data_fields(dataframe):
    keys = list(dataframe.columns.values)
    keys.remove('ID')
    types = []
    for k in keys:
        if isinstance(dataframe[k][0], str):
            types.append('text')
        elif isinstance(dataframe[k][0], float):
            types.append('real')
        elif isinstance(dataframe[k][0], int):
            types.append('integer')
        elif isinstance(dataframe[k][0], bool):
            types.append('text')
        else:
            types.append('text')

    outString = 'ID text primary key '
    for i, k in enumerate(keys):
        outString = outString + ', {} {}'.format(k, types[i])
    return outString


def copyR(logfile):
    """Generate a LOG FILE"""
    outputf = open(logfile, 'w')
    outputf.write('+'+'='*77+'+ \n')
    tl = 'IM AWARE DB Logfile.'
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = 'logging file for the database driver'
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = ' '
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = ' Version: '+__version__
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    outputf.write('|'+' '*77+'| \n')
    tl = __copyright__+' (c) ' + __author__
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    outputf.write('+'+'='*77+'+' + '\n')
    outputf.write('\n')
    outputf.close()
    return




"""
Begin generating database
"""
# Empty lists 
tableNames = []
schema = []
key = 1

# Load data from schema csv template
database_structure = create_all_schemas(fieldsDesc)

# Tabel 0 : ANM 
tableNames.insert(0, "ANM") 
schema.insert(0," create table " + tableNames[0] + " (") 
directory = str(Path(__file__).resolve().parent.parent)+'/ANM_dam_data' #home+'/im_aware_collab/SRC/IM-AWARE-GIS/ANM_dam_data'
ANM = pd.read_excel(directory +'/'+'processed_ANM 06-2021.xlsx',engine='openpyxl')
ANM.columns.values
schema[0] += gen_ANM_DB_data_fields(ANM) +')'

# Tabel 1: Asset_Data
tableNames.insert(1, "Asset_Data") 
schema.insert(1," create table " + tableNames[1] + " (") 
schema[1] += "ID integer primary key, Date text, Misc data, Modification text, "
schema[1] += "Dam_ID integer, File_Link text, Owner text) "

#Tabel 2: Flooding_Model_Desctiprion
table_name = list(database_structure.keys())[0]
tableNames.insert(2, table_name)
schema.insert(2, database_structure[table_name]['sql_create_table'])

#Tabel 3: 'Analysis_Results'
table_name = list(database_structure.keys())[1]
tableNames.insert(3, table_name) 
schema.insert(3, database_structure[table_name]['sql_create_table']) 

tableNames.insert(4, "Asset_Record") 
schema.insert(4," create table " + tableNames[4] + " (") 
schema[4] += "Asset_Record_ID text primary key, Dam_ID text, Entry_ID text, "
schema[4] += "Date text, Comment text, Owner text)"

#table 5: INSAR data 
tableNames.insert(5,"INSAR")
schema.insert(5," create table " + tableNames[5] + " (" ) 
schema[5] += "(DamID text not null, Scene text not null," 
schema[5] += "Variable text not null, Path text primary key)"

WorkFolder = Path(sys.argv[1])

if sys.argv[2]:
    Warehouse_name = sys.argv[2]
else:
    Warehouse_name = 'Warehouse'
    

WarehouseFolder = WorkFolder.joinpath(Warehouse_name)
KeyFolder = WarehouseFolder.joinpath("Keys")
#DbFile = WorkFolder.joinpath("DataWarehouse.db")


print(folder_creator(WarehouseFolder))
print(folder_creator(KeyFolder))

for x in tableNames:
    key_file_creator(str(KeyFolder.joinpath(f'{x}.txt')))
    try:
        os.mkdir(str(WarehouseFolder.joinpath(x)))
    except:
        print('dir {} alreagy exists'.format(str(WarehouseFolder.joinpath(x))))

dbfile = str(WarehouseFolder.joinpath('datawarehouse.db'))


logfile = str(WarehouseFolder)+'{}_log.txt'.format(Warehouse_name)
copyR(logfile)
outputf = open(logfile,'a')
outputf.write(15*'=')
outputf.write('<<HEAD PARAMS>>\n')
outputf.write('PATH = [{}]\n'.format(str(WarehouseFolder)))
outputf.write('DB = [{}]\n'.format(str(dbfile)))
outputf.write('<<END PARAMS>>\n')
outputf.write(4*'\n')
outputf.write(15*'=')

for x in schema :
    do_in_DB(x,dbfile)

time_stamp = datetime.now()
outputf.write('Instanciation Complete : {}'.format(time_stamp.strftime("%d/%m/%Y-%H:%M:%S")))
outputf.close()


#"Id text primary key, Dam_Name text not null, Company text, CPF_CNPJ text, "
#schema[0] += "Tailings_Material text, Height real, Stored_Volume real , Building_method text, Risk text "
#schema[0] += "Potential_Damage text, Type text, PNSB text, Emergency_Level integer, Current_Status text, Lat real"
#schema[0] += "Long real, Control integer )"
#" create table " + tableNames[2] + " (")
#schema[2] += "Dam_ID text, Date_Time text, Particle_Num_Dist text, Particle_Mass_Dist text,"  #Dist means distribution or range
#schema[2] += "Particle_Radius_Dist text, Damping_Dist text, Volume_Factor_Dist text, Latitute_Offset_Dist text,"
#schema[2] += "Longitude_Offset_Dist text, Tailings_Density_Dist real, max_simulated_time real, Owner text, Comment text,"
#schema[2] += "Code_Version text, Number_of_Children integer, ID text primary key) "

#tableNames.insert(3, "Analysis_Results")
#schema.insert(3," create table " + tableNames[3] + " (")
#schema[3] += "Analysis_Id text primary key, File_Address text, Output_Summary_File text, Simulation_Time text,"
#schema[3] += "Max_Distance real, Max_Velocity real, Total_Energy real, Flooding_Area  real,"
#schema[3] += "Type_of_Analysis text, Parent integer, Repeat integer)"
#schema[3] += "Particle_Number integer, Particle_Mass real, Particle_Radius_real,"

#schema[3] += "Damping real, Volume_Factor real, Latitude_Offset real, Longitude_Offset real, Tailings_Density real,"
#schema[3] += "Max_Distance real, Max_Velocity real, Total_Energy real, Flooding_Area real,"
#schema[3] += "Analysis_ID text, Evaluation_Time real, Type_of_Analysis text, Parent_ID text, Tree_Level integer,"
#schema[3] += "Repeat integer, File_Address text, Output_Summary text, ID text primary key)"
