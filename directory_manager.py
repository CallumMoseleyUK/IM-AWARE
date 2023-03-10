from pathlib import Path  
import os 
import sys

def clean_dir_string(dirFunc):
    def inner(*args):
        path = dirFunc(*args)
        return str(Path(path)).replace('\\','/')
    return inner

@clean_dir_string
def get_work_dir(*args):
    '''
    Returns the top level im_aware_collab directory
    '''
    if not args:
        fullPath = Path.cwd()
    else:
        fullPath = args[0]
    return Path(str(fullPath).split("SRC")[0])

@clean_dir_string
def get_code_dir(*args):
    '''
    Returns the im_aware_collab/SRC/IM-AWARE-GIS Git repository
    '''
    workDir = Path(get_work_dir(*args))
    return workDir.joinpath('SRC/IM-AWARE-GIS')

@clean_dir_string
def get_warehouse_dir(*args):
    '''
    Returns the im_aware_collab/IMAWARE data warehouse folder
    '''
    workDir = Path(get_work_dir(*args))
    return workDir.joinpath('IMAWARE')

@clean_dir_string
def get_key_dir(*args):
    '''
    Returns the im_aware_collab/IMAWARE/Keys data warehouse folder
    '''
    warehouseDir = Path(get_warehouse_dir(*args))
    return warehouseDir.joinpath('Keys')

## TODO: Change when moved to MySql
@clean_dir_string
def get_database():
    '''
    Returns the directory of the datawarehouse.db database
    '''
    warehouseDir = Path(get_warehouse_dir())
    return warehouseDir.joinpath('datawarehouse.db')

@clean_dir_string
def get_insar_folder(*args):
    '''
    Returns the INSAR warehouse directory
    '''
    warehouseDir = Path(get_warehouse_dir(*args))
    return warehouseDir.joinpath('INSAR')

@clean_dir_string
def get_dem_dir(*args):
    warehouseDir = Path(get_warehouse_dir(*args))
    return warehouseDir.joinpath('Sim_Raw/data_DEM')


# def get_user_folder(*args):
#     '''
#     Returns the INSAR warehouse directory
#     '''
#     warehouseDir = get_warehouse_dir(*args)
#     return warehouseDir.parent.joinpath('USER')
