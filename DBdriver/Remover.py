#datawarehouse remover 
#andres alonso Rodriguez PhD, March 22,2021
import shutil 
from pathlib import Path 


WorkFolder = Path.cwd()
WarehouseFolder = WorkFolder.joinpath('Warehouse')


shutil.rmtree(WarehouseFolder)