import DBFunctions 
import pandas as pd
from pathlib import Path 
#Example with a pandas dataframe
# '/home/sgdcalle/im_aware_collab/SRC/IM-AWARE-GIS/ANM_dam_data/'
PATH = str(Path(__file__).resolve().parent.parent)+'/ANM_dam_data/'
data = pd.read_excel(PATH+'processed_ANM 06-2021.xlsx',engine='openpyxl')
DBFunctions.pandasToDb(data,'ANM',['ID'])

#Example with a flood simulation
#record = {'Analysis_Id' : 'ABC', 'Output_Summary' : 'xxxxx', 'Simulation_Time' : DBFunctions.current_t(),
#         'Max_Distance' : 12000.3, 'Max_Velocity' : 30.3, 'Total_Energy': 100.1, 'Flooding_Area' : 7.2,
#         'Particle_Number' : 120, 'Particle_Mass': 1.1, 'Particle_Radius': 2.3, 'Max_Time': 600.4,
#         'Damping' : 0.15, 'Volume_Factor': 0.55, 'Latitude_Offset' : 0.001, 'Longitude_Offset' : 0.003,
#         'Type_of_Analysis' : 'deterministic', 'Parent': '???' , 'Repeat' : 0}
#record['ID'] = '{}@{}'.format(record['Analysis_Id'],record['Simulation_Time'])
#mainPath = Path(__file__).resolve().parent
#filepath = mainPath.joinpath('NE.pptx')
#DBFunctions.insertFile(record,'Analysis_Results',filepath)
