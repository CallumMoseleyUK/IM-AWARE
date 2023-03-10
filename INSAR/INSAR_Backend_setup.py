import Insar_Image_Processing as IIP

import pandas as pd 
import pathlib
import os 

dam_neighborhood_km = 2
# damData = pd.read_csv('Dam_Coordinates.csv').to_dict(orient = "records")

dirname = pathlib.Path(str(pathlib.Path.cwd()).split("SRC")[0])

filename = os.path.join(
                dirname, 'SRC/IM-AWARE-GIS/ANM_dam_data/Extra_Sites_05_2022.xlsx')
a = IIP.INSAR_PROCESSING(dam_neighborhood_km, process_raw_files=True, file_address = filename )


IIP.update_INSAR_table()

# a =  IIP.INSAR_PROCESSING( process_raw_files=False, )