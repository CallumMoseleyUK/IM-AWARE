import importlib
from pathlib import Path
import sys
import os
import dill
sys.path.append(str(Path(__file__).parent.parent))

from source_data.weather_data import External_Data, quarter_label, week_label, month_label, convert2date, split_data
from INSAR.insar_data import INSAR_DATA, gen_mask
from DAMS import ANM_DAMS, DAM

import numpy as np
from datetime import datetime
import pandas as pd

import pathos.multiprocessing as mp

import matplotlib.pyplot as plt


#Choose Dam
dams = ANM_DAMS()
dam_id = dams.dams_df['ID'][23]

# Load Insar Data
path_to_insar_results = ''
Insar = INSAR_DATA(dam_id, path_to_insar_results)


# Load weather data for dam site 
Weather = External_Data(dam_id)
Weather.get_daily_data(n_stations=1)


# Convert to weekly weather data
Weather.gen_preciptiation_data(period='W')
Weather.gen_temperature_data(period='W')
Weather.temp.get_extremes()



# Generate index for weather data
Start_Date = [datetime.strptime(i, '%Y-%m-%d') for i in Insar.Start_Date]
End_Date = [datetime.strptime(i, '%Y-%m-%d') for i in Insar.End_Date ]
Mid_Date = [(Start_Date[i] + (End_Date[i]-Start_Date[i])/2).date() for i in range(len(Start_Date))]
Mid_Date = np.sort(Mid_Date)

# Weekly label generator
Q_label = [week_label(d) for d in Mid_Date]



# Process insar data 
gkr_length_scale = np.sqrt(np.sum(np.diff(Insar.point_loc[['Lat', 'Long']].values[:2,:],axis=0)**2))
Insar.fit_gaussian_regressor('vert',gkr_length_scale)
N_interp = 40
Z_out = Insar.get_matrix('vert',  n_interp = N_interp)

XI, YI, xi, yi, mask = gen_mask(Insar.point_loc[['Lat', 'Long']].values, n_interp=N_interp)



# Get insar statistics 
max_z = Insar.get_GKR_statistic('nanmax')
min_z = Insar.get_GKR_statistic('nanmin')
mean_z = Insar.get_GKR_statistic('nanmean')
std_z = Insar.get_GKR_statistic('nanstd')

# Write insar statistics to table 
cols = ['INSAR_{}'.format(i) for i in ['max','min','mu','std']]
Data = pd.DataFrame(np.vstack([max_z,min_z,mean_z,std_z]).T, 
            columns = cols, 
            index = Q_label)


# Generate weather statistics 
A = Weather.temp.get_period_stat('nanmax')
Data['Temp_max'] = A
A = Weather.temp.get_period_stat('nanmin')
Data['Temp_min'] = A
A = Weather.temp.get_period_stat('nanmean')
Data['Temp_mu'] = A
A = Weather.temp.get_period_stat('nanstd')
Data['Temp_mustd'] = A
A = Weather.temp.high.get_period_stat('nanstd')
Data['Temp_maxstd'] = A
A = Weather.temp.low.get_period_stat('nanstd')
Data['Temp_minstd'] = A
A = Weather.temp.high.get_period_stat('nanmax')
Data['Temp_maxmax'] = A
A = Weather.temp.low.get_period_stat('nanmin')
Data['Temp_maxmin'] = A

cum_sum = []
for di in range(len(Start_Date)):
    cum_sum.append(np.nansum(Weather.prcp.data.loc[Start_Date[di]:End_Date[di]]))
Data['Prcp_cumsum'] = cum_sum

B = Weather.prcp.get_period_stat('nanmax')
Data['Prcp_max'] = B
B = Weather.prcp.get_period_stat('nanmin')
Data['Prcp_min'] = B
B = Weather.prcp.get_period_stat('nanmean')
Data['Prcp_mu'] = B
B = Weather.prcp.get_period_stat('nanstd')
Data['Prcp_mustd'] = B