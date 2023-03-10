from codecs import latin_1_decode
from pathlib import Path
import os 
import numpy as np
import pandas as pd
import handy_plotter as hpl
import folium

workpath = Path(__file__).parent
os.chdir(workpath)
import insar_data


def data_slicer(data,n):
    data_slice = data.iloc[:,:n]
    raw_cols = data_slice.columns
    out ={}
    for i in range(len(raw_cols)):
        if i == 0:
            di=0
        if i > 0:
            lat_long1 = raw_cols[i][1:-1].split(',')
            lat_long0 = raw_cols[i-1][1:-1].split(',')
            lat1 = float(lat_long1[0])
            long1 = float(lat_long1[1])
            lat0 = float(lat_long0[0])
            long0 = float(lat_long0[1])
            di += int(round((np.pi*6371*1000/90)*((lat1-lat0)**2 + (long1-long0)**2)**0.5,0))   
        out[str(di)] = data.iloc[:,i]
        outPd = pd.DataFrame(out)
        outPd.index.names = ['Date']
        outPd = outPd.reset_index()    
    
    return outPd

def panda2d_to_panda1d (panda, col_name):
    cls = panda.columns[1:]
    out = []
    for each in cls:
        out_dict = {'Date' : panda['Date'], 'z': panda[each], 'distance' : int(each)  }
        out_panda = pd.DataFrame(out_dict)
        out.append(out_panda)        
    return  pd.concat(out)

def mapper (panda):
    map = folium.Map(location = )


dataPATH = Path(__file__).parent.parent.parent.parent.joinpath('IMAWARE').joinpath('INSAR_RESULTS')
print(workpath)
sites = {'rock':'Rock16', 'dam':'SulSupe_VALESA'}
data = { 'rock': insar_data.INSAR_DATA(sites['rock'], str(dataPATH) ).data['cum_disp'], 'dam': insar_data.INSAR_DATA(sites['dam'], str(dataPATH)).data['cum_disp'] }
d = data_slicer(data['rock'],10)
e = panda2d_to_panda1d(d,'distance [mm]')
f = hpl.simple_line_plot(e,{'x':'Date', 'y':'z', 'colour':'distance'}, {'x':'Date', 'y':'cm displacement [mm]'})
f.save('rock.html')
g = data_slicer(data['dam'],10)
h = panda2d_to_panda1d(g,'distance [mm]')
i = hpl.simple_line_plot(h,{'x':'Date', 'y':'z', 'colour':'distance'}, {'x':'Date', 'y':'cm displacement [mm]'})
i.save('dam.html')
sites2 = {'rock':'Rock_17', 'dam':'CampoGr_VALESA'}
data2 = { 'rock': insar_data.INSAR_DATA(sites2['rock'], str(dataPATH) ).data['cum_disp'], 'dam': insar_data.INSAR_DATA(sites2['dam'], str(dataPATH)).data['cum_disp'] }
d2 = data_slicer(data2['rock'],10)
e2 = panda2d_to_panda1d(d2,'distance [mm]')
f2 = hpl.simple_line_plot(e2,{'x':'Date', 'y':'z', 'colour':'distance'}, {'x':'Date', 'y':'cm displacement [mm]'})
f2.save('rock2.html')
g2 = data_slicer(data2['dam'],10)
h2 = panda2d_to_panda1d(g2,'distance [mm]')
i2 = hpl.simple_line_plot(h2,{'x':'Date', 'y':'z', 'colour':'distance'}, {'x':'Date', 'y':'cm displacement [mm]'})
i2.save('Dam2.html')
sites3 = sites = {'rock':'Rock9', 'dam':'CampoGr_VALESA'}
data3 = { 'rock': insar_data.INSAR_DATA(sites3['rock'], str(dataPATH) ).data['cum_disp'], 'dam': insar_data.INSAR_DATA(sites3['dam'], str(dataPATH)).data['cum_disp'] }
d3 = data_slicer(data3['rock'],10)
e3 = panda2d_to_panda1d(d3,'distance [mm]')
f3 = hpl.simple_line_plot(e3,{'x':'Date', 'y':'z', 'colour':'distance'}, {'x':'Date', 'y':'cm displacement [mm]'})
f3.save('rock3.html')
g3 = data_slicer(data3['dam'],10)
h3 = panda2d_to_panda1d(g3,'distance [mm]')
i3 = hpl.simple_line_plot(h3,{'x':'Date', 'y':'z', 'colour':'distance'}, {'x':'Date', 'y':'cm displacement [mm]'})
i3.save('Dam3.html')
sites4 = {'rock':'Rock0', 'dam':'Alemxc3_GERDAUA'}
data4 = { 'rock': insar_data.INSAR_DATA(sites4['rock'], str(dataPATH) ).data['cum_disp'], 'dam': insar_data.INSAR_DATA(sites4['dam'], str(dataPATH)).data['cum_disp'] }
d4 = data_slicer(data4['rock'],10)
e4 = panda2d_to_panda1d(d4,'distance [mm]')
f4 = hpl.simple_line_plot(e4,{'x':'Date', 'y':'z', 'colour':'distance'}, {'x':'Date', 'y':'cm displacement [mm]'})
f4.save('rock4.html')
g4 = data_slicer(data4['dam'],10)
h4 = panda2d_to_panda1d(g4,'distance [mm]')
i4 = hpl.simple_line_plot(h4,{'x':'Date', 'y':'z', 'colour':'distance'}, {'x':'Date', 'y':'cm displacement [mm]'})
i4.save('Dam4.html')