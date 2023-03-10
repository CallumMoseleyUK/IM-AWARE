from pathlib import Path
from sqlite3.dbapi2 import Date
from altair.vegalite.v4.schema.core import Data
from folium.folium import Map
from numpy.core.arrayprint import DatetimeFormat
import pandas as pd
import numpy as np
import sys 
import matplotlib.pyplot as plt
import cv2
import datetime
sys.path.append(str(Path(__file__).parent.parent))

import INSAR.handy_plotter as hpl
import DBdriver.DBFunctions as DBF
from source_data.earth_data import Interactive_map as i_map
from source_data.earth_data import define_dot_markers 

import folium
import math
from io import StringIO
from google.cloud import storage
import io 
import os
from  source_data.GCPdata import GCP_HANDLER

#home_addr = os.path.expanduser('~')
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = home_addr + '/cloudsql/imaware-cloud-storage.json'

#storage_client = storage.Client()
#bucket_name = 'im_aware_collab'
#bucket = storage_client.get_bucket(bucket_name)

# from Insar_Image_Processing import INSAR_IMAGE, INSAR_PROCESSING
import datetime

repo = Path(__file__).parent.absolute().joinpath('Repository') 

import os

# os.chdir(Path(__file__).parent.parent)
from DAMS import ANM_DAMS, DAM
# os.chdir(Path.cwd().joinpath('DBdriver'))

# print(Path.cwd())

# os.chdir(Path(__file__).parent)


class INSAR_FILE():

    def __init__(self, dam, file_address, variable, datadf):
        self.dam = dam
        self.param = variable
        # self.file_address = self._DBquery(dam,scene,variable)
        self.file_address = file_address
        self.data = self.disp_to_mm(datadf)
        self.time_interval = self.get_scene()
        self.mid_interval = self.time_interval[0] + (self.time_interval[1] - self.time_interval[0])/2
        self.geo_extent = self.get_coordinate_extents()
        self.basic_stats = {'max': self.get_max(), 'min': self.get_min(), 'std': self.get_std(), 
                            'mean': self.get_mean(), 'Qs' : self.get_qrs()}
        #self.kde_plot = self.insar_kde()
    
    def disp_to_mm (self,dataf):
        if self.param == 'vert':
            dataf['z'] = 1000*dataf['z']
        return dataf

    def get_scene(self):
        date_interval = date_converter(str(Path(self.file_address).stem.split('_')[0]),str(Path(self.file_address).stem.split('_')[1]))
        return date_interval

    def get_coordinate_extents(self):
        return [self.data['Long'].min(), self.data['Long'].max(), self.data['Lat'].min(), self.data['Lat'].max()]

    def vert_d_correlogram(self, cord):
        corrs = []
        nBins = 24
        wdata = self.data.copy(deep = True)
        wdata['d'] = 6371*10**3*math.pi/360*((cord[0] - wdata['Lat'])**2 + (cord[1] - wdata['Long'])**2)**0.5
        bin_criteria = wdata['d'].max()/nBins
        wdata['dbin'] = wdata['d']/bin_criteria
        wdata['dbin'] = wdata['dbin'].apply(np.ceil).astype(int)
        data1 = wdata[wdata['dbin'] == 1]['z']
        corrs.append({'corr' : 1.00, 'd' : bin_criteria, 'n': len(data1) }) 
    
        for i in range(2,nBins+1,1):
            data2 = wdata[wdata['dbin'] == i]['z']
            sample_size = max(len(data1),len(data2))
            if len(data1) < sample_size: 
                dataA = data1.sample(sample_size, replace = True)
                dataB = data2
            elif len(data2) < sample_size:
                dataB =  data2.sample(sample_size, replace = True)
                dataA = data1
            else :
                dataA = data1
                dataB = data2
            corrs.append({'corr': np.corrcoef(dataA,dataB)[0,1], 'd': i*bin_criteria, 'n':sample_size})
            out_data = pd.DataFrame(corrs)
            #fig = hpl.simple_line_plot(out_data,{'x': 'd', 'y':'corr', 'other': ['n']},{'x' : 'distance [m]', 'y': 'Pearson correalation'})
        #return {'fig': fig, 'data' : out_data }
        return out_data

    def vert_angle_correlogram(self, cord):
        corrs = []
        nBins = 36
        wdata = self.data.copy(deep = True)
        wdata['angle'] = (180/np.pi)*np.arctan2((wdata['Long'] - cord[1]),(wdata['Lat']-cord[0]))
        bin_criteria = 360/nBins
        wdata['angle_bin'] = wdata['angle']/bin_criteria
        wdata['angle_bin'] = wdata['angle_bin'].apply(np.ceil).astype(int)
        data1 = wdata[wdata['angle_bin'] == 1]['z']
        corrs.append({'corr' : 1.00, 'angle' : bin_criteria, 'n': len(data1) }) 
        bin_range = wdata['angle_bin'].drop_duplicates().sort_values().to_numpy()
        for i in bin_range:
            data2 = wdata[wdata['angle_bin'] == i]['z']
            sample_size = max(len(data1),len(data2))
            if len(data1) < sample_size: 
                dataA = data1.sample(sample_size, replace = True)
                dataB = data2
            elif len(data2) < sample_size:
                dataB =  data2.sample(sample_size, replace = True)
                dataA = data1
            else :
                dataA = data1
                dataB = data2       
            corrs.append({'corr': np.corrcoef(dataA,dataB)[0,1], 'angle': i*bin_criteria, 'n':sample_size})
            out_data = pd.DataFrame(corrs)
            #fig = hpl.simple_line_plot(out_data,{'x': 'angle', 'y':'corr', 'other': ['n']},{'x' : 'angle [deg_dec]', 'y': 'Pearson correalation'})
            # return {'fig':fig, 'data' : out_data }
        return out_data
    
    def get_max(self):
        record = self.data.sort_values(by='z',ascending=False).head(1)
        return {'max': record['z'].to_numpy()[0], 'lat': record['Lat'].to_numpy()[0], 'long':record['Long'].to_numpy()[0]}

    def get_min(self):
        record = self.data.sort_values(by='z',ascending=True).head(1)
        return {'min': record['z'].to_numpy()[0], 'lat': record['Lat'].to_numpy()[0], 'long':record['Long'].to_numpy()[0]}

    def get_std(self):
        return self.data['z'].std()

    def get_mean(self):
        return self.data['z'].mean()

    def get_quantile(self, quantile):
        return self.data['z'].quantile(quantile)

    def get_qrs(self):
        return {'Q10': self.data['z'].quantile(0.1),'Q25' : self.data['z'].quantile(0.25) , 'median': self.data['z'].median(),
        'Q75': self.data['z'].quantile(0.75),'Q90': self.data['z'].quantile(0.9)}

    def get_n_closest_to(self,point,n):

        if  n == 1:
            self.data['d2_to_point'] = (self.data['Lat'] - point[0])**2+(self.data['Long'] - point[1])**2
            out =  self.data.sort_values(by='d2_to_point', ascending=True).head(1)
            return out['z'].to_numpy()[0]
        else:
            self.data['d2_to_point'] = (self.data['Lat'] - point[0])**2+(self.data['Long'] - point[1])**2
            out =  self.data.sort_values(by='d2_to_point', ascending=True).head(n)
            out = out.reset_index(drop=True)
            return out['z'], out[['Lat','Long']], out['Variable'].iloc[0], out[['d0','d1']].iloc[0]
            # Z, position, variable_name, date_range

    def get_interp(self,point):
        self.data['d2_to_point'] = (self.data['Lat'] - point[0])**2+(self.data['Long'] - point[1])**2
        out =  self.data.sort_values(by='d2_to_point', ascending=True).head(3)
        zs = np.transpose(np.array(out['z']))
        xs = np.transpose(np.array([out['Lat'],out['Long'],np.ones(3)]))
        bs = np.matmul(np.linalg.inv(xs),zs)
        return point[0]*bs[0]+point[1]*bs[1]+bs[2]

    def _comp_statistic(self, statistic,*args):
        if statistic == 'max':
            return self.get_max()
        if statistic == 'min': 
            return self.get_min()
        if statistic == 'mean':
            return self.get_mean()
        if statistic == 'std':
            return self.get_std()
        if statistic == 'closest': 
            return self.get_n_closest_to(args[0],args[1])
        if statistic == 'interp':
            return self.get_interp(args[0])
        if statistic == 'quantile':
            return self.get_quantile(args[0])

    def make_contour_plot(self):
        Ihopefig = hpl.triangular_contour_for_map(self.data['Long'], self.data['Lat'], self.data['z'])
    
        return Ihopefig

    def make_tooltips(self,*ratio):

        if not ratio:
            rt = 0.05
        else:
            rt = ratio[0]

        tooltip_data = self.data.sample(frac = rt )     
        markers = []
        for i in range(0,len(tooltip_data)):
            place_crds= [tooltip_data.iloc[i]['Lat'], tooltip_data.iloc[i]['Long']]
            if self.param == 'vert':
                tlp = round(tooltip_data.iloc[i]['z'],1)
            else:
                tlp = round(tooltip_data.iloc[i]['z'],2)
            markers.append(folium.CircleMarker(location = place_crds, radius = 0.5 , tooltip = tlp, 
            line_color = 'blue', fill_color = 'blue'))
        return markers

    def insar_kde(self):
        if self.param == 'vert':
            indata = self.data.copy(deep=True)
            indata['z'] = indata['z']
            return hpl.basic_kde(indata,'z','Vertical Displacement [mm]')
        
        if self.param == 'corr':
            return hpl.basic_kde(self.data,'z', 'Coherence')

    def get_paired_path(self,path):
        if path.find('vert'):
            return path.replace('vert','corr',1)
        if path.find('corr'):
            return path.replace('corr','vert',1)


    def coh_disp (self,path):
     
        gcp = GCP_HANDLER([]) 
        panda1 = gcp.load_csv_insar(path)
        panda2 = gcp.load_csv_insar(self.get_paired_path(path))

        if len(panda1[panda1['Variable'] == 'vert'])> 0:
            file1 = panda1
            file2 = panda2
        elif len(panda1[panda1['Variable'] == 'corr']) > 0:
            file1 = panda2
            file2 = panda1

        file1['zvert'] = 1000*file1['z']
        file1 = file1.drop(columns=['z','Variable'])

        file2['zcorr'] = file2['z']
        file2 = file2.drop(columns=['z','d0','d1','Variable'])
        out = file1.merge(file2, how = 'inner', on = ['Lat','Long'])
        return out

    # def comp_statistic(self, statistic):
    #     return self.basic_stats[statistic]


    #def _DBquery(self,dam,scene,variable):
            
        #sqlc = "select * from INSAR where Dam_ID == '{}' AND Scene == '{}' ".format(dam,scene)
        #sqlc += "AND Variable == '{}';".format(variable)
        #qry_out =  DBF.collect_from_DB(sqlc)
        #if len(qry_out) == 1:
            #return Path(qry_out[0][3])
        #else:
                #print('Data does not exists')
                #quit()

    #def make_map (self,file_out):
        #test_map = i_map(self.dam)
        #test_map.terrain_model()
        #test_map.region_of_interest()
        #test_map.generate_map_layers()
        #plot_place = self.make_contour_plot(file_out)
        #tooltps = self.make_tooltips()
        #test_map.add_marker(tooltps)
        #test_map.add_png_layer(str(plot_place), self.geo_extent[0], self.geo_extent[1 ],self.geo_extent[2], self.geo_extent[3 ] )
        #test_map.finalise()
        #return test_map

    #def _load_file(self,file_path):
        #split_path = str(file_path).split('im_aware_collab')[1]
        #split_path = split_path[1:-13] 
        #blob = bucket.get_blob(prefix = split_path)
        #blob = blob.download_as_string()
        #return pd.read_csv(io.BytesIO(blob))



class INSAR_ANALYSIS(DAM):

    def __init__(self, *damId):  # path_to_insars
        super(DAM).__init__()
        #super(QUERY_in_INSAR).__init__()
        #self.file_depot = Path(__file__).parent.parent.parent.parent.joinpath('IMAWARE/Insar_Front_Depot')
        #os.makedirs(self.file_depot,exist_ok = True)
        # self.file_depot = path_to_insars
        self._read_data()

        if damId:
            self.choose_dam(damId[0])
            #self.get_record_list()
            self.full_record_list = self.get_record_list()
            self.record_list = self.full_record_list
            self.full_record_table = gen_record_table(self.full_record_list)
            self.record_table = self.full_record_table.copy(deep=True)

    def get_record_list(self):
        '''
        Retrieve INSAR database records for current dam
        '''
        record_list = DBF.query_by_dam(self.dam_uid,'INSAR')
        if not len(record_list) >0:
            print('no insar data for this dam')
            return None
        return record_list

    def filter_by_variable(self,variable):
        try:
            self.record_table = self.full_record_table[self.full_record_table['Variable'] == variable]
        except:
            print('Error, variable not valid')

    # def generate_insars(self):

    def filter_by_date(self,*date):
        # if variable:
        #     self.filter_by_variable(variable[0])
        t1,t2 = date_converter(*date)
        idD = np.logical_and(self.record_table['Start-Date']>=t1,self.record_table['End-Date']<=t2)
        self.record_table = self.record_table[idD]

    def reset_record_table(self):
        self.record_table = self.full_record_table.copy(deep=True)
        self.record_table = self.record_table.reset_index(drop=True)

    def down_sample_record_table(self,freq = 3):
        self.record_table = self.record_table.iloc[::freq,:]

    def generate_insars_list(self, variable ):
        # setup toolbar
        toolbar_width = 100
        sys.stdout.write("Downloading INSAR variable: {}\n[{}]".format(variable, " " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (1))  # return to start of line, after '['
        
        INSARS = []
        table = self.record_table[self.record_table['Variable'] == variable].copy(deep=True)
        
        prog_lim = np.array_split(np.arange(len(table)), toolbar_width)
        if len(table) < toolbar_width:
            prog_lim = np.linspace(0,100,len(table))
        #     print('ERROR: need to fix toolbar for iterating through an array less than len 100')

        prog_lim = np.array([pl.max() for pl in prog_lim])       
        # prog_lim = len(table)/toolbar_width
        # if prog_lim>1:
        #     n_k = 1
        # else:
        #     n_k = int(1/prog_lim)
        prog = 0
        progress_c = 0
        for i, idx in enumerate(table.index):
            # self.record_table.loc[i,'Varible'] 
            #DBF.collect_from_DB('','INSAR')
            #path_me = self.record_table.loc[i, 'Path']
            #gcp = GCP_HANDLER()
            #datai = gcp.load_csv_insar(path_me)
            GCPH = GCP_HANDLER(table.loc[idx].to_dict())
            #GCPH.gen_blob()
            datai = GCPH.load_csv()[0]
            INSARS.append(INSAR_FILE(
                self.dam_uid, self.record_table.loc[idx, 'Path'],  variable, datai))
            
            n_k = np.where(i == prog_lim)[0]
            if n_k.size > 0:
                progress_c += 1
                sys.stdout.write(
                    '\r'+'[{}] - {}%'.format(progress_c*'#', progress_c))
                sys.stdout.flush()

            # prog = prog + 1
            # if prog >= prog_lim:
            #     progress_c += 1
            #     sys.stdout.write(
            #         '\r'+'[{}] - {}%'.format(progress_c*n_k*'#', progress_c*n_k))
            #     sys.stdout.flush()
            #     prog = 0
        
        sys.stdout.write(
                    '\n\tComplete.\n')
        return INSARS
    
    def gen_INSAR(self):
        variables = np.unique(self.record_table['Variable'])
        
        insar_dict = {}
        for v in variables: 
            insar_dict[v] = self.generate_insars_list(v)
        self.INSAR_DICT = insar_dict
        return insar_dict

    def compute_statistic_time_series(self, statistic):

        if not hasattr(self, 'INSAR_DICT'):
            self.gen_INSAR()
        all_vars = self.record_table['Variable'].drop_duplicates().tolist()
        out = {}
        pds = {}

        for v in all_vars:
            out[v] = []
        
        for each in self.INSAR_DICT:
    
            for i in range(len(self.INSAR_DICT[each])):
                out_dict ={}
                variable = self.INSAR_DICT[each][i].param
                if type(self.INSAR_DICT[each][i].basic_stats[statistic]) is not dict:
                    out_dict[statistic] = self.INSAR_DICT[variable][i].basic_stats[statistic]
                else:
                    for sub_param in self.INSAR_DICT[each][i].basic_stats[statistic]:      
                        out_dict[sub_param] = self.INSAR_DICT[variable][i].basic_stats[statistic][sub_param]
                out_dict['t_Mid'] = self.INSAR_DICT[variable][i].mid_interval
                out[variable].append(out_dict)        

        for v in all_vars:
            pds[v] = pd.DataFrame(out[v]) 
        
        return pds 

    def get_statistic_table(self,statistic,variable): #args
        full_series = self.compute_statistic_time_series(statistic) 
        return full_series[variable].sort_values(by = 't_Mid' )

    def get_vert_coh_correlation(self):
        data = self.record_table
        verts = data[data['Variable'] == 'vert']
        cohrs = data[data['Variable'] == 'corr']

        dat, idx = np.unique(verts.loc[:, 'Start-Date'], return_index=True)
        verts = verts.iloc[idx, :]
        verts = verts.reset_index(drop=True)
        dat, idy = np.unique(cohrs.loc[:, 'Start-Date'], return_index=True)
        cohrs = cohrs.iloc[idy, :]
        cohrs = cohrs.reset_index(drop=True)

        out = []
        for i in verts.index:
            try:

                GCPH0 = GCP_HANDLER(verts.loc[i].to_dict())
                #GCPH0.gen_blob()
                data0 = GCPH0.load_csv()[0]

                idx = np.where(cohrs.loc[:, 'Start-Date'] == verts.loc[i, 'Start-Date'])[0][0]

                GCPH1 = GCP_HANDLER(cohrs.loc[idx].to_dict())
                #GCPH1.gen_blob()
                data1 = GCPH1.load_csv()[0]
                # gcp = GCP_HANDLER()
                # datai = gcp.load_csv_insar(verts.iloc[i]['Path'])
                file1 = INSAR_FILE(verts.loc[i,'Dam_ID'],verts.loc[i,'Path'],'vert',data0)
                scene = dates2scene(file1.time_interval)
                
                file2 = INSAR_FILE(verts.loc[i,'Dam_ID'], cohrs.loc[idx, 'Path'], 'corr', data1)
                out.append({'t_Mid': file1.mid_interval ,'corr' : file1.data['z'].corr(file2.data['z'])})
            except Exception as e:
                print(e)
                #print('data from either {} or {} is missing'.format(file1.file_address, file2.file_address))
        file1 = None
        file2 = None
        return pd.DataFrame(out).sort_values(by='t_Mid' , ascending = False)   

    def join_verts_cohs(self):
        pass

    def time_histories(self,coordinates, n_closest=50):
        out = []
        record = {}
        
        for v in self.INSAR_DICT.keys():
            th = []
            for each in self.INSAR_DICT[v]:
                if each.param not in record.keys():
                    record[each.param] = pd.DataFrame()
                Z, position, variable_name, date_range = each.get_n_closest_to(coordinates,n_closest)
                col_pos = ['({},{})'.format(*position.loc[i,:]) for i in position.index]
                record[each.param].loc[date_range[0], 'End_Date'] = date_range[1]
                record[each.param].loc[date_range[0], col_pos] = Z.values
                
                # record[each.param]
                # record['t_Mid'] = each.mid_interval
                # th.append(record)
                # record = {}
            # raw_th = pd.DataFrame(th)
            # raw_th = raw_th.set_index('t_Mid')    
            # out.append(raw_th)
        return record #pd.concat(out, axis=1)

    def cummulative_disp(self,coordinates, n_closest=50):
        th = self.time_histories(coordinates, n_closest = n_closest)
        if 'vert' in list(th.columns):
            th['Cm_vert'] = pd.DataFrame.cumsum(th['vert'])
            return th
        else :
            print('No vertical displacement data')
            return None     

    # def time_histories_with_intervals(self, coordinates, n_closest=50):
    #     out = []
    #     record = {}
    #     dvars = list(self.INSAR_DICT.keys())
    #     for i in range(len(dvars)):
    #         th = []
    #         for each in self.INSAR_DICT[dvars[i]]:
    #             record[each.param] = each.get_n_closest_to(coordinates,n_closest)
    #             record['Start_Time'] = each.time_interval[0]
    #             record['End_Time'] = each.time_interval[1]
    #             record['t_Mid'] = each.mid_interval
    #             th.append(record)
    #             record = {}
    #         raw_th = pd.DataFrame(th)
    #         # if i == 0:
    #         #     out.append(raw_th.copy(deep=True))
    #         # else:
    #             # raw_th.drop(columns = ['Start_Time','End_Time'], inplace=True) 
    #         out.append(raw_th.copy(deep=True))
    #     return out      

    # def robust_disp_time_histories(self,coordinates, n_closest=50):
    #     data = self.time_histories_with_intervals(coordinates, n_closest= n_closest)
    #     data['Disp_rate_mmDay'] = abs(data['vert']/(data['Start_Time'] - data['End_Time']).astype('timedelta64[D]'))
    #     # data = data.sort_values(by=['Disp_rate_mmDay','End_Time','Start_Time'], ascending=False)
    #     data['vert_mm'] = data['vert']
    #     # data = data.drop_duplicates(subset = ['End_Time']).drop(columns = ['t_Mid','Start_Time','vert'])
    #     data['Cm_vert_mm'] = pd.DataFrame.cumsum(data['vert_mm'])
    #     # data =  data.sort_values(by='End_Time', ascending=False)
    #     return data 

    # def plot_robust_time_histories(self,coordinates):
    #     data = self.robust_disp_time_histories(coordinates)
    #     fig1 = hpl.simple_line_plot(data,{'x': 'End_Time', 'y' : 'Cm_vert_mm', 'other' : []},
    #         {'x': 'Time', 'y' : 'Vertical Displacement [mm]'})
    #     return fig1

    def make_maps(self,out_folder,var):
        maps = []
        panda = []
        i=0
        for each in self.INSAR_DICT[var]:
            a_map = each.make_map(out_folder)
            maps.append(a_map)
            panda.append({'dam':each.dam, 'scene' : dates2scene(each.time_interval), 'mid_interval' : each.mid_interval, 'variable': each.param, 'map_index':i })
            i += 1
        return maps,pd.DataFrame(panda).sort_values(by='mid_interval', ascending= False)

    def plot_Quantiles(self,flag):

        if flag == 'Normal':
            qs = ['Q25', 'Q75']
        if flag == 'Wide':
            qs = ['Q10','Q90']

        data1 = self.get_statistic_table('Qs','vert')
        for each in qs:
            data1[each] = data1[each]
        data2 = self.get_statistic_table('Qs','corr')
        
        vert_axis_label = {'x': "time", 'y': "vertical displacement [mm]"}
        corr_axis_label = {'x' : "time", 'y': "coherence"}

        out1 = hpl.bounds_and_mid_plot(data1,{'x': 't_Mid' ,'y_low' : qs[0], 'y_mid': 'median', 'y_up' : qs[1] },vert_axis_label)
        out2 = hpl.bounds_and_mid_plot(data2,{ 'x':'t_Mid','y_low' : qs[0], 'y_mid': 'median', 'y_up' : qs[1] },corr_axis_label)

        return {'fig' : {'vert' : out1, 'corr' : out2 } , 'data' :{'vert' : data1, 'corr':data2} }

    def plot_mean_std(self):

        vert_axis_label = {'x': "time", 'y': "vertical displacement [mm]"}
        corr_axis_label = {'x' : "time", 'y': "coherence"}

        data1a = self.get_statistic_table('mean','vert')
        data1a['mean'] = data1a['mean']
        data1b = self.get_statistic_table('std', 'vert')
        data1b['std'] = 1*data1b['std']
        data1 = data1a.merge(data1b,how = 'inner')
        data1['mean + std'] = data1['mean'] + data1['std']
        data1['mean - std'] = data1['mean'] - data1['std'] 
        data2a = self.get_statistic_table('mean','corr')
        data2b = self.get_statistic_table('std', 'corr')
        data2 = data2a.merge(data2b,how = 'inner')
        data2['mean + std'] = data2['mean'] + data2['std']
        data2['mean - std'] = data2['mean'] - data2['std']

        out1 = hpl.bounds_and_mid_plot(data1,{'x': 't_Mid' ,'y_low' : 'mean - std' , 'y_mid': 'mean', 'y_up' : 'mean + std'},vert_axis_label)
        out2 = hpl.bounds_and_mid_plot(data2,{ 'x':'t_Mid','y_low' : 'mean - std', 'y_mid': 'mean', 'y_up' : 'mean + std' },corr_axis_label)

        return {'fig':{'vert' : out1, 'corr' : out2}, 'data' :{'vert' : data1 , 'corr':data2}}

    def plot_max_min(self, variable):

        dataA =  self.get_statistic_table('max', variable)
        dataB = self.get_statistic_table('min', variable)

        if variable == 'vert':
            axis_label = {'x': "time", 'y': "vertical displacement [mm]"}
            dataA['max'] = 1*dataA['max']
            dataB['min'] = 1*dataB['min']

        elif variable == 'corr':
            axis_label = {'x' : "time", 'y': "coherence"}

        data = dataA.merge(dataB,how = 'inner', on='t_Mid')
        out = hpl.band_plot(data,{'x': 't_Mid' ,'y_low' : 'min' , 'y_up' : 'max'},axis_label)
        max_markers = define_dot_markers(data,'lat_x','long_x','max','t_Mid','crimson')
        min_markers = define_dot_markers(data,'lat_y','long_y','min','t_Mid', 'blue')
        return [max_markers,min_markers]
        #markers = max_markers + min_markers
        #maper = self.make_summary_map(markers)

        return {'fig': out, 'data':data} #'map':maper

    def plot_vert_coh_correlation(self):
        data = self.get_vert_coh_correlation()
        return { 'fig' : hpl.simple_line_plot(data,{'x': 't_Mid', 'y' : 'corr', 'other':[]},{'x': 'Time', 'y' : 'Pearson Correlation'}), 'data': data} 

    def make_summary_map (self,markers):
        test_map = i_map(self.dam_uid)
        # test_map.terrain_model()
        # test_map.region_of_interest()
        # test_map.generate_map_layers()
        test_map.add_marker(markers)
        test_map.finalise()
        return test_map
 
def gen_record_table(recordlist):
        # generate a pandas array with all the metadata (no insar csvs!)
        table = pd.DataFrame(recordlist)
        date0 = []
        date1 = []
        for ind in table.index:
            d0, d1 = scene2dates(table.loc[ind,:]['Scene'])
            date0.append(d0)
            date1.append(d1)
        table['Start-Date'] = date0
        table['End-Date'] = date1
        return table

def date_converter(*args):
    t1 = datetime.datetime.strptime(args[0],'%Y-%m-%d')
    if len(args) ==1:
        t2 = datetime.datetime.today()
        # t2 = datetime.datetime.strptime(str(t2),'%Y-%m-%d')
    else:
        t2 = datetime.datetime.strptime(args[1],'%Y-%m-%d')
    return t1,t2

def scene2dates(scene):
    d0 = datetime.datetime.strptime(scene.split('_')[0],'%Y-%m-%d')
    d1 = datetime.datetime.strptime(scene.split('_')[1],'%Y-%m-%d') 
    return d0, d1

def dates2scene (dates):
    sd0 = dates[0].strftime('%Y-%m-%d')
    sd1 = dates[1].strftime('%Y-%m-%d')
    return '{}_{}'.format(sd0,sd1)

def get_insar_prefix(path):
    if 'corr' in path:
        prefix = path.split(
            'im_aware_collab/')[1].split('_corr')[0]
    elif 'vert' in path:
        prefix = path.split(
            'im_aware_collab/')[1].split('_corr')[0]
    else:
        prefix = None
    return prefix


def make_contour(insar):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.tricontourf(insar.data['Long'], insar.data['Lat'],
                   insar.data['z'], levels=10, cmap='Accent_r', alpha=0.5)
    posx = [min(insar.data['Long']),
            max(insar.data['Long']),
            min(insar.data['Lat']),
            max(insar.data['Lat'])]
    img = get_img_from_fig(fig, dpi=180)
    img[:, :, -1] = 160
    return posx, img

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True,
                bbox_inches='tight', pad_inches=0.0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img


def generate_insar_images(damID, names, dataDf):
    insar_files = []
    contours = {}
    positions = []
    for i in range(len(names)):
        if 'vert' in names[i]:
            var = 'vert'
        elif 'corr' in names[i]:
            var = 'corr'
        else:
            var = None
        insar_files.append(INSAR_FILE(damID, names[i], var, dataDf[i]))
        pos, img = make_contour(insar_files[-1])
        nam = get_insar_layer_name(names[i])
        contours[nam] = img
        positions.append(pos)
    return insar_files, contours, positions


def get_insar_layer_name(path):
    return ':'.join(path.split('_')[-3:]).replace('.csv', '')

def str2date(d):
    return datetime.date(*[int(i) for i in d.split('-')])

def mid_date(d0,d1):
    return d0 + (d1 - d0)/2

def get_mid_dates(start_dates, end_dates):
    t_mid = []
    for i in range(len(start_dates)):
        d0 = str2date(start_dates[i])
        d1 = str2date(end_dates[i])
        t_mid.append(mid_date(d0,d1))
    return t_mid

def get_date_interval(start_dates, end_dates):
    t_int = []
    for i in range(len(start_dates)):
        d0 = str2date(start_dates[i])
        d1 = str2date(end_dates[i])
        t_int.append((d1 - d0).days)
    return t_int

def post_process_time_histories(time_histories, dam_loc):
    
    d_var = list(time_histories.keys())
    loc_str = time_histories[d_var[1]].columns.to_list()
    t0 = time_histories[d_var[1]].index.values
    t1 = time_histories[d_var[1]]['End_Date'].values
    
    loc_str.remove('End_Date')
    location = np.array([eval(loc) for loc in loc_str])
    
    # compute distance and azimouth
    dist = (6371*10**3*np.pi/180) * \
        ((dam_loc[0]-location[:,0])**2 + (dam_loc[1]-location[:,1])**2)**0.5 
    az = np.arctan2(dam_loc[1]-location[:,1], dam_loc[0]-location[:,0])

    distance_pd = pd.DataFrame(
        np.hstack([location, np.vstack([dist, az]).T]),
        columns = ['Lat', 'Long', 'dist2dam', 'Az'] )
    
    # compute times
    mid_time = get_mid_dates(t0, t1) 
    interval_time = np.array(get_date_interval(t0, t1))

    # compute displacmenet rate
    if 'vert' in d_var:
        disp_rate = abs(
            time_histories['vert'].loc[:,loc_str].T/interval_time).T
        disp_rate.index = [D.strftime("%Y-%m-%d") for D in mid_time]

        cum_disp = time_histories['vert'].loc[:,loc_str].cumsum(axis=1)
        cum_disp.index = [D.strftime("%Y-%m-%d") for D in mid_time]
    
        return disp_rate, cum_disp, distance_pd
#if __name__ == '__main__':
#    exec(open('Test_INSAR.py').read())


'''class QUERY_in_INSAR():

    def __init__(self):
        
        out = self._filter_INSAR_records('ALL')
        out[['d0','d1']] = out['Scene'].str.split('_', expand = True )
        out = out.drop(columns = ['Scene'])
        out = self.scene_to_dates(out)
        self.all_data = out

    def _dbexample(self,damID):
        table = 'INSAR'
        recs = DBF.query_by_dam(damID,table)
        return recs
    def _filter_INSAR_records (self,*params):

        if len(params) ==3:

            qry_elements = [ "select * from INSAR where ", "Dam_ID == '{}' ".format(params[0]), 
                    " Scene == '{}' ".format(params[1]), " Variable == '{}' ".format(params[2])]

            raw_qry = []
            raw_qry.append(qry_elements[0])

            i = 1
            for x in params:
                if x != '':
                    raw_qry.append(qry_elements[i])
                i+=1
        
            qout = ""
            for j in range(len(raw_qry)):
                if j < 2:
                    qout += raw_qry[j]
                else: 
                    qout += 'and' + raw_qry[j]    
        
        if params[0] == 'ALL':
            qout = "select * from INSAR"

        return self._insar_to_pandas(DBF.collect_from_DB(qout))   

    def _filter_by_date(self,t1,t2,*args):
        
        if t2 == '' :
            t2 = datetime.datetime.today()
        else: 
            t2 = datetime.datetime.strptime(t2,'%Y-%m-%d')
        if t1 == '':
            t1 = datetime.datetime(2000,1,1)
        else:
            t1 = datetime.datetime.strptime(t1,'%Y-%m-%d')
    
        if args[0] == None:
            pandas = self.all_data
        else: 
            pandas = args[0]     

        pandas = pandas.loc[(pandas['d0'] > t1) & (pandas['d1'] < t2)]
        return pandas 

    def get_scene(self,closest_date, *data):

        closest_date = datetime.datetime.strptime(closest_date,'%Y-%m-%d')

        if not data:
            data = self.all_data
    
        data['day_delta'] = abs((data['d1']-closest_date).dt.days)
        data = data.sort_values(by = ['day_delta', 'Variable'], ascending = True)
        return data[['d0','d1']].head(1)

    def _insar_to_pandas(self,qry_out):
        
        records = []     
        for each in qry_out:
            records.append({'Dam_ID' : each[0], 'Scene' : each[1], 'Variable': each[2]})
        
        return pd.DataFrame(records)  

    def all_data_for_dam(self,Dam_ID):
        return self._filter_INSAR_records(Dam_ID,'','')

    def scene_to_dates (self,pandas_df):
        pandas_df['d1'] = pd.to_datetime(pandas_df['d1'])
        pandas_df['d0'] = pd.to_datetime(pandas_df['d0'])
        return pandas_df'''
