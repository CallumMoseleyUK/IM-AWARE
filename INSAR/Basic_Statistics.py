from pathlib import Path
import sys

from numpy import clongdouble
sys.path.append(str(Path(__file__).parent.parent))

from DBdriver.DBFunctions import *
from INSAR.INSAR_ANALYSIS import *
from source_data.GCPdata import *
import pandas as pd
from setup_stats_in_cloud import *
import os
import shutil
import handy_plotter as hpl

class QUERY_IN_INSAR:

    def __init__(self):
        self.dam_list = self.dams_with_insar()

    def path_to_insar(self,dam,scene,variable):
        sqlq = "select Path from INSAR where Dam_ID = '{}' and Scene = '{}' ".format(dam,scene) 
        sqlq += "and Variable = '{}' ".format(variable)
        return collect_from_DB(sqlq)[0][0]

    def collect_scenes(self,dam):
        sqlq = "select distinct Scene from INSAR where Dam_ID = '{}'".format(dam)
        return self.single_field_query_unpacker(collect_from_DB(sqlq))

    def dams_with_insar(self):
        sqlq = 'select distinct Dam_ID from INSAR'
        return self.single_field_query_unpacker(collect_from_DB(sqlq))

    def single_field_query_unpacker(self,query_results):
        out =[]
        for each in query_results:
            out.append(each[0])
        return out 

    def dam_crds(self,dam):
        sql1 = "select * from ANM where ID = '{}'".format(dam)
        out = collect_from_DB(sql1)
        return [out[0][-3],out[0][-2]]


class IFILE (QUERY_IN_INSAR) :
    
    def __init__(self,dam,scene,var):
        self.path = super().path_to_insar(dam,scene,var)
        self.dam = dam
        self.scene = scene
        self.var = var
        gcp = GCP_HANDLER([])
        self.Ifile = INSAR_FILE(self.dam,self.path,self.var,gcp.load_csv_insar(self.path))

    def bootstraping_sampling(self,data1,data2):
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
        return dataA,dataB,sample_size
        
    def simple_sampling(self,data1,data2):
        sample_size = min(len(data1),len(data2))
        if len(data1) > sample_size: 
            dataA = data1.sample(sample_size)
            dataB = data2
        elif len(data2) > sample_size:
            dataB =  data2.sample(sample_size)
            dataA = data1
        else :
            dataA = data1
            dataB = data2
        return dataA,dataB,sample_size

    def distance_correlogram(self, cord, sampling_method):
        data = self.Ifile.data
        corrs = []
        nBins = 24
        wdata = data.copy(deep = True)
        wdata['d'] = 6371*10**3*math.pi/360*((cord[0] - wdata['Lat'])**2 + (cord[1] - wdata['Long'])**2)**0.5
        bin_criteria = wdata['d'].max()/nBins
        wdata['dbin'] = wdata['d']/bin_criteria
        wdata['dbin'] = wdata['dbin'].apply(np.ceil).astype(int)
        data1 = wdata[wdata['dbin'] == 1]['z']
        corrs.append({'corr' : 1.00, 'd' : round(bin_criteria,0), 'n': len(data1) }) 
    
        for i in range(2,nBins+1,1):
            data2 = wdata[wdata['dbin'] == i]['z']
        
            if sampling_method == 'boot':
                dataA,dataB,sample_size = self.bootstraping_sampling(data1,data2)
        
            if sampling_method == 'simple':
                dataA,dataB,sample_size = self.simple_sampling(data1,data2 )

            corrs.append({'corr': np.corrcoef(dataA,dataB)[0,1], 'd': round(i*bin_criteria,0), 'n':sample_size})
        return pd.DataFrame(corrs)

    def orientation_correlogram(self,cord,sampling_method):
        data = self.Ifile.data
        corrs = []
        nBins = 36
        wdata = data.copy(deep = True)
        wdata['angle'] = (180/np.pi)*np.arctan2((wdata['Long'] - cord[1]),(wdata['Lat']-cord[0]))
        bin_criteria = wdata['angle'].max()/nBins
        wdata['angle_bin'] = wdata['angle']/bin_criteria
        wdata['angle_bin'] = wdata['angle_bin'].apply(np.ceil).astype(int)
        data1 = wdata[wdata['angle_bin'] == 1]['z']
        angle_bins = wdata['angle_bin'].drop_duplicates().sort_values()
        corrs.append({'corr' : 1.00, 'angle' : round(bin_criteria,0), 'n': len(data1) }) 
    
        for i in angle_bins:
            data2 = wdata[wdata['angle_bin'] == i]['z']
            
            if sampling_method == 'boot':
                dataA,dataB,sample_size = self.bootstraping_sampling(data1,data2)
        
            if sampling_method == 'simple':
                dataA,dataB,sample_size = self.simple_sampling(data1,data2 )

            corrs.append({'corr': np.corrcoef(dataA,dataB)[0,1], 'angle': round(i*bin_criteria,0), 'n':sample_size})
        return pd.DataFrame(corrs)

    def func_montecarlo(self,n,fun,fun_params,group_var, summ_var):
        big_panda =[]
        for i in range(0,n):
            sim = fun(*fun_params)
            sim['simulation'] = i+1
            big_panda.append(sim)
        big_panda = pd.concat(big_panda)
        mean_panda = big_panda[[summ_var,group_var]].groupby([group_var]).mean()
        std_panda = big_panda[[summ_var,group_var]].groupby([group_var]).std()
        return {'mean':mean_panda, 'std': std_panda}

    def montecarlo_convergence(self,fun,fun_params,var,nsims):
        if fun.__name__ == 'distance_correlogram':
            group_var = 'd'
            xlabel = 'distance [m]'
        elif fun.__name__  == 'orientation_correlogram':
            group_var = 'angle'
            xlabel = 'azimuth [deg]'

        summ_var = var
        means = []
        stds = []
        base = 5
        for i in range(1,nsims,1):
            n = base**i
            runs = self.func_montecarlo(n,fun,fun_params,group_var, summ_var)
            runs['mean']['5^n'] = str(i)
            runs['std']['5^n'] = str(i)
            means.append(runs['mean'])
            stds.append(runs['std'])
        mean_panda = pd.concat(means).reset_index()
        mean_fig = hpl.simple_line_plot(mean_panda,{'x':group_var, 'y': summ_var, 'colour' : '{}^n'.format(str(base)), 'other' :['{}^n'.format(str(base))]},{'x': xlabel, 'y': 'Pearson Correlation'})
        std_panda = pd.concat(stds).reset_index()
        std_fig = hpl.simple_line_plot(std_panda,{'x':group_var, 'y': summ_var, 'colour' : '{}^n'.format(str(base)), 'other' :['{}^n'.format(str(base))] },{'x': xlabel, 'y': 'Pearson Correlation'})
        return {'mean': {'fig':mean_fig,'data': mean_panda}, 'std':{'fig':std_fig ,'data':std_panda}}
    
    def export_MC_convergence(self,fun,fun_params,var,nsims,string_path):
        out =  self.montecarlo_convergence(fun,fun_params,var,nsims)
        
        outfile = {}
        outfile['fig_mean'] = string_path + '/{}_{}_{}_mean_{}.html'.format(self.dam,self.scene,self.var,nsims)
        outfile['panda_mean'] = string_path + '/{}_{}_{}_mean_{}.csv'.format(self.dam,self.scene,self.var,nsims)
        outfile['fig_std'] = string_path + '/{}_{}_{}_std_{}.html'.format(self.dam,self.scene,self.var,nsims)
        outfile['panda_std'] = string_path +'/{}_{}_{}_std_{}.csv'.format(self.dam,self.scene,self.var,nsims)
        cc = CLOUD_CONNECT()
        cc.upload_pandas(out['mean']['data'],str(outfile['panda_mean']))
        cc.upload_pandas(out['std']['data'],str(outfile['panda_std']))
        cc.upload_altair(out['mean']['fig'],str(outfile['fig_mean']))
        cc.upload_altair(out['std']['fig'],str(outfile['fig_std']))

class Out_Single_Scene(IFILE):

    def __init__(self,out_dir, dam,scene,var, storage):
        super().__init__(dam,scene, var)
        self.formats = {'fig':'html'}
        self.base_path = str(out_dir)
        self.params = {'dam':self.dam, 'scene': self.scene, 'var': self.var}
        self.cloud_storage = 'andres_test_cloud/static'
        if storage == 'local':
            self.local_storage = True 
            if not os.path.exists(self.base_path):
                os.makedirs(self.base_path)
        
        elif storage == 'cloud':
            self.local_storage = False          

    def push_scatter_front(self):

        pandas = self.Ifile.coh_disp(str(self.path))
        
        if len(pandas) > 4750: 
            pandas= pandas.sample(4500)
        sct  = hpl.basic_scatter(pandas, {'x':'zvert', 'y':'zcorr'}, {'x':'vertical displacement [mm]', 'y':'coherence'})
        
        fname = file_name_maker( self,self.base_path + '/scatter_',self.params,'fig')
        
        if self.local_storage:
           return  {'fig': sct , 'address': local_push(fname,sct)}
        else:
           return  {'fig': sct , 'address': cloud_push(self,fname,sct)}

    def push_kde_front(self):

        kde = self.Ifile.insar_kde()
        fname = file_name_maker( self,self.base_path + '/kde_',self.params,'fig')
        
        if self.local_storage:
            return  {'fig': kde , 'address': local_push(fname,kde)}
        else:
            return  {'fig': kde , 'address': cloud_push(self,fname,kde)}

    def delete_folder_contents(self):
        shutil.rmtree(self.dir)
    

    def plot_correlogram(self,data,type):
        if type == 'dist':
            return hpl.simple_line_plot(data,{'x':'d', 'y': 'corr'},{'x':'distance[m]', 'y':'Pearson Correlation'})
        elif type == 'or':
            return hpl.simple_line_plot(data,{'x':'angle', 'y': 'corr'},{'x':'Azimuth [deg]', 'y':'Pearson Correlation'})

    def push_dist_correlogram(self,cord, sampling_method):
        data = self.distance_correlogram(cord,sampling_method)
        plot = self.plot_correlogram(data,'dist')
        fname = file_name_maker(self, self.base_path + '/corr_dist_',self.params,'fig')
        if self.local_storage:
            return  {'fig': plot , 'address': local_push(fname,plot)}
        else:
            return  {'fig': plot , 'address': cloud_push(self,fname,plot)}

    def push_or_correlogram(self,cord,sampling_method):
        data = self.orientation_correlogram(cord,sampling_method)
        plot = self.plot_correlogram(data,'or')
        fname = file_name_maker(self, self.base_path + '/corr_or_',self.params,'fig')
        if self.local_storage:
            return  {'fig': plot , 'address': local_push(fname,plot)}
        else:
            return  {'fig': plot , 'address': cloud_push(self,fname,plot)}

class Out_Insar_Analysis():
    
    def __init__(self,insar_analysis_object):
        insar_analysis_object.gen_INSAR()
        self.collection = insar_analysis_object

    def plot_time_history(self, cords):
        return self.collection.plot_robust_time_histories(cords)

    def plot_quantiles(self,flag):
        return self.collection.plot_Quantiles(flag)

    def plot_mean_std(self):
        return self.collection.plot_mean_std()    

def file_name_maker(out_obj,pre,dict,type):
        out = '{}'.format(pre)
        for each in dict.keys():
            out += '{}_'.format(dict[each])
        out = out[0:-2]
        return '{}.{}'.format(out,out_obj.formats[type])

def cloud_push(out_obj,fname,html_plot):
        f_cloud_name = "{}/{}".format(out_obj.cloud_storage,fname)
        AltairUpload(html_plot,f_cloud_name)
        return f_cloud_name

def local_push(fname,html_plot):
        html_plot.save(fname)
        return fname


if __name__ == '__main__':
    #cc = cloud_connect()
    #cc.create_bucket('andres_test_cloud')
    #cloud_path = 'andres_test_cloud/moo/test.html'
    #file_path = 'E:\im_aware_collab\SRC\IM-AWARE-GIS\INSAR\KDE.txt'
    #cc.upload_file(cloud_path,file_path)   
    #a = cc.download_file(cloud_path)
    b = QUERY_IN_INSAR()
    c = b.dam_list
    d = b.collect_scenes(c[2])
    e = IFILE(c[2],d[2],'vert')
    f = {'dam': c[2], 'scene' : d[2], 'var' : 'vert' }
    g = Out_Single_Scene('moo', c[2],d[2],'vert','cloud')
    h = g.push_kde_front()
    i = g.push_scatter_front()
    crds = []
    for each in c:
        crds.append(b.dam_crds(each))
    j = g.push_dist_correlogram(crds[2],'boot')
    k = g.push_or_correlogram(crds[2],'boot')

    l = INSAR_ANALYSIS(c[2])
    m = Out_Insar_Analysis(l)
    n = m.plot_time_history(crds[2])
    o = m.plot_quantiles('Wide')



