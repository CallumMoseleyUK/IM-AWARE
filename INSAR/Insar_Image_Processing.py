from pathlib import Path
import os
import sys

print(Path.cwd())
sys.path.append(str(Path.cwd()))

if str(Path.cwd()).split('/')[-1] == 'INSAR': 
     os.chdir(Path.cwd().parent)
import directory_manager as dbm
#

sys.path.append(dbm.get_work_dir())
import DBdriver.DBFunctions as DB
from DAMS import ANM_DAMS, DAM

repo = DB.WarehouseFolder.joinpath('INSAR')
# os.mkdir(repo)

rawRepo = repo.joinpath('RawData')

procRepo = repo.joinpath('ProcessedData')
rOIsRepo = repo.joinpath('ROIs')

if not os.path.isdir(procRepo):
    print('making {}'.format(procRepo))
    os.mkdir(procRepo)

if not os.path.isdir(rOIsRepo):
    print('making {}'.format(rOIsRepo))
    os.mkdir(rOIsRepo)

import sys 
import rasterio
from pyproj import Proj, transform, transformer
import pandas as pd
import numpy as np
import datetime 
import shutil
import zipfile
import os
import json
import sys 



class INSAR_IMAGE :

    __verbose__ = True

    def __init__(self,*inFile):
        if inFile:
           self._initialize_class(inFile) 

    def _initialize_class(self,inFile):
        self.outProj = Proj(init='epsg:4326')
        read_image = self._image_reader(inFile)
        self.raster = read_image['raster']
        self.resolution = read_image['resolution']
        self.extension = read_image['extension']
        self.cornerNW = read_image['cornerNW']
        self.cornerSE = read_image['cornerSE']
        self.type = read_image['type']
        self.d0 = read_image['d0']
        self.d1 = read_image['d1']
        
        
    def _image_reader(self,inFile): 
        out = {}
        with rasterio.open(inFile) as raster:
                
            # Collection of raster size and coordinate systems from tags within the image
            rSize = [raster.width,raster.height]
            rCrs = raster.crs
                
            #Definition of the coordinate transform to latitude and longitude
            trns = transformer.Transformer.from_crs(rCrs,self.outProj.crs)
            trns2 = transformer.Transformer.from_crs(self.outProj.crs,rCrs)

            rowId = np.array(range(rSize[1]))
            colId = np.array(range(rSize[0]))
            X = raster.xy(0*colId,colId)[0]
            Y = raster.xy(rowId,0*rowId)[1]
                
            #Color Bands are read from the raster
            r = pd.DataFrame(raster.read(1),columns= X, index = Y)

            out['raster'] =  {'bands': r, 'xy': [X,Y] , 'trs' : [trns,trns2] }
            out['resolution'] = X[1]-X[0]
            out['extension'] = {'x': (X[len(X)-1] - X[0])/1000, 'y': (Y[len(Y)-1] - Y[0])/1000}
            out['cornerNW'] = {'lat':trns.transform(X[0],Y[0])[1], 'long':trns.transform(X[0],Y[0])[0] }
            out['cornerSE'] = {'lat':trns.transform(X[len(X)-1],Y[len(Y)-1])[1], 'long': trns.transform(X[len(X)-1],Y[len(Y)-1])[0] }
                
            nameFile = inFile.name.split('_')
            out['type'] = nameFile[8].split('.')[0].strip()
            out['d0'] = '{}-{}-{}'.format(nameFile[1][0:4],nameFile[1][4:6],int(nameFile[1][6:8]))
            out['d1'] = '{}-{}-{}'.format(nameFile[2][0:4],nameFile[2][4:6],int(nameFile[2][6:8]))
        return out     
            
    # TODO should have a method to set the desired region 

    # TODO then method to extract with some default values passed as key value arguments on the input
    def insar_region_extractor(self,location,window_ext):

        raster = self.raster 

        #function that extracts data in the region of interest (rOI) from the raster
        #input parameters
        #raster: raster extracted by the rgbReader Function
        #location, coordinats of the center of the rOI in [latitude, longitude] format
        #output parameters:
        # 1. dictionary with the parameter 'data' which is a dataset that contains the red, bue and green values of each pixel with 
        #    it centroid coordinates in both catersial raster local coordinates and latitude and longitude
        # 2. r,g,b values of the raster

        bands = raster['bands']
        xy = raster['xy']
        trans = raster['trs']
        unit = 'm'

        xyloc = trans[1].transform(location[1],location[0])

        #Extraction of the region of interest (rectangle around the dam)
    
        window = 10**3*window_ext #lenght of the square observation window around the dam in metres
         
        x = np.array(xy[0])
        y = np.array(xy[1])

        xBand = [xyloc[0]-window/2,xyloc[0]+window/2]
        yBand = [xyloc[1]-window/2,xyloc[1]+window/2]
        
        dx_x = np.logical_and(x>xBand[0],x<xBand[1])
        dx_y = np.logical_and(y>yBand[0],y<yBand[1])

        # TODO: region extractor and the dataframe builder should be two different methods. 

        R = bands.loc[dx_y,dx_x]
        red = np.hstack(R.values)
        xa,ya = np.meshgrid(x[dx_x],y[dx_y])
        xa = np.hstack(xa)
        ya = np.hstack(ya)
        if not all(red ==0 ):
            refined_data = pd.DataFrame(np.concatenate([[xa],[ya]],axis=0).T,columns=['x','y'])
            refined_data['Variable'] = self.type
            refined_data['z'] = red
            refined_data['Lat'] = trans[0].transform(refined_data['x'],refined_data['y'])[1]
            refined_data['Long'] = trans[0].transform(refined_data['x'],refined_data['y'])[0]
            refined_data['d0'] = self.d0
            refined_data['d1'] = self.d1
            return {'data': refined_data, 'image': R}
        else: 
            return 'Error loading data'
    
    def insar_processor(self, place, wE):
        
        #function that integrates reading of an RGB raster and the extraction of a region of interest rOI
        #input parameters:
        # 1. file = raster file name
        # 2. place = coordinates of the center of the rOI in [latitude, longitude] format
        #output table describing the region of interest and the square matrices that contain RBG data. 
        
        rOI = self.insar_region_extractor(place,wE)
        try:
            rOI['data'] = rOI['data'].drop(columns = ['x','y'])
        except:
            return rOI
            # return rOI['data'] 
        return rOI['data']

    def data_exporter (self , damName, location,wE):
        scene = '{}_{}'.format(self.d0,self.d1)
        pth = rOIsRepo.joinpath(damName)
        fName = str(pth.joinpath('{}_{}-{}.csv'.format(scene,
                    self.type, DB.get_current_time_forPath())))
        if not os.path.exists(str(pth)) :
            os.mkdir(str(pth))
        try :
            self.insar_processor(location,wE).to_csv(fName, index=False)

            #TODO: This is wrong I think. You are checking if the file exists straight after you made it!
            if os.path.exists(str(fName)):
                #print ('duplicate file') 
                print(
                    '\tCompleted insar processing for {} - \t\nwriting file to: \n\t\t{}'.format(damName, fName))

        except:
            #os.rmdir(str(pth))
            er = 'Cannot extract position from INSAR : Lat {} Long {} \n {} Dam outside INSAR image'.format(location[0],location[1],damName)
            print(er)
            return er

    def region_ok (self , location ):

            if (location[0] < self.cornerNW['lat'] and location[0] > self.cornerSE['lat'] and location[1] > self.cornerNW['long'] and location[1] < self.cornerSE['long']) :
                return True 
            
            else:
                return False

class INSAR_PROCESSING (INSAR_IMAGE,ANM_DAMS): #ANM_DAMS

    # TODO inherit the DAMS class 

    #TODO add methods to not repeatedly analyse the same files. Once the file has been 
    # processed we should add it to a csv, and then always read that csv in the init and check 
    # which files have not been processed yet. 
    def __init__(self, *args, process_raw_files=True, file_address= None):

        super(INSAR_IMAGE).__init__()
        super(ANM_DAMS).__init__()
        self._read_data(file_address=file_address)
        self.rawfiles_log = self._read_files_log()
        self.rawErrorfiles_log = self._read_error_files_log()
        #print(dir(self))
        
        if args:
            if process_raw_files:
                self.raw_files_extractor()
            if len(args) ==1 :
                self.extract_data_from_processedfile(args[0])
            else:
                self.extract_data_from_processedfile(args[0],args[1])
  
    def _read_files_log(self):
        out = []
        if os.path.isfile(str(rawRepo.joinpath('FileLog.json'))):
            with open(rawRepo.joinpath('FileLog.json')) as f:
                paths = json.load(f)
            for each in paths:
                out.append(Path(each))
            return out
            
        else:
            return []
    
    def _read_error_files_log(self):
        out = []
        if os.path.isfile(rawRepo.joinpath('Error_FileLog.json')):
            with open(rawRepo.joinpath('Error_FileLog.json')) as f:
                paths = json.load(f)
            for each in paths:
                out.append(Path(each))
            return out
            
        else:
            return []

    def extract_data_from_processedfile (self,wE, *damList) :
        if not damList:
            damList = self.dams_df
        else: 
            damList = damList[0]

        for file in list(procRepo.iterdir()):
            
            if check_if_insar(file): 
                self._initialize_class(file)
                print('Extracting data in region of the dams \n File: \n\t{}'.format(file))
                for idd in damList.index :
                    dam = damList.loc[idd]
                    if self.region_ok([dam['Lat'],dam['Long']]) :
                        self.data_exporter(dam['ID'],[dam['Lat'],dam['Long']],wE)
                    else:
                        print('Dam Outside the INSAR region')

    def raw_file_processor(self,file):
        contents = [] 
        with zipfile.ZipFile(file,'r') as zf:
            fname = file.name.split('.')[0]
            oFile1 = '{}_corr.tif'.format(fname)
            oFile2 = '{}_vert_disp.tif'.format(fname)
            zf.extractall()
            zPath = Path.cwd().joinpath(fname)
            
            if self.__verbose__ : 
                if not os.path.isfile(zPath.joinpath(oFile1)):
                    print('No coor {}'.format(fname))
                if not os.path.isfile(zPath.joinpath(oFile2)):
                    print('No vert {}'.format(fname))
            
            shutil.copyfile(zPath.joinpath(oFile1),procRepo.joinpath(oFile1))
            contents.append('corr')
            if self.__verbose__ : 
                if os.path.isfile(procRepo.joinpath(oFile1)): 
                    print('corr from {} was read ok'.format(file))
                else:
                    print('No corr file {}'.format(file))
            shutil.copyfile(zPath.joinpath(oFile2),procRepo.joinpath(oFile2))
            if self.__verbose__ :
                if os.path.isfile(procRepo.joinpath(oFile2)):  
                    print('displacement from {} was read ok'.format(file))
                else:
                    print('No vert file {}'.format(file))

            contents.append('ver_disp')
            shutil.rmtree(zPath)

    def _individual_file_processor(self, each):
        try:
            self.raw_file_processor(each)
            self.rawfiles_log.append(each)
        except:
            self.rawErrorfiles_log.append(each)
            print('error extracting file {}'.format(each))

    def _multi_file_processor(self,files):
        for each in files:
            self._individual_file_processor(each)
        rawfiles_str = [str(x) for x in self.rawfiles_log] 
        rawErrorfiles_str = [str(x) for x in self.rawErrorfiles_log]
        
        with open(rawRepo.joinpath('FileLog.json'), 'w') as f:
            json.dump(rawfiles_str,f)
        with open(rawRepo.joinpath('Error_FileLog.json'), 'w') as f:
            json.dump(rawErrorfiles_str, f)

    def raw_files_extractor(self):
        
        files = list(rawRepo.iterdir())   
        
        if len(self.rawfiles_log) == 0 :
            self._multi_file_processor(files)
        else:
            new_files = []
            for each in files:
                if each not in self.rawfiles_log and self.check_file_type(each,'zip'):
                    new_files.append(each) 
            self._multi_file_processor(new_files)
        
    def check_file_type(self,path,fType):
        if '.{}'.format(fType) in str(path):
            return True
        else: 
            return False

def check_if_insar(fileName):
    if ".tif" in str(fileName):
        return True
    else :
        return False

def update_INSAR_table():
    
    for damAddress in list(rOIsRepo.iterdir()):
        if damAddress.is_dir():
            dam = str(damAddress.name).split('.')[0]
            for scene in list(damAddress.glob('*.csv')):
                scene_info = str(scene.name).split('_')
                time_lapse = '{}_{}'.format(scene_info[0],scene_info[1])
                variable = scene_info[2].split('.')[0].split('-')[0]
                path = str(scene)
                qry = "insert into INSAR (Dam_ID, Scene, Variable, Path )"
                qry += " values ('{}','{}','{}','{}');".format(dam,time_lapse,variable,path)
                DB.do_in_DB(qry)

def create_INSAR_table():
    create_insar_table = "create table INSAR (Dam_ID text not null, Scene text not null," 
    create_insar_table += "Variable text not null, Path text primary key);" 
    DB.do_in_DB(create_insar_table)





        

