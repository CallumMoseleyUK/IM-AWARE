from platform import release
from dam_break.dam_break import DAM_BREAK, DEM_DATA
from dam_break.dambreak_lib import DAMBREAK_SIM
import matplotlib.pyplot as plt
import math
from pathlib import Path
import os
from source_data.file_handler import FILE_HANDLER
from source_data.earth_data import Interactive_map
import io
import numpy as np

'''Popup links'''
#https://gis.stackexchange.com/questions/313382/click-event-on-maps-with-folium-and-information-retrieval
#https://stackoverflow.com/questions/67628175/how-to-copy-a-markers-location-on-folium-map-by-clicking-on-it

'''Site parameters'''
siteID= 'Bru_fail'
#siteID = 'Alemxc3_GERDAUA'
simID = '%s' % siteID
siteLatitude,siteLongitude = -20.119722, -44.121389
#siteLatitude,siteLongitude = 53.406047659770145, -2.9801006918520625
damHeight = 80.0
numObj = 200
releaseVolume = 2685782.0
materialDensity  = 1594.0
maxTime = 200.0
timeStep = 0.2
dampingCoeff = 0.04

'''Map parameters'''
resolution=1
skipPoints=1

'''Relevant directories'''
currentFolder = Path(os.getcwd())
warehouseFolder = currentFolder.parent.parent.joinpath('IMAWARE')
analysisResultsFolder = warehouseFolder.joinpath('Analysis_Results')
analysisImagesFolder = warehouseFolder.joinpath('Analysis_Images')
demDirectory = Path(os.getcwd())
demDirectory = demDirectory.parent.parent.joinpath('IMAWARE/Sim_Raw/data_DEM')
demDirectory = str(demDirectory).replace('\\','/')

warehouseFolder = str(warehouseFolder).replace('\\','/')

''' Estimate pond radius '''
pondRadius = math.sqrt(releaseVolume/(math.pi * damHeight))

''' Run simulation '''
simObject = DAM_BREAK(siteLat=siteLatitude,siteLon=siteLongitude,
                pondRadius=pondRadius,
                nObj=numObj,
                tailingsVolume=releaseVolume,
                tailingsDensity=materialDensity,
                maxTime=maxTime,
                timeStep=timeStep,
                dampingCoeff=dampingCoeff,
                demDirectory=demDirectory)
simObject._bVerbose = True
simObject.run_simulation()

simRecord = simObject.get_database_record(simID)
fileHandler = FILE_HANDLER()
(fileName,csvName) = simObject.save_results(siteID,simID,fileHandler=fileHandler,warehouseFolder=warehouseFolder)
simRecord['File_Address'] = csvName
simRecord['File_Handler'] = fileHandler
resultsObject = DAMBREAK_SIM(srcInput=simRecord,bAbsolutePath=True,demDirectory=demDirectory)

maskPath = os.path.dirname(csvName) + '/%s.png' % 'speed'
mask,X,Y = resultsObject.fit_speed_mask(resultsObject.max_time(),resolution=resolution,skipPoints=skipPoints)
resultsObject.save_mask(maskPath,mask,X,Y)    

image = resultsObject.fileHandler.load_image(maskPath)

minLon,maxLon,minLat,maxLat = np.min(X),np.max(X),np.min(Y),np.max(Y)
minQuantity = np.min(mask)
maxQuantity = np.max(mask)
position = (minLon,maxLon,minLat,maxLat)

#imap = Interactive_map([siteLatitude,siteLongitude])
#imap = Interactive_map(siteID)
imap = Interactive_map([siteLatitude,siteLongitude])
colorMap=None
imap.add_png_layer(image, float(position[0]), float(
    position[1]), float(position[2]), float(position[3]), Label='Speed', colorMap=colorMap)

imap.finalise()
imap.show_map()

#imap.save_map('MAP_DEBUG/map.html')
imap.Map.save('MAP_DEBUG/map.html')