from pathlib import Path 
import rasterio
import pandas as pd
from pyproj import Proj, transform, transformer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import plotly.express as px

dataRep = Path.cwd().joinpath('Repository')

def rgbReader (inFile):

    #Function to read RGB raster files
    #input parameters: 
    #inFile = Full address of the raster file with extension

    #output parameters:
    #dictionary with data extracted from the bands ('band' parameter). the cartesian coordinates of the edges of the raster ('xy')
    # and transforms ('trs') from latidue and longitude to xy (trns2) and from xy to latitude and longitue (trns)

    # Coordinate sytem on which usual latitude and longtiude coordinates are based
    outProj = Proj(init='epsg:4326')

    with rasterio.open(inFile) as raster:
        
        # Collection of raster size and coordinate systems from tags within the image
        rSize = [raster.width,raster.height]
        rCrs = raster.crs
        
        #Definition of the coordinate transform to latitude and longitude
        trns = transformer.Transformer.from_crs(rCrs,outProj.crs)
        trns2 = transformer.Transformer.from_crs(outProj.crs,rCrs)

        #rowId = np.array(range(rSize[0]))
        #colId = np.array(range(rSize[1]))
        #xy = raster.xy(rowId,colId)

        #Transformation of raster column and row numbers into cartesian coordinates
        rowId = np.array(range(rSize[0]))
        ColId = np.array(range(rSize[1]))
        xy = raster.xy(rowId,ColId)
        X,Y = trns.transform(xy[0],xy[1])
        
        #Color Bands are read from the raster
        r = pd.DataFrame(raster.read(1),columns= Y, index = X)
        g = pd.DataFrame(raster.read(2),columns= Y,index = X)
        b = pd.DataFrame(raster.read(3),columns= Y,index= X)

    return  {'bands': [r, g, b], 'xy': xy , 'trs' : [trns,trns2]}

def insarReader (inFile):

    #Function to read RGB raster files
    #input parameters: 
    #inFile = Full address of the raster file with extension

    #output parameters:
    #dictionary with data extracted from the bands ('band' parameter). the cartesian coordinates of the edges of the raster ('xy')
    # and transforms ('trs') from latidue and longitude to xy (trns2) and from xy to latitude and longitue (trns)

    # Coordinate sytem on which usual latitude and longtiude coordinates are based
    outProj = Proj(init='epsg:4326')

    with rasterio.open(inFile) as raster:
        
        # Collection of raster size and coordinate systems from tags within the image
        rSize = [raster.width,raster.height]
        rCrs = raster.crs
        
        #Definition of the coordinate transform to latitude and longitude
        trns = transformer.Transformer.from_crs(rCrs,outProj.crs)
        trns2 = transformer.Transformer.from_crs(outProj.crs,rCrs)

        rowId = np.array(range(rSize[1]))
        colId = np.array(range(rSize[0]))
        X = raster.xy(0*colId,colId)[0]
        Y = raster.xy(rowId,0*rowId)[1]
        
        #Color Bands are read from the raster
        r = pd.DataFrame(raster.read(1),columns= X, index = Y)

    return  {'bands': r, 'xy': [X,Y] , 'trs' : [trns,trns2]}


def rgbRegionExtractor(raster,location):

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

    xyloc = trans[1].transform(location[1],location[0])

    #Extraction of the region of interest (rectangle around the dam)
    
    window = 1*10**3 #lenght of the square observation window around the dam in metres
         
    x = np.array(xy[0])
    y = np.array(xy[1])

    xBand = [xyloc[0]-window/2,xyloc[0]+window/2]
    yBand = [xyloc[1]-window/2,xyloc[1]+window/2]
        
    dx_x = np.logical_and(x>xBand[0],x<xBand[1])
    dx_y = np.logical_and(y>yBand[0],y<yBand[1])

    R = bands[0].loc[dx_y,dx_x]
    G = bands[1].loc[dx_y,dx_x]
    B = bands[2].loc[dx_y,dx_x]
    red = np.hstack(R.values)
    green = np.hstack(G.values)
    blue = np.hstack(B.values)
    xa,ya = np.meshgrid(x[dx_x],y[dx_y])
    xa = np.hstack(xa)
    ya = np.hstack(ya)
    refined_data = pd.DataFrame(np.concatenate([[xa],[ya]],axis=0).T,columns=['x','y'])
    refined_data['red'] = red
    refined_data['blue'] = blue
    refined_data['green']= green
    refined_data['lat'] = trans[0].transform(refined_data['x'],refined_data['y'])[1]
    refined_data['long'] = trans[0].transform(refined_data['x'],refined_data['y'])[0]
    return {'data': refined_data, 'image': [R,G,B]}


def insarRegionExtractor(raster,location):

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
    
    window = 1*10**3 #lenght of the square observation window around the dam in metres
         
    x = np.array(xy[0])
    y = np.array(xy[1])

    xBand = [xyloc[0]-window/2,xyloc[0]+window/2]
    yBand = [xyloc[1]-window/2,xyloc[1]+window/2]
        
    dx_x = np.logical_and(x>xBand[0],x<xBand[1])
    dx_y = np.logical_and(y>yBand[0],y<yBand[1])

    R = bands.loc[dx_y,dx_x]
    red = np.hstack(R.values)
    xa,ya = np.meshgrid(x[dx_x],y[dx_y])
    xa = np.hstack(xa)
    ya = np.hstack(ya)
    refined_data = pd.DataFrame(np.concatenate([[xa],[ya]],axis=0).T,columns=['x','y'])
    refined_data['disp_'+ unit ] = red
    refined_data['lat'] = trans[0].transform(refined_data['x'],refined_data['y'])[1]
    refined_data['long'] = trans[0].transform(refined_data['x'],refined_data['y'])[0]
    return {'data': refined_data, 'image': R}

def rgbProcessor (file, place):
    
    #function that integrates reading of an RGB raster and the extraction of a region of interest rOI
    #input parameters:
    # 1. file = raster file name
    # 2. place = coordinates of the center of the rOI in [latitude, longitude] format
    #output table describing the region of interest and the square matrices that contain RBG data. 
    
    ras = rgbReader(file)
    rOI = rgbRegionExtractor(ras,place)
    return rOI    

def insarProcessor (file, place):
    
    #function that integrates reading of an RGB raster and the extraction of a region of interest rOI
    #input parameters:
    # 1. file = raster file name
    # 2. place = coordinates of the center of the rOI in [latitude, longitude] format
    #output table describing the region of interest and the square matrices that contain RBG data. 
    
    ras = insarReader(file)
    rOI = insarRegionExtractor(ras,place)
    return rOI   


def rgbPlotter(rOI):

    #function to plot data from the region of interest (rOI)
    #input: Dictionary describing the region of interest, outcome of the rgbProcessor function
    #output: plot of the region of interest centered around provided coordinates by the user

    COL = np.dstack([rOI['image'][0].values,rOI['image'][1].values,rOI['image'][2].values])
    plt.imshow(COL,extent=[min(rOI['data']['x'].values),max(rOI['data']['x'].values),min(rOI['data']['y'].values),max(rOI['data']['y'].values)],aspect='auto')
    plt.scatter(rOI['data']['x'].mean(),rOI['data']['y'].mean(),color='red',marker='+',s=1000)
    plt.show()

def insarPlotter(rOI):

    #function to plot data from the region of interest (rOI)
    #input: Dictionary describing the region of interest, outcome of the rgbProcessor function
    #output: plot of the region of interest centered around provided coordinates by the user

    plt.imshow(rOI['image'].values,extent=[min(rOI['data']['x'].values),max(rOI['data']['x'].values),min(rOI['data']['y'].values),max(rOI['data']['y'].values)],aspect='auto')
    plt.scatter(rOI['data']['x'].mean(),rOI['data']['y'].mean(),color='red',marker='+',s=1000)
    plt.show()

brum = [-20.119722,-44.121389]
file = dataRep.joinpath('RGBT.jp2')
out = rgbProcessor(file,brum)
rgbPlotter(out)

file2 = dataRep.joinpath('InsarTest.tif')
out2 = insarProcessor(file2,brum)
insarPlotter(out2)

import seaborn as sns
import matplotlib.pyplot as plt

inplot = out2['data'].dropna()
print(inplot.head(10))
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(inplot['long'], inplot['lat'], inplot['disp_m'])
plt.show()