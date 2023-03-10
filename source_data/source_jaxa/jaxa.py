## Library to query JAXA DEM model
# Circumvents login credientals, if needed apply for access at:
# https://www.eorc.jaxa.jp/ALOS/en/aw3d30/index.htm
#

from io import StringIO, BytesIO
import numpy as np
from math import floor, ceil
import os.path
import zipfile
import requests
from PIL import Image
from source_data.file_handler import FILE_HANDLER
#from source_data.GCPdata import GCP_IO
from pathlib import Path

def get_map(lat,long,dataDir,fileHandler=None):
    '''
    Retrives a DEM map for given coordinates, downloading it from JAXA if necessary
        Output: (mapZ,tifDir,mapLat,mapLong)
    '''
    ## Format URL for given coordinates
    if lat<0:
        tileCodeLat = "S%.3i" % abs(floor(lat))
        mapCodeLat = "S%.3i" % abs(floor(lat/5)*5)
    else:
        tileCodeLat = "N%.3i" % abs(floor(lat))
        mapCodeLat = "N%.3i" % abs(floor(lat/5)*5)

    if long<0:
        tileCodeLong = "W%.3i" % abs(floor(long))
        mapCodeLong = "W%.3i" % abs(floor(long/5)*5)
    else:
        tileCodeLong = "E%.3i" % abs(floor(long))
        mapCodeLong = "E%.3i" % abs(floor(long/5)*5)

    tileCode = tileCodeLat + tileCodeLong
    mapCode = mapCodeLat + mapCodeLong

    ## Define file directories
    fname = "%s.zip" % tileCode
    url = "https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2012/%s/%s" % (mapCode, fname)
    zipDest = "ALPSMLC30_%s_DSM.zip" % tileCode
    tifFile = "ALPSMLC30_%s_DSM.tif" % tileCode
    tifDir = "%s/%s/%s" % (dataDir, tileCode, tifFile)
    contentsDir = "%s/%s" % (dataDir,tileCode)

    print('DEM url ', url)
    
    ## Check if tif is on the cloud already
    if fileHandler==None:
        # Default to Google Cloud Storage
        fileHandler = FILE_HANDLER()

    if not fileHandler.file_exists(tifDir):
        download_url(url,dataDir,fileHandler)

    ## Import digital surface model
    mapZ = fileHandler.load_image(tifDir,'16L')

    ## Calculate lower and upper limits for longitude/latitude for the given tile
    mapLat = [float(floor(lat)),float(ceil(lat))]
    mapLong = [float(floor(long)),float(ceil(long))]

    return (mapZ,tifDir,mapLat,mapLong)

def download_url(url, savePath, fileHandler, chunk_size=128):
    ## Download zip
    r = requests.get(url, stream=True)
    zipData = BytesIO()
    for chunk in r.iter_content(chunk_size=chunk_size):
        zipData.write(chunk)

    ## Save contents
    with zipfile.ZipFile(zipData) as extractedFile:
        for fileName in extractedFile.namelist():
            if fileName[-1] == '/':
                fileHandler.mkdir(savePath+'/'+fileName)
                continue
            saveFile = '%s/%s' % (savePath,fileName)
            fileData = extractedFile.read(fileName)
            fileHandler.save_bytes(fileData,saveFile)
    