import numpy as np
import folium
import os
from PIL import Image
from io import StringIO
import os
import io
import pandas as pd
from json2html import *
import time
from pathlib import Path

class FILE_HANDLER():
    '''
    Base class for handling files used by IMAWARE models. Inherit this class and overload save_bytes() to accomodate storage platform (e.g. local, cloud, etc)
    '''
    def __init__(self):
        pass

    def file_exists(self,path):
        path = self._clean_path(path)
        return Path(path).exists()

    def mkdir(self,path):
        '''Recursively creates a directory'''
        path = self._clean_path(path)
        os.makedirs(path, exist_ok = True)

    def _clean_path(self,path):
        ''' Turns all slashes into forward slashes'''
        return str(Path(path)).replace('\\','/')

    def save_bytes(self,bytesIn,new_path):
        new_path = self._clean_path(new_path)
        self.mkdir(os.path.dirname(new_path))
        with open(new_path,'wb') as file:
            file.write(bytesIn)

    def save_text(self,textIn,new_path):
        bytesIn = bytes(textIn,'utf-8')
        self.save_bytes(bytesIn,new_path)
    
    def save_csv(self,dataFrame,new_path):
        bytesIO = io.BytesIO()
        dataFrame.to_csv(bytesIO,index=False)
        bytesIn = bytesIO.getvalue()
        self.save_bytes(bytesIn, new_path)

    def save_image(self,imageArray,new_path,format='png'):
        bytesIO = io.BytesIO()
        pilImage = Image.fromarray(imageArray.astype(np.uint8))
        pilImage.save(bytesIO, format=format)
        bytesIn = bytesIO.getvalue()
        self.save_bytes(bytesIn,new_path)

    def load_bytes(self,src_path):
        src_path = self._clean_path(src_path)
        with open(src_path,'rb') as file:
            data = file.read()
        return data

    def load_text(self,src_path,format='utf-8'):
        return self.load_bytes(src_path).decode(format)

    def load_csv(self,src_path):
        data = self.load_bytes(src_path)
        return pd.read_csv(io.BytesIO(data))

    def load_csv_as_numpy(self,src_path):
        dataFrame = self.load_csv(src_path)
        return dataFrame.to_numpy()
    
    def load_image(self,src_path,format='RGBA'):
        data = self.load_bytes(src_path)
        if format == '16L':
            image = Image.open(io.BytesIO(data))
        else:
            image = Image.open(io.BytesIO(data)).convert(format)
        return np.array(image).astype(np.uint32)
