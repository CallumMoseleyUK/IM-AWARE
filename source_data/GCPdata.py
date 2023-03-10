'''
---
'''
import numpy as np
import folium
import os
from google.cloud import storage
from google.cloud import vision
from PIL import Image
from io import StringIO
import os
import io
import pandas as pd
from json2html import *
import time

#import directory_manager
from pathlib import Path
from source_data.file_handler import FILE_HANDLER
'''
---
'''

home_addr = os.path.expanduser('~')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = home_addr + \
    '/cloudsql/imaware-cloud-storage.json'

class GCP_IO(FILE_HANDLER):
    '''
    Simple GCP input-output handler.
    '''

    bucket_name = 'im_aware_collab'

    def __init__(self,bucketName='im_aware_collab',bSplitPathAtBucket=True):
        self._bSplitPathAtBucket = bSplitPathAtBucket
        self.bucket_name = bucketName
        self.bucket, self.storage_client = GCPconnector(bucketName=self.bucket_name)
        super().__init__()

    # - Misc
    def _generate_blob(self,src_path):
        if type(src_path) == 'str':
            #Typecast to Path object converts all slashes to double backslashes, easier to replace with forward slashes.
            src_path = Path(src_path)
        src_path = str(src_path).replace('\\','/')
        if self._bSplitPathAtBucket:
            src_path = src_path.split(self.bucket_name+'/')[-1]
        return self.bucket.blob(src_path)

    def mkdir(self,path):
        '''
        Google Cloud Platform automatically creates directories as needed.
        '''
        pass

    def file_exists(self,src_path):
        blob = self._generate_blob(src_path)
        return blob.exists()

    def delete_file(self,src_path):
        blob = self._generate_blob(src_path)
        blob.delete()

    ## - Saving
    def save_bytes(self,bytesIn,new_path):
        blob = self._generate_blob(new_path)
        for i in range(10):
            try:
                blob.upload_from_string(bytesIn)
            except Exception as e:
                print('Attempt %i failed to save, trying again: ' % (i))
                print(e)
                time.sleep(1.0)

    def save_text(self, textIn, new_path):
        super().save_text(textIn,new_path)

    def save_csv(self, dataFrame, new_path):
        super().save_csv(dataFrame,new_path)
        
    def save_image(self, imageArray, new_path, format='png'):
        super().save_image(imageArray,new_path,format)

    ## - Loading
    def load_bytes(self,src_path):
        '''
        Download file at src_path as a bytes object
        '''
        src_path = self._clean_path(src_path)
        blob = self._generate_blob(src_path)
        return blob.download_as_string()

    def load_text(self,src_path,format='utf-8'):
        return super().load_text(src_path,format)

    def load_csv(self,src_path):
        return super().load_csv(src_path)

    def load_image(self,src_path,format='RGBA'):
        return super().load_image(src_path,format='RGBA')
    
class GCP_HANDLER(GCP_IO):

    bucket_name = 'im_aware_collab'

    def __init__(self, simulation_dict):
        super().__init__()
        if len(simulation_dict) > 0:
            self.sim_dict = simulation_dict
            if 'File_Address' in list(simulation_dict.keys()):
                path = simulation_dict['File_Address']
            else:
                path = simulation_dict['Path']
            self.path = path.split('im_aware_collab/')[-1]
        self.path = self._clean_path(self.path)
        self.blob = self._generate_blob(self.path)

    # def load_map_image(self, suffix = 'speed'):
    #     path = [i for i in self.list_files_with_matching(prefix)]
    #     return GCPimageLoader(self.blob)

    # def load_map_position(self):
    #     return GCPimageLoader_position(self.blob)

    def load_map_images(self):
        ''' Loads all map images associated with the current analysis'''

        if 'Analysis_Results' in self.path:
            prefix = self.path.replace('Analysis_Results', 'Analysis_Images').replace('.csv', '') + '/'
        names, blobs = self.list_files_with_matching(prefix)
        Dimages = {}
        posx = {}
        histData = {}
        for i in range(len(names)):
                #.dat files contain position/min/max/mean of values in a corresponding .png render.
                if '.dat' in names[i]:
                    posBlob = blobs[i]
                    var = names[i].split('/')[-1].replace('.dat', '')
                    posx[var] = GCPimageLoader_position(posBlob)
                #.png files are pre-rendered transparent images to overlay on a map
                elif '.png' in names[i]:
                    var = names[i].split('/')[-1].replace('.png', '')
                    Dimages[var] = GCPimageLoader(blobs[i])
                #.hist files are histogram data corresponding to a .png and .dat file.
                elif '.hist' in names[i]:
                    var = names[i].split('/')[-1].replace('.hist', '')
                    histData[var] = GCPtextLoader(blobs[i])

        return posx, Dimages, histData

    def load_csv(self,path=None):
        if path==None:
            path = self.path
        path = self._clean_path(path)
        prefix = path
        if '_vert' in path:
            prefix = path.split('_vert')[0]
        if '_corr' in path:
            prefix = path.split('_corr')[0]
        names, blobs = self.list_files_with_matching(prefix)
        CSVs = {}
        for i in range(len(names)):
            if '.csv' in names[i]:
                CSVs[i] = super().load_csv(names[i])
        return CSVs

    def load_csv_insar(self,path):
         '''
         Loads a file in csv format, correcting for inconsistencies in INSAR filenames
         '''
         path = self._clean_path(path)
         prefix = path.split('{}'.format(self.bucket_name)+'/')[-1]
         prefix = prefix[0:-13]
         blob = self.list_files_with_matching(prefix)
         if len(blob[0]) == 1:
            blobc = blob[1][0].download_as_string()
            return pd.read_csv(io.BytesIO(blobc))
         if len(blob[0]) > 1:
            print('duplicate file {}'.format(prefix))
            with open('error_log.txt', 'w') as f:
                f.write('duplicate file {}'.format(prefix))
            return 'error'


    def list_all_files(self):
        blobs = self.storage_client.list_blobs(self.bucket_name)
        for b in blobs:
            print(b.name)

    def list_files_with_matching(self, prefix, delimiter=None):
        blobs = self.storage_client.list_blobs(
            self.bucket_name, prefix=prefix, delimiter=delimiter)

        names = []
        blobber = []
        for blob in blobs:
            names.append(blob.name)
            blobber.append(blob)

        return names, blobber
        # if delimiter:
        #     print("Prefixes:")
        #     for prefix in blobs.prefixes:
        #         print(prefix)

    def save_text(self, textIn, new_path):
        '''
        Saves text to the Google Cloud Storage bucket
        '''
        super().save_text(textIn,new_path)

    def save_csv(self, dataFrame, new_path):
        '''
        Uploads a Pandas dataframe as a csv to bucket
        '''
        super().save_csv(dataFrame,new_path)

    def save_image(self, imageArray, new_path, format='png'):
        '''
        Uploads a numpy array as an image to bucket (default png format)
        '''
        super().save_image(imageArray,new_path,format)


def GCPconnector(bucketName='im_aware_collab'):
    # try:
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()
    # # MAP.add_dambreak_layer('speed')
    # bucket_name = 'im_aware_collab'
    bucket = storage_client.get_bucket(bucketName)

    return bucket, storage_client
    # except Exception as e:
    #     print(e)
    #     return False, False


# bucket, storage_client = GCPconnector()

'''NOTE:  (Callum) I've tried to deprecate all these free-floating methods'''
def GCPblobber(path, bucket, bReplaceBackslash=False):
    ## Google Cloud interprets backslashes as part of a file name, not a separator
    if bReplaceBackslash:
        path = str(path).replace('\\','/')
    blob = bucket.blob(path)
    # blob = blob.download_as_string()
    return blob

def GCPtextLoader(blob):
    data = blob.download_as_string()
    return data.decode('utf-8')

def GCPimageLoader_position(blob):
    data = GCPtextLoader(blob)
    posix = data.split(',')
    return [float(posix[0]), float(posix[1]), float(posix[2]), float(posix[3]), float(posix[4]), float(posix[5]), posix[6], posix[7]]
    #return [float(p) for p in posix[:-1]]


def GCPimageLoader(blob,format='RGBA'):
    image = blob.download_as_string()
    if format == '16L':
        image = Image.open(io.BytesIO(image))
    else:
        image = Image.open(io.BytesIO(image)).convert(format)
    array = np.array(image).astype(np.uint32)
    return array


def GCPcsvLoader(blob,bReturnAsNumpy=False):
    blob = blob.download_as_string()
    if bReturnAsNumpy:
        pass
    else:
        return pd.read_csv(io.BytesIO(blob))


def GCPupload(blob, filename):
    blob.upload_from_filename(filename)


def raw_connector():
    home_addr = os.path.expanduser('~')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = home_addr+'/cloudsql/imaware-cloud-storage.json'
    return storage.Client()

def AltairUpload(data,destination):
    
    bucket_name = destination.split('/')[0]
    sc = raw_connector()
    bucket = sc.get_bucket(bucket_name)
    blob = bucket.blob(destination)
    s = io.StringIO(data.to_html())
    upload = s.getvalue()
    blob.upload_from_string(upload) 

##### CALLUM IMPLIMETNATION
# class GCP_HANDLER():
#     '''
#     Handles saving/loading to and from Google Cloud Storage
#     '''
#     def __init__(self,bucketName='im_aware_collab'):
#         ### Locate service account credentials
#         home_addr = os.path.expanduser('~')

#         os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = home_addr + '/cloudsql/imaware-cloud-storage.json'
#         ## Connect to data warehouse bucket
#         self.bucketName = bucketName
#         self.file_address_prefix = 'https: // storage.cloud.google.com' + '/'+ self.bucketName
#         t = self._gcp_connector(self.bucketName)
#         if t:
#             self.bucket, self.storageClient = t
#         else: 
#             print('_gcp_connector failed')

#     def _gcp_connector(self,bucketName):
#         '''
#         Returns the bucket object for the data warehouse
#         '''
#         try:
#             vision_client = vision.ImageAnnotatorClient()
#             # MAP.add_dambreak_layer('speed')
#             storageClient = storage.Client()
#             bucket = storageClient.get_bucket(bucketName)
#             return bucket, storageClient
#         except Exception as e:
#             print(e)
#             return False

#     def _gcp_blobber(self,path,bSplitPath=True):
#         '''
#         Loads the given file as a blob.
#         Additionally splits the path at the bucket name
#         '''
#         if bSplitPath:
#             path = str(path)
#             new_path = path.split(self.bucketName)[-1]
#             new_path = new_path[1:]
#         else:
#             new_path = path

#         new_path = str(Path(new_path))
#         print('NEW PATH --- ', new_path)
#         return self.bucket.blob(new_path)

#     def _load_file_as_string(self,filePath):
#         '''
#         Loads file as string
#         '''

#         blob = self._gcp_blobber(filePath)
#         return blob.download_as_string()

#     def load_image(self,filePath):
#         '''
#         Loads an image in numpy array format
#         '''
#         blob = self._load_file_as_string(filePath)
#         image = Image.open(io.BytesIO(blob)).convert('RGBA')
#         return np.array(image).astype(np.uint32)

#     def load_csv(self,filePath):
#         '''
#         Loads a file in csv format
#         '''
#         blob = self._load_file_as_string(filePath)
#         return pd.read_csv(io.BytesIO(blob))

#     def load_csv_insar(self,path):
#         '''
#         Loads a file in csv format, correcting for inconsistencies in INSAR filenames
#         '''
#         prefix = path.split(self.bucketName+'/')[-1]
#         prefix = prefix[0:-13]
#         blob = self.list_files_with_matching(prefix)
#         blob = blob.download_as_string()
#         return pd.read_csv(io.BytesIO(blob))


#     def load_text(self,filePath,format='utf-8'):
#         '''
#         Loads a file in text format (default UTF-8)
#         '''
#         blob = self._load_file_as_string(filePath)
#         return blob.decode(format)

#     def save_text(self,textIn,destPath):
#         '''
#         Saves text to the Google Cloud Storage bucket
#         (pretty much just saves raw data, all saving functions use this)
#         '''
#         blob = self._gcp_blobber(destPath)
#         blob.upload_from_string(textIn)

#     def save_csv(self,dataFrame,destPath):
#         '''
#         Uploads a Pandas dataframe as a csv to bucket
#         '''
#         s = io.StringIO()
#         dataFrame.to_csv(s,index=False)
#         textIn = s.getvalue()
#         self.save_text(textIn,destPath)

#     def save_image(self,imageArray,destPath,format='png'):
#         '''
#         Uploads a numpy array as an image to bucket (default png format)
#         '''
#         b = io.BytesIO()
#         pilImage = Image.fromarray(imageArray.astype(np.uint8))
#         pilImage.save(b, format=format)
#         textIn = b.getvalue()
#         self.save_text(textIn,destPath)

#     def save_figure(self,fig,destPath):
#         # TODO: Do something really clever here and generalise figure saving for different formats.
#         pass

#     def list_all_files(self):
#         '''
#         Lists all files in the bucket
#         '''
#         blob = self.storageClient.list_blobs(self.bucketName)
#         for b in blob:
#             print(b.name)

#     def ls(self, delimiter=None):
#         '''
#         TODO: make sure this works
#         '''
#         if not hasattr(self, 'path'):
#             self.gen_blob()
#         blobs = self.storageClient.list_blobs(
#             self.bucket_name, prefix=self.path, delimiter=delimiter)
#         if delimiter:
#             print("Prefixes:")
#             out =[]
#             for prefix in blobs.prefixes:
#                 out.append(prefix)
#             blobs = out
#         return blobs

#     def list_files_with_matching(self,prefix):
#         '''
#         Lists all files with matching prefix
#         '''
#         blob_iter = self.bucket.list_blobs(prefix=prefix)
#         blobs = []

#         for blob in blob_iter:
#             blobs.append(blob)

#         if len(blobs) ==1:
#             return blobs[0]
#         else:
#             return blobs

    
