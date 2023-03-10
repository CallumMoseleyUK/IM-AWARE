from pathlib import Path
import os 
import io
import sys
from google.cloud import storage
sys.path.append(str(Path(__file__).parent.parent))

class CLOUD_CONNECT:

    def __init__(self) :
        home_addr = os.path.expanduser('~')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = home_addr+'/cloudsql/imaware-cloud-storage.json'
        self.storage_client = storage.Client()

    def create_bucket(self,bucket_name):
        new_bucket = self.storage_client.create_bucket(bucket_name)
        return new_bucket
    
    def fetch_bucket(self,bucket_name):
        return self.storage_client.get_bucket(bucket_name)

    def create_blob(self,bucket,cloud_path):
        blob = bucket.blob(cloud_path)
        return blob 
 
    def upload_file(self,cloud_path,file_path):
        bucket_name = cloud_path.split('/')[0]
        bucket = self.fetch_bucket(bucket_name)
        blob = self.create_blob(bucket,cloud_path)
        with open(file_path,'r') as f:
            file = f.read()
        blob.upload_from_string(file)

    def download_file(self,cloud_path):
        bucket_name = cloud_path.split('/')[0]
        bucket = self.fetch_bucket(bucket_name)
        blob = self.create_blob(bucket,cloud_path)
        return io.BytesIO(blob.download_as_string())   

    def upload_pandas(self,panda,cloud_path): 
        bucket_name = cloud_path.split('/')[0]
        bucket = self.fetch_bucket(bucket_name)
        blob = self.create_blob(bucket,cloud_path)
        s = io.StringIO()
        panda.to_csv(s,index=False)
        upload = s.getvalue()
        blob.upload_from_string(upload)
    
    def upload_altair(self,alt,cloud_path):
        alt.save('s.html')
        self.upload_file(cloud_path,'s.html')
        os.remove('s.html')

        
       
if __name__ == '__main__':
    cc = CLOUD_CONNECT()
    cc.create_bucket('andres_test_cloud')
    cloud_path = 'andres_test_cloud/moo/test.html'
    file_path = 'E:\im_aware_collab\SRC\IM-AWARE-GIS\INSAR\KDE.txt'
    cc.upload_file(cloud_path,file_path)   
    a = cc.download_file(cloud_path)