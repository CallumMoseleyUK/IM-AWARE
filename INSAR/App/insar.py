from flask import Flask,render_template,request,redirect
from pathlib import Path 
import sys
import os 
from io import StringIO
import pathlib

repo = pathlib.Path(__file__).parent.absolute().joinpath('Repository')
rawRepo = repo.joinpath('RawData')
procRepo = repo.joinpath('ProcessedData')
rOIsRepo = repo.joinpath('ROIs')
workFolder = pathlib.Path(__file__).parent


appPath = Path(__file__).parent
mainPath = appPath.parent
EEpath = mainPath.parent.joinpath('source_data')
repositoryPath = mainPath.joinpath('Repository')
roisPath = repositoryPath.joinpath('ROIs')
sys.path.append(str(mainPath))
#sys.path.append(str(EEpath))
templatePath = appPath.joinpath('templates')
print(repositoryPath)
#import earth_data as edata
import InsarModule as insar

app = Flask(__name__)
damList = []
scenes = []
dam = ''

allFiles = list(roisPath.iterdir())
for file in allFiles:
    if str(file.name).split('.')[1] == 'csv':
        damList.append(str(file.name).split('.')[0])
  
@app.route("/", methods = ['GET','POST'])
def firstWindow(): 
    global dam
    if request.method == 'POST':
       dam = request.form['Dam']
       global damO
       damO = insar.damInsarChild(dam)
       return redirect('/SS') 

    else:
        return render_template('Insar.html', Dams = damList )  

@app.route("/SS", methods = ['GET','POST'])
def secondWindow():
    global scene
    if request.method == 'POST':
        scene = request.form['Scene']
        d0 = scene.split('to')[0].strip()
        d1 = scene.split('to')[1].strip()
        maps = damO.mapMaker(d0,d1)
        os.chdir(str(templatePath))
        maps['los'].save('los.html')
        maps['vert'].save('vert.html')
        maps['corr'].save('corr.html') 
        os.chdir(str(mainPath))
#        edata.INITIALISE()
#        edata.basemaps['Google Satellite'].add_to(maps['vert'])
        return render_template('Maps.html', Dam = dam, TL = scene, Map1 = maps['vert'] )

    else:
        return render_template('SceneSelection.html', Scenes = damO.params['TimeSpans'], Dam = dam)

if __name__ == '__main__':
    app.debug = True
    app.run()