from PyQt5 import QtWidgets, QtWebEngineWidgets
from folium.plugins import Draw
import folium, io, sys, json, os
from pathlib import Path
from source_data.earth_data import Interactive_map, add_ee_layer, basemaps
from dam_break.dambreak_sim import DAMBREAK_SIM
from dam_break.dam_break import DAM_BREAK
from source_data.file_handler import FILE_HANDLER
import directory_manager
from datetime import datetime
import math

## NOTE: notes.
#https://stackoverflow.com/questions/66418148/pyqt-embed-qwebengineview-in-main-window
#https://doc.qt.io/qt-6/qwebenginepage.html (see createWindow(QWebEnginePage::WebWindowType type))

class FLOOD_APP():
    ''' App for running flood simulations '''

    drawOptions={
            'polyline':True,
            'rectangle':True,
            'polygon':True,
            'circle':False,
            'marker':True,
            'circlemarker':False}

    latitude,longitude = -20.119722, -44.121389
    mapResolution = 1
    mapSkipPoints = 1
    simInfoDict = {}
    zoomStart = 12
    mapWidth = 3

    resultsDirectory = directory_manager.get_warehouse_dir()
    demDirectory = directory_manager.get_dem_dir()

    damID = 'default_dam'
    simID = 'default_sim'
    simulationSettings = {
            'siteLat': latitude,
            'siteLon': longitude,
            'pondRadius': 103.0,
            'nObj': 100,
            'tailingsVolume': 2685782.0,
            'tailingsDensity': 1594.0,
            'maxTime': 120.0,
            'timeStep': 0.2,
            'dampingCoeff': 0.04,
            'demDirectory': demDirectory,
            'fileHandler': None
    }
    settingLabels = {
            'nObj': 'Particle number',
            'pondRadius': 'Pond radius (m)',
            'tailingsVolume': 'Release volume (m^3)',
            'tailingsDensity': 'Material density (kg/m^3)',
            'dampingCoeff': 'Damping factor',
            'maxTime': 'Sim time (s)',
            'timeStep': 'Time step (s)'
    }
    textFields = {}

    simulation = None
    fileHandler = None
    imap = None

    def __init__(self,fileHandler,*args):
        '''
        Parameters:
            fileHandler : Instance of a FILE_HANDLER to be used
            args : sys.argsv
        '''
        self._init_QApplication(*args)

        self.fileHandler = fileHandler
        self.update_sim_settings(fileHandler=self.fileHandler)

    def _init_QApplication(self,*args):
        self.qApplication = QtWidgets.QApplication(*args)

    def _init_main_window(self):
        ''' Structures the main window '''
        self.mapView = QtWebEngineWidgets.QWebEngineView()
        self.pushButton = QtWidgets.QPushButton()
        self.pushButton.setText('Stop server')
        self.pushButton.clicked.connect(self._button_pushed)

        self.mainWindow = QtWidgets.QMainWindow()
        centralWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout(centralWidget)
        self.mainWindow.setCentralWidget(centralWidget)

        self._init_text_fields()
        self.layout.addWidget(self.mapView,0,0,self.layoutRows,self.mapWidth)
        self.layout.addWidget(self.pushButton,self.layoutRows,self.mapWidth+1)

        self.layout.setColumnStretch(0,1)
        self.layout.setColumnStretch(1,1)
        self.layout.setRowStretch(0,1)
        self.layout.setRowStretch(1,1)

        self.webEnginePage = WEB_ENGINE_PAGE(self,self.mapView)
        self.mapView.setPage(self.webEnginePage)
        
    def _init_text_fields(self):
        ''' Adds a label and text field for each simulation settings '''
        i = 0
        for key in self.settingLabels.keys():
            textLabel = self.settingLabels[key]
            textValue = self.simulationSettings[key]

            labelWidget = QtWidgets.QLabel()
            labelWidget.setText(textLabel)
            self.layout.addWidget(labelWidget,i,self.mapWidth+1,1,1)
            i += 1

            textEdit = QtWidgets.QTextEdit()
            textEdit.setPlaceholderText(textLabel)
            textEdit.setPlainText(str(textValue))
            self.layout.addWidget(textEdit,i,self.mapWidth+1,1,1)
            self.textFields[key] = textEdit
            i += 1
        self.layoutRows = i

    def _update_map(self):
        self._update_map_data()
        self._update_map_view()

    def _update_map_data(self):
        ''' Updates interactive map and saves to self.mapData as bytes'''
        self.imap = APP_MAP(self.latitude,self.longitude,zoom_start=self.zoomStart)
        draw = Draw(draw_options=self.drawOptions, edit_options={'edit':False})
        self.imap.Map.add_child(draw)

        for key in self.simInfoDict.keys():
            self._render_sim_images(self.simInfoDict[key])
        self.imap.finalise()

        self.mapData = io.BytesIO()
        self.imap.Map.save(self.mapData, close_file=False)

    def _update_map_view(self,htmlCode=None):
        '''Regenerates map viewer HTML'''
        if htmlCode==None:
            htmlCode = self.mapData.getvalue().decode()
        self.mapView.setHtml(htmlCode)

    def _button_pushed(self):
        '''Button push event'''
        print('Button pushed')

    def read_settings(self):
        ''' Reads settings from text fields and updates sim settings '''
        kwargs = {}
        for key in self.textFields.keys():
            kwargs[key] = float(self.textFields[key].toPlainText())
        self.update_sim_settings(**kwargs)
        
    def run(self):
        ''' Runs the app until the window is closed '''
        self._init_main_window()
        self._update_map()

        self.mainWindow.show()
        self.exit()

    def exit(self):
        self.mapData.close()
        sys.exit(self.qApplication.exec_())

    def update_sim_settings(self,**kwargs):
        ''' Updates current simulation settings '''
        for k in kwargs.keys():
            self.simulationSettings[k] = kwargs[k]
        self.latitude = self.simulationSettings['siteLat']
        self.longitude = self.simulationSettings['siteLon']

    def run_flood_simulation(self):
        ''' Runs the flood simulation for the selected point '''
        self.simID = self.get_sim_ID()
        self.simulation = DAM_BREAK(**self.simulationSettings)
        self.simulation._bVerbose = True
        self.simulation.run_simulation()
        (fileName,csvName) = self.simulation.save_results(self.damID,self.simID,fileHandler=self.fileHandler,warehouseFolder=self.resultsDirectory)
        
        self.simRecord = self.simulation.get_database_record(self.simID)
        self.simRecord['File_Address'] = csvName
        self.simRecord['File_Handler'] = self.fileHandler
        self.simResultsHandler = DAMBREAK_SIM(srcInput=self.simRecord,bAbsolutePath=True,demDirectory=self.demDirectory)
        self.renderPath = os.path.dirname(csvName)
        mask,maskX,maskY = self.simResultsHandler.fit_speed_mask(
                                                    self.simResultsHandler.max_time(),
                                                    resolution = self.mapResolution,
                                                    skipPoints = self.mapSkipPoints)
        maskPath = self.renderPath + '/speed_%s.png' % self.simID
        self.simResultsHandler.save_mask(maskPath,mask,maskX,maskY)

        # Create new sim info
        extent = self.simResultsHandler.get_lon_lat_bounds(maxTime=self.simResultsHandler.max_time())
        newSimInfo = SIM_INFO(self.latitude,self.longitude,
                            self.simulationSettings,
                            label=self.get_sim_ID(),
                            extent=extent,
                            speed_render=maskPath)
        self.simInfoDict[self.simID] = newSimInfo

        # Render new sim info
        self._update_map()

    def get_sim_ID(self):
        ''' Generates a unique ID for a simulation'''
        time_stamp = datetime.now()
        fileID = '{}-{}'.format(self.damID,
                                time_stamp.strftime("%Y%m%d-%H%M%S"))
        return fileID

    def _render_sim_images(self,simInfo):
        ''' Renders all images associated with a simulation '''
        simInfo.render(self.imap,self.fileHandler)

class WEB_ENGINE_PAGE(QtWebEngineWidgets.QWebEnginePage):

    def __init__(self,app,webEngineView):
        super().__init__(webEngineView)
        self.loadFinished.connect(self.handleLoadFinished)

        self.app = app
        self.featureHandlers = {
            'Point': self.click_point,
            'Polygon': self.click_poly
        }
    
    def handleLoadFinished(self):
        print('Application loaded')

    def javaScriptAlert(self,securityOrigin,msg):
        ''' overload to prevent javascript alert popups ''' 
        pass

    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        print(msg)
        try:
            feature = json.loads(msg)
        except:
            print('Json message cannot be interpreted')
            print(msg)
            return
        featureType = feature['geometry']['type']
        print('Feature type selected: ',featureType)
        coords = feature['geometry']['coordinates']
        self.featureHandlers[featureType](coords)

    def click_point(self,coords):
        print('clicking point')
        print(coords)
        self.app.update_sim_settings(siteLat=coords[1],siteLon=coords[0])
        self.app.read_settings()
        self.app.run_flood_simulation()
    
    def click_poly(self,coords):
        print('clicking poly')
        print(coords)

    def click_line(self,coords):
        print(coords)

class APP_MAP(Interactive_map):
    def __init__(self,latitude,longitude,zoom_start=13):
        self.latitude = latitude
        self.longitude = longitude
        self.zoom_start = zoom_start
        super().__init__(latitude,longitude)

    def _read_data(self):
        pass

    def choose_dam(self,*args):
        pass

    def get_loc(self,longlat=False):
        if longlat:
            return [self.longitude,self.latitude]
        return [self.latitude,self.longitude]

    def region_of_interest(self,*angle_of_interest):
        return None

    def _initialise_map(self, zoom_start=13):
        Map = folium.Map(location=self.get_loc(), zoom_start=zoom_start,
                         control_scale=True)
        Map.add_ee_layer = add_ee_layer

        Map.add_ee_layer(Map, self.terrain, self.vis_elev, 'JAXA')
        basemaps['Google Maps'].add_to(Map)
        basemaps['Google Satellite Hybrid'].add_to(Map)
        return Map

class SIM_INFO:
    ''' Container class for simulation parameter information '''
    latitude = FLOOD_APP.latitude
    longitude = FLOOD_APP.longitude
    simSettings = {}
    extent = (0,1,0,1)
    images = {}
    label = 'default_sim_info'

    def __init__(self,lat,lon,simSettings,label='default_sim_info',extent=(0,1,0,1),**images):
        self.latitude = lat
        self.longitude = lon
        self.extent = extent
        self.images = images
        self.label = label
        self._update_sim_settings(simSettings)

    def _update_sim_settings(self,simSettings):
        self.simSettings = simSettings

    def render(self,interactiveMap,fileHandler):
        minLon,maxLon,minLat,maxLat = self.extent
        for key in self.images.keys():
            imageLabel = self.label + '_' + key
            imageDir = self.images[key]
            image = fileHandler.load_image(imageDir)
            interactiveMap.add_png_layer(image,minLon,maxLon,minLat,maxLat,
                                        Label=imageLabel,colorMap=None)
        interactiveMap.add_marker_point(self.longitude,self.latitude,
                                label=self.label,tooltip=self.get_tooltip())

    def get_tooltip(self):
        latStr,lonStr = self.format_coords(self.latitude,self.longitude)
        tooltip = '<h3 align="center" style="font-size:16px"><b>%s</b></h3>' % self.label
        tooltip += '<br><b>Latitude</b> : %s<br><b>Longitude</b>: %s<br>' % (latStr,lonStr)

        for key in FLOOD_APP.settingLabels.keys():
            settingName = FLOOD_APP.settingLabels[key]
            tooltip += '<b>%s</b> : %s<br>' % (settingName,str(self.simSettings[key]))
        return tooltip

    def format_coords(self,lat,lon):
        ''' Formats coordinates in the form <degrees>° <minutes>\' <seconds>\'\' S etc'''
        if lat<0.0:
            latStr = '%s° %s\' %s\'\' S'
        else:
            latStr = '%s° %s\' %s\'\' N'
        latStr = latStr % self.decompose_coord(lat)

        if lon<0.0:
            lonStr = '%s° %s\' %s\'\' W'
        else:
            lonStr = '%s° %s\' %s\'\' E'
        lonStr = lonStr % self.decompose_coord(lon)
        return latStr,lonStr

    def decompose_coord(self,coord):
        ''' Decomposes an angle in degrees to degrees, minutes and seconds'''
        degrees = abs(coord)
        minutes = (degrees % 1)*60.0
        seconds = (minutes % 1)*60.0
        return math.floor(degrees),math.floor(minutes),math.floor(seconds)
        
if __name__ == '__main__':
    app = FLOOD_APP(FILE_HANDLER(),sys.argv)
    app.run()