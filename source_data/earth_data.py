import os
import datetime
import json

import pandas as pd
import numpy as np

### Mapping Imports ###
import ee
import ee.mapclient
from ee import oauth

#import geemap.colormaps as cmps
import branca.colormap as cm
from branca.element import MacroElement
from jinja2 import Template

import folium
from folium import plugins
#import geemap.colormaps as cm

from google_auth_oauthlib.flow import Flow
#import restee as ree

### IM AWARE Imports ###
from DAMS import ANM_DAMS, DAM
from dam_break.dambreak_lib import DAMBREAK_SIM
from IPython.display import display
import DBdriver.DBFunctions as dbf

import jinja2
from source_data.GCPdata import GCP_HANDLER
import io



def INITIALISE():
    ##################### DAM STUFF ########################
    try:
        #ee.Authenticate()
        ee.Initialize()
    except:
        print('ERROR: Attempted to initialise a Google Earth Enngine with your accont. \n - Without google earth this wont work!')
'''
basemap_list = ['Google Maps','Google Satellite','Google Terrain','Google Satellite Hybrid','Esri Satellite']
vis_rbg = {'min': 0, 'max': 3300, 'bands': ['B4', 'B3', 'B2']}
#vis_elev = {'min': 0,'max': 4000,'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

#palette = cm.palettes.dem
vis_elev = {'min': 800, 'max': 1900, 'palette':['black','white']} #palette}#['white', 'blue','pink','red','black','green','yellow']}  # 'blue','green',palette} 


default_cmap = ["darkblue","yellow","orange","red"]#['black','red','yellow','white']
#,"blue","cyan","green",
'''


class Interactive_map(DAM):
    """Class for the interactive map generation used in the online APP. The
    ...
    Attributes
    ----------
    DAM : class 
        The DAM class also defined in this script holds all the methods for selecting a dam and 
        using it in the visualisations. 
    Methods
    -------
    basemap_options()
        Prints the options that can be used as the background to the map
    _initialise_map()
        Is a private method that initialises the folium map object
    
    terrain_model()
        Initialises the terrain model used in the visualisation
        #TODO: Currently hard coded JAXA, generalise to other sources
    visual_data()
        Loads sentinel data. 
        #TODO: Generalise to format the output of the sentinel data into pannels. 
        #       And add methods to deal with the cloud masking, and image segentation
    generate_layers()
        Adds the layers to the map
        #TODO: Needs to be generalised to accept any type of layer object, with key 
        #       value pairs for the layer name and response
    show_map()
        Generated the interactive window.
    """
    __simulation_results__ = False
    _full_dam_list = False
    _verbose_ = True
    __init_terrain__ = False

    basemap_list = ['Google Maps','Google Satellite','Google Terrain','Google Satellite Hybrid','Esri Satellite']
    vis_rbg = {'min': 0, 'max': 3300, 'bands': ['B4', 'B3', 'B2']}
    vis_elev = {'min': 800, 'max': 1900, 'palette':['black','white']} #palette}#['white', 'blue','pink','red','black','green','yellow']}  # 'blue','green',palette} 

    zoom_start = 12
    layerControl = None
    
    def __init__(self, *args):
        super(DAM).__init__()
        INITIALISE() #initialise earth engine
        self._read_data()
        self.choose_dam(*args)

        self.range_of_view_degrees = .05
        #print('Map target : {}'.format(self.map_label()))

        # Just some dummy dates for the sentinel stuff
        start_date = '2020-07-15'
        end_date = '2021-08-01'
        # if self.__init_terrain__:  #TODO add this in, fix error on line 128
        self.terrain_model()
        self.visual_data(start_date, end_date)
        self.region_of_interest()
        self.generate_map_layers()

    def _read_data(self,file_address=None):
        super()._read_data(file_address=file_address)

    def region_of_interest(self, *angle_of_interest):
        if angle_of_interest:
            self.range_of_view_degrees = angle_of_interest[0]
        LB = (self.loc[0]-self.range_of_view_degrees,
              self.loc[1]-self.range_of_view_degrees)
        RT = (self.loc[0]+self.range_of_view_degrees,
              self.loc[1]+self.range_of_view_degrees)
        self.polygon_roi = [LB, (LB[0], RT[1]), RT, (RT[0], LB[1])]
        #self.roi = ee.Geometry.Rectangle([LB, RT])
        self.roi = ee.Geometry.Point(self.loc[0],self.loc[1]).buffer(10000)
        return self.roi

    def basemap_options(self):
        return self.basemap_list

    def _initialise_map(self, zoom_start=12):
        Map = folium.Map(location=self.loc, zoom_start=zoom_start,
                         control_scale=True)
        #Map = geemap.Map()
        Map.add_ee_layer = add_ee_layer
        
        if not self._full_dam_list:
            Map = dam_pointer(Map, self.dams_df)
        else:
            Map = dam_pointer(Map, self._full_dams_df)

        #Map.add_ee_layer(Map,self.dem.updateMask(dem.gt(0)), vis_params, 'DEM')
        Map.add_ee_layer(Map, self.terrain, self.vis_elev, 'JAXA')
        basemaps['Google Maps'].add_to(Map)
        basemaps['Google Satellite Hybrid'].add_to(Map)
        return Map

    def terrain_model(self):
        # Import the USGS or JAXA ground elevation image.
        #dem_usgs = ee.Image('USGS/SRTMGL1_003')
        dem = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2')
        self.terrain = dem.select('DSM')
        print('Using {} terrain model'.format('JAXA'))

    def visual_data(self, start_date, end_date):
        self.dates = [start_date, end_date]
        self.poi = ee.Geometry.Point(self.get_loc(longlat=True))
        self.cop_dat = ee.ImageCollection(
            'COPERNICUS/S2').filterBounds(self.poi).filterDate(start_date, end_date)

    def add_visual(self):
        self.Map.add_ee_layer(self.Map, self.cop_dat.mean(),
                              self.vis_rbg, 'Sentenal')  # s2.mean()

    def generate_map_layers(self):
        if not hasattr(self, 'roi'):
            self.region_of_interest()

        # Add the elevation model to the map object.
        self.Map = self._initialise_map(zoom_start=self.zoom_start)
        #self.Map.add_ee_layer(self.Map,dem.updateMask(dem.gt(0)), vis_params, 'DEM')
        #self.Map.add_ee_layer(self.Map, self.terrain, vis_elev, 'JAXA')
        #self.Map.add_colorbar(vis_elev, label="Elevation (m)",
        #                 layer_name='JAXA')
        #folium.vector_layers.Polygon(
        #    self.polygon_roi, color='blue', weight=10).add_to(self.Map)
        # Add a layer control panel to the map.

    def add_image_layer(self, path_to_file, minLon, maxLon, minLat, maxLat, bEnforceWarehouseFolder=False):
        '''
        Overlays a png image over the interactive map
        NOTE: will only accept files relative to the data warehouse (IMAWARE). Absolute paths are adjusted to accordingly.
            This allows compatibility with different filesystems, and legacy database entries which store absolute paths.
        - path_to_file : path to png
        - minLon, maxLon, minLat, maxLat : lower and upper limits for longitude and latitude.
        '''
        # if bEnforceWarehouseFolder:
        #     try:
        #         path_to_file = str(path_to_file).split('IMAWARE')[1]
        #     except:
        #         pass
        #     path_to_file = str(
        #         directory_manager.get_warehouse_dir()) + path_to_file
        # print('path_to_file: ', path_to_file)

        imageObj = self.gcp.load_image(path_to_file)

        img = folium.raster_layers.ImageOverlay(
            image=imageObj, bounds=[
                                  [minLat, minLon], [maxLat, maxLon]])
        img.add_to(self.Map)

    def add_png_layer(self, imageObj, minLon, maxLon, minLat, maxLat, Label='test', bEnforceWarehouseFolder=False, colorMap=None):
        '''
        Overlays a png image over the interactive map
        NOTE: will only accept files relative to the data warehouse (IMAWARE). Absolute paths are adjusted to accordingly.
            This allows compatibility with different filesystems, and legacy database entries which store absolute paths.
        - path_to_file : path to png
        - minLon, maxLon, minLat, maxLat : lower and upper limits for longitude and latitude.
        '''
        # if bEnforceWarehouseFolder:
        #     try:
        #         path_to_file = str(path_to_file).split('IMAWARE')[1]
        #     except:
        #         pass
        #     path_to_file =str(directory_manager.get_warehouse_dir()) + path_to_file
        # print('path_to_file: ',path_to_file)

        ## Import GCP file as object
        # imageObj = self.gcp.load_image(path_to_file)
        
        img = folium.raster_layers.ImageOverlay(
            image=imageObj, bounds=[[minLat, minLon], [maxLat, maxLon]], name=Label, colorMap=None)
        #img.add_to(self.Map)
        self.Map.add_child(img)
        if colorMap:
            self.Map.add_child(colorMap)
            self.Map.add_child(BindColormap(img,colorMap))

    def add_marker_point(self,lon,lat,label='default_label',tooltip='default_tooltip'):
        self.Map.add_child(folium.Marker(location=[lat,lon],
                                        popup=label,
                                        tooltip=tooltip))

    def add_overlay_layer(self,path_to_folder,png=None,dat=None,bEnforceWarehouseFolder=False):
        '''
        Adds an overlay specified by a set of files in a folder.
        - path_to_folder : the folder containing the overlay files.
        - png : the specific png or list of pngs to render. Defaults to rendering all pngs.
        - dat : data file for the overlay. Format [minLon,maxLon,minLat,maxLat,Misc]. Misc is returned by this function.
        '''
        # if png file is not specified, overlay all of the pngs in the folder
        if not png:
            if self._verbose_:
                print('WARNING: No png given. Assuming it is the only png file in the directory {}'.format(path_to_folder))
            png = [i for i in os.listdir(path_to_folder) if '.png' in i]
        if not isinstance(png,list):
            png = [png]

        # if .dat file is not specified, find one in the folder automatically
        if not dat:
            if self._verbose_:
                print('WARNING: No dat given. Assuming it is the only dat file in the directory {}'.format(path_to_folder))

            dat = [i for i in os.listdir(path_to_folder) if '.dat' in i]
        # Use first .dat in list
        if isinstance(dat,list):
            dat = dat[0]
        datPath = path_to_folder + '/%s' % dat
        
        # Extract layer data
        with open(datPath, 'r') as f:
            dataStr = f.readlines()[0]
        data = dataStr.split(',')
        minLon = float(data[0])
        maxLon = float(data[1])
        minLat = float(data[2])
        maxLat = float(data[3])

        for p in png:
            self.add_png_layer(path_to_folder + '/%s' % p, minLon, maxLon, minLat, maxLat, bEnforceWarehouseFolder=bEnforceWarehouseFolder)
        
        # Return additional layer data
        return data[4:]
        
    def add_dambreak_layer(self,record,outputSelect='speed'):
        '''
        Overlays a pre-rendered flood over the map
        NOTE: probably too specific to dambreak for Interactive_Map, consider moving
        NOTE: Deprecated
        '''
        print('add_dambreak_layer is deprecated')
        damID = self.dam_uid
        #record1 = dbf.query_by_dam(damID,'Flooding_Model_Description')
        #record1 = record1[0]
        #record2 = dbf.query_by_analysis(record1['ID'])
        #record2 = record2[0]
        record2 = record
        folder = os.path.splitext(record2['File_Address'])[0]
        folder = folder.replace('Analysis_Results','Analysis_Images')
        fimg = '%s.png' % outputSelect
        fdata = 'position.dat'

        data = self.add_overlay_layer(folder,fimg,fdata)
        minEnergy = float(data[0])
        maxEnergy = float(data[1])
        minSpeed = float(data[2])
        maxSpeed = float(data[3])
        minAltitude = float(data[4])
        maxAltitude = float(data[5])

        #toxic implementation, find another way
        minOutput = eval('min%s' % outputSelect.capitalize())
        maxOutput = eval('max%s' % outputSelect.capitalize())

        # add color bar
        colors = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
        index = np.linspace(minOutput,maxOutput,len(colors))
        colormap = cm.LinearColormap(colors=colors,
                             index=index, vmin=minOutput, vmax=maxOutput,
                             caption='%s (m/s)' % outputSelect.capitalize())

        colormap.add_to(self.Map)

    def add_marker(self,tooltip_list):
        for each in tooltip_list:
            each.add_to(self.Map)

    def finalise(self):
        '''
        Run this method after adding all other features.
        NOTE: Add anything that has to happen at finalisation here.
        '''
        if self.layerControl!=None:
            self.layerControl.reset()
            return
        self.layerControl = folium.LayerControl()
        #latLonPopup = LatLngPopup()
        #clickPopup = ClickPopup(location=self.loc)
        self.Map.add_child(self.layerControl)
        #self.Map.add_child(latLonPopup)
        #clickPopup.add_to(self.Map)

    def save_map(self,destPath):
        '''
        Saves the map as an interactive HTML file on Google Cloud Storage at the given destination path
        NOTE: Legacy code, may not work
        '''
        s = io.StringIO()
        self.Map.save(s)
        htmlIn = s.getvalue()
        gcp = GCP_HANDLER()
        gcp.save_text(htmlIn,destPath)

    def show_map(self):
        # self.finalise()
        # Add fullscreen button
        plugins.Fullscreen().add_to(self.Map)
        display(self.Map)

    def return_figure(self, figsize=[1400,900]):
        fig = folium.Figure(width=figsize[0], height=figsize[1])
        self.Map.add_to(fig)
        self.fig = fig
    
    def show_figure(self):
        if not hasattr(self,'fig'):
            self.return_figure()
        #display(self.fig)'

class BindColormap(MacroElement):
    '''Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    '''
    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
        self.layer = layer
        self.colormap = colormap
        self._template = Template(u"""
        {% macro script(this, kwargs) %}
            {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
            {{this._parent.get_name()}}.on('overlayadd', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
                }});
            {{this._parent.get_name()}}.on('overlayremove', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
                }});
        {% endmacro %}
        """)  # noqa

class LinearColormapVisible(cm.LinearColormap):
    '''
    Same as LinearColormap but with better contrasted font.
    '''
    def __init__(self, colors, index=None, vmin=0., vmax=1., caption=''):
        super().__init__(colors=colors,index=index,vmin=vmin, vmax=vmax,caption=caption)
        with open('source_data/color_scale_white.js','r') as templateFile:
            self._template = Template(templateFile.read())

class ClickPopup(folium.Marker):
    def __init__(self,location=[0,0]):
        popup = f'<input type="text" value="{location[0]}, {location[1]}" id="myInput"><button onclick="myFunction()">Copy location</button>'
        super().__init__(location=location,popup=popup)

    def add_to(self, parent, name=None, index=None):
        super().add_to(parent, name, index)
        ''' Adds mouse position to top right'''
        '''
        formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"
        plugins.MousePosition(
            position="topright",
            separator=" | ",
            empty_string="NaN",
            lng_first=True,
            num_digits=20,
            prefix="Coordinates:",
            lat_formatter=formatter,
            lng_formatter=formatter).add_to(parent)
        '''
        jinja2python = folium.MacroElement().add_to(parent)
        jinja2python._template = jinja2.Template("""
            {% macro script(this, kwargs) %}
            function myFunction() {
            /* Get the text field */
            var copyText = document.getElementById("myInput");

            /* Select the text field */
            copyText.select();
            copyText.setSelectionRange(0, 99999); /* For mobile devices */

            /* Copy the text inside the text field */
            document.execCommand("copy");
            document.getElementById("latitude").value = e.latlng.lat.toFixed(4);
            }
            {% endmacro %}
        """)
        #return self


class LatLngPopup(MacroElement):
    """
    When one clicks on a Map that contains a LatLngPopup,
    a popup is shown that displays the latitude and longitude of the pointer.

    """
    _template = Template(u"""
            {% macro script(this, kwargs) %}
                var {{this.get_name()}} = L.popup();
                function latLngPop(e) {
                data = e.latlng.lat.toFixed(4) + "," + e.latlng.lng.toFixed(4);
                    {{this.get_name()}}
                        .setLatLng(e.latlng)
                        .setContent( "<br /><a href="+data+"> click </a>")
                        .openOn({{this._parent.get_name()}})
                    }
                {{this._parent.get_name()}}.on('click', latLngPop);

            {% endmacro %}
            """)  # noqa

    def __init__(self):
        super(LatLngPopup, self).__init__()
        self._name = 'LatLngPopup'


def define_dot_markers(panda,lat_name,long_name,tooltip_name,pop_name, colour):
    markers = []
    for i in range(0,len(panda)):
        crds = [panda.iloc[i][lat_name],panda.iloc[i][long_name] ]
        tlp = str(panda.iloc[i][tooltip_name])
        pop = str(panda.iloc[i][pop_name]) 
        markers.append(folium.Circle(location = crds, radius = 0.5 , tooltip = tlp, popup = pop, 
            color = colour, fill = True))
    return markers

def add_ee_layer(self, ee_object, vis_params, name):
    """ Function to add a layer to the folium map
    ...
    """
    try:
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True).add_to(self)
        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):
            folium.GeoJson(
                data=ee_object.getInfo(),
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)

    except:
        print("Could not display {}".format(name))

#TODO: ANDRES: I'd like you to look into the folium Popup method. It would be great if we can dispaly a 
#               variety of intormeation (like we are doing with the dam pointer, in a popup). That 
#               could be links to interesting documents. Details about the simulation results completed
#               Or anthing else. Optional dam pointer functions that could give different information
#               for all dams based on a selection would be brilliant. 

def dam_pointer(Map, dam_pd):
    """ Function to add a pointer to the map for each dam site
    """
    for idd in dam_pd.index:
        dam = dam_pd.iloc[idd]
        label = '{} - ({})'.format(dam.Dam_Name, dam.Company)
        #print(label)
        dam_loc = [dam.Lat, dam.Long]
        folium.Marker(location=dam_loc, popup=label).add_to(Map)
    return Map

# Adding the layer method to the folium maps
folium.Map.add_ee_layer = add_ee_layer


def generate_cmap(palette, magnitude):
    color_dict = {}
    col = np.linspace(magnitude[0], magnitude[1], len(palette), endpoint=True)
    for ind, C in enumerate(palette):
        print(ind)
        color_dict[col[ind]] = C
    return color_dict


def colouration(values, color_dict):
    if not len(values):
        values = [values]
    col = []
    key_V = np.array(list(color_dict.keys()))
    for V in values:
        col.append(color_dict[key_V[np.argmin(abs(key_V-V))]])
    return col


def poly_colour_line(latlon, values, feat_group, cmap, resolution=4, weight=2.5, opacity=1):
    c = 0
    for p in latlon[1::resolution]:
        #col = get_color(values[c], VBounds)
        if c == 0:
            p1 = latlon[1::resolution][c]
        ########### Now apply an resolution method... add [::param] to for loop
        #####                                         add average to the values in the range in cmap()
        intp = list(np.concatenate(
            [c - np.array(range(int(resolution/2))), c + np.array(range(1, int(resolution/2)))]))
        folium.PolyLine([p1, p], color=cmap(np.max(values[intp])),
                        weight=weight, opacity=opacity).add_to(feat_group)  #
        p1 = p
        c += 1
    return feat_group




# Add custom basemaps to folium
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps',
        overlay=True,
        control=True
    ),
    'Google Satellite': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ),
    'Google Terrain': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Terrain',
        overlay=True,
        control=True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=True,
        control=True
    )
}
