# Driver for the IM AWARE web app
# 
# Flask documentation, etc
# https://pymbook.readthedocs.io/en/latest/flask.html
# https://flask.palletsprojects.com/en/2.0.x/quickstart/
# https://www.py4u.net/discuss/278063
#
# Background threading:
# https://stackoverflow.com/questions/14384739/how-can-i-add-a-background-thread-to-flask
# https://stackoverflow.com/questions/21214270/how-to-schedule-a-function-to-run-every-hour-on-flask
#
# Load template with form data:
# https://www.tutorialspoint.com/flask/flask_sending_form_data_to_template.htm
#

## Define directoriies
import os
from turtle import bgcolor, fillcolor
from flask import Flask, request, render_template
import sys
import pathlib

workFolder = pathlib.Path(__file__).parent.parent
webFolder = workFolder.joinpath('web_app')
staticFolder = webFolder.joinpath('static')
sys.path.append(str(workFolder))
from dam_break.dambreak_lib import DAMBREAK_STAT
import numpy as np
from DAMS import DAM
from source_data.earth_data import Interactive_map, INITIALISE, LinearColormapVisible
import DBdriver.DBFunctions as dbf
import branca.colormap as cm
from shutil import copy, copyfileobj

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import six
#from INSAR.interface_Insar import *
from INSAR.INSAR_ANALYSIS import * 
import directory_manager
from source_data.GCPdata import GCP_HANDLER

class WEB_APP:
    bDebug = True
    f_main = 'main.html'
    f_results = 'dam_break.html'
    f_preresults = 'preresults.html'
    damList = []
    simList = []
    damID = ''
    simID = ''
    dataRecord = {}
    simRecord = {}
    mode = 'dam_break'
    plotSelection1 = 'speed'
    plotSelection2 = 'speed'
    plotSelections = ['Speed','Altitude']
    colorBarColors = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]

    ## Analysis mode data
    modeData = {}
    modeData['name'] = {'dam_break': 'Dam Break', 'insar': 'INSAR'}
    modeData['table'] = {'dam_break': 'Flooding_Model_Description', 'insar': 'INSAR'}
    modeData['main_page'] = {'dam_break': 'preresults.html', 'insar': 'main.html'}
    modeData['results_page'] = {'dam_break': 'preresults.html', 'insar': 'results_insar.html'}
    modeData['display_field'] = {'dam_break': 'ID', 'insar': 'Scene'}
    #modeData['output_list'] = {'dam_break': ['Speed','Altitude','Energy'], 'insar': ['Vert','Corr']}
    modeData['plot_list'] = {'dam_break': ['Speed','Altitude'], 'insar': ['kde','other']}

    ## Exclusions and amendments when displaying database records
    dataFrameRename = {}
    dataFrameRename['ID'] = ''
    dataFrameRename['Code_Version'] = ''
    dataFrameRename['Comments'] = ''
    dataFrameRename['Particle_Mass_Dist'] = ''
    dataFrameRename['Particle_Radius_Dist'] = ''
    dataFrameRename['Path'] = ''
    dataFrameRename['Owner'] = ''
    dataFrameRename['File_Address'] = ''
    dataFrameRename['Parent_ID'] = ''
    dataFrameRename['Analysis_ID'] = ''
    dataFrameRename['Tree_Level'] = ''
    dataFrameRename['Output_Summary'] = ''
    dataFrameRename['Max_Distance'] = ''
    dataFrameRename['Max_Velocity'] = ''
    dataFrameRename['Total_Energy'] = ''
    dataFrameRename['Flooding_Area'] = ''
    dataFrameRename['Repeat'] = ''
    

    def __init__(self):
        self.modeData['output_list'] = {'dam_break': self.get_sims_for_set_dambreak, 'insar': lambda *args : ['Vert','Corr']}
        self.modeData['analysis_handle'] = {'dam_break': self.set_dambreak_record,'insar': self.generate_insar_analyses}
        self.modeData['col_plotter'] = {'dam_break': self.generate_col_dambreak, 'insar': self.generate_col_insar}
        self.modeData['sim_lister']= {'dam_break': self.get_all_sets_dambreak, 'insar': self.get_all_sims_insar}

        self.set_mode(self.mode)


    def set_mode(self,modeStr):
        '''
        Switches app mode between dam_break and insar
        '''
        self.mode = modeStr
        self.f_results = self.modeData['results_page'][self.mode]
        self.damList = dbf.get_all_dams(self.get_db_table())
        self.plotSelections = self.get_plot_list()
        self.generate_col = self.modeData['col_plotter'][self.mode]

    def get_all_sets_dambreak(self,damID):
        '''
        Retrieves all database records for the Flooding_Model_Description table
        '''
        self._simSetList = dbf.query_by_dam(damID,'Flooding_Model_Description')
        return self._simSetList

    def get_sims_for_set_dambreak(self,*args):
        '''
        Retrieves all simulations belonging to the specified set
        '''
        self._all_dambreak_analyses = dbf.query_by_analysis(self.simID)
        return [r['ID'] for r in self._all_dambreak_analyses]

    #def get_all_sims_dambreak(self,damID):
    #    self._all_dambreak_analyses = dbf.query_all_analyses(damID)
    #    return self._all_dambreak_analyses
    
    def get_all_sims_insar(self,damID):
        return dbf.query_by_dam(damID,self.get_db_table())

    def get_db_table(self):
        '''
        Returns the database table corresponding to the selected analysis
        '''
        return self.modeData['table'][self.mode]

    def get_main_page(self):
        '''
        Returns the main html page file name
        '''
        return self.modeData['main_page'][self.mode]

    def get_results_page(self):
        '''
        Returns the results html page file name
        '''
        return self.modeData['results_page'][self.mode]

    def get_output_list(self,*args):
        '''
        Retrieves the output list for the drop down menu
        '''
        return self.modeData['output_list'][self.mode](*args)

    def get_plot_list(self):
        '''
        Retrieves the plot list for the plot drop down menus
        '''
        return self.modeData['plot_list'][self.mode]

    def set_dambreak_record(self,*args):
        '''
        Sets the current database (Floodingh_Model_Description) record in dambreak mode.
        '''
        simSelect = args[0]
        print('Generating dambreak analysis for sim %s' % simSelect)
        self.simID = simSelect
        self.simRecord = [i for i in self._simSetList if i['ID'] == self.simID][0]
        
    def generate_dambreak_analyses(self,*args):
        '''
        Generate html map and plots for dambreak analysis
        '''
        outputSelect = args[0]
        self.outputRecord = dbf.query_by_ID(outputSelect,'Analysis_Results')
        self.GCPH = GCP_HANDLER(self.outputRecord)
        #self.GCPH.gen_blob()
        posx, imgs, histData = self.GCPH.load_map_images()
        self.generate_map('generated_map.html', imgs, posx)

    def generate_insar_analyses(self,*args):
        '''
        Generate html map and plots for INSAR analysis
        '''
        if args:
            scene,variable = args[0:2]
        else:
            print('Error generating analyses')

        self.simRecord = dbf.query_result(self.get_db_table(),'Scene=\'%s\' AND Dam_ID=\'%s\' AND Variable=\'%s\'' % (scene,self.damID,variable))[0]
        self.GCPH = GCP_HANDLER(self.simRecord)
        #self.GCPH.gen_blob()
        prefix = get_insar_prefix(self.simRecord['Path'])
        names, blobs = self.GCPH.list_files_with_matching(prefix)
        CSVs = self.GCPH.load_csv()
        insar_files, contours, positions = generate_insar_images(
            self.damID, names, CSVs)

        # Format position data as a dictionary corresponding to each image
        posDict = {}
        i = 0
        for k in list(contours.keys()):
            posDict[k] = positions[i]
            i += 1

        self.generate_map('generated_map.html', contours, posDict)

    def generate_map(self, fname, images, position):
        '''
        Generates a HTML format map to be embedded in the results page
        '''
        #try:
        MAP = Interactive_map(self.damID)
        c=0
        colorData = {}
        for K in list(position.keys()):
            if len(position[K])>4:
                colorData[K] = position[K][4:8]
            else:
                colorData[K] = None

            # Skip if no matching image
            if not K in list(images.keys()):
                continue

            if colorData[K]:
                unit = colorData[K][3].replace('\'','').replace(' ','')
                layerColorMap = self.generate_colorbar(colorData[K][0],colorData[K][1],K,unit)
            else:
                layerColorMap = None
                
            MAP.add_png_layer(images[K], float(position[K][0]), float(
                position[K][1]), float(position[K][2]), float(position[K][3]), Label=K, colorMap=layerColorMap)
            c+=1
        
        # Finalise map
        MAP.finalise()
        mapDir = str(staticFolder.joinpath(fname))
        # Save map to directory
        MAP.Map.save(mapDir)

    def generate_histograms(self,nBins=30):
        '''
        Creates a histogram image from the selected set of dam break simulations
        '''
        quantities = ['Flooding_Area','Max_Distance','Max_Velocity','Total_Energy']
        units = ['m^2','m','m/s','MJ/m^2']
        dambreakStat = DAMBREAK_STAT(self.simRecord)

        for i,quantity in enumerate(quantities):
            fileDest = staticFolder.joinpath('%s.png' % quantity)
            xlabel = '%s (%s)' % (quantity, units[i])
            ylabel = 'Frequency'

            # Histogram of the given quantity
            histData,binEdges = dambreakStat.hist_quantity(quantity,nBins=nBins)
            binWidth = abs(binEdges[1]-binEdges[0])

            # Find reference point of currently selected simulation
            refData = self.outputRecord[quantity]

            # Plot and save
            #fig = plt.figure()
            plt.bar(binEdges[:-1], histData, width = binWidth)
            plt.xlim(min(binEdges), max(binEdges))
            plt.axvline(x=refData,color='r')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(fileDest,transparent=True,bbox_inches='tight',pad_inches=0.0)
            plt.close()

    def generate_colorbar(self,minOutput,maxOutput,outputName,units='-'):
        '''
        Returns a color bar object to the interactive map
        '''
        colors = self.colorBarColors
        index = np.linspace(minOutput,maxOutput,len(colors))
        caption = '%s (%s)' % (outputName,units)
        colormap = LinearColormapVisible(colors=colors,
                            index=index, vmin=minOutput, vmax=maxOutput,
                            caption=caption)

        self.generate_histograms(nBins=30)

        return colormap

    def generate_col_dambreak(self,*args):
        '''
        Generate a plot for one of the page columns
        '''
        pName,cName = args[0:2]
        pName = pName.lower()
        #try:
        # if not isinstance(self.simRecord, dict):
        #     print('RECORD NOT DICT')
        #     record = self.simRecord
        # else:
        record = self.simRecord
        folder = os.path.splitext(record['File_Address'])[0]
        folder = folder.replace('Analysis_Results','Analysis_Images')
        fsrc = folder + '/' + ('%s_time.png' % pName)
        plotData = self.GCPH.load_image(fsrc)

        fdest = staticFolder.joinpath(cName)
        #copy(fsrc,fdest)
        copyfileobj(plotData,fdest)
        #except:
        #    print('Unable to generate requested column, using previous plot')

    def generate_col_insar(self,*args):
        pName,cName = args[0:2]
        print('ARGS: ', args)
        fsrc = self.insarFront.push_plot_front(self.insarFile,self.simRecord['Scene'],pName,str(staticFolder.joinpath(cName)))
        #fdest = staticFolder.joinpath(cName)
        #print(fsrc)
        #print(fdest)
        #copy(fsrc,fdest)

    def dataFrameFromRecord(self,rec):
        '''
        Formats a dataframe from a dictionary. Removes unwanted fields, renames others etc
        '''
        rec = rec.copy()
        for k in self.dataFrameRename.keys():
            try:
                val = rec[k]
                newKey = self.dataFrameRename[k]
                rec.pop(k)
                if newKey!='':
                    rec[newKey] = val
            except:
                pass
        
        return pd.DataFrame(rec.items(),columns=('Parameter','Value'))

    def generate_record_table_html(self, *record):
        '''
        Generates a database record table in html format
        '''
        if not record:
            record = webObj.simRecord
        else:
            record = record[0]
        df = self.dataFrameFromRecord(record)
        return df.to_html()

    def generate_record_table(self, col_width=4.2, row_height=0.625, font_size=18,
                        header_color='#1abc9c', row_colors=['#f1f1f2', 'w'], edge_color='w',
                        bbox=[0, 0, 1, 1], header_columns=0,
                        ax=None, **kwargs):
        '''
        Generates a png table from the current simRecord
        DEPRECATED
        '''
        data = self.dataFrameFromRecord(self.simRecord)

        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

        #prepare for saving:
        # draw canvas once
        plt.gcf().canvas.draw()
        # get bounding box of table
        points = mpl_table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
        # add 10 pixel spacing
        points[0,:] -= 10; points[1,:] += 10
        # get new bounding box in inches
        nbbox = matplotlib.transforms.Bbox.from_extents(points/plt.gcf().dpi)
        plt.savefig(staticFolder.joinpath('record_table.png'), bbox_inches=nbbox)
        return ax

app = Flask(__name__)

### Entry point for the application ###
@app.route("/")
def main():
    return render_template(webObj.f_main, dams = [])

@app.route("/index/", methods=["GET", "POST"])
def onClick_index():
    print('-------INDEX--------')
    return render_template('index.html')

@app.route("/about/", methods=["GET", "POST"])
def onClick_about():
    print('-------ABOUT-------')
    return render_template('about.html')

## Switch between analysis modes, e.g. dam break and INSAR
@app.route("/mode_select_dambreak", methods=["GET", "POST"])
def onClick_select_dambreak():
    webObj.set_mode('dam_break')
    outputList = []
    columnList = webObj.get_plot_list()
    return render_template(webObj.f_preresults, dams = webObj.damList, outputList = outputList, columnList = columnList)


@app.route("/mode_select_insar", methods=["GET", "POST"])
def onClick_select_insar():
    webObj.set_mode('insar')
    outputList = webObj.get_output_list()
    columnList = webObj.get_plot_list()
    return render_template(webObj.f_main, dams = webObj.damList, outputList = outputList, columnList = columnList)

#def onClick_select_mode(form):
#    webObj.set_mode(webObj.mode)
#    outputList = webObj.get_output_list()
#    columnList = webObj.get_plot_list()
#    return render_template(webObj.f_main, dams = webObj.damList, outputList = outputList, columnList = columnList)


@app.route("/dam_select/", methods=["GET", "POST"])
def onClick_dam():
    '''
    Dam selection click event
    '''
    form = request.form
    webObj.damID = form['damList']
    print('Dam ID: ', form['damList'])

    # Find analysis entry for the selected dam
    table = webObj.get_db_table()
    primaryKey = dbf.get_primary_key(table)
    displayField = webObj.modeData['display_field'][webObj.mode]
    webObj.simRecords = webObj.modeData['sim_lister'][webObj.mode](webObj.damID)
    webObj.simList = []
    for r in webObj.simRecords:
        webObj.simList.append(r[displayField])
    webObj.simList.sort() #sort into alphabetical order

    outputList = webObj.get_output_list()
    return render_template(webObj.get_main_page(),dams = webObj.damList, sims = webObj.simList, outputList = outputList, columnList=webObj.get_plot_list(), form=request.form)

@app.route("/analysis_select/", methods=["GET", "POST"])
def onClick_analysis():
    form = request.form
    simSelect = form['simList']

    try:
        maskSelect = form['plot_select'].lower()
    except:
        maskSelect = 'speed'

    columnList=webObj.get_plot_list()
    webObj.plotSelection1 = columnList[0]
    webObj.plotSelection2 = columnList[1]

    funcHandle = webObj.modeData['analysis_handle'][webObj.mode]
    funcHandle(simSelect,maskSelect)

    ## Generate table from database record
    #webObj.generate_record_table()
    tableHtml = webObj.generate_record_table_html()

    print("Simulation selected: ", webObj.simRecord)#[primaryKey])

    return render_template(webObj.get_results_page(),dams = webObj.damList, sims = webObj.simList, tableHtml = tableHtml, outputList=webObj.get_output_list(), columnList=webObj.get_plot_list(), form=request.form)

@app.route("/sim_select/", methods=["GET","POST"])
def onClick_sim():
    '''
    Used by dam break to select individual simulations from a set
    '''
    resultsPage = 'results.html'
    simSelect = request.form['outputList']
    webObj.generate_dambreak_analyses(simSelect)
    tableHtml = webObj.generate_record_table_html(webObj.simRecord)
    tableHtml2 = webObj.generate_record_table_html(webObj.outputRecord)
    return render_template(resultsPage,dams = webObj.damList, sims = webObj.simList, tableHtml = tableHtml, tableHtml2 = tableHtml2, outputList=webObj.get_output_list(), columnList=webObj.get_plot_list(), form=request.form)

@app.route("/col1/", methods=["GET", "POST"])
def onClick_column1():
    form = request.form
    print('onClick_column1')
    print(form)
    webObj.plotSelection1 = form['dropdown_col1']
    webObj.generate_col(webObj.plotSelection1,'col1.png')
    tableHtml = webObj.generate_record_table_html()
    return render_template(webObj.get_results_page(),dams = webObj.damList, sims = webObj.simList, tableHtml = tableHtml, form=request.form, outputList=webObj.get_output_list(), columnList=webObj.get_plot_list())

@app.route("/col2/", methods=["GET", "POST"])
def onClick_column2():
    form = request.form
    print('onClick_column2')
    print(form)
    webObj.plotSelection2 = form['dropdown_col2']
    webObj.generate_col(webObj.plotSelection2,'col2.png')
    tableHtml = webObj.generate_record_table_html()
    return render_template(webObj.get_results_page(),dams = webObj.damList, sims = webObj.simList, tableHtml = tableHtml, form=request.form, outputList=webObj.get_output_list(), columnList=webObj.get_plot_list())

@app.route("/hist_example/", methods=["GET", "POST"])
def onClick_histogram():
    '''
    Example use for generate_histogram() method. (NOTE: not assigned to a button in the app)
    '''
    webObj.generate_histograms(nBins=30)


### Run app if this script is run directly ###
if __name__ == "__main__":

    webObj = WEB_APP()
    app.debug = webObj.bDebug
    if len(sys.argv)<2:
        hostIP = ""
    else:
        hostIP = sys.argv[1]
    
    if hostIP=="":
        app.run()
    else:
        app.run(host=hostIP)
                
                
