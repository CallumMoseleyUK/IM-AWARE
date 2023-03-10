import sys
import numpy as np
import multiprocessing as mp
import os
import subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime
from time import perf_counter, sleep
import DBdriver.DBFunctions as dbf
import DBdriver.Field_Extractor as fe
from dam_break.dambreak_lib import DAMBREAK_SIM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from source_data.GCPdata import *
from dam_break.dambreak_lib import DAM_BREAK

def gen_simulation_ID(dam_id, simtype='DAMBREAK'):

    time_stamp = datetime.now()
    print('Generating simulation file ID')
    print('\tDate : {}/{}/{}'.format(time_stamp.day,
                                     time_stamp.month, time_stamp.year))
    print('\tTime Stamp : {}'.format(time_stamp.time()))
    fileID = '{}-{}-{}'.format(dam_id, simtype,
                               time_stamp.strftime("%Y%m%d-%H%M%S"))

    print(70*'-')
    print('\t FileID = {}'.format(fileID))
    print(70*'-')

    return fileID

def check_record_validity(record,table):
    '''
    Tests the validity of a database record against a given table
        - Returns the number of valid entries 
    '''
    tableFields = dbf.get_fields(table)
    validEntries = len([i for i in list(record.keys()) if i in tableFields])

    if validEntries < len(tableFields):
        print("WARNING: record for table %s is incomplete" % table)
    return validEntries

def update_database(record,table):
    '''
    Updates the given table with the given record in dictionary format
    '''

    rValid = True #check_record_validity(record,table)
    e = ""
    if rValid:
        try:
            keyFields = ["ID"]
            keyValues = [record["ID"]]
            result = dbf.insert_into_DB(table,keyFields,keyValues,record)
        except:
            e = sys.exc_info()[0]
            print("update_database exception: ", e)
            with open("log_update_database.txt","w") as logf:
                logf.write(str(e))
    
    return (result,e)
def get_git_hash():
    '''
    Reads the current Git revision hash
    '''
    #return subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode()
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode()

## Generates histogram data for a 2d np array
def hist_from_map(data,gcpHandler,saveDir,nBins=30):
    data = data.copy()
    data = data.reshape(-1)
    data = data[np.where(np.logical_not(np.isnan(data)))]

    hist, binEdges = np.histogram(data,bins=nBins)

    dataStr = str(list(hist)) + '\n' + str(list(binEdges))

    gcpHandler.save_text(dataStr,saveDir)
    return hist, binEdges


## Function to generate image files from a database record
def generate_image(simRecord,damID,maxTime):
    dateTime = datetime.now().strftime("[%m/%d/%Y, %H:%M:%S]: ")
    dbObj = None
    resolution=1
    skipPoints=1
    bFullRange=True
    simID = simRecord["ID"]

    WarehouseFolder = Path("IMAWARE")
    AnalysisFolder = WarehouseFolder.joinpath("Analysis_Results")
    ImageFolder = WarehouseFolder.joinpath("Analysis_Images")
    gcpHandler = GCP_IO()
    simRecord["File_Handler"] = gcpHandler

    print('Pre-rendering for %s' % simID)

    # Define sim path and create if needed
    imFolder = ImageFolder.joinpath(damID,simID)
    # Map files
    dataFile = str(imFolder.joinpath("position.dat"))
    energyFile = str(imFolder.joinpath("energy.png"))
    speedFile = str(imFolder.joinpath("speed.png"))
    altFile = str(imFolder.joinpath("altitude.png"))
    depthFile = str(imFolder.joinpath("depth.png"))
    bMapCriterion = not (gcpHandler.file_exists(dataFile) and
                        gcpHandler.file_exists(energyFile) and 
                        gcpHandler.file_exists(speedFile) and 
                        gcpHandler.file_exists(altFile) and
                        gcpHandler.file_exists(depthFile))

    # Plot files
    energyTimeFile = str(imFolder.joinpath("energy_time.png"))
    speedTimeFile = str(imFolder.joinpath("speed_time.png"))
    altTimeFile = str(imFolder.joinpath("altitude_time.png"))

    bPlotCriterion = not (gcpHandler.file_exists(energyTimeFile) and 
                        gcpHandler.file_exists(speedTimeFile) and 
                        gcpHandler.file_exists(altTimeFile))

    ## Load dambreak file only if there are incomplete files
    if bMapCriterion or bPlotCriterion:
        try:
            dbObj = DAMBREAK_SIM(simRecord)
            #if no file is found, log the exception
        except Exception as dbObjExcept:
            print(dbObjExcept)
            dateTime = datetime.now().strftime("[%m/%d/%Y, %H:%M:%S]: ")
            logName = 'logs/prerenders.txt'
            os.makedirs(os.path.dirname(logName),exist_ok=True)
            with open(logName,'a') as logFile:
                logFile.write('{} : {} ERROR: {}\n'.format(dateTime,simID,str(dbObjExcept)))

            simRecord["File_Handler"] = None
            return simRecord
            
    if bMapCriterion:
        # do analysis, save images to Analysis_Images/Speed/damID etc
        # save image bounds [lon0,lat0,lon1,lat1] to same folder
        mask,maskX,maskY,vxMask,vyMask,vzMask,speedMask,altMask,eMask,depthMask = dbObj.fit_all_masks(maxTime,resolution,bFullRange,skipPoints)
        mask,maskX,maskY = dbObj.fit_mask(maxTime,resolution,bFullRange,skipPoints)
        ######speedMask,maskX.maskY = dbObj.fit_speed_mask()
        # Put unit conversions here
        eMask = eMask/1e6

        # Calculate final inundated area
        cellArea = dbObj.get_cell_area() * resolution
        floodArea = np.nansum(mask) * cellArea

        # Calculate maximum distance travelled
        _,maxDist = dbObj.most_distant_particle()

        #  Set map bounds and min/max data
        bounds = minLon,maxLon,minLat,maxLat = (np.min(maskX),np.max(maskX),np.min(maskY),np.max(maskY))
        colourScales = (np.nanmin(eMask),np.nanmax(eMask),np.nanmin(speedMask),np.nanmax(speedMask),np.nanmin(altMask),np.nanmax(altMask),np.nanmin(depthMask),np.nanmax(depthMask))
        minEnergy,maxEnergy,minSpeed,maxSpeed,minAltitude,maxAltitude,minDepth,maxDepth = colourScales

        print("Saving maps for %s" % simID)
        # Save map plots
        formatFunc = lambda x : str(x).replace('(','').replace(')','')
        dataText = formatFunc(bounds+colourScales)
        gcpHandler.save_text(dataText,dataFile)

        # Save position/boundary data for each map
        dataEnergy = formatFunc((minLon,maxLon,minLat,maxLat,minEnergy,maxEnergy,np.nanmean(eMask),'MJ/m^2'))
        dataSpeed = formatFunc((minLon,maxLon,minLat,maxLat,minSpeed,maxSpeed,np.nanmean(speedMask),'m/s'))
        dataAlt = formatFunc((minLon,maxLon,minLat,maxLat,np.nanmin(altMask),np.nanmax(altMask),np.nanmean(altMask),'m'))
        dataDepth = formatFunc((minLon,maxLon,minLat,maxLat,np.nanmin(depthMask),np.nanmax(depthMask),np.nanmean(depthMask),'m'))
        gcpHandler.save_text(dataEnergy,str(imFolder.joinpath('energy.dat')))
        gcpHandler.save_text(dataSpeed,str(imFolder.joinpath('speed.dat')))
        gcpHandler.save_text(dataAlt,str(imFolder.joinpath('altitude.dat')))
        gcpHandler.save_text(dataDepth,str(imFolder.joinpath('depth.dat')))

        # Save histograms
        nBins = 30
        hist_from_map(eMask,gcpHandler,nBins=nBins,saveDir=str(imFolder.joinpath('energy.hist')))
        hist_from_map(speedMask,gcpHandler,nBins=nBins,saveDir=str(imFolder.joinpath('speed.hist')))
        hist_from_map(altMask,gcpHandler,nBins=nBins,saveDir=str(imFolder.joinpath('altitude.hist')))
        hist_from_map(depthMask,gcpHandler,nBins=nBins,saveDir=str(imFolder.joinpath('depth.hist')))
        
        # Save additional database data
        updatedFields = {'Max_Distance': maxDist,
                        'Max_Velocity': np.nanmax(speedMask),
                        'Total_Energy': np.nanmax(eMask),
                        'Flooding_Area': floodArea}
        dbf.update_record('Analysis_Results',simID,updatedFields)

        # Save maps
        dbObj.save_mask(energyFile,eMask,maskX,maskY)
        dbObj.save_mask(speedFile,speedMask,maskX,maskY)
        dbObj.save_mask(altFile,altMask,maskX,maskY)
        dbObj.save_mask(depthFile,depthMask,maskX,maskY)
        print("Saved maps for %s" % simID)
    else:
        print('Skipping map renders')

    print("Saving plots for %s" % simID)
    
    # Time series plots
    if bPlotCriterion:
        time = dbObj.simTime
        if not energyTimeFile.exists():
            energyTime = dbObj.get_total_energy(time)
            dbObj.save_plot(energyTimeFile,time,energyTime,'Time (s)', 'Kinetic Energy (J)')
        if not speedTimeFile.exists():
            speedTime = dbObj.get_mean_speed(time)
            dbObj.save_plot(speedTimeFile,time,speedTime,'Time (s)','Speed (m/s)')
        if not altTimeFile.exists():
            altTime = dbObj.get_mean_altitude(time)
            dbObj.save_plot(altTimeFile,time,altTime,'Time (s)', 'Altitude (m)')
        #if not areaTimeFile.exists():
        #    areaTime = dbObj.get_flood_area(time,resolution=5,bFullRange=True,skipPoints=1) # set to low resolution as an approximation? Skipping the full range will make a huge difference
        #    dbObj.save_plot(str(areaTimeFile),time,areaTime,'Time (s)', 'Flood area (m^2)')
        print("Saved plots for %s" % simID)

    else:
        print('Skipping plot renders')
    ## Log progress
    dateTime = datetime.now().strftime("[%m/%d/%Y, %H:%M:%S]: ")
    logName = 'logs/prerenders.txt'
    os.makedirs(os.path.dirname(logName),exist_ok=True)
    with open(logName,'a') as logFile:
        logFile.write('{} : {} Maps: {}, Plots: {}\n'.format(dateTime,simID,bMapCriterion,bPlotCriterion))

    # Returning an unserialisable class instance will break multiprocessing
    simRecord["File_Handler"] = None
    return simRecord

def run_sim(data):
    '''
    Wrapper function for running the dam break simulation Matlab code
        Inputs:
        dam_data - dam data structure from ANM data
        simID - unique string ID
        lat - latitude (deg)
        lon - longitude (deg)
        den - density (kg/m^3)
        vol - volume (m^3)
        pondR - pond radius (m)
        tMax - time to run simulation (s)
        nEle - number of elements to simulate
        cVisc - viscous damping coefficient
    '''
    [modelID,simID,damID,lat,lon,pondR,nEle,vol,den,tMax,cVisc,latOff,lonOff,volFactor] = data
    timeStep = 0.2

    #simID = "%s_%i" % (gen_simulation_ID(damID),j)
    print("Running simulation: " ,simID)

    #logFileName = "dam_break/logs/sim_log_%s.txt" % gen_simulation_ID("")
    logFileName = "dam_break/logs/sim_log_%s.txt" % modelID.split("DAMBREAK")[1]
    os.makedirs(os.path.dirname(logFileName), exist_ok=True)
    logfile = open(logFileName,'a')
    logfile.write("Starting simulation: %s \n" % simID)
    logfile.close()

    ## Apply volume factor
    vol = volFactor*vol

    ## Folder to save results (relative to im_aware_collab)
    resultDir = "IMAWARE/Analysis_Results/%s" % damID
    #resultDir = "IMAWARE/Analysis_Results/%s" % simID

    ## Start simulation
    startTime = perf_counter()
    inputString = "{\'%s\',%.15f,%.15f,%.15f,%i,%.15f,%.15f,%.15f,%.15f,\'%s\'}" %\
        (simID,float(lon),float(lat),float(pondR),int(nEle),float(vol),float(den),float(tMax),float(cVisc),str(resultDir))

    ## Run simulation
    dbSim = DAM_BREAK(lat,lon,pondR,nEle,vol,den,tMax,timeStep,cVisc)
    dbSim.run_simulation()
    fileName,csvName = dbSim.save_results(damID,simID)

    # simOut: [Particle_Number,Particle_Mass,Particle_Radius,Max_Simulated_Time,Damping,simID]
    #simOut = eng.main_func([simID,float(lon),float(lat),float(pondR),int(nEle),float(vol),float(den),float(tMax),float(cVisc)])

    # Extract parameters from simOut
    #[Particle_Number,Particle_Mass,Particle_Radius,Max_Simulated_Time,Damping,simID,File_Address] = simOut
    Particle_Number = dbSim.nObj
    Particle_Mass = dbSim.get_particle_mass()
    Particle_Radius = dbSim.get_particle_radius()
    Max_Simulated_Time = dbSim.maxTime
    Damping = dbSim.dampingCoeff
    File_Address = csvName


    # Update simID with unique ID generated in Matlab (appends '_i' from 0-Inf if conflicts)
    evaluationTime = perf_counter() - startTime
    
    # Update Analysis_Results
    record = {"Particle_Number": int(Particle_Number),\
        "Particle_Mass": float(Particle_Mass),\
        "Particle_Radius": float(Particle_Radius),\
        "Damping": cVisc,\
        "Volume_Factor": volFactor,\
        "Latitude_Offset": latOff,\
        "Longitude_Offset": lonOff,\
        "Tailings_Density": den,\
        "Max_Distance": -1,\
        "Max_Velocity": -1,\
        "Total_Energy": -1,\
        "Flooding_Area": -1,\
        "Analysis_ID": modelID,\
        "Evaluation_Time": evaluationTime,\
        "Type_of_Analysis": "DAMBREAK",\
        "Parent_ID": modelID,\
        "Tree_Level": 0,\
        "Repeat": 0,\
        "File_Address": File_Address,\
        "Output_Summary": '',\
        "ID": simID}
    update_database(record,"Analysis_Results")

    ## Generate pre-render images
    generate_image(record,damID,Max_Simulated_Time)

    # Update log
    with open(logFileName,"a") as logfile:
        logfile.write("Simulation finished: %s\n" % simID)
        logfile.write("     Evaluation time: %s seconds\n" % evaluationTime)
        logfile.write("     Record:\n     %s\n" % str(record))
        logfile.write("     Git hash: %s\n" % get_git_hash())
    
    return record


def parse_distribution(strInput):
    '''
    Parses a string input as a tuple distribution
    e.g '[interval,0,1,10]' >> ('interval',0,1,10) 
    '''
    strInput = strInput.replace("(","")
    strInput = strInput.replace(")","")
    strInput = strInput.replace(" ","")
    args = strInput.split(",")
    distr = [args[0]]
    for i in args[1:]:
        distr.append(float(i))
    
    return tuple(distr)

def evaluate_distribution(distr):
    '''
    Evaluates a tuple format parameter distribution
    e.g. ('interval',0,1,10) >> np.linspace(0,1,10)
    '''
    # Move this somewhere more general?
    funcDict = {"constant": lambda x : x[0],\
                "rand_uniform": lambda x : np.random.uniform(x[0],x[1]),\
                "rand_norm": lambda x : np.random.normal(x[0],x[1])}
    
    return funcDict[distr[0]](distr[1:])

'''
Input: name of file containing model arguments
'''
if __name__ == '__main__':
    ## Collect and validate inputs
    try:
        inputFile = sys.argv[1]
    except:
        print('Must provide input file')

    with open(inputFile,'rb') as file:
        inputData = file.read().decode(errors='replace').strip().replace('\r','')
    inputDict = {}
    for inputStr in inputData.split('\n'):
        k,v = inputStr.replace(' ','').split('=')
        inputDict[k] = v
    
    ## Model inputs
    username = os.getlogin()
    simNum = int(inputDict['simNum'])
    tMax = float(inputDict['tMax'])
    Particle_Num_Dist = parse_distribution(str(inputDict['Particle_Num_Dist']))
    Damping_Dist = parse_distribution(str(inputDict['Damping_Dist']))
    Latitude_Offset_Dist = parse_distribution(str(inputDict['Latitude_Offset_Dist']))
    Longitude_Offset_Dist = parse_distribution(str(inputDict['Longitude_Offset_Dist']))
    Volume_Factor_Dist = parse_distribution(str(inputDict['Volume_Factor_Dist']))
    Tailings_Density_Dist = parse_distribution(str(inputDict['Tailings_Density_Dist']))
    
    ## Generate simulation inputs and populate Analysis_Results table
    dirname = os.path.dirname(os.path.abspath("__file__"))
    filename = os.path.join(dirname, 'ANM_dam_data/processed_ANM 06-2021.xlsx')
    ANM = pd.read_excel(filename,engine="openpyxl")
    conditions = (ANM['Tailings_Material']=='iron') & (ANM['Height']>40) & (ANM['Building_Method']=='upstream')
    
    candidate_dams = ANM[conditions]
    candidate_dams = candidate_dams.reset_index(drop=True)

    funcData = []
    timeStamp = str(datetime.now())
    for i, dam_data in enumerate(candidate_dams.iloc):
        # Create Flooding_Model_Details record
        modelID = gen_simulation_ID(dam_data.ID)
        modelRecord = {"Dam_ID": dam_data.ID,\
            "Date_Time": timeStamp,\
            "Particle_Num_Dist": str(Particle_Num_Dist).replace("\'",""),\
            "Particle_Mass_Dist": " ",\
            "Particle_Radius_Dist": " ",\
            "Damping_Dist": str(Damping_Dist).replace("\'",""),\
            "Volume_Factor_Dist": str(Volume_Factor_Dist).replace("\'",""),\
            "Latitude_Offset_Dist": str(Latitude_Offset_Dist).replace("\'",""),\
            "Longitude_Offset_Dist": str(Longitude_Offset_Dist).replace("\'",""),\
            "Tailings_Density_Dist": str(Tailings_Density_Dist).replace("\'",""),\
            "Max_Simulated_Time": tMax,\
            "Owner": username,\
            "Comments": " ",\
            "Code_Version": get_git_hash(),\
            "Number_of_Children": 0,\
            "ID": modelID}

        for j in range(simNum):
            # Populate input set for run_sim()
            latOff = evaluate_distribution(Latitude_Offset_Dist)
            lonOff = evaluate_distribution(Longitude_Offset_Dist)
            volF = evaluate_distribution(Volume_Factor_Dist)
            damping = evaluate_distribution(Damping_Dist)
            density = evaluate_distribution(Tailings_Density_Dist)
            nEle = evaluate_distribution(Particle_Num_Dist)
            simID = "%s_%i" % (gen_simulation_ID(dam_data.ID),j)
            volume = dam_data.Stored_Volume
            # TODO: make a better estimate of pond radius, rather than scaling
            # - reference values: Feijao, volume=9.57e6 and pondR=300
            pondR = 300.0*(volF*float(dam_data.Stored_Volume)/9.57E6)**(1.0/3.0)
            # Append input list for one simulation run to funcData
            inputList = (modelID, simID, dam_data.ID,float(dam_data.Lat), float(dam_data.Long), float(pondR), int(nEle), float(volume), float(density), float(tMax), float(damping), float(latOff),float(lonOff),float(volF))
            funcData.append(inputList)

            # Update Analysis_Details record
            modelRecord["Number_of_Children"] += 1

        # Save Flooding_Model_Description record
        update_database(modelRecord,"Flooding_Model_Description")

    ## Run simulations
    
    t = perf_counter()
    nProcessors = mp.cpu_count()
    print("Starting parallel pool: %i workers" % nProcessors)
    pool = mp.Pool(nProcessors)
    print("Parallel pool started, starting simulations")
    results = pool.map(run_sim, funcData)
    print("All simulations finished")
    pool.close()
    print("Parallel pool closed")
    #print(results)
    t = perf_counter() - t
    print(t)
    
