import os
import numpy as np
from math import pi
from collections.abc import Iterable
import matplotlib.pyplot as plt
from pathlib import Path
import io

import directory_manager
import sys

from source_data.GCPdata import GCP_HANDLER
sys.path.append(directory_manager.get_code_dir())
try:
    from source_data.source_jaxa import jaxa
except:
    print('dambreak_lib.py could not load jaxa.py')

class DAMBREAK_SIM:
    '''
    Class to hold and query time histories of simulated particles from the dam break model.
        srcFileName - the source csv file containing N particle time histories.
        (Note that particle number N in the file name is approximate due to initialisation
        method used to populate the point of dam break, actual number may vary)
            Csv name format - 'data_siteIndex_N_simTime'
            Csv row format - [time,lat0,long0,vx0,vy0,vz0,...
                            lat1,long1,vx1,vy1,vz1,...
                            ...,
                            lat(N-1),long(N-1),vx(N-1),vy(N-1),vz(N-1),
                            latN,longN,vxN,vyN,vzN]
    '''
    #_bVerbose = False
    simRecord = {}
    simN = int(0) #number of particles in simulation
    simTime = np.array([]) # array of time values over the simulation (seconds)
    simLat = np.array([]) # latitude data (degrees)
    simLon = np.array([]) #longitude data (degrees)
    simAltitude = np.array([]) # altitude (m)
    simVx = np.array([]) #velocity x component (m/s)
    simVy = np.array([]) #velocity y component (m/s)
    simVz = np.array([]) #velocity z component (m/s)
    simStartLat = 0 #simulation starting point latitude (degrees)
    simStartLon = 0 #simulation starting point longitude (degrees)
    simResX = 111e3
    simResY = 111e3
    _earthRadius = 6371e3 #earth radius (m)
    _lonAdjust = 1.0 #latitude-dependent adjustment factor for longitudinal resolution
    
    # If any of these are None, they are re-fitted when mask fitting functions are called
    mask = np.array([])
    maskX = np.array([])
    maskY = np.array([])
    dMask = np.array([])
    vxMask = np.array([])
    vyMask = np.array([])
    vzMask = np.array([])
    speedMask = np.array([])
    altMask = np.array([])
    eMask = np.array([])
    depthMask = np.array([])
    # Likewise, if min/max time span specifiedd is different, masks are re-fitted.
    _maskMinTime = -1
    _maskMaxTime = -1

    demData = None
    fileHandler = None

    # number of states per particle
    CONST_STATES = 6

    def __init__(self,srcInput=None,bAbsolutePath=False,demDirectory=None):
        
        if isinstance(srcInput,dict):
            self.simRecord = srcInput
            srcFileName = self.simRecord["File_Address"]
            try:
                self.fileHandler = self.simRecord['File_Handler']
            except:
                #Default to Google Cloud file handler
                self._init_gcp_handler()
        else:
            srcFileName = srcInput
            self._init_gcp_handler()

        self.demDirectory = demDirectory

        #if srcFileName:
        if not bAbsolutePath:
            srcFileName = srcFileName.split('im_aware_collab/')[-1]
        self._load_innundation_data(srcFileName)

    def _init_gcp_handler(self):
        self.fileHandler = GCP_HANDLER(self.simRecord)

    def _load_innundation_data(self, srcFileName):
        """
        Used to load simulation data.
        """
        #print('self.gcpHandler',self.gcpHandler)
        #print('self.gcpHandler.bucket',self.gcpHandler.bucket)
        #print('srcFileName',srcFileName)
        try:
            self.data = self.fileHandler.load_csv_as_numpy(srcFileName)
        except OSError:
            print("Simulation file %s failed to load" % srcFileName)
            self.data = np.array([])
        else:
            data = self.data
            rows = data.shape[0]
            cols = data.shape[1]
            
            if (cols-1) % self.CONST_STATES == 0:
                self.simN = int((cols - 1)/self.CONST_STATES)
                self.simTime = data[:,0]
                
                self.simLat = data[:,1:-1:self.CONST_STATES]
                self.simLon = data[:,2:-1:self.CONST_STATES]
                self.simAltitude = data[:,3:-1:self.CONST_STATES]
                self.simVx = data[:,4:-1:self.CONST_STATES]
                self.simVy = data[:,5:-1:self.CONST_STATES]
                self.simVz = data[:,6::self.CONST_STATES]
                self.simStartLat = np.mean(self.simLat[0,:])
                self.simStartLon = np.mean(self.simLon[0,:])
                self._lonAdjust = np.cos(self.simStartLat*pi/180)
                self.simResY = self._earthRadius*(pi/180)
                self.simResX = self.simResY * self._lonAdjust
            else:
                self.data = np.array([])
                print("Simulation file %s has invalid format" % srcFileName)

    def _time_index(self,timeArray):
        '''
        Returns indices where the time series is equal to an input time array
            e.g. self._timeIndex(np.array([10,25,60])) will return indices where time = 10, 25 and 60 seconds
        '''
        # If empty array/list:
        if isinstance(timeArray,Iterable) and not len(timeArray):
            timeArray = self.simTime
        # If scalar:
        if not isinstance(timeArray,Iterable):
            timeArray = [timeArray]
        
        ind = np.zeros(self.simTime.shape)
        for j in timeArray:
            ind[np.argmin(np.abs(self.simTime-j))] = 1
        
        return np.where(ind)

    def get_lon_lat_bounds(self,maxTime=None):
        '''NOTE: Runs slowly if no mask fitting functions have been called for this instance.'''
        if maxTime == None:
            maxTime = self.max_time()
        if len(self.mask) == 0:
            self.mask,self.maskX,self.maskY = self.fit_mask(self,maxTime,resolution=5,bFullRange=True,skipPoints=1)
        return np.min(self.maskX),np.max(self.maskX),np.min(self.maskY),np.max(self.maskY)

    
    def num_particles(self):
        '''
        Returns number of particles
        '''
        return self.simN

    def max_time(self):
        return self.simTime[-1]
    
    def get_latitude(self,index,timeArray=[]):
        '''
        Returns the latitude history for the particle corresponding to index.
        '''
        tInd = self._time_index(timeArray)
        return np.squeeze(self.simLat[tInd,index])

    def get_longitude(self,index,timeArray=[]):
        '''
        Returns the longitude history for the particle corresponding to index.
        '''
        tInd = self._time_index(timeArray)
        return np.squeeze(self.simLon[tInd,index])

    def get_path(self,index,timeArray=[]):
        '''
        Returns the latitude and longitude histories for the particle corresponding to index
            Row 0: latitude. Row 1: longitude
            Optionally specify an array of specific times after simulation start to retrieve those values.
        '''
        return np.vstack((self.get_latitude(index,timeArray), self.get_longitude(index,timeArray)))

    def get_altitude(self,index,timeArray=[]):
        '''
        Returns the altitude history of a particle corresponding to index
        '''
        tInd = self._time_index(timeArray)
        return np.squeeze(self.simAltitude[tInd,index])

    def get_depth(self,index,timeArray=[]):
        '''
        Gets the depth of a particle (height above the terrain elevation)
        TODO: add a simDepth parameter to self, populate it for all particles the first time this method is called, making future calls faster. Or just do it in __init__.
        '''
        # Instantiate digital elevation model if not already done
        if not self.demData:
            self.demData = DEM_DATA(self.simStartLat,self.simStartLon,self.fileHandler,demDirectory=self.demDirectory)

        # Get get particle position
        zParticle = self.get_altitude(index,timeArray)
        lat = self.get_latitude(index,timeArray)
        lon = self.get_longitude(index,timeArray)
        radius = self.simRecord['Particle_Radius']

        # Get depth
        depth = []
        for i in range(len(zParticle)):
            depth += [np.clip(zParticle[i] - self.demData.get_elev(lat[i],lon[i]) + radius,0,None)]

        return depth

    def get_velocity(self,index,timeArray=[]):
        '''
        Returns the velocity for the particle corresponding to index
            Row 0: X component. Row 1: Y component. Row 2: Z component
            Optionally specify an array of specific times after simulation start to retrieve those values.
        '''
        #tInd = self._time_index(timeArray)
        velocityOut = np.vstack((self.get_velocity_x(index,timeArray), self.get_velocity_y(index,timeArray), self.get_velocity_z(index,timeArray)))
        return velocityOut

    def get_velocity_x(self,index,timeArray=[]):
        tInd = self._time_index(timeArray)
        return np.squeeze(self.simVx[tInd,index])
    def get_velocity_y(self,index,timeArray=[]):
        tInd = self._time_index(timeArray)
        return np.squeeze(self.simVy[tInd,index])
    def get_velocity_z(self,index,timeArray=[]):
        tInd = self._time_index(timeArray)
        return np.squeeze(self.simVz[tInd,index])
    

    def get_velocity2D(self,index,timeArray=[]):
        '''
        Returns the 2D velocity for the particle corresponding to index
            Row 0: X component. Row 1: Y component
            Optionally specify an array of specific times after simulation start to retrieve those values.
        '''
        V = self.get_velocity(index,timeArray)
        return V[:,:-1]

    def get_speed(self,index,timeArray=[]):
        '''
        Returns speed of the paricle for a corresponding index
        '''
        V = self.get_velocity(index, timeArray)
        S = np.sqrt(V[0]**2+V[1]**2+V[2]**2)
        return S 

    def get_speed_bounds(self,timeArray=[]):
        '''
        Returns Bounds on speed of all particles
        '''
        S = []
        for i in range(self.num_particles()):
            S.append(self.get_speed(i, timeArray))
        S = np.concatenate(S)
        return [min(S), max(S)]

    def get_energy(self,index,timeArray=[]):
        '''
        Returns the kinetic energy of the particle for a corresponding index
        '''
        E = 0.5 * self.simRecord['Particle_Mass'] * self.get_speed(index,timeArray)**2
        return E

    def get_distance(self,index,timeArray=[]):
        '''
        Returns distance travelled by a particle given by index from the simulation starting point
        '''
        geo = self.get_path(index,timeArray)
        geo0 = np.array([self.simStartLat,self.simStartLon])
        dgeo = geo - np.tile(geo0,(geo.shape[1],1)).T
        dpos = self._geo2pos(dgeo[0,:],dgeo[1,:])
        return np.sqrt(np.sum(dpos**2,axis=0))

    def get_max_distance(self,index,timeArray=[]):
        '''
        Calculates the maximum distance travelled by a particle from the starting point at its position
        '''
        dist = self.get_distance(index,timeArray)
        maxDist = np.max(dist)
        #maxTime = timeArray[dist==maxDist]
        #if multiple maxima are found, take the earliest
        #if isinstance(maxTime,Iterable):
        #    maxTime = maxTime[0]
        #maxLat,maxLon = self.get_path(index,maxTime)
        return maxDist#,maxLat,maxLon,maxTime

    def most_distant_particle(self):
        '''
        Returns the index of the most distant particle at specified timeStep
        '''
        iMax = -1
        maxDist = -1.0
        for i in np.arange(self.num_particles()):
            dist = self.get_max_distance(i)
            if dist>maxDist:
                #latMax,lonMax = self.getPath(i,timeStep)
                iMax = i
                maxDist = dist
        return iMax,maxDist

    def get_cell_area(self):
        '''
        Returns the area of a single map cell
        '''
        return self.simResX*self.simResY/(3600**2)

    def fit_all_masks(self,maxTime,resolution=5,bFullRange=True,skipPoints=1):
        '''
        Fits a mask covering the inundation area and returns average velocities for each cell. Each value indicates the number of particles within a cell of the mask.
            maxTime - maximum simulation time (maxTime=self.time[-1] for full simulation)
            resolution - mask resolution (resolution=1 means same resolution as DEM, 1/3600 degrees)
            bFullRange - if true, all points in time series are considered, else only the points at which each
                particle's distance from the dam is maximum.
            skipPoints - number of time steps to skip (only has an effect when bFullRange=True)
        NOTE: Map fitting can be sped up by restating the calculation as a matrix operation.
                If this happens, split this function up into separate functions for each quantity.
                Otherwise, the quickest way seems to be to calculate all masks at once.
        '''
        if isinstance(maxTime,list):
            minTime = min(maxTime)
            maxTime = max(maxTime)
        else:
            minTime = 0.0

        ''' Return previous maps if already calculated'''
        if self.mask != None and minTime==self._maskMinTime and maxTime==self._maskMaxTime:
            return self.dMask,self.maskX,self.maskY,self.vxMask,self.vyMask,self.vzMask,self.speedMask,altMask,eMask,depthMask
        self._maskMinTime = minTime
        self._maskMaxTime = maxTime

        ''' Calculate new maps'''

        iTimeRange = (self.simTime<=maxTime) & (self.simTime>=minTime)
        timeArray = self.simTime[iTimeRange]

        if not bFullRange:
            print('bFullRange must be set to true until fixed')
            bFullRange = True

        if not self.demData:
            self.demData = DEM_DATA(self.simStartLat,self.simStartLon,self.fileHandler,self.demDirectory)

        pts = []
        vpts = []
        altpts = []
        depthpts = []
        for i in np.arange(self.num_particles()):
            # find maximum distance/position travelled by particle i
            if bFullRange:
                lati,loni = self.get_path(i,timeArray)
                vxi,vyi,vzi = self.get_velocity(i,timeArray)
                alti = self.get_altitude(i,timeArray)
                depthi = self.get_depth(i,timeArray)
                #pts.append(np.concatenate([[loni[::skipPoints]],[lati[::skipPoints]]]))
                #vpts.append(np.concatenate([[vxi[::skipPoints]],[vyi[::skipPoints]],[vzi[::skipPoints]]]))
                #altpts.append(alti[::skipPoints])
                #depthpts.append(depthi[::skipPoints])
                pts.append(np.vstack((loni[::skipPoints],lati[::skipPoints])))
                vpts.append(np.vstack((vxi[::skipPoints],vyi[::skipPoints],vzi[::skipPoints])))
                altpts.append(np.array(alti[::skipPoints]))
                depthpts.append(np.array(depthi[::skipPoints]))
                
            else:
                # switch dimensions here
                maxDist,maxLat,maxLon,_ = self.get_max_distance(i,timeArray)
                #pts[0,i] = maxLon
                #pts[1,i] = maxLat
                pts.append(np.concatenate([[maxLon],[maxLat]]))
        pts = np.concatenate(pts,axis=1)
        vpts = np.concatenate(vpts,axis=1)
        altpts = np.concatenate(altpts,axis=0)
        depthpts = np.concatenate(depthpts,axis=0)

        mapPixelX = self.demData.mapZ.shape[1]
        mapPixelY = self.demData.mapZ.shape[0]
        X = np.arange(np.min(pts[0,:]),np.max(pts[0,:])+resolution/mapPixelX,resolution/mapPixelX)
        Y = np.arange(np.min(pts[1,:]),np.max(pts[1,:])+resolution/mapPixelY,resolution/mapPixelY)

        dMask = np.zeros([len(X),len(Y)])
        vxMask = np.zeros([len(X),len(Y)])
        vyMask = np.zeros([len(X),len(Y)])
        vzMask = np.zeros([len(X),len(Y)])
        altMask = np.zeros([len(X),len(Y)])
        depthMask = np.zeros([len(X),len(Y)])

        dx = X[1]-X[0]
        dy = Y[1]-Y[0]
        samples = pts

        cellArea = self.get_cell_area()
        
        for i,x in enumerate(X[:-1]):
            for j,y in enumerate(Y[:-1]):
                iInCell = np.logical_and(np.logical_and(x<=samples[0],samples[0]<=x+dx),np.logical_and(y<=samples[1],samples[1]<=y+dy))
                if np.any(iInCell):
                    dMask[i,j] = np.sum(iInCell)
                    vxMask[i,j] = np.nanmean(vpts[0][iInCell])
                    vyMask[i,j] = np.nanmean(vpts[1][iInCell])
                    vzMask[i,j] = np.nanmean(vpts[2][iInCell])
                    altMask[i,j] = np.nanmean(altpts[iInCell])

                    ## - Depth method: maximum
                    depthMask[i,j] = max(depthMask[i,j],max(depthpts[iInCell],key=abs))
                    ## - Depth method: volume/area
                    #particleVolume = np.sum(iInCell) * (self.simRecord['Particle_Radius'] * 2) ** 3
                    #depthMask[i,j] = depthMask[i,j] + particleVolume / cellArea
                    ## - Depth method: centroid
                    #depthMask[i,j] = depthMask[i,j] + np.mean(depthpts[iInCell])

        vxMask[vxMask==0] = np.nan
        vyMask[vyMask==0] = np.nan
        vzMask[vzMask==0] = np.nan
        altMask[altMask==0] = np.nan
        depthMask[depthMask==0] = np.nan
        speedMask = np.sqrt(vxMask**2 + vyMask**2 + vzMask**2)

        try:
            eMask = 0.5*(self.simRecord["Particle_Mass"] * dMask*speedMask**2) / cellArea
        except:
            print("No database record found")
            eMask = 0.5*(dMask*speedMask**2) / cellArea
        
        X,Y = np.meshgrid(X,Y)
        X = X.T
        Y = Y.T

        # Store these so they can be queried more than once without running the whole thing
        self.mask = dMask>0
        self.maskX = X
        self.maskY = Y
        self.dMask = dMask
        self.vxMask = vxMask
        self.vymask = vyMask
        self.vzMask = vzMask
        self.speedMask = speedMask
        self.altMask = altMask
        self.eMask = eMask
        self.depthMask = depthMask        
        return dMask,X,Y,vxMask,vyMask,vzMask,speedMask,altMask,eMask,depthMask

    def fit_velocity_mask(self,maxTime,resolution=5,bFullRange=True,skipPoints=1):
        '''
        Fits a mask covering the inundation area and returns average velocities for each cell. Each value indicates the number of particles within a cell of the mask.
            maxTime - maximum simulation time (maxTime=self.time[-1] for full simulation)
            resolution - mask resolution (resolution=1 means same resolution as DEM, 1/3600 degrees)
            bFullRange - if true, all points in time series are considered, else only the points at which each
                particle's distance from the dam is maximum.
            skipPoints - number of time steps to skip (only has an effect when bFullRange=True)
        '''
        _,maskX,maskY,vxMask,vyMask,vzMask,speedMask,_,_,_ = self.fit_all_masks(maxTime,resolution,bFullRange,skipPoints)
        return vxMask,vyMask,vzMask,speedMask,maskX,maskY

    def fit_speed_mask(self,maxTime,resolution=5,bFullRange=True,skipPoints=1):
        _,_,_,speedMask,maskX,maskY = self.fit_velocity_mask(maxTime,resolution,bFullRange,skipPoints)
        return speedMask,maskX,maskY

    def fit_density_mask(self,maxTime,resolution=5,bFullRange=True,skipPoints=1):
        '''
        Fits a mask covering the inundation area. Each value indicates the number of particles within a cell of the mask.
            maxTime - maximum simulation time (maxTime=self.time[-1] for full simulation)
            resolution - mask resolution (resolution=1 means same resolution as DEM, 1/3600 degrees)
            bFullRange - if true, all points in time series are considered, else only the points at which each
                particle's distance from the dam is maximum.
            skipPoints - number of time steps to skip (only has an effect when bFullRange=True)
        '''
        dMask,X,Y,_,_,_,_,_,_,_ = self.fit_all_masks(maxTime,resolution,bFullRange,skipPoints)
        return dMask,X,Y

    def fit_mask(self,maxTime,resolution=5,bFullRange=True,skipPoints=1):
        '''
        Fits a boolean mask covering the inundation area.
            maxTime - maximum simulation time (maxTime=self.time[-1] for full simulation)
            resolution - mask resolution (resolution=1 means same resolution as DEM, 1/3600 degrees)
            bFullRange - if true, all points in time series are considered, else only the points at which each
                particle's distance from the dam is maximum.
            skipPoints - number of time steps to skip (only has an effect when bFullRange=True)
        '''
        mask,X,Y = self.fit_density_mask(maxTime,resolution,bFullRange,skipPoints)
        return mask>0,X,Y

    def fit_energy_mask(self,maxTime,resolution=5,bFullRange=True,skipPoints=1):
        dMask,maskX,maskY,vxMask,vyMask,vzMask,speedMask,altMask,eMask,depthMask = self.fit_all_masks(maxTime,resolution,bFullRange,skipPoints)
        return eMask,maskX,maskY

    def fit_altitude_mask(self,maxTime,resolution=5,bFullRange=True,skipPoints=1):
        _,maskX,maskY,_,_,_,_,altMask,_,_ = self.fit_all_masks(maxTime,resolution,bFullRange,skipPoints)
        return altMask,maskX,maskY

    def get_flood_area(self,timeArray=[],resolution=5,bFullRange=True,skipPoints=1):
        '''
        Calculates approximate inundated area for a set of time values
        '''
        try:
            if len(timeArray)==0: #array is empty
                timeArray = self.simTime
        except: #array is scalar
            timeArray = [timeArray]

        a = []
        for t in timeArray:
            a.append(self._get_flood_area(t,resolution,bFullRange,skipPoints))

        if len(a)==1:
            a = a[0]

        return a
    def _get_flood_area(self,maxTime,resolution=5,bFullRange=True,skipPoints=1):
        '''
        Calculates approximate inundated area
        '''
        mask,X,Y = self.fit_mask(maxTime,resolution,bFullRange,skipPoints)
        dLon = X[1,0] - X[0,0]
        dLat = Y[0,1] - Y[0,0]
        dPos = self._geo2pos(dLat,dLon)

        cellArea = dPos[0] * dPos[1]
        return cellArea * np.sum(mask)
        
    def _get_total_quantity(self,method,timeArray=[]):
        '''
        Calculates the total quantity (specified by method) over the given time array
        '''
        for i in range(self.num_particles()):
            if i==0:
                s = method(i,timeArray)
            else:
                s = s + method(i,timeArray)
        return np.squeeze(s)

    def get_total_energy(self,timeArray=[]):
        '''
        Calculates total energy of all particles for the given time array
        '''
        return self._get_total_quantity(self.get_energy,timeArray)

    def _get_mean_quantity(self,method,timeArray=[]):
        '''
        Calculates mean quantity (specified by method) over the given time array
        '''
        return self._get_total_quantity(method,timeArray) / self.num_particles()
        
    def get_mean_speed(self,timeArray=[]):
        '''
        Calculates mean flow speed over the given time array
        '''
        return self._get_mean_quantity(self.get_speed,timeArray)

    def get_mean_altitude(self,timeArray=[]):
        '''
        Calculates mean altitude over the given time array
        '''
        return self._get_mean_quantity(self.get_altitude,timeArray)

    def angle_of_reach(self):
        '''
        Calculation of the angle of reach, the slope of energy loss line along the path  
        results are calculated for each particle
        '''
        partId = [*range(1,self.num_particles(),1)]
        dists = []
        lats = []
        longs = []
        zend = []
        zinit = []
        areach = []
        for each in partId: 
            dists.append(max(self.get_distance(each)))
            lat,lon = self.get_path(each,[self.max_time()])
            lats.append(lat[0])
            longs.append(lon[0])
            ze = self.get_altitude(each,[self.max_time()])
            zend.append(ze[0][0])
            zi = self.get_altitude(each,[0])
            zinit.append(zi[0][0])
            areach.append(180/(np.pi)*np.arctan((zinit[-1] - zend[-1])/dists[-1]))
        return areach

    def save_mask_io(self,mask,X,Y):
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        plt.axis("off")
        plt.contourf(X,Y,mask)
        
        imageObj = io.BytesIO()
        plt.savefig(imageObj,transparent=True,bbox_inches='tight',pad_inches=0.0)
        plt.close()
        return imageObj

    def save_mask(self,fileName,mask,X,Y):
        imageObj = self.save_mask_io(mask,X,Y)
        self.fileHandler.save_bytes(imageObj.getvalue(),fileName)
        
    def save_plot_io(self,x,y,xlabel,ylabel):
        fig = plt.figure()
        plt.plot(x,y)
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        imageObj = io.BytesIO()
        plt.savefig(imageObj,transparent=True,bbox_inches='tight',pad_inches=0.0)
        plt.close()
        return imageObj

    def save_plot(self,fileName,x,y,xlabel,ylabel):
        imageObj = self.save_plot_io(x,y,xlabel,ylabel)
        self.fileHandler.save_bytes(imageObj.getvalue(),fileName)

    def _geo2pos(self,latIn,lonIn):
        '''
        Conversion between latitude/longitude and cartesian positions
        '''
        return np.array([lonIn*self.simResX,latIn*self.simResY])

    def _pos2geo(self,x,y):
        lonOut = x/self.simResX
        latOut = y/self.simResY
        return np.array([latOut,lonOut])

class DEM_DATA:
    '''
    Handles data from JAXA's Digital Elevation Model.
    '''
    #Directory to store DEM data
    #demDirectory = directory_manager.get_dem_dir()
    demDirectory = 'IMAWARE/Sim_Raw/data_DEM'
    #Resolution per horizontal pixel (meters/pixel)
    mapResX = 30
    #Resolution per vertical pixel (meters/pixel)
    mapResY = mapResX
    # Coordinate range for map
    mapLon = [0,1]
    mapLat = [0,1]
    # Additional parameters
    _earthRadius = 6371e3 #earth radius (m)
    _lonAdjust = 1.0 #latitude-dependent adjustment factor for longitudinal resolution


    def __init__(self,lat,lon,fileHandler=None,demDirectory=None):
        if demDirectory != None:
            self.demDirectory = demDirectory
        self.mapZ,self.tifDir,self.mapLat,self.mapLon = jaxa.get_map(lat,lon,self.demDirectory,fileHandler=fileHandler)
        pxPerDeg_X = self.mapZ.shape[1]/(self.mapLon[1]-self.mapLon[0])
        pxPerDeg_Y = self.mapZ.shape[0]/(self.mapLat[1]-self.mapLat[0])
        self._lonAdjust = np.cos(lat*pi/180)
        self.mapResY = self._earthRadius*(pi/180) / pxPerDeg_Y
        self.mapResX = self._earthRadius*(pi/180) * self._lonAdjust / pxPerDeg_X
        #self.mapZ = np.flipud(self.mapZ)

        self.fileHandler = fileHandler

    def get_elev(self,lat,lon,bCartesian=False):
        '''
        Returns the elevation in meters at the given latitude and longitude
        '''
        #Retrieve elev in cardinal directions
        hNorth,hEast,hSouth,hWest,iAlpha,jAlpha = self._get_cardinal_elevs(lat,lon,bCartesian)

        # Calculate interpolant elevation
        h = (hSouth + iAlpha*(hNorth - hSouth) + 
            hWest + jAlpha*(hEast-hWest)) / 2
        return h

    
    def get_surface_normal(self,lat,lon,bCartesian=False):
        '''
        Returns the surface normal of the DEM at a given latitude and longitude
        '''
        #Retrieve elev in cardinal directions
        hNorth,hEast,hSouth,hWest,iAlpha,jAlpha = self._get_cardinal_elevs(lat,lon,bCartesian=bCartesian)
        
        # Calculate spatial derivatives of elevation
        dh_dx = (hEast - hWest)/(2*self.mapResX)
        dh_dy = (hNorth - hSouth)/(2*self.mapResY)
        dh_dz = -1

        # Calculate surface normal as a unit vector (positive up)
        surfNorm = -np.array([dh_dx,dh_dy,dh_dz])/np.sqrt(dh_dx**2 + dh_dy**2 + 1)
        return surfNorm

    def _get_cardinal_elevs(self,lat,lon,bCartesian=False):
        '''
        Retrives elevations in each cardinal direction.
        Returns (hNorth,hEast,hSouth,hWest,iAlpha,jAlpha)
        where iAlpha and jAlpha are the alpha values in each axis, for the purpose of interpolation, y = y0 + alpha*(y1-y0)
        '''
        # Convert lat and lon to pixel units
        if bCartesian:
            pX,pY = self._pos2px(lat,lon)
        else:
            pX,pY = self._geo2px(lat,lon)
        i = round(pY)
        j = round(pX)

        # Read height values from each cardinal direction
        hEast = self._read_Z(i,j+1)
        hWest = self._read_Z(i,j-1)
        hNorth = self._read_Z(i-1,j)
        hSouth = self._read_Z(i+1,j)

        # Calculate alpha values in each axis for interpolation
        iAlpha = ((pY % 1)+1) / 2
        jAlpha = ((pX % 1)+1) / 2

        return hNorth,hEast,hSouth,hWest,iAlpha,jAlpha

    def _read_Z(self,i,j):
        '''
        Reads a value from the map matrix while preventing out-of-bounds indices
        '''
        I = int(np.clip(i,0,self.mapZ.shape[0]-1))
        J = int(np.clip(j,0,self.mapZ.shape[1]-1))
        return float(self.mapZ[I,J])

    def _geo2px(self,lat,lon):
        '''
        Converts geographical coordinates to pixel values (positive right, down)
        NOTE: these are currently returns as float values so that the modulo can be used for interpolation
        '''
        pX = (lon - self.mapLon[0]) * ((self.mapZ.shape[1]-1) / (self.mapLon[1]-self.mapLon[0]))
        pY = self.mapZ.shape[0] - (lat - self.mapLat[0]) * ((self.mapZ.shape[1]-1) / (self.mapLat[1]-self.mapLat[0]))
        return pX,pY

    def _geo2pos(self,lat,lon):
        pX,pY = self._geo2px(lat,lon)
        x = pX*self.mapResX
        y = (self.mapZ.shape[0] - pY)*self.mapResY
        return x,y
        
    def _pos2geo(self,x,y):
        xRange = [0,self.mapResX*self.mapZ.shape[1]]
        yRange = [0,self.mapResY*self.mapZ.shape[0]]
        lon = self.mapLon[0] + (self.mapLon[1] - self.mapLon[0]) * (x-xRange[0]) / (xRange[1]-xRange[0])
        lat = self.mapLat[0] + (self.mapLat[1] - self.mapLat[0]) * (y-yRange[0]) / (yRange[1]-yRange[0])
        return lat,lon

    def _pos2px(self,x,y):
        '''
        Converts cartesian coordinates to pixel values (positive right, down)
        NOTE: these are currently returns as float values so that the modulo can be used for interpolation
        '''
        pX = x/self.mapResX
        pY = self.mapZ.shape[0] - y/self.mapResY
        return pX,pY
    