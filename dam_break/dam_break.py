from math import sqrt
import sys
import directory_manager
import io
import numpy as np
from dam_break.dambreak_lib import DEM_DATA
import math
import pickle
import os
from source_data.GCPdata import GCP_IO
from pathlib import Path

class DAM_BREAK:
    '''
    Simulates a dam break for a single site, modeling tailings as a set of discrete spherical particles.
    '''
    _timeStep = 0.2
    _timeStepIndex = 0
    _collisionTick = 5
    _collisionCount = 0
    _bTruncateTimeHistory = True
    _bVerbose = False
    _progressPercentage = 0.0
    CONST_STATES_TO_SAVE = 6

    ## -- Default parameters
    nObj = 1000 # number of particles
    pondRadius = 300.0 # radius of the dam's pond
    pondHeight = 80.0 # height of the dam
    tailingsVolume = 10.0e6 # volume of released tailings
    tailingsDensity = 1594.0 # density of released tailings
    maxTime = 300.0 # number of seconds to simulate
    frictionCoeff = 1.0 # friction coefficient between particles and the floor
    dampingCoeff = 0.04 # damping coefficient affecting angular rotation of particles
    collisionFrictionCoeff = 0.0 # friction coefficient between particles

    time = []
    particles = []

    def __init__(self,
                siteLat=-23.0,siteLon=-43.5,
                pondRadius=300.0,
                nObj=1000,
                tailingsVolume=10.0e6,
                tailingsDensity=1594.0,
                maxTime=300.0,
                timeStep=0.2,
                dampingCoeff=0.04,
                demDirectory='IMAWARE/Sim_Raw/data_DEM',
                fileHandler=None):
        '''
        Constructor for DAM_BREAK object, representing a single (deterministic) simulation run.
        '''
        ## Set site location and retrieve digital elevation model (DEM) data
        self.siteLat = siteLat
        self.siteLon = siteLon
        self.pondRadius = pondRadius
        self.tailingsVolume = tailingsVolume
        self.tailingsDensity = tailingsDensity
        self.maxTime = maxTime
        self._timeStep = timeStep
        self.dampingCoeff = dampingCoeff
        self.time = np.arange(0.0,self.maxTime+self._timeStep,self._timeStep)

        ## Calculate site location in cartesian coordinates
        print('Loading digital elevation model...')
        self.demData = DEM_DATA(siteLat,siteLon,fileHandler=fileHandler,demDirectory=demDirectory)
        print('...digital elevation model loaded.')
        siteX,siteY = self.demData._geo2pos(self.siteLat,self.siteLon)
        siteZ = self.demData.get_elev(self.siteLat,self.siteLon)
        self.sitePos = [siteX,siteY,siteZ]

        ## Calculate pond parameters
        self.pondHeight = self.tailingsVolume/(math.pi*self.pondRadius**2)

        ## Initialise particles
        self.init_particles(nObj)

    def get_database_record(self,simID):
        record = {"Particle_Number": int(self.nObj),\
                "Particle_Mass": float(self.get_particle_mass()),\
                "Particle_Radius": float(self.get_particle_radius()),\
                "Damping": self.dampingCoeff,\
                "Volume_Factor": -1,\
                "Latitude_Offset": 0,\
                "Longitude_Offset": 0,\
                "Tailings_Density": self.tailingsDensity,\
                "Max_Distance": -1,\
                "Max_Velocity": -1,\
                "Total_Energy": -1,\
                "Flooding_Area": -1,\
                "Analysis_ID": '',\
                "Evaluation_Time": self.maxTime,\
                "Type_of_Analysis": "DAMBREAK",\
                "Parent_ID": '',\
                "Tree_Level": 0,\
                "Repeat": 0,\
                "File_Address": '',\
                "Output_Summary": '',\
                "ID": simID}
        return record

    def init_particles(self,nObj):
        self.particles = []
        nRadial = math.floor( (nObj*self.pondRadius / (math.pi*self.pondHeight))**(1/3))
        particleRadius = self.pondRadius/(2*nRadial)
        nLayers = round(self.pondHeight/(2*particleRadius))

        # Keep parameters in valid ranges
        nRadial = max((nRadial,1))
        nLayers = max((nLayers,1))
        particleRadius = max((particleRadius,0.0001))

        # For each fluid layer
        i = 0
        for layer in range(nLayers):
            #For each ring in the current layer
            for r in range(nRadial):
                n = round(2*math.pi*r)
                distFromPond = (2*r-1)*particleRadius
                theta = np.linspace(0,2*math.pi,n)
                for t in theta:
                    i += 1
                    particle = DAM_BREAK_PARTICLE(self.demData,
                                mass=1.0,
                                radius=particleRadius,
                                inertia=1.0,
                                frictionCoeff=self.frictionCoeff,
                                dampingCoeff=self.dampingCoeff,
                                collisionFrictionCoeff=self.collisionFrictionCoeff,
                                time=self.time
                                )
                    posX = self.sitePos[0] + distFromPond*math.cos(t)
                    posY = self.sitePos[1] + distFromPond*math.sin(t)
                    posZ = self.demData.get_elev(posX,posY,bCartesian=True) + layer*2*particle.radius
                    particle.set_pos([posX,posY,posZ],0)
                    self.particles += [particle]
        self.nObj = len(self.particles)

        # Calculate particle mass and radius after total number of particles is known
        for p in self.particles:
            p.set_mass(self.tailingsDensity*self.tailingsVolume/self.nObj)
        
    def update(self):
        '''
        Moves the simulation forward by one time step
        '''
        # Update each particle
        for p in self.particles:
            p.apply_kinematics(self._timeStepIndex,self._timeStep)
            p.apply_terrain_collision(self._timeStepIndex)
            if self._collisionCount==0:
                for other in self.particles:
                    p.apply_particle_collision(self._timeStepIndex,other)

        self._timeStepIndex += 1
        self._collisionCount = (self._collisionCount+1) % self._collisionTick
    
    def run_simulation(self):
        '''
        Runs the simulation up to maxTime (seconds) by timeStep (seconds)
        '''
        self._timeStepIndex = 1
        for t in self.time[1:]:
            if self._bVerbose:
                #print('Time = %1.1f' % t)
                self._update_progress_bar()
            self.update()
        
        # Desample state time histories to once per collision event
        if self._bTruncateTimeHistory:
            self.time = self.time[::self._collisionTick]
            for p in self.particles:
                p.X = p.X[::self._collisionTick]

    def _update_progress_bar(self):
        self._progressPercentage += 100.0*self._timeStep/self.maxTime
        integerProgress = math.floor(self._progressPercentage)
        sys.stdout.write('\r'+'[{}] - {}%'.format(integerProgress*'#', integerProgress))
        sys.stdout.flush()
        if integerProgress>=100:
            print('\n')

    def results_to_array(self):
        '''
        Converts the results to a numpy array of the format
        [time,lat_0,lon_0,vx_0,vy_0,vz_0,...lat_n,lon_n,vx_n,vy_n,vz_n]
        '''
        data = np.array(self.time).reshape((len(self.time),1))
        for p in self.particles:
            state = p.get_state(range(len(self.time)))
            lat,lon = self.demData._pos2geo(state[:,0],state[:,1])
            state[:,0] = lat
            state[:,1] = lon
            data = np.hstack((data,state[:, 0:self.CONST_STATES_TO_SAVE]))
        return data

    def generate_results_io(self):
        '''
        Generates a stringIO object containing csv data for the current simulation.
        '''
        data = self.results_to_array() 
        stringIO = io.StringIO()
        np.savetxt(stringIO,data,delimiter=',',fmt='%1.9f')
        return stringIO

    def save_results(self,damID,simID,fileHandler=None,warehouseFolder='IMAWARE'):
        '''
        Save results as csv (defaults to using Google Cloud - update later)
        '''
        warehouseFolder = Path(warehouseFolder)
        resultsFolder = warehouseFolder.joinpath('Sim_Raw/results_py')
        csvFolder = warehouseFolder.joinpath('Analysis_Results').joinpath(damID)
        csvFolder = str(csvFolder).replace('\\','/')

        # Save serialised output
        fileName = ''
        #fileName = resultsFolder.joinpath('%s.dat' % simID)
        #with open(fileName,'wb') as f:
        #    pickle.dump(self,f)
        #print('Simulation %s serialised successfully' % simID)

        # Save csv formatted output
        csvName = csvFolder + '/%s.csv' % simID
        stringIO = self.generate_results_io()
        dataStr = stringIO.getvalue()
        if fileHandler==None:
            fileHandler = GCP_IO()
        fileHandler.save_text(dataStr,csvName)
        print('Simulation %s output saved successfully' % simID)

        return (fileName,csvName)

    def get_particle_mass(self):
        return self.particles[0].mass
    def get_particle_radius(self):
        return self.particles[0].radius
    
class DAM_BREAK_PARTICLE:
    '''
    Represents a single particle in a dam break simulation
    '''
    ## - Default parameters
    mass = 1.0
    radius = 1.0
    inertia = 1.0
    frictionCoeff = 1.0
    dampingCoeff = 0.04
    dampingTimeConstant = 1.0/dampingCoeff
    collisionFrictionCoeff = 0.0

    ## - Initial states
    _n_states = 9
    surfNorm = np.array([0,0,1])
    surfZ = 0.0
    X = np.array([])
    _gravity = 9.81
    mapCell = [-1,-1]
    demData = None

    def __init__(self,demData,mass,radius,inertia,frictionCoeff,dampingCoeff,collisionFrictionCoeff,time):
        self.demData = demData
        self.mass = mass
        self.radius = radius
        self.inertia = inertia
        self.frictionCoeff = frictionCoeff
        self.dampingCoeff = dampingCoeff
        self.dampingTimeConstant = 1.0/dampingCoeff
        self.collisionFrictionCoeff = collisionFrictionCoeff
        self.X = np.zeros((len(time),self._n_states))

    ## State get methods
    def get_state(self,tIndex):
        return self.X[tIndex,:]
    def get_pos(self,tIndex):
        return self.X[tIndex,0:3]
    def get_vel(self,tIndex):
        return self.X[tIndex,3:6]
    def get_angvel(self,tIndex):
        return self.X[tIndex,6:9]

    ## State set methods
    def set_state(self,state,tIndex):
        self.X[tIndex,:] = state
    def set_pos(self,position,tIndex):
        x = self.get_state(tIndex)
        x[0:3] = position
        self.set_state(x,tIndex)
    def set_vel(self,velocity,tIndex):
        x = self.get_state(tIndex)
        x[3:6] = velocity
        self.set_state(x,tIndex)
    def set_angvel(self,angularVelocity,tIndex):
        x = self.get_state(tIndex)
        x[6:9] = angularVelocity
        self.set_state(x,tIndex)

    ## Parameter set methods
    def set_mass(self,mass):
        self.mass = mass
        self.inertia = (2/5)*self.mass*self.radius**2

    ## Model methods
    def update_surf_data(self,positionX,positionY):
        self.surfNorm = self.demData.get_surface_normal(positionX,positionY,bCartesian=True)
 
    def apply_kinematics(self,tIndex,timeStep):
        '''
        Applies kinematic equations of motion to the particle for a time step
        '''
        # Particle states
        position = self.get_pos(tIndex-1) #position
        velocity = self.get_vel(tIndex-1) #velocity
        angularVelocity = self.get_angvel(tIndex-1) #angular velocity

        # Update map cell and recalculate surface normal if needed
        pX,pY = self.demData._pos2px(position[0],position[1])
        currentCell = [round(pX),round(pY)]
        if currentCell != self.mapCell:
            self.update_surf_data(position[0],position[1])
            self.mapCell = currentCell

        # Calculate forces
        frictionScaled = self.frictionCoeff * self.mass
        weight = np.array([0,0,-self.mass*self._gravity])
        friction = -frictionScaled * (velocity + np.cross(self.radius*self.surfNorm, angularVelocity))
        torque = np.cross(-self.radius*self.surfNorm,friction)

        # Friction adjustment to prevent reversing direction
        dVelFriction = friction / self.mass
        dVelFriction[0] = min(dVelFriction[0],-velocity[0],key=abs)
        dVelFriction[1] = min(dVelFriction[1],-velocity[1],key=abs)
        dVelFriction[2] = min(dVelFriction[2],-velocity[2],key=abs)

        dAngDamping = -(1.0 / self.dampingTimeConstant) * angularVelocity
        dAngDamping[0] = min(dAngDamping[0],-angularVelocity[0],key=abs)
        dAngDamping[1] = min(dAngDamping[1],-angularVelocity[1],key=abs)
        dAngDamping[2] = min(dAngDamping[2],-angularVelocity[2],key=abs)

        # Time derivatives
        dVelocity = weight / self.mass + dVelFriction # acceleration
        dAngVelocity = torque / self.inertia + dAngDamping # angular acceleration
        dState = np.hstack((velocity,dVelocity,dAngVelocity)) # State time derivative = [velocity, acceleration, angular acceleration]

        # Apply state change
        xi = self.X[tIndex-1,:] + dState*timeStep
        self.X[tIndex,:] = xi

    def apply_terrain_collision(self,tIndex):
        '''
        Applies collision with the terrain using the digital elevation model
        '''
        position = self.get_pos(tIndex)
        # Surface elevation is interpolated so must be recalculated even within the same cell
        self.surfZ = self.demData.get_elev(position[0],position[1],bCartesian=True)
        if position[2] < self.surfZ:
            position[2] = self.surfZ
            velocity = self.get_vel(tIndex)
            dVelocity = np.dot(velocity,self.surfNorm)
            velocity = velocity - min(dVelocity,0) * self.surfNorm
            self.set_pos(position,tIndex)
            self.set_vel(velocity,tIndex)
        
    def apply_particle_collision(self,tIndex,other):
        '''
        Apply collision between self and another DAM_BREAK_PARTICLE
        '''
        # Do not calculate collision with self
        if self==other:
            return
        # Distance threshold for collision detection
        distanceThresh = self.radius + other.radius
        distanceThreshSq = distanceThresh**2

        # Check particles are colliding
        posA = other.get_pos(tIndex)
        posB = self.get_pos(tIndex)

        displacement = posB - posA
        distanceSq = np.dot(displacement,displacement)
        if distanceSq >= distanceThreshSq or distanceSq == 0:
            return

        # Calculate collision normal
        dist = sqrt(distanceSq)
        collisionNormal = displacement/dist
        collisionFriction = self.collisionFrictionCoeff

        # Calculate specific impulse
        velA = other.get_vel(tIndex)
        velB = self.get_vel(tIndex)
        dVel = velB - velA
        massRatioB = other.mass/(other.mass+self.mass)
        massRatioA = self.mass/(other.mass+self.mass)
        dVelB = (collisionFriction - massRatioB)*np.dot(dVel,collisionNormal)*collisionNormal - dVel*collisionFriction
        dVelA = (collisionFriction - massRatioA)*np.dot(dVel,collisionNormal)*collisionNormal - dVel*collisionFriction
        
        # Adjust velocities
        self.set_vel(velB + dVelB, tIndex)
        other.set_vel(velA - dVelA, tIndex)
        
        # Encroachment correction
        encroachDist = distanceThresh - dist
        self.set_pos(posB + massRatioB*encroachDist*collisionNormal, tIndex)
        other.set_pos(posA - massRatioA*encroachDist*collisionNormal, tIndex)
        