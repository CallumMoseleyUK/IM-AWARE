from dam_break.dambreak_lib import DAMBREAK_SIM
from DBdriver import DBFunctions as dbf
from sklearn.neighbors import KernelDensity
import numpy as np
from stats_mod import ecdf

class DAMBREAK_SET:
    '''
    Container class for sets of simulation data defined by a Flooding_Model_Description database table record.
    '''
    damRecord = {}
    setRecord = {}
    simRecords = []
    setID = ''

    def __init__(self,setRecord):
        '''
        record - a database entry from table Flooding_model_Description in dictionary format.
        '''
        self.damID = setRecord['Dam_ID']
        self.damRecord = dbf.query_by_ID(self.damID,'ANM')
        self.setID = setRecord['ID']
        self.setRecord = setRecord
        self.simRecords = dbf.query_by_analysis(self.setID)

    def get_dam_volume(self):
        return float(self.damRecord['Stored_Volume'])
    
    def get_dam_height(self):
        return float(self.damRecord['Height'])

    def get_data_set(self,quantityString):
        '''
        Returns the set of data corresponding to a given database field name
        '''
        dataSet = np.array([])
        return [record[quantityString] for record in self.simRecords]

class DAMBREAK_STAT(DAMBREAK_SET):
    '''
    Class for handling statistical analysis of simulation data sets defined by a Flooding_Model_description record
    '''
    def __init__(self,setRecord):
        super().__init__(setRecord)

    def calculate_ecdf(self,quantityString):
        '''
        Calculates the ECDF for the given quantity. Returns P,X.
        '''
        dataSet = self.get_data_set(quantityString)
        P,X = ecdf(dataSet)
        return P,X

    def hist_data(self,data,nBins=30):
        '''
        Calculates the histogram data for a given data set.
        '''
        histFreq,binEdges = np.histogram(data, bins=nBins)
        return histFreq,binEdges

    def hist_quantity(self,quantity,nBins=30):
        '''
        Calculates the histogram of a given quantity, corresponding to a field in the database.
        '''
        data = [r[quantity] for r in self.simRecords]
        return self.hist_data(data,nBins=nBins)
        
    def hist_area(self,nBins=30):
        return self.hist_quantity('Flooding_Area',nBins=nBins)

    def hist_max_distance(self,nBins=30):
        return self.hist_quantity('Max_Distance',nBins=nBins)

    def hist_max_velocity(self,nBins=30):
        return self.hist_quantity('Max_Velocity',nBins=nBins)

    def hist_total_energy(self,nBins=30):
        return self.hist_quantity('Total_Energy',nBins=nBins)

    def kde_quantity(self,quantity,bandwidthFactor=1.0):
        '''
        Fit data using Kernel Density Estmation
        '''
        data = [r[quantity] for r in self.simRecords]
        dataReshaped = np.array(data).reshape(-1,1)
        dataReshaped = dataReshaped/max(dataReshaped)

        #bandwidth = np.var(dataReshaped) * bandwidthFactor
        bandwidth = bandwidthFactor
        print(bandwidth)
        kde = KernelDensity(kernel='exponential', bandwidth = bandwidth).fit(dataReshaped)
        #kde = KernelDensity(kernel='gaussian').fit(dataReshaped)
        
        density = kde.score_samples(dataReshaped)
        return density,dataReshaped
    
    def kde_area(self,bandWidthFactor=1.0):
        return self.kde_quantity('Flooding_Area',bandwidthFactor=bandWidthFactor)