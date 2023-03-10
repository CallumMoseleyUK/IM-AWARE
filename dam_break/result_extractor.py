import dambreak_lib as dbr
import os 
import pandas as pd
from pathlib import Path
import numpy as np
import DBFunctions as dbf

basePath = Path(os.path.realpath(__file__)).parent

#def angle_of_reach(self, index, timearray=[]):
    #cords = self.getPath(index,timearray=[])
    #lsr = self.getMaxDistance(index,timearray=[])
    #aor = np.arctan(z(cords[0][0], cords[0][1]) - z(cords[len(cords)][0], cords[len(cords)][1]))/lsr[len(lsr)][0])

def angle_of_reach(simulID):
    sql1 = "select Parent_ID from Analysis_Results where ID = '{}'".format(simulID)      
    parent = dbf.collect_from_DB(sql1)[0][0]
    sql2 = "select Max_Simulated_Time from Flooding_Model_Description where ID = '{}'".format(parent)
    maxTime = dbf.collect_from_DB(sql2)[0][0]
    sql3 = "select File_Address from Analysis_Results where ID = '{}'".format(simulID)
    address = dbf.collect_from_DB(sql3)[0][0]
    file = address.joinpath('{}.csv'.format(simulID))
    a = dbr.DAMBREAK_SIM()
    a._load_innundation_data(file)
    partId = [*range(1,a.num_particles(),1)]
    dists = []
    lats = []
    longs = []
    zend = []
    zinit = []
    areach = []
    for each in partId: 
        dists.append(max(a.get_distance(each)))
        lat,lon = a.get_path(each,[maxTime])
        lats.append(lat[0])
        longs.append(lon[0])
        ze = a.get_altitude(each,[maxTime])
        zend.append(ze[0][0])
        zi = a.get_altitude(each,[0])
        zinit.append(zi[0][0])
        areach.append(180/(np.pi)*np.arctan((zinit[-1] - zend[-1])/dists[-1]))
    out = pd.DataFrame({'Particle_ID' : partId, 'Max_Reach': dists, 'End_Latitude' : lats, 'End_Longitude': longs, 'Zend' : zend, 'Zinit': zinit, 'reach_angle' : areach})
    return out 
        
#print(get_simul_from_ID('Xingu_VALESA-DAMBREAK-20210901-174150_9'))
