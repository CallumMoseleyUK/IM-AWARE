## WIP script for detecting dam ponds from JAXA digital elevation maps
## Dependencies: opencv-python, opencv-contrib-python
# Canny method: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
from data_connector import demObject
import folium 
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from sklearn.neighbors import KernelDensity

from scipy import ndimage
from scipy import stats
from scipy.integrate import cumtrapz, trapz

from itertools import groupby, product
import os 

class data_holder():
    pass

class Pond:
    def __init__(self,mapObj,limits):
        self.subObj = mapObj.submap(limits)
        self.Terrain = self.subObj.getImg()
        self.X = self.subObj.X
        self.Y = self.subObj.Y
        self.resolution = self.subObj.res

        setattr(self,'gen_data',data_holder)
        setattr(self.gen_data,'Terrain',self.Terrain)

    def Edge_detection(self,sigma=0.33):
        ## Run edge detection

        #TODO: Why does the edge detection flip the data! There is clearly a bug here that 
        # needs fixing Callum. This will mean the np.flip used in Find_islands then becomes redundent!
        # This needs checking and correcting so that all the figures match! Until this is done I don't
        # trust any results! 

        img = np.uint8(self.Terrain) #loss of precision, is this a problem?
        mapEdge = auto_canny(img,sigma)#.astype("bool")
        id_edge = np.where(mapEdge)

        X_edge = self.X[id_edge[1]]
        Y_edge = self.Y[id_edge[0]]

        self.edge = np.concatenate([[X_edge],[Y_edge]])
        return self.edge

    def Edge_PMF_compute(self,bandwidth_factor=1.5):

        if not hasattr(self,'edge'):
            self.Edge_detection()

        # FIX HACK LATER
        Xm,Ym = np.meshgrid(self.X,self.Y)
        xx = np.hstack(Xm)
        yy = np.hstack(Ym)
        self.space_mesh = np.concatenate([[xx],[yy]])
        KDE = KernelDensity(kernel='exponential', bandwidth=self.resolution*bandwidth_factor).fit(self.edge.T)
        edge_probability = np.exp(KDE.score_samples(self.space_mesh.T))

        edge_PMF = np.reshape(edge_probability,[len(self.Y),len(self.X)])
        self.edge_probability = edge_probability
        self.edge_PMF = edge_PMF
        return self.edge_PMF

    def Find_islands(self,mass_threshold=.79, samples=500,bandwidth_factor=1.5):
        if not hasattr(self,'edge_PMF'):
            self.Edge_PMF_compute()

        X = self.X
        Y = self.Y
        edge_CPMF = compliment(X,Y,np.flip(np.flip(self.edge_PMF,axis=0),axis=1))
        samps = rejection_sample(X,Y,edge_CPMF,samples,sampling_threshold=mass_threshold)
        island_mask = generate_mask(X,Y,self.edge_PMF,samps,resolution=1)

        KDE_t = KernelDensity(kernel='exponential', bandwidth=self.resolution*bandwidth_factor).fit(samps.T)
        P_CPMF = np.exp(KDE_t.score_samples(self.space_mesh.T))
        P_CPMF = np.reshape(P_CPMF,[len(Y),len(X)])

        #Add a border of zeros 
        #island_mask = np.pad(island_mask, pad_width=1, mode='constant', constant_values=0)
        labeled_mask, n_features = ndimage.label(island_mask, np.ones((3,3)))

        setattr(self.gen_data,'mask',island_mask)
        setattr(self.gen_data,'labelled_mask',labeled_mask)
        setattr(self.gen_data,'n_islands',n_features)
        setattr(self.gen_data,'island_samples',samps)
        setattr(self.gen_data,'CPMF',P_CPMF)
        setattr(self.gen_data,'Compliment_EPMF',edge_CPMF)
        setattr(self.gen_data,'PMF',self.edge_PMF)
        
    def Compute_island_statistics(self,origin):
        """
        Scoring islands by distance from a point.
        """

        Data_dict = {}
        island_scores = []
        island_centroid = []

        for ind in range(1,self.gen_data.n_islands+1):
            Data_dict[ind] = {}

            iY,iX = np.where(np.flip(self.gen_data.labelled_mask,axis=0)==ind)
            pX,pY = self.X[iX],self.Y[iY]

            distance = np.sqrt((pX-origin[0])**2+(pY-origin[1])**2)
            pVal = stats.norm.pdf(distance, loc=0, scale=1000)

            Data_dict[ind]['cells'] = np.concatenate([[pX],[pY]])
            Data_dict[ind]['distances'] = distance
            Data_dict[ind]['scores'] = pVal
            Data_dict[ind]['avg_dist_score'] = np.mean(pVal)
            
            distance = np.sqrt((np.median(pX)-origin[0])**2+(np.median(pY)-origin[1])**2)
            pVal = stats.norm.pdf(distance, loc=0, scale=1000)
            Data_dict[ind]['centroid'] = [np.median(pX),np.median(pY)]
            island_centroid.append(Data_dict[ind]['centroid'])
            Data_dict[ind]['centroid_score'] = pVal
            island_scores.append([Data_dict[ind]['avg_dist_score'],Data_dict[ind]['centroid_score']])
        
        self.island_centroid = np.concatenate([island_centroid],axis=1)
        self.island_scores = np.concatenate([island_scores],axis=1)

        setattr(self.gen_data,'island_data',Data_dict)
        self._compute_island_shape_statistics()
        self.summary_island_data['centroid_x'] = self.island_centroid[:,0]
        self.summary_island_data['centroid_y'] = self.island_centroid[:,1]
        self.summary_island_data['d_score_Av'] = self.island_scores[:,0]
        self.summary_island_data['d_score_Cx'] = self.island_scores[:,1]
        
        
    def _compute_island_shape_statistics(self):
        #Area
        #Height
        #width, length
        if not np.shape(self.Terrain)==np.shape(self.gen_data.labelled_mask):
            print('TODO: implimented method for variable mesh resolution')
        unit_area = (self.X[1]-self.X[0])*(self.Y[1]-self.Y[0]) 
        Area = []
        Height = []
        Width = []
        Length = []
        D = pd.DataFrame(columns=['Area','Width','Length','AvgAlt','minAlt','maxAlt'])
        for ind in range(1,self.gen_data.n_islands+1):
            
            N_ua = np.sum(self.gen_data.labelled_mask==ind)
            A = unit_area * N_ua

            H =  self.Terrain[self.gen_data.labelled_mask==ind]

            W = np.max(self.gen_data.island_data[ind]['cells'][0])- np.min(self.gen_data.island_data[ind]['cells'][0])
            L = np.max(self.gen_data.island_data[ind]['cells'][1])- np.min(self.gen_data.island_data[ind]['cells'][1])

            self.gen_data.island_data[ind]['Area'] = A
            self.gen_data.island_data[ind]['Height'] = H
            self.gen_data.island_data[ind]['Width'] = W
            self.gen_data.island_data[ind]['Length'] = L
            d = np.array([A,W,L,np.mean(H),np.min(H),np.max(H)])
            D.loc[ind] = d

        self.summary_island_data = D        

    def _field_plotter(self,title,data,data_label,cmap='jet'):
        plt.figure(figsize=(12,10))
        plt.title(title)
        col = plt.imshow(data,extent= [self.subObj.X[0],self.subObj.X[-1],self.subObj.Y[-1],self.subObj.Y[1]],cmap=cmap)
        plt.xlabel('W-E (m)')
        plt.ylabel('N-S (m)')
        plt.colorbar(col,label=data_label)

    def plot_score(self):
        cmap = cm.Reds
        scores = np.log(self.island_scores[:,0])
        norm = Normalize(vmin=np.min(scores), vmax=np.max(scores))

        plt.figure(figsize=(12,10))
        for i in range(1,self.gen_data.n_islands+1):
            score  = np.log(self.gen_data.island_data[i]['avg_dist_score'])
            plt.scatter(self.gen_data.island_data[i]['cells'][0],self.gen_data.island_data[i]['cells'][1],
                color=cmap(norm(score)),label='I: {}  S: {:.2e}'.format(i,score))
        plt.legend(loc=[0,0])

    def plot_island_samples(self):
        plt.figure(figsize=(12,10))
        plt.scatter(self.gen_data.island_samples[0],self.gen_data.island_samples[1])
        plt.xlabel('W-E (m)')
        plt.ylabel('N-S (m)')

    def plot_edge(self,site_location=[]):
        plt.figure(figsize=(12,10))
        plt.scatter(self.edge[0],self.edge[1])
        if site_location:
            plt.scatter(site_location[0],site_location[1],marker='+',s=1000,c='red')
        plt.xlabel('W-E (m)')
        plt.ylabel('N-S (m)')

    def plot(self,source='Terrain'):

        plot_list = ['Terrain','PMF','CPMF','mask','labelled_mask']
        data_label = ['Height', r'$P_{edge}$', r'$P_{zone}$', 'Mask', 'Label']
        
        if source=='all':
            plotter = plot_list
        else:
            if source not in plot_list:
                print('ERROR: {} not availiable to plot. Choose from: \n\t {}'.format(source,plot_list))
                return
            else:
                plotter = [source]

        for name in plotter:
            DAT = getattr(self.gen_data,name)
            label = data_label[plot_list.index(name)]
            self._field_plotter(name,DAT,label)

def plot_terrain(mapObj,title='',cmap='jet'):
    plt.figure(figsize=(12,10))
    plt.title(title)
    col = plt.imshow(mapObj.getImg(),extent= [mapObj.X[0],mapObj.X[-1],mapObj.Y[-1],mapObj.Y[1]],cmap=cmap)
    plt.xlabel('W-E (m)')
    plt.ylabel('N-S (m)')
    plt.colorbar(col,label='Height')

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(65535, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    edged[edged>0] = 1
    return edged

def normalize(X):
    return (X - np.min(X))/(np.max(X)-np.min(X))

def joint_cdf(X, Y, Z,adj_scale=True):
    (xmin, xmax, ymin, ymax, nx, ny)= (X[0], X[-1],Y[0],Y[-1],len(X),len(Y))
    
    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    print('APPROXIMATION ERROR : {:.3f}%'.format((1-np.sum(Z*dS))*100))

    A_Internal = Z[1:, 1:]
    # sides: up, down, left, right
    (A_d_0, A_l_0) = (0.5 * Z[0, 1:], 0.5 * Z[1:, 0])
    (A_d_1, A_l_1) = (0.5 *A_d_0[0],0.5 *A_l_0[0])
    A_d = np.concatenate([[A_d_1],A_d_0])
    A_l = np.concatenate([[A_l_1],A_l_0])
    
    Z_intg = np.cumsum(A_Internal,axis=1)
    Z_intg = np.cumsum(Z_intg,axis=0)

    Ztmp = np.zeros(Z.shape)
    Ztmp[0,:] = A_d
    Ztmp[:,0] = A_l
    Ztmp[1:,1:] = Z_intg*dS
    #if adj_scale:
    #    Ztmp = (Ztmp-np.min(Ztmp))/(np.max(Ztmp)-np.min(Ztmp))
    return Ztmp

def joint_Integral(X, Y, Z):
    (xmin, xmax, ymin, ymax, nx, ny)= (X[0], X[-1],Y[0],Y[-1],len(X),len(Y))
    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    A_Internal = Z[1:-1, 1:-1]
    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (Z[0, 1:-1], Z[-1, 1:-1], Z[1:-1, 0], Z[1:-1, -1])
    # corners
    (A_ul, A_ur, A_dl, A_dr) = (Z[0, 0], Z[0, -1], Z[-1, 0], Z[-1, -1])

    return dS * (np.sum(A_Internal)\
                + 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))\
                + 0.25 * (A_ul + A_ur + A_dl + A_dr))

def generate_mask(X,Y,Z,samples,resolution=2):
    mask = np.zeros(np.shape(Z[::resolution,::resolution]))
    X = X[::resolution]
    Y = Y[::resolution]
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
    # TODO: Check variables from loops are right way round (working code below)
    '''
    for j,x in enumerate(X[:-1]):
        for i,y in enumerate(Y[:-1]):
            if np.any(np.logical_and(np.logical_and(x<=samples[0],samples[0]<=x+dx),np.logical_and(y<=samples[1],samples[1]<=y+dy))):
                mask[i,j] = 1
    '''
    for i,y in enumerate(X[:-1]):
        for j,x in enumerate(Y[:-1]):
            if np.any(np.logical_and(np.logical_and(x<samples[0],samples[0]<x+dx),np.logical_and(y<samples[1],samples[1]<y+dy))):
                mask[i,j] = 1

    return mask

def Manhattan(A, B):
    return abs(A[0] - B[0]) + abs(A[1] - B[1])

def Euclid(A,B):
    return  np.sqrt(abs(A[1] - B[1])**2 + abs(A[0] - B[0])**2)

def is_inside(s,l):
    len_s = len(s) 
    return any(s == l[i:len_s+i] for i in range(len(l) - len_s+1))

def removeDuplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]

def compliment(X, Y, Z):
    #cZ = joint_cdf(X, Y, Z)

    (xmin, xmax, ymin, ymax, nx, ny)= (X[0], X[-1],Y[0],Y[-1],len(X),len(Y))
    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    #Z_comp = dS/np.ones(Z.shape)
    Z_comp = 1 - Z

    Zintg = joint_Integral(X,Y,Z_comp)
    Z_comp = Z_comp/Zintg
    return np.flip(np.flip(Z_comp,axis=1),axis=0)


def either_side(x,val):
    l = np.where(x[1:]<val)[0]
    r = np.where(x[:-1]>val)[0]
    try:
        left = l[-1]+1
    except:
        left = 0
    try:
        right = r[0]
    except:
        right = len(r)-1
    return left, right

def acceptance(xp,yp,x,y):
    _xi, xi_ = either_side(x,xp)
    (_y, y_) = (y[_xi],y[xi_])

    y_threshold = np.interp(xp,[x[_xi],x[xi_]],[_y, y_])
    return yp < y_threshold

def rejection_sample(X,Y,Z,Nsamples,sampling_threshold=1):

    bounds = [[X[0],X[-1]],[np.min(Z),np.max(Z)]]
    keep_samples = []

    if sampling_threshold<1:
        dZ = np.max(Z) - np.min(Z)
        Z[Z<(np.min(Z)+(dZ*sampling_threshold))] = -1

    for i in range(len(Y)):
        test_samp = np.concatenate([[np.random.uniform(bounds[0][0],bounds[0][1],Nsamples)],
                                        [np.random.uniform(bounds[1][0],bounds[1][1],Nsamples)]],axis=0)
        idk = np.zeros(Nsamples)
        for j in range(Nsamples):
            idk[j] = acceptance(test_samp[0,j],test_samp[1,j],X,Z[i,:]) * 1
        dY = (Y[1]-Y[0])*.01
        idk = np.where(idk)[0]
        samps = np.concatenate([[test_samp[0,idk]],[Y[i]+np.random.uniform(-dY,dY/2,len(idk))]])

        keep_samples.append(samps)

    return np.concatenate(keep_samples,axis=1)
  
if __name__ == '__main__':

    """
    Workflow Demonstration
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'ANM_dam_data/processed_ANM 06-2021.xlsx')
    ANM = pd.read_excel(filename)
    conditions = (ANM['Tailings_Material'] == 'iron') & (ANM['Height'] > 40) & (ANM['Building_Method'] == 'upstream')

    candidate_dams = ANM[conditions]
    candidate_dams = candidate_dams.reset_index(drop=True)

    ## Site parameters
    siteLat = -20.122
    siteLong = -44.122
    mapRes = 30
    
    #np.argmin(abs(ANM['Lat']- siteLong))
    #np.argmin(abs(ANM['Long']- siteLong))
    
    #dam_data = ANM.iloc[226]

    #TODO: Specifying these limits needs to be done automatically!
    limits = [93358,97204,93256,97102] # [xmin,xmax,ymin,ymax] {metres})
    #dist_sigma = 1500 #standard deviation for the distance falloff

    ## Instantiate DEM object
    mapObj = demObject.fromCoords(siteLat,siteLong,mapRes)


    #conditions = (ANM['Tailings_Material']=='iron') & (ANM['Height']>40) & (ANM['Building_Method']=='upstream')
    #i = 2
    #candidate_dams = ANM[conditions]
    #candidate_dams = candidate_dams.reset_index(drop=True)
    dam_data = candidate_dams.iloc[i]
    dam_loc = [dam_data.Lat, dam_data.Long]
    dam_loc = [siteLat,siteLong]
    label = '{} - ({})'.format(dam_data.Dam_Name,dam_data.Company)
    delta_geo = .0005  
    
    LB = (dam_data.Lat-delta_geo,dam_data.Long-delta_geo)  # (siteLat-delta_geo,siteLong-delta_geo)#
    RT = (dam_data.Lat+delta_geo,dam_data.Long+delta_geo)  # (siteLat+delta_geo,siteLong+delta_geo)#

    def WTK(loc):
        return '{},{}'.format(round(loc[0],6),round(loc[1],6))

    ROI = [LB,(LB[0],RT[1]),RT,(RT[0],LB[1])]

    m = folium.Map(dam_loc, zoom_start=20)
    folium.Marker(location=dam_loc,popup=label).add_to(m)
    folium.vector_layers.Polygon(ROI,color='blue',weight=10).add_to(m)
    m

    ### POND MODEL ###
    pond = Pond(mapObj,limits)
    pond.Find_islands()
    pond.Compute_island_statistics([95000,95000])
    

    pond.plot()
    pond.plot_edge()

    pond.plot(source='all')
    pond.plot_score()

    pond.summary_island_data

    from data_connector import demObject
    """
    multiple_demonstration
    """
    conditions = (ANM['Tailings_Material']=='iron') & (ANM['Height']>40) & (ANM['Building_Method']=='upstream')
    
    candidate_dams = ANM[conditions]
    candidate_dams = candidate_dams.reset_index(drop=True)
    # 54,

    i = 0
    #i=54
    dam_data = candidate_dams.iloc[i]
    
    lat = dam_data['Lat']
    lon = dam_data['Long']

    print('DAM {} \n\t Lat : {} - Long : {}'.format(dam_data.Dam_Name,lat,lon))
    mapObj = demObject.fromCoords(lat,lon,30)
    mapObj.render(plt.figure(figsize=(10,10)))

    DAM = demObject(mapObj.getImg(),30)
    DAM.fromMapObj(mapObj)
    site = DAM.getSitePos()

    range_lim = 1500
    limits = [site[1]-range_lim,site[1]+range_lim,site[0]-range_lim,site[0]+range_lim]
    limits= [38000,70000,72000,104000]
    pond = Pond(mapObj,limits)
    pond.plot()
    pond.Edge_detection()
    pond.plot_edge(site)

    pond.Find_islands()
    pond.Compute_island_statistics([site[0],site[1]])
    pond.summary_island_data

    pond.plot(source='all')
    pond.plot_score()
        


    #extent = [self.Y[0], self.Y[-1],self.X[0], self.X[-1]]

    print('DAM {} \n\t Lat : {} - Long : {}'.format(dam_data.Dam_Name,lat,lon))
    extent = [mapObj.latRange[0],mapObj.latRange[1],mapObj.longRange[-1],mapObj.longRange[0]]

    plt.figure(figsize=(12,12))
    plt.imshow(np.flipud(mapObj.getImg()), extent=extent)
