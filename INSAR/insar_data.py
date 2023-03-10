
from pathlib import Path
from pykalman import KalmanFilter

from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import TruncatedSVD
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import ConvexHull
from scipy.stats import norm
from scipy.stats import multivariate_normal

import math

import numpy as np
import pandas as pd
import os
import sys
import dill 

import time

from shapely.geometry import Point
from shapely.geometry import Polygon

#obj = INSAR_DATA('Rock0','D:\Work\IM_AWARE\im_aware_collab\IMAWARE\insar_working_dir\INSAR_RESULTS')
class INSAR_DATA:
    '''
    Container class for processed INSAR data.
    Provides analysis tools for Kalman filtration, singular value decomposition
    and fast fourier transformation.
    '''
    _fileLabels = ['vert','corr','cum_disp','disp_rate','point_loc']

    def __init__(self,damID,insarFolder):
        '''
        damID = ID of the dam corresponding to its primary key in the IMAWARE database
        insarFolder = folder to retrieve data from
        '''
        self.data = {}
        damFolder = Path(insarFolder).joinpath(damID)
        fileList = os.listdir(damFolder)

        for fileName in fileList:
            try:
                route_to_data = damFolder.joinpath(fileName)
                fileData = pd.read_csv(route_to_data,index_col=0)
            except:
                continue
            
            for label in self._fileLabels:
                if label in fileName and fileName[0]!='.':
                    self.data[label] = fileData

        self.__post_process_data()

    def __getattr__(self,name):
        try:
            if name in list(self.data.keys()):
                return self.data[name]
            #else:
            #    for k,v in self.data.items():
            #        if name in v.columns:
            #            return v.loc[name]

        except AttributeError:
            raise AttributeError('%s not in data dictionary' % name)

    def __post_process_data(self):
        '''
        Processing done after object initialisation
        '''
        self.Start_Date = self.__getattr__('vert').index
        self.End_Date = self.__getattr__('vert')['End_Date'].values
        self.data['vert'] = self.data['vert'].drop(columns='End_Date')
        self.data['corr'] = self.data['corr'].drop(columns='End_Date')
        
        
    def get_date_range(self):
        return [self.Start_Date[0], self.End_Date[-1]]

    def kalman(self,variableName,*ism):
        '''
        Applies a Kalman filter to insar data of the given variable name.
        NOTE: what does "ism" need to be?
        TODO: return something
        '''
        data = self.__getattr__(variableName)
        if ism: ism = ism[0]
        else: ism = np.zeros(len(data.columns))

        kf = KalmanFilter(initial_state_mean = ism, n_dim_obs=len(data.columns) )
        out_em = kf.em(data,em_vars='all')
        smoothed_state_means, smoothed_state_covariances = out_em.smooth(data)
        stds = []
        for each in smoothed_state_covariances:
            stds.append(np.sqrt(np.diag(each)))
        return smoothed_state_means, smoothed_state_covariances, stds

    def svd(self,variableName):
        data = self.__getattr__(variableName).iloc[:,:n]
        crr = data.corr()
        evects,evals, evect2 = np.linalg.svd(crr)
        evals_sum = np.sum(evals)
        evals_ratio = evals*(1/evals_sum)
        evals_ratio = np.cumsum(evals_ratio)
        out = pd.DataFrame({'var_ratio':evals_ratio, 'eigval':evals})
        nevals = len(out)
        out_eigvects = evects[:,:nevals]
        return {'eigenvals' : out, 'eigvects' : out_eigvects}
        pass

    def fft(self, variableName):
        ''' WRITE ME'''
        data = self.__getattr__(variableName)
        ft = np.fft.ifftshift(data.values)
        ft = np.fft.fft2(ft)
        ft = np.fft.fftshift(ft)
        return ft

    def fit_gaussian_regressor(self,variableName, length_scale):
        data = self.__getattr__(variableName)
        store = {}
        for i in data.index:
            g0 = GKR(self.point_loc[['Lat', 'Long']].values,
                    data.loc[i, :], 
                    length_scale)
            store[i] = g0
        setattr(self, 'GKR_{}'.format(variableName), store)

    def __gen_mask(self,variableName, n_interp=20):
        
        XI, YI, xi, yi, mask = gen_mask(
            self.point_loc[['Lat', 'Long']].values, n_interp = n_interp)
        # setattr(self, 'Mask_{}'.format(variableName), mask)
        # setattr(self, 'Matrix_LOC_{}'.format(variableName), [XI, YI])
        return XI, YI, mask


    def get_matrix(self, variableName, n_interp=20):
        """
        Returns the Gaussian Kernal Regression smoothed matrix

        Output: np.ndarray
                [Lat, Lon, Var]
        """
        self.XI, self.YI, self.mask = self.__gen_mask(variableName,n_interp=n_interp)
        
        data = self.__getattr__(variableName)
        out = np.zeros([n_interp, n_interp, len(data.index)])
        
        gkn_obj = getattr(self, 'GKR_{}'.format(variableName))

        toolbar_width = 100
        sys.stdout.write("Processing kernel smoothing: {}\n[{}]".format(variableName, " " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (1))  # return to start of line, after '['
        prog_lim = np.array_split(np.arange(len(data)), toolbar_width)
        if len(data) < toolbar_width:
            print('ERROR: need to fix toolbar for iterating through an array less than len 100')

        prog_lim = np.array([pl.max() for pl in prog_lim])
        prog = 0
        progress_c = 0

        for i, idd in enumerate(data.index):
            out[:, :, i] = interp(gkn_obj[idd], self.mask, self.XI, self.YI)
            n_k = np.where(i == prog_lim)[0]
            if n_k.size > 0:
                progress_c += 1
                sys.stdout.write(
                    '\r'+'[{}] - {}%'.format(progress_c*'#', progress_c))
                sys.stdout.flush()


        sys.stdout.write(
            '\n\tComplete.\n')

        self.smoothed_data = out
        return out

    def get_GKR_statistic(self, statistic):
        A = getattr(np, '{}'.format(statistic))(self.smoothed_data, axis=0)
        return getattr(np, '{}'.format(statistic))(A, axis=0)

    def keys(self):
        return self.__dict__.keys()


    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return dill.load(f)



def gen_mask(coords, n_interp=100):
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_interp)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_interp)

    XI, YI = np.meshgrid(xi, yi)
    x, y = XI.flatten(), YI.flatten()
    points = np.vstack((x, y)).T

    hull = ConvexHull(coords)
    polygon = hull.points[hull.vertices]
    polygon = Polygon(polygon)

    idm = np.zeros(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        idm[i] = polygon.contains(Point(*p))
    mask = np.reshape(idm, np.shape(XI))

    return XI, YI, xi, yi, mask


def interp(GKN_obj, mask, XI, YI):
    xi = np.reshape(XI, np.size(XI))
    yi = np.reshape(YI, np.size(YI))
    Z_out = GKN_obj.predict(np.vstack([xi, yi]).T)
    # f = NearestNDInterpolator(coords, values)
    # #f = interp2d(coords[:, 0], coords[:, 1], values,kind='nearest')
    # Z = f(xi, yi)
    Z_out = np.reshape(Z_out, np.shape(XI))
    Z_out[~mask] = np.nan
    return Z_out


def generate_matrix(coords, var_df, n_interp=100):
    XI, YI, xi, yi, mask = gen_mask(coords, n_interp=n_interp)
    out = np.zeros([n_interp, n_interp, len(var_df.index)])
    for i, idv in enumerate(var_df.index):
        out[:, :, i] = interp(
            coords, var_df.loc[idv, :].values, mask, xi, yi)
    return XI, YI, out


class GKR:
    def __init__(self, x, y, b):
        self.x = np.array(x)
        self.y = np.array(y)
        self.b = b

    '''Implement the Gaussian Kernel'''

    def gaussian_kernel(self, z):
        return (1/np.sqrt(2*np.pi))*np.exp(-0.5*z**2)

    '''Calculate weights and return prediction'''

    def _predict(self, X):
        kernels = np.array([self.gaussian_kernel(
            (np.linalg.norm(xi-X))/self.b) for xi in self.x])
        weights = np.array([len(self.x) * (kernel/np.sum(kernels))
                           for kernel in kernels])
        return np.dot(weights.T, self.y)/len(self.x)

    def predict(self,X):
        Z = np.zeros([X.shape[0]])
        for i in range(X.shape[0]):
            Z[i] = self._predict(X[i,:])
        return Z


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.interpolate import interp2d
    from matplotlib import animation
    from mpl_toolkits import mplot3d
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    """
    Generate an INSAR data object
    """
    PATH = str(Path(__file__).parent.parent.parent.parent.joinpath('IMAWARE').joinpath('INSAR_RESULTS'))
    #obj = INSAR_DATA('Rock0', '/Volumes/WORK/IMAWARE/im_aware_collab/INSAR_RESULTS')
    obj = INSAR_DATA('Bru_fail', str(PATH))
    
    try:
        os.mkdir(PATH)
    except:
        print('file exists')
    """
    Choose a length scale for the Gaussian kernel regressor
    """
    b = np.sqrt(np.sum(np.diff(obj.point_loc[['Lat', 'Long']].values[:2,:],axis=0)**2))
    
    """
    Fit the Gaussian kernel regressor
    """
    obj.fit_gaussian_regressor('vert',b)

    """
    Compute the smoothed-upsampled mesh array 
    """
    N_interp = 50
    Z_out = obj.get_matrix('vert',  n_interp = N_interp)

    """
    Compute some extra stuff for plotting 
    """
    XI, YI, xi, yi, mask = gen_mask(obj.point_loc[['Lat', 'Long']].values, n_interp=N_interp)
    x_t = np.reshape(XI, np.size(XI))
    y_t = np.reshape(YI, np.size(YI))
    

    """
    Choose an interesting time period
    """
    time_interval = 0
    time_interval += 1

    """
    Plot the interesting time point raw data points and the smoothed interpolated Gaussian weighted regression
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Scene dates ({} - {})'.format(obj.Start_Date[time_interval], obj.End_Date[time_interval]),fontsize=20)
    cm = ax[0].scatter(obj.point_loc['Lat'].values, obj.point_loc['Long'].values,
                  c=obj.vert.iloc[time_interval, :], s=1000, cmap='hot')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cm, cax=cax, orientation='vertical')

    cm = ax[1].contourf(XI, YI, Z_out[:,:,time_interval], cmap='hot')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cm, cax=cax, orientation='vertical')
    # plt.savefig(PATH+'/vert_plot-dates ({} - {}).png'.format(obj.Start_Date[time_interval], obj.End_Date[time_interval]))
    """
    Plot the 2 on 3D axes 
    """

    fig, ax = plt.subplots(1, figsize=( 10, 10))
    fig.suptitle('Scene dates ({} - {})'.format(obj.Start_Date[time_interval], obj.End_Date[time_interval]),fontsize=20)
    ax = plt.axes(projection='3d')
    ax.contour3D(XI, YI, Z_out[:,:,time_interval], 1000, cmap='hot')
    ax.scatter(obj.point_loc['Lat'].values, obj.point_loc['Long'].values,
                  obj.vert.iloc[time_interval, :], s=200)
    # plt.savefig(PATH+'/test_1.html3d_plot-dates ({} - {}).png'.format(obj.Start_Date[time_interval], obj.End_Date[time_interval]))
    """
    Plot full time series 
    """
    from IPython.display import clear_output
    plt.figure()
    cm0 = plt.imshow(Z_out[:, :, 0], cmap='jet')
    
    for i in range(173,len(obj.Start_Date)):
        cm = plt.imshow(Z_out[:, :, i], cmap='jet')
        plt.colorbar(cm)
        plt.title('Dates ({} - {})'.format(obj.Start_Date[i], obj.End_Date[i]))
        plt.savefig(
            PATH+'/Dates-{}-{}.png'.format(obj.Start_Date[i], obj.End_Date[i]))
        plt.draw()
        plt.pause(.5)
        clear_output(wait=True)


    import imageio
    # PATH= '/Volumes/WORK/IMAWARE/im_aware_collab/SRC/IM-AWARE-GIS/working_directory/dynamic_gif_2/'
    with imageio.get_writer(PATH+'/' + 'brufail_gif.gif', mode='I') as writer:
        for filename in os.listdir(PATH):
            image = imageio.imread(PATH+'/'+filename)
            time.sleep(.5)
            writer.append_data(image)
    
    # Z_out = Z[:,:,time_interval]
    # # = g0.predict(np.vstack([x_t,y_t]).T)
    # # Z_out = np.reshape(Z_out, np.shape(XI))
    # # Z_out[~mask] = np.nan

    # fig = plt.figure(figsize=(20,20))
    # cm = 

    # fig = plt.figure(figsize=(20,20))
    # cm = plt.contourf(XI,YI,Z_out)
    # plt.colorbar(cm)

    # fig = plt.figure(figsize=(20,20))
    # ax = plt.axes(projection='3d')
    # ax.contour3D(XI,YI,Z_out, 1000, cmap='hot')
    # ax.scatter(obj.point_loc['Lat'].values,obj.point_loc['Long'].values,obj.vert.iloc[time_interval, :], s=200)



    
    
    
    # plt.colorbar(cm)

    # ax.view_init(30, 35)
    # fig

    # ax.view_init(30, 75)
    # fig


    # coords0 = obj.point_loc[['Lat', 'Long']].values
    # coords1 = obj2.point_loc[['Lat', 'Long']].values

    # XI0, YI0, out0 = generate_matrix(coords0, obj.vert, n_interp = 100)
    # XI1, YI1, out1= generate_matrix(coords1, obj2.vert, n_interp = 100)
    
    # plt.figure(figsize=(20,20))
    # plt.plot(np.max(out0[20:-20,20:-20],axis=0))

    

    # fig, ax = plt.subplots(figsize=(10, 10))
    # im = plt.imshow(out0[:,:,0], cmap='jet', vmin=-20, vmax=20) # vmax=np.max(out0[~np.isnan(out0)])) vmin=np.min(out0[~np.isnan(out0)]),

    # def init():
    #     im.set_data(out0[:,:,0])

    # def animate(i):
    #     im.set_data(out0[:,:,i])
    #     return im

    # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=174,
    #                             interval=1)
    
    
    # plt.ion()
    
    # for i in range(331):
        
    #     try:
    #         ax[0].imshow(out0[:,:,i])
    #     except:
    #         print(None)
        
    #     try:
    #         ax[1].imshow(out1[:,:,i])
    #     except:
    #         print(None)
    #     # drawing updated values
    #     plt.draw()
    
    #     # This will run the GUI event
    #     # loop until all UI events
    #     # currently waiting have been processed
    #     time.sleep(0.1)
        
    # ft0 = obj.fft('vert')
    # ft1 = obj2.fft('vert')
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(abs(ft0))
    # plt.figure(figsize=(10,10))
    # plt.imshow(abs(ft1))

    # 
    # 

    # X, Y, Z = generate_matrix(coords, obj.vert, n_interp = 100)

    
    # Z[np.isnan(Z)] = 0
    # u, s, vh = np.linalg.svd(Z)
    # plt.imshow(s)

    # generate_matrix(coords, values, n_interp=100)

    
    

    # svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    # svd.fit(Z)
    # U, V = 


    # plt.scatter(obj.point_loc['Lat'],
    #             obj.point_loc['Long'])
    # plt.plot(hull.points[hull.vertices,0],hull.points[hull.vertices,1])



    # xi = np.linspace(obj.point_loc['Lat'].min(),obj.point_loc['Lat'].max(),1000)
    # yi = np.linspace(obj.point_loc['Long'].min(),obj.point_loc['Long'].max(),1000)
    # XI, YI = np.meshgrid(xi,yi)

    # S = 0 
    
    # Z = f(xi,yi)

    # # plt.figure(figsize=(10,10))
    # # ck = plt.contourf(XI, YI, Z)
    # # plt.colorbar(ck)
    # S += 1
    # plt.figure(figsize=(10,10))
    # ck = plt.scatter(obj.point_loc['Lat'],
    #             obj.point_loc['Long'], c = obj.vert.iloc[S, :], s=400)
    # plt.colorbar(ck)

    # plt.figure(figsize=(10,10))
    # plt.plot(obj.vert)
