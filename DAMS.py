### General Imports ###

### General Imports ###

import os
import datetime
import json

import pathlib
import pandas as pd
import numpy as np
import math


class ANM_DAMS():
    """Class for storing the full ANM dam data set.
    ...
    Attributes
    ----------
    
    
    _full_dams_df : DataFrame
            This is the full public ANM data set, translated to english
    dams_df : DataFrame
            This is the reduced data set with several conditions applied
    Methods
    -------
    _read_data : This is a private method to load the dataset csv file
    data_labels : Returns the data labels.
    """

    def __init__(self):
        self._read_data()

    def _read_data(self, file_address=None):
        #home = pathlib.Path.home()  # os.path.dirname(__file__)
        #dirname = os.path.join(home, 'im_aware_collab')
        #filename = os.path.join(
        #    dirname, 'ANM_dam_data/processed_ANM 06-2021.xlsx')
        dirname = pathlib.Path(str(pathlib.Path.cwd()).split("SRC")[0])
        if file_address ==None:
            filename = os.path.join(
                dirname, 'SRC/IM-AWARE-GIS/ANM_dam_data/processed_ANM 06-2021.xlsx')
        else:
            filename = file_address
        filename = filename.replace('\\','/')
        ANM = pd.read_excel(filename, engine="openpyxl")
        #self._inclusion_conditions['Tailings_Material'] = 'iron'
        #self._inclusion_conditions = {}
        conditions = (ANM['Tailings_Material'] == 'iron') & (
            ANM['Height'] > 40) & (ANM['Building_Method'] == 'upstream')
        self._full_dams_df = ANM
        self.dams_df = self._full_dams_df[conditions].reset_index(
            drop=True)  # self.filter_dams()

    def data_labels(self):
        return self._full_dams_df.columns.values
        #def filter_dams(self):
        #self.dams_df = filter_dict(self._full_dams_df, self._inclusion_conditions)
        #return self.dams_df


class DAM(ANM_DAMS):
    def __init__(self, *args):
        super().__init__()
        if args:
            self._read_data()
            self.choose_dam(args[0])

    def choose_dam(self, *args):
        if args:
            for x in args:
                ind = x
            if isinstance(ind, int):
                self.dam_data = self.dams_df.iloc[ind]
                self.dam_idx_number = ind
                self.dam_uid = self.dam_data['ID']

            elif isinstance(ind, str):
                ind = np.where(self.dams_df.ID == ind)[0][0]
                self.dam_data = self.dams_df.iloc[ind]
                self.dam_idx_number = ind
                self.dam_uid = self.dam_data['ID']
            elif isinstance(ind,list):
                self.dam_data = self.dams_df.iloc[0]
                self.dam_data.Lat = ind[0]
                self.dam_data.long = ind[1]
        else:
            print('If you wish to choose a dam specify its name, or the indicie of \n\t{}'.format(
                self.dams_df[['Dam_Name', 'Company']]))
            self.dam_data = self.dams_df.iloc[0]

        self.loc = [self.dam_data.Lat, self.dam_data.Long]
        self.label = '{} - ({})'.format(self.dam_data.Dam_Name,
                                        self.dam_data.Company)

        print('Loading {} data'.format(self.label))

    def get_loc(self, longlat=False):
        if longlat:
            return [self.dam_data.Long, self.dam_data.Lat]
        return [self.dam_data.Lat, self.dam_data.Long]

    def get_risk_level(self):
        return self.dam_data.Risk

    def get_potential_damage(self):
        return self.dam_data.Potential_Damage

    def get_material(self):
        return self.dam_data.Tailings_Material

    def get_volume(self):
        return self.dam_data.Stored_Volume

    def get_emergency_level(self):
        return self.dam_data.Emergency_Level

    def get_map_label(self):
        return self.label


def measure(lat1, lon1, lat2, lon2):
    R = 6378.137  # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * \
        math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d * 1000
