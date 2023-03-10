import pandas as pd
import geopy.distance
import numpy as np
from datetime import date
from pathlib import Path

import sys


import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).parent.parent))
from google.cloud import bigquery
from source_data.earth_data import INITIALISE
from DAMS import ANM_DAMS, DAM

def closest_station(dam_loc, sta_loc):
    try:
        return geopy.distance.geodesic(dam_loc, sta_loc).km
    except:
        return np.nan


MINAS_EXTENT = {'lat': [-24.675, -13.222],
                'lon': [-54.731, -36.541]}

def standard_query(data_id, condition=None): return """
            SELECT *
            FROM `{0}` a
            {1}
            LIMIT
            50000
            """.format(data_id, condition)

loc_cond = "WHERE(a.latitude > {0} AND a.latitude < {1}) AND(a.longitude > {2} AND a.longitude < {3})".format(
    *(list(MINAS_EXTENT.values())[0] + list(MINAS_EXTENT.values())[1]))

# possible_stations.to_csv('stations_data_check.csv')

class Inventory():
    dataset_id = 'bigquery-public-data.ghcn_d'

    def __init__(self):
        self.client = bigquery.Client()
        self.__process_data_subsets__()

    def __process_data_subsets__(self):
        # Make an API request.
        tables = self.client.list_tables(self.dataset_id)
        sub_sets = {}
        for i, table in enumerate(tables):
            sub_sets[i] = {'id': table.dataset_id, 'table_id': table.table_id}

        self.sub_sets_df = pd.DataFrame.from_dict(sub_sets).T
        self.sub_sets_ids = self.dataset_id + \
            '.{}'.format(
                [i for i in self.sub_sets_df['table_id'] if 'inventory' in i][0])

    def get_inventory_ids(self):
        return self.sub_sets_ids

    def get_inventory_df(self):
        return self.sub_sets_df


class Weather_API(Inventory):
    def __init__(self, LOC=None, Lattitude=None, Longitude=None, N_stations=1, start=None, end=None):
        super().__init__()

        # assert all(Lattitude,Longitude) or Loc, 'Must provide a Lattitude and Longitude or a LOC'
        if not Lattitude == None:
            self.LOC = [Lattitude, Longitude]
        else:
            self.LOC = LOC

        self.start = start
        self.end = end
        self.N_stations = N_stations

        self.__date_update__()

    def __date_update__(self, default_start_date=2010, default_end_date=date.today().year):
        if self.start == None:
            self.start = default_start_date
        if self.end == None:
            self.end = default_end_date

    def update_inventory(self, loc_cond=loc_cond):
        #### HERE

        self.station_inv_string = standard_query(self.sub_sets_ids, loc_cond)
        ### Here
        self.inventory = (
            self.client.query(self.station_inv_string)
            .result()
            .to_dataframe(
                create_bqstorage_client=True,
            )
        )
        return self.inventory

    def post_process_station_inventory(self, *data_requirements, requirment_method='all'):

        assert requirment_method in [
            'all', 'any'], 'requirement_method must be all or any'

        if not data_requirements:
            data_reqs = ['PRCP', 'TAVG', 'TMAX', 'TMIN']
        else:
            if hasattr(data_requirements, len):
                #Assuming list :
                data_reqs = data_requirements[0]
            else:
                data_reqs = []
                for i, r in data_requirements.items():
                    data_reqs.append(r)

        self.inventory['distance'] = [closest_station(
            self.LOC,  l) for l in self.inventory[['latitude', 'longitude']].values]
        self.inventory = self.inventory.sort_values(
            'distance').reset_index(drop=True)

        filter_inventory = self.inventory[['id', 'element']]
        filter_inventory = filter_inventory.pivot(
            index="id", columns="element", values="element")

        possible_stations = filter_inventory.dropna(
            subset=data_reqs, how='any')

        for i in possible_stations.index:
            possible_stations.loc[i, ['latitude',	'longitude', 'firstyear', 'lastyear',	'distance']] = self.inventory[self.inventory['id'] == i][[
                'latitude',	'longitude', 'firstyear', 'lastyear',	'distance']].iloc[0, :]

        self.possible_stations = possible_stations.sort_values('distance')

    def station_date_check(self, start=None, end=None):
        if not start == None:
            self.start = start
        if not end == None:
            self.end = end

        self.possible_stations['date_check'] = np.nan
        for i in self.possible_stations.index:
            S, E = self.possible_stations.loc[i, ['firstyear',	'lastyear']]
            self.possible_stations.loc[i, 'date_check'] = (
                S <= self.start and E >= self.end)

    def get_possible_stations(self):
        return self.possible_stations

    def fetch(self, full_filter=True):
        if not hasattr(self, 'inventory'):
            self.update_inventory()

        #Assuming the general user wants to get all prcp and temp data
        self.post_process_station_inventory()
        self.station_date_check()
        if full_filter:
            stations = self.possible_stations.loc[self.possible_stations['date_check'], :]
            return stations
        return self.possible_stations


class Stations():
    def __init__(self):
        self.api = Weather_API()

    def nearby(self, LOC, start=None, end=None):
        self.api.start = start
        self.api.end = end
        self.api.LOC = LOC
        self.api.__date_update__()
        self.stations = self.api.fetch()

    def fetch(self, n_stations=1):
        self.station_ids = self.stations.index[:n_stations].values
        self.station_data = self.stations.loc[self.station_ids, :]

        DATA = {}
        for Sid in self.station_ids:
            data_store = []
            for date_id in np.arange(self.start, self.end + 1):
                station_cond = "WHERE a.id='''{0}'''".format(Sid)
                data_id = self.dataset_id + \
                    '.{}'.format(
                        [i for i in self.sub_sets_df['table_id'] if str(date_id) in i][0])
                data_string = standard_query(data_id, station_cond)
                data = (
                    self.client.query(data_string)
                    .result()
                    .to_dataframe(
                        create_bqstorage_client=True,
                    ))

                data["date"] = pd.to_datetime(data["date"])
                data = data.sort_values(by="date")
                filtered_data = data.pivot(
                    index="date", columns="element", values="value")
                
                # Convert 10s of a degree to degree
                cols = [c for c in filtered_data if 'T' in c]
                filtered_data[cols] = filtered_data[cols]/10

                data_store.append(filtered_data)
            DATA[Sid] = pd.concat(data_store)

        if len(DATA.keys()) == 1:
            return DATA[Sid]
        else:
            return DATA

    def __getattr__(self, name):
        try:
            try:
              return getattr(self.api, name)
            except:
                return self.station_data.loc[:,name]

        except AttributeError:
            raise AttributeError('%s not in availiable data' % name)



if __name__ == '__main__':
    n_stations=3

    dams = ANM_DAMS()
    i = 22
    # i += 1

    LOC = dams.dams_df.loc[i, ['Lat',	'Long']].values

    S = Stations()
    S.nearby(LOC)
    
    Data = S.fetch(n_stations=n_stations)

    
    fig, ax = plt.subplots(n_stations, 1, figsize=(20, n_stations*10))
    c = 0 
    for k, v in Data.items():
        v.plot(ax = ax[c])
        ax[c].set_title('id = {} - LOC = {} - Distance = {}km'.format(k, [S.latitude[k], S.longitude[k]], S.distance[k]))
        c += 1


    mpl.rcParams['figure.figsize'] = '16, 12'
    # m = Basemap(projection='kav7', lon_0=-90, resolution='l', area_thresh=1000.0)

    ll_lon, ur_lon, ll_lat, ur_lat = -60, -20, -30, 0
    m = Basemap(projection='merc', lon_0=0, resolution='l',
            llcrnrlon = ll_lon, urcrnrlon = ur_lon,
            llcrnrlat = ll_lat, urcrnrlat = ur_lat,
           )
    m.shadedrelief(scale=0.2)
    m.drawcoastlines()
    m.drawcountries()
    
    m.drawparallels(np.arange(-90., 99., 30.))
    junk = m.drawmeridians(np.arange(-180., 180., 60.))

    xd, yd = m(LOC[1], LOC[0])
    m.scatter(x, y, c='C1')

    x, y = m(S.longitude.values, S.longitude.values)
    m.scatter(x, y, c='C0')
