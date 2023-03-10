import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


from DAMS import ANM_DAMS, DAM
# from GCPdata import GCP_IO
from INSAR.insar_data import INSAR_DATA

import meteostat as mts 
# from meteostat import Stations, Daily, Hourly

from source_data.weather_api import Stations

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt



home_path = str(Path(__file__).parent.parent.parent.parent)


def week_label(date):
    return 'W-{}-Y-{}'.format(date.strftime("%V"), date.year)

def month_label(date):
    return 'M-{}-Y-{}'.format(date.month, date.year)

def quarter_label(date):
    if date.month <= 3:
        quarter = 1
    elif date.month >3 and date.month <= 6:
        quarter = 2
    elif date.month >6 and date.month <= 9:
        quarter = 3
    else:
        quarter = 4
    # 
    return 'Q-{}-Y-{}'.format(quarter, date.year)

def year_label(date):
    return 'Y-{}'.format(date.year)

def start_year_split(dates):
    tmp = [datetime.strptime(i, '%Y-%m-%d') for i in dates]
    for i, d in enumerate(tmp):
        if d.month == 1:
            idF = i
            break
    return dates[:idF], dates[idF:]
    
def convert2date(dates):
    return [datetime.strptime(i, '%Y-%m-%d') for i in dates]

def split_data(dates, period, units):
    if period == 'Weekly':
        def f(d): return int(d.strftime("%V"))
    elif period == 'Monthly':
        def f(d): return d.month   
    elif period == 'Quarterly':
        def f(d): 
            return pd.Timestamp(d).quarter
    elif period == 'Annually':
        def f(d): return d.year
    return [f(d) for d in dates]
    


"""

FROM HERE TOMORROW!!!! 




"""

def split_array(array, i):
    return np.arange(i), array[i:]

def group_adjacent(array):
    l_a = len(array)
    groups = []
    start = 0
    while len(array)>0:
        for i, a in enumerate(array):
            if i == 0:
                continue
            if a != array[i-1]:
                A, array = split_array(array, i)
                groups.append(start + A)
                start = start + len(A)
                break
            if i == len(array)-1:
                groups.append(np.arange(start, l_a))
                array = []
    return groups
            

class Processing_Methods():
    """ Class with methods for dealing with Precipitation and Temperature data
    
    """

    period_options = {'Weekly' : {'key': 'Weekly', 'period': 7, 
                                  'other_labels': [7, 'W', 'w', 'week', 'Week'],
                                  'gen': week_label,
                                  'units': 51 },
                      'Monthly' : {'key':  'Monthly', 'period': 30, 
                                   'other_labels': [30, 'M', 'm', 'month', 'Month'],
                                   'gen': month_label,
                                   'units': 12 },
                      'Quarterly': {'key': 'Quarterly', 'period': int(356/4), 
                                   'other_labels': ['q', 'Q', 'quarter', 'Quarter'],
                                   'gen': quarter_label,
                                    'units': 4},
                      'Annually': {'key':  'Annually', 'period': 356, 
                                    'other_labels': ['a', 'A', 'Annual', 'y', 'Y', 'Year'], 
                                    'gen': year_label, 
                                    'units': 1}}
    def __init__(self, label='PRCP'):
        self.col_label = label
        self.prune_data(label)

    def prune_data(self, column):
        """
        Removes unneccessary data in the data frame
        """
        # if len(self.data.columns)>1:
        self.data = self.data[column]
        # self.data = self.data.dropna()

    def process_data(self, window):
        """
        Used to process the data into a specified set of periods 
        i.e weekly, monthly, quarterly, annually
        """
        self.__split_time_series__(window)
        return self.period_df

    def __check_period_string__(self, period):
        """
        Allows the user to specify one of the 4 periods in a variety of different ways.

        i.e. 
        Annual can be choosen by passing any of the following strings 
                                ['a', 'A', 'Annual', 'y', 'Y', 'Year']
        """
        if period in list(self.period_options.keys()):
            self.period = period
            self._period_opt = self.period_options[self.period]
            return self.period_options[period]['period']
        else:
            for k, v in self.period_options.items():
                for p in v['other_labels']:
                    if period == p:
                        self.period = k
                        self._period_opt = self.period_options[self.period]
                        return self.period_options[k]['period']
            missing_option(period, self.period_options)

    def __split_time_series__(self, window):
        """
        Private method used to split the data into whatever period choosen by the user.
        """
        iDU = split_data(
            self.data.index, self._period_opt['key'], self._period_opt['units'])

        g = group_adjacent(iDU)

        self.n_periods = len(g) #int(np.ceil(len(self.data.values)/window))
        # split data into periods 
        period_data = [self.data.values[idX] for idX in g]
        self._period_dates = [self.data.index[idX] for idX in g]
        
        # gen period mid date
        self._period_mid_date = [i[int(len(i)/2)] for i in self._period_dates]
        # gen period labels
        self._period_labels = self.__gen_period_label__(self._period_mid_date, self.period)
        # gen df
        self.period_df = pd.DataFrame(period_data, index = self._period_labels)
        # Fill row na with mean 
        row_mean = np.nanmean(self.period_df.values, axis=1)
        inds = np.where(np.isnan(self.period_df.values))
        self.period_df.values[inds] = np.take(row_mean, inds[0])

        missing_index = pd.isnull(
            self.period_df).all(1)
        if any(missing_index):
            print('WARNING: You have periods filled with NaN, filling them with the averages\n\t of the same period in the rest of the time series.')
            missing_index = self.period_df.index[missing_index]
            data_df = self.period_df.copy(deep=True)
            for i in missing_index:
                data_df = replace_missing_period_data(data_df, i)
            self.period_df = data_df

    
    def __gen_period_label__(self, dates, period):
        """
        generates period labels. 
        i.e 1 Jan - 7 Jan = week 1
        or  1 Jan - 30 March = quarter 1
        """

        lab = []
        for d in dates:
            lab.append(self.period_options[period]['gen'](d))
        return lab

    def get_period_label(self, date):
        """
        Get the period label associated with any date choosen by the user
        """

        if date in self._period_mid_date:
            return self._period_labels[self._period_mid_date == date]
        else:
            for i, dl in enumerate(self._period_dates):
                if date >= dl[0] and date < dl[-1]:
                    return self._period_labels[i]
    
    def get_period_date_range(self, date):
        """
        Get the date range for any period (choosen by date) choosen by the user.
        """

        label = self.get_period_label()
        idL = self._period_labels == label
        return [self._period_dates[idL][0], self._period_dates[idL][-1]]

    def get_rolling(self, window, statistic, fillmethod= 'bfill'):
        """
        Get any daily rolling statistic choosen by the user  
        statistic = mean, min, max, ect.
        window must be an integer
        """
        return getattr(getattr(self.data.rolling(window), statistic)(),fillmethod)()

    def get_stat(self, statistic):
        """
        Get any statistic of the full daily time serie
        """
        return getattr(np, statistic)(self.data.values)
    
    def __period_stat__(self, statistic):
        return pd.DataFrame(getattr(np, statistic)(self.period_df.values, axis=1), 
                index=self._period_labels, 
                columns=['{}({})'.format(statistic, self.col_label)])
    
    def get_period_stat(self,statistic):
        """
        Get any statistic for each of the periods choosen by the user.

        You can pass a list of statistics and the method will return a data frame
        with all of those statistics
        """

        if isinstance(statistic, list):
            return pd.concat([self.__period_stat__(s) for s in statistic], axis=1)
        else:
            return self.__period_stat__(statistic)

    # def savgol(self, window, polynomial):

    # def distribution(self, period):
    #     p = self.__check_period_string__(period)

        
def missing_option(period, period_options):
    raise Exception('{} not in options. Choose from {}'.format(period,list(period_options.keys()) ))


def replace_missing_period_data(data_df, missing_index):
    I = [int(s) for s in missing_index.split('-') if s.isdigit()][0]

    idR = [i for i in data_df.index if '-{}-'.format(I) in i]

    # Fill col na with mean
    col_mean = np.nanmean(data_df.loc[idR, :])
    data_df.loc[missing_index, :] = col_mean
    return data_df

class Temperature(Processing_Methods):
    """
    Class holding temperature data and statisitics methods
    """
    def __init__(self, data_df, period, label = 'TAVG'):
        self.raw_data = data_df.copy(deep=True)
        self.data = data_df
        super().__init__(label=label)
        duration = self.__check_period_string__(period)
        self.process_data(duration)
    
    def get_extremes(self):
        """
        By default the class works with average daily temperatures.
        By running this method you can access all the same tools for the 
        high and low extremes too. 
        """

        self.high = Temperature(self.raw_data, self.period, label='TMAX')
        duration = self.high.__check_period_string__(self.period)
        # self.high.prune_data('tmax')
        self.high.process_data(duration)
        self.low = Temperature(self.raw_data, self.period, label='TMIN')
        duration = self.low.__check_period_string__(self.period)
        # self.low.prune_data('tmin')
        self.low.process_data(duration)
        

class Precipitation(Processing_Methods):
    """
    Class holding precipitation data and statisitics methods
    """
    def __init__(self, data_df, period):
        self.data = data_df
        super().__init__(label = 'PRCP')
        duration = self.__check_period_string__(period)
        self.process_data(duration)
        

class External_Data(DAM):
    """
    Class used by users to process weather data for dam sites. 

    We assume the closest weather station to the dam site is sufficient. 
    #TODO in the future we can generalise to handle model methods too. 
    """
    data_types = {'temp':{'options' : ['T', 't', 'temp', 'Temp', 'TEMP', 'Temperature']},
                  'prcp': {'options': ['P', 'p', 'prcp', 'Prcp','PRCP', 'Precipitation']}}

    def __init__(self, damID, *date_range):
        """
        initialise the class by passing a damID. 

        By default the class gathers data for the full range of dates from the INSAR 
        database. Instead the user can pass a different range if desiered.  
        """
        super().__init__(damID)
        if date_range:
            self.set_date_range(date_range[0])
        else:
            self.__insar_date_range()

    def __insar_date_range(self, insarFolder=home_path+'/INSAR_RESULTS'):
        """
        Private method to get INSAR date range.
        """
        #TODO : rebuild to work with the GCP handler to pull files directly from there 
        ins_obj = INSAR_DATA(self.dam_uid, insarFolder)
        date_strings = ins_obj.get_date_range()
        self.insar_dates = [ins_obj.Start_Date, ins_obj.End_Date]
        self.set_date_range(date_strings)
    
    def set_date_range(self, date_range):
        """
        Method used to set the date range in the correct format. Allows user to pass 
        hyphonated string dates or datetime objects.
        """

        if isinstance(date_range[0], str):
            self.Date_Range = [datetime.strptime(i, '%Y-%m-%d') for i in date_range]
        elif isinstance(date_range[0], datetime):
            self.Date_Range = date_range
        else:
            print('ERROR: must pass dates as [datetime, datetime] or [str(Y-m-d), str(Y-m-d)]')

    def get_daily_data(self, n_stations=1):
        """
        Method uses BigQuery Weather_ApI module to download precipiation and temperature data for 
        the closest stations to the dam site 
        """
        stations = Stations()
        stations.nearby(self.get_loc())
        
        # self.station = stations.fetch(n_stations)

        Data = stations.fetch(n_stations=n_stations)
        
        self.raw_data = Data
        # data = Daily(
        #     self.station.index[-1], start=self.Date_Range[0], end=self.Date_Range[1])
        # self.raw_data = data.fetch()
        # self.raw_data.index = [i.to_pydatetime().date() for i in self.raw_data.index]
        # self.poor_station_check()
        return Data

    def get_hourly_data(self, n_stations=1):
        """
        Method uses metostat module to download precipiation and temperature data for 
        the closest stations to the dam site 
        """

        print('Warning! BigQuery doesnt have hourly data, Reverting to use Metostat!\n This means you may be using a weather station very far from the target!')
        stations = mts.Stations()
        stations.nearby(*self.get_loc())
        
        self.station = stations.fetch(n_stations)
        
        data = mts.Hourly(
            self.station.index[-1], start=self.Date_Range[0], end=self.Date_Range[1])
        self.raw_data_hourly = data.fetch()
        # self.raw_data.index = [i.to_pydatetime().date() for i in self.raw_data.index]
        # self.poor_station_check()

    def poor_station_check(self, data_proportion = .7):
        choice = 1
        while len(self.raw_data.index) < data_proportion*len(self.insar_dates[0]):
            print('WARNING: Poor data quality from station {}. Moving to next'.format(choice))
            choice += 1
            self.get_daily_data(n_stations=choice)


    def gen_temperature_data(self, period = 'Weekly'):
        """ Method to process the temperature data for the dam site. """
        self.temp = Temperature(self.raw_data, period)
        self._period_opt = self.prcp._period_opt

    def gen_preciptiation_data(self, period = 'Weekly'):
        """ Method to process the precipitation data for the dam site. """
        self.prcp = Precipitation(self.raw_data, period)
        self._period_opt = self.prcp._period_opt

    def get_data(self, data_type):
        """ Method to get the temperature of the precipitation classes """
        for k, v in self.data_types.items():
            if data_type in v['options']:
                return getattr(self, k)
            
    def __repr__(self):
        s = 70 * '=' + '\n'
        s0 = [s]
        s0.append('External Data:\n\n')
        s0.append('\t Dam ID: {} \n\t LOC: {}'.format(self.dam_uid, self.loc))
        s0.append('\n\t Date Range: {}'.format([i.strftime('%Y-%m-%d') for i in self.Date_Range]))
        if hasattr(self, '_period_opt'):
            s0.append('\n\t Period: {} \n\t Days per period {}'.format(
                self._period_opt['key'], self._period_opt['period']))

        s0.append('\n\n\tClass Contains:')
        s0.append('\n\t\t - Temperature Data : {}'.format(hasattr(self, 'TEMP')))
        s0.append('\n\t\t - Precipitation Data : {}\n'.format(hasattr(self, 'PRCP')))
        s0.append(s)
        return ''.join(s0)

        

if __name__ == '__main__':

    B = External_Data('Rock0')
    B.get_daily_data()
    B.gen_preciptiation_data(period='Quarterly')
    B.gen_temperature_data(period='Q')

    series_max = B.prcp.get_stat('max')
    
    v_df = B.prcp.get_period_stat('var')
    m_df = B.prcp.get_period_stat('mean')
    stats_df = B.prcp.get_period_stat(['min','mean','max','var'])

    rolling_avg_precipitation = B.prcp.get_rolling(3, 'mean')

    rolling_avg_temp = B.temp.get_rolling(3, 'mean')

    Temperature_data = B.get_data('temp')
