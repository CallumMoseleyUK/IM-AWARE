import altair as alt
import matplotlib.pyplot as plt
import os 
from pathlib import Path 
import numpy as np
#from source_data.GCPdata import *


def save_fig(plotter,end_folder,end_filename_wext ):
    if plotter == 'plt':
       call_route = Path.cwd()
       os.chdir(end_folder)
       plt.savefig('{}.png'.format(end_filename_wext),transparent = True)
       os.chdir(call_route)

def triangular_contour_for_map(x, y, z):
    fig1, ax1 = plt.subplots()
    tcf = ax1.tricontourf(x,y,z, levels = 10, cmap = 'Reds', alpha = 0.5)
    #cmap = tcf.get_cmap()
    #inds = np.linspace(min(z),max(z),10)
    #cols = [cmap(i) for i in inds]
    return tcf #{'cols':cols,'inds':inds}

def bounds_and_mid_plot(panda,bounds_labels,ax_labels):

    x_axis = alt.Axis(title = ax_labels['x'])
    y_axis = alt.Axis(title = ax_labels['y'])

    line = alt.Chart(panda).mark_line().encode(
        x = alt.X(bounds_labels['x'], axis = x_axis),
        y = alt.Y(bounds_labels['y_mid'], axis = y_axis),
        tooltip = [bounds_labels['x'], bounds_labels['y_mid']]
        ).interactive()
    
    band = alt.Chart(panda).mark_area(opacity = 0.5).encode(
       x = alt.X(bounds_labels['x']),
       y = alt.Y(bounds_labels['y_low']),
       y2 = alt.Y2(bounds_labels['y_up']),
       tooltip =[bounds_labels['x'], bounds_labels['y_low'], bounds_labels['y_up']]).interactive()

    return band+line

def band_plot(panda,bounds_labels,ax_labels):

    x_axis = alt.Axis(title = ax_labels['x'])
    y_axis = alt.Axis(title = ax_labels['y'])
    
    band = alt.Chart(panda).mark_area(opacity = 0.5).encode(
       x = alt.X(bounds_labels['x'], axis = x_axis),
       y = alt.Y(bounds_labels['y_low'], axis = y_axis),
       y2 = alt.Y2(bounds_labels['y_up']),
       tooltip = [bounds_labels['x'],bounds_labels['y_low'], bounds_labels['y_up']] ).interactive()
       
    return band

def simple_line_plot(panda,var_names,ax_labels):
    
    x_axis = alt.Axis(title = ax_labels['x'])
    y_axis = alt.Axis(title = ax_labels['y'])

    tpc = [var_names['x'], var_names['y']]
    line = alt.Chart(panda).mark_line().encode(
 
        x = alt.X(var_names['x'], axis = x_axis),
        y = alt.Y(var_names['y'], axis = y_axis),
        color = var_names['colour'], 
        tooltip = tpc,
        ).interactive()
        
    return line


        

def step_plot(panda,var_names,ax_labels):

    x_axis = alt.Axis(title = ax_labels['x'])
    y_axis = alt.Axis(title = ax_labels['y'])

    tpc = [var_names['x'], var_names['y']]

    if 'other' in var_names.keys():
        for each in var_names['other']:
            tpc.append(each)

    if 'colour' in var_names.keys():
        colour = alt.Color(var_names['colour'], scale = alt.Scale(scheme='tableau20'))    
        
    else:
        colour = alt.value('red')

    line = alt.Chart(panda).mark_rect().encode(
 
        x = alt.X(var_names['x'], axis = x_axis),
        y = alt.Y(var_names['y'], axis = y_axis),
        color = colour, 
        tooltip = tpc,
        ).interactive()
    
    return line


def basic_kde(panda,var, ax_label):

    x_axis = alt.Axis(title = ax_label)

    return alt.Chart(panda).transform_density(var, as_ = [var, 'Empirical PDF']).mark_area().encode(
        x = alt.X('{}:Q'.format(var), axis = x_axis), y = 'Empirical PDF:Q')

def basic_scatter(panda,vars,ax_label):

    panda = panda.rename(columns={'zvert': vars['x'], 'zcorr' : vars['y']})
    x_axis = alt.Axis(title = ax_label['x'])
    y_axis = alt.Axis(title = ax_label['y'])

    figure = alt.Chart(panda).mark_circle(size=40).encode(
        x = alt.X(vars['x'],axis = x_axis),
        y = alt.Y(vars['y'],axis = y_axis),
        tooltip = [vars['x'],vars['y']]
        ).interactive()
    return figure
