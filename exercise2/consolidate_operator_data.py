"""
    Usage: consolidate_operator_data.py <date>

    Author: EK

"""


from docopt import docopt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

# deploymentpath = '/home/nb-2/edo/Data/ussk/a_consolidation/'
from config import *
dir_plots = 'conveyor_birdseye_plots/'

FPS = 0.5
# if 'compressed' in dir_detections:
#     FPS = 0.5
Nfr = int(FPS*60*60)
Nrois = 420


def roi2y(iroi):
    return (Nrois - iroi)*231/Nrois

def y2roi(y):
    return Nrois - int(Nrois/231*y) 

y2roi = np.vectorize(y2roi)

def cluster_1d(points, eps):
    """Simple 1D clusters assuming gaps btw data"""
    

    if len(points) == 0:
        return None 

    else:
        clusters = []

        #can omit if the input is already sorted 
        points_sorted = sorted(points)

        curr_point = points_sorted[0]
        curr_cluster = [curr_point]
        for point in points_sorted[1:]:
            if point <= curr_point + eps:
                curr_cluster.append(point)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)
        
        return clusters
    
def movement_stamps(movement, eps=5, membershipsizethreshold=5):
    """Calc. repr. timestamp for clusters timestamps sign. motion". """
    # start clustering algo 1D
    clusters = cluster_1d(movement, eps)

    if clusters is None:
        return np.array([]),np.array([])

    else:    
        movement_end = np.array([max(cluster) for cluster in clusters if len(cluster) > membershipsizethreshold ])
        movement_start = np.array([min(cluster) for cluster in clusters if len(cluster) > membershipsizethreshold ]) 

        return  movement_start, movement_end

def consolidate_operator_states(df, eps=5):

    moving_frames = df.loc[df.state != 0, 'nfr'].values
    movement_start, movement_end = movement_stamps(moving_frames, eps=eps)

    return movement_start, movement_end

def load_detections(t_hour, minute_delta = 1):
    
    Nfr1 = int(minute_delta*60*FPS) #1 minute
    tnow = t_hour
    mat = np.zeros((Nrois, Nfr)) # 1 hour prefilling
    for i in range(60//minute_delta):
        fnamedet = dir_detections + tnow.strftime('%Y_%m/%d/score_mat_stationary_%H-%M.npy')
        dets = np.flip(np.load(fnamedet), axis=0)
        mat[:,i*Nfr1:(i+1)*Nfr1] = dets[:,:Nfr1]
        tnow += timedelta(minutes=minute_delta)
    
    return mat

def load_op_states(t_hour, minute_delta=1):

    opfilesnotfound = []
    FPSop = 1 #fixed after 1/4/23
    tnow = t_hour
    Nfr1 = minute_delta*60*FPSop
    df = pd.DataFrame()
    for i in range(60//minute_delta):
        fnameop = dir_op_state + tnow.strftime('%Y_%m/%d/%H_%M_%S.csv')
        try:
            dff = pd.read_csv(fnameop,header=None, usecols=[0,1,4], names=['t','nfr','state'])

        except FileNotFoundError: 
            opfilesnotfound.append(fnameop)
            dff = pd.DataFrame(columns = ['t','nfr','state'])

        dff['nfr'] = i*Nfr1 + dff.nfr
        df = pd.concat((df, dff))

        tnow += timedelta(minutes=minute_delta)
    return df, opfilesnotfound

def discretise_state(movement_start, movement_end, Nfr = 7200):

    states = np.zeros(Nfr)

    # TODO find the first one
    state = 0
    for i in range(Nfr):
        
        if i in movement_start:
            state = 1
        elif i in movement_end:
            state = 0

        states[i] = state

    return states

def make_trajectory(start, movement_start, movement_end, speeds, Nfr=7200, y0=8.25, ylim=231):
    traj = np.ones(Nfr)*y0

    traj_t0s = movement_start[movement_start >= start]
    Ntrajmoves = len(traj_t0s)
    traj_t1s = movement_end[-Ntrajmoves:]
    traj_speeds = speeds[-Ntrajmoves:]
    
    y = y0
    for nmove in range(Ntrajmoves):

        t0now = traj_t0s[nmove]
        t1now = traj_t1s[nmove]
        speednow = traj_speeds[nmove]
        #move part 
        traj[t0now:t1now] = y + speednow*np.arange(t1now-t0now)
        y =  min(y + speednow*(t1now-t0now), ylim)
        if y > ylim:
            traj[t1now:] = ylim
            break
        
        if nmove < Ntrajmoves - 1:
            t_next = traj_t0s[nmove+1]
        else:
            t_next = Nfr
        #stationary part
        traj[t1now:t_next] = y

    return traj


def phi(y_0, speed, t ):
    """Calculate phi at certain t given a y_level and a set of parameters we like to find. """

    return y_0 + speed*t

def cost_point(t, y, levels, speed):
    """ Calculate cost for a single point given a space of potential function evaluations.
    
        Input:  t, y - properties of a point
                y_levels_cam (pd.DataFrame) y_levels serving as comparison baseline for specific cam 
                speed - float
        Output: smallest distance to a y_level given the paramaters        
        """
    
    best_cost = 1000
    
    for level in levels:

        y_potential = phi(level,speed,t)
        
        cost = abs(y - y_potential)
        #decide on which level the potential y should be 
        if cost < best_cost:
            best_cost = cost 
            best_y = y_potential

    # print('IN', y, 'POT', best_y, 'cost', best_cost)    
    return best_cost 

def cost_of_region(speed,points,levels):
    """Calculate the cost (measure of fit) per region.
    Input: 
        speed : float
        points : (Nrois, Nfr) np.array
        levels : (M,) np.array
                         
    Output: cost_region: float
    """

    cost_region = 0
    for point in points:
        cost_region += cost_point(point[0], point[1], levels, speed)
    
    return cost_region
