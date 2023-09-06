"""
Usage: cd <fildeDirname>; python3 conveyor_tracking_dashboard.py

"""

import copy
from glob import glob
import dash_bootstrap_components as dbc
from datetime import date, datetime, timedelta
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from dash.dependencies import Input, Output, State
import io
import plotly.io as pio
scope = pio.kaleido.scope
from PIL import Image
import numpy as np
import pandas as pd
import os
import logging
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash
import dash_daq as daq
from dash.dash_table import DataTable

from consolidate_operator_data import consolidate_operator_states, load_detections, load_op_states
from config import dir_plots, dir_con_translations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# region custom globals

Nrois = 420
FPS = 0.5
y_carousel = 13.5
HEIGHT, WIDTH = Nrois, int(FPS*60*60)

cols_params = ['t_start', 't_end', 'speed', 'y_eff0', 'y_eff1', 'new_coil']
colors = ['red', 'yellow', 'orange']
colors_shifts = ['pink','orange','brown','magenta','red','green','yellow']
# history for undo
trajectories_state = []
done_trajs_state = []
params_state = []
starts_state = []
trajs_annot_state = []

# endregion

# region helper functions

def y2roi(y):
    return int(Nrois/231*y)
y2roi = np.vectorize(y2roi)

def roi2y(iroi):
    return iroi*231/Nrois
roi2y = np.vectorize(roi2y)


def get_time_data(nframe, init_time):
    # assuming global var FPS
    return init_time + timedelta(seconds=nframe/FPS)

def datetime2nfr(t):
    return (t.minute*60 + t.second)*FPS

# endregion

# region loading

def load_data(init_time):
    global trajectories_state, done_trajs_state, params_state, starts_state, trajs_annot_state
    # cutoff = 3600

    trajectories_state = []
    done_trajs_state = []
    params_state = []
    starts_state = []
    trajs_annot_state = []

    heatmap = load_detections(init_time)  # [:,:cutoff]
    h, w = heatmap.shape

    try:
        movement_start, movement_end, speeds = load_translations(init_time)

    except FileNotFoundError as e:
        print("Computing own translations..., Initial translations from operator screen loading")
        try:
            df, opfilesnot_found = load_op_states(init_time)
        except FileNotFoundError as e:
            print("Can't load the operator states")
            pass
            # TODO this error handling

        movement_start, movement_end = consolidate_operator_states(df, eps=10//FPS)

        #handle the difference in fps btw. op. data & vids
        FPSop = 1
        movement_start = (movement_start//(FPSop/FPS)).astype(int)
        movement_end = (movement_end//(FPSop/FPS)).astype(int)
        speeds, costs = [0.35]*len(movement_start), [0]*len(movement_start)
        

    y_eff_0 = np.zeros(len(movement_start))
    y_eff_1 = np.ones(len(movement_start))*231
    new_coil = np.ones(len(movement_start))

    # parameteres of the conveyor motions 
    params = [movement_start, movement_end, speeds, y_eff_0, y_eff_1, new_coil]
    
    starts = get_starts_beginning(heatmap, ylims = (y_carousel,100))
    coil_v_y = {init_time - timedelta(seconds=i): start for i, start in enumerate(starts)}

    times_starts = list(coil_v_y.keys())

    return heatmap, params, starts, times_starts

def load_translations(init_time):
    yyyy, mm, dd, HH = init_time.strftime('%Y-%m-%d-%H').split('-')
    path_trans = dir_con_translations + \
        f"{yyyy}_{mm}/{dd}/auto_con_translations_h{HH}.csv"
    df = pd.read_csv(path_trans)

    movement_start = df.nfr_start.values
    movement_end = df.nfr_end.values
    speeds = df.nfr_speed.values

    return movement_start, movement_end, speeds

# endregion

# region saving

def save_translations(params, init_time):

    df = pd.DataFrame()
    df['nfr_start'] = params[0]
    df['nfr_end'] = params[1]
    df['nfr_speed'] = params[2]
    df['speed'] = df.nfr_speed*FPS
    df['t_start'] = df.nfr_start.apply(lambda x: get_time_data(x, init_time))
    df['t_end'] = df.nfr_start.apply(lambda x: get_time_data(x, init_time))
    df['duration'] = (df.nfr_end - df.nfr_start)/FPS
    df['dy'] = df.speed * df.duration
    df['cost'] = 0

    subdir = dir_con_translations + init_time.strftime('%Y_%m/%d/')
    os.makedirs(subdir, exist_ok=True)
    HH = init_time.strftime('%H')
    fname = subdir + f'annot_con_translations_h{HH}.csv'

    cols = ['t_start', 't_end', 'dy', 'cost', 'speed',
            'duration', 'nfr_start', 'nfr_end', 'nfr_speed']
    df[cols].to_csv(fname,
                    date_format='%Y-%m-%d %H:%M:%S.%f',
                    float_format='%.3f',
                    index=False,
                    )
    return fname

def savefigure(figure, init_time, flag=''):
    outdir = dir_plots + init_time.strftime('%Y_%m/%d/')
    HH = init_time.strftime('%H')
    os.makedirs(outdir, exist_ok=True)

    tannot = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    fname = outdir + f'{HH}_{flag}{tannot}.jpg'

    pio.write_image(fig=figure, file=fname, format='png', width=1800, height=600)
    scope._shutdown_kaleido()

    return fname

# endregion

# region method

def get_starts_beginning(heatmap, ylims=(y_carousel, 231), prominence=0.15, height=0.20):
    """Find the starting points just from the first few frames of the heatmap.
    
    Parameters
    ----------
    heatmap : np.array
        (Nrois x Nfr)  Matrix of the conveyor detections.

    Returns
    -------
    start_inferred : np.array
        Array of the order of the default rois.
    """

    region_start = heatmap[:, :60]     # first 2 minutes of the hour
    scores_avg = region_start.mean(axis=1)
    smoothed = gaussian_filter1d(scores_avg, sigma=1)
    peaks = find_peaks(smoothed, height=height)[0]

    # keep in mind the indexing of scoring matrix here
    starts_inferred = Nrois - peaks
    # filter by y 
    starts_inferred = starts_inferred[
        (starts_inferred < y2roi(ylims[1])) & (starts_inferred > y2roi(ylims[0]))]

    return starts_inferred

def make_trajectory(start, movement_start, movement_end, speeds, y_eff0, y_eff1,
                    Nfr=7200, y0=y_carousel, ylim=100):
    """ Make a trajectory.

    Parameters
    ----------
    start : int
        Starting frame

    movement_start : np.array
        (N,) array of frames of when the global shift starts.
    
    movement_end : np.array
        (N,) Array of frames of when the global shift ends.
    
    speeds : np.array 
        (N,) Array of factors for the time difference btw. start & end. 
    
    y_eff0 : np.array
        (N,) array of values defining the left limit of the y range for which the shift is applied.

    y_eff1 : np.array
        (N,) array of values defining the right limit of the y range for which the shift is applied.
    
    Nfr : int 
        Number of frames 

    y0 : float 
        Value of y at the starting frame.
    
    ylim : float 
        Limit of y range.

    Returns
    --------    
    traj : (Nfr,) np.array
    """

    traj = np.ones(Nfr)*y0

    traj_t0s = movement_start[movement_start >= start]
    Ntrajmoves = len(traj_t0s)
    traj_t1s = movement_end[-Ntrajmoves:]
    traj_speeds = speeds[-Ntrajmoves:]
    y_eff0 = y_eff0[-Ntrajmoves:]
    y_eff1 = y_eff1[-Ntrajmoves:]

    y = y0
    for nmove in range(Ntrajmoves):
        
        y_eff0now = y_eff0[nmove]
        y_eff1now = y_eff1[nmove]

        t0now = traj_t0s[nmove]
        t1now = traj_t1s[nmove]
        speednow = traj_speeds[nmove]
        #move part 
        if y_eff0now < y < y_eff1now:
            moved =  y + speednow*np.arange(t1now-t0now)
            traj[t0now:t1now] = np.where(moved > y_eff1now, y_eff1now, moved)
            y =  moved[-1]
        else:
            traj[t0now:t1now] = y

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

# endregion

# region helper functions for callbacks

def get_trajectories(params, starts_left, done_trajs= None, trajectories = None):
        
    if done_trajs is None:
        movement_start, movement_end, speeds, y_eff0, y_eff1, new_coil = params
        trajs = []

        for start_l in starts_left:
            traj = make_trajectory(
                0, movement_start, movement_end, speeds, y_eff0, y_eff1, y0=roi2y(start_l), Nfr=WIDTH)
            trajs.append(traj)
        
        for start in movement_start:
            traj = make_trajectory(start, movement_start, movement_end, speeds, y_eff0, y_eff1, y0=y_carousel, Nfr=WIDTH)
            trajs.append(traj)
    
    else: 
        movement_start, movement_end, speeds, y_eff0, y_eff1, new_coil = params
        trajs = []
        N_starts_l = len(starts_left)

        for i in range(N_starts_l):
            if done_trajs[i] == 0:
                start_l = starts_left[i]
                traj = make_trajectory(
                    0, movement_start, movement_end, speeds, y_eff0, y_eff1, y0=roi2y(start_l), Nfr=WIDTH)  
            else: 
                traj = trajectories[i]
            trajs.append(traj)
        
        for i in range(len(params[0])):
            if done_trajs[N_starts_l+i] == 0:
                if new_coil[i] == 1:
                    start = movement_start[i]
                    traj = make_trajectory(start, movement_start, movement_end, speeds, y_eff0, y_eff1, y0=y_carousel, Nfr=WIDTH)
                else:
                    traj = np.zeros(WIDTH)
            else:
                traj = trajectories[N_starts_l + i]
            trajs.append(traj)

    return trajs


def get_trajs_roi(trajs):
    trajs_roi = [y2roi(traj)  for traj in trajs]
    return trajs_roi


def params_from_transdata(data, params_prev):
    Ntrans = len(data)
    dates_start = [datetime.strptime(data[i]['t_start'], '%H:%M:%S') for i in range(Ntrans)]
    dates_end = [datetime.strptime(data[i]['t_end'], '%H:%M:%S') for i in range(Ntrans)]
    movement_start = np.array([datetime2nfr(ds) for ds in dates_start]).astype(int)
    movement_end = np.array([datetime2nfr(de) for de in dates_end]).astype(int)

    sorted_idx = np.argsort(movement_start)
    movement_start = movement_start[sorted_idx]
    movement_end = movement_end[sorted_idx]

    speeds = np.array([float(data[i]['speed'])/FPS for i in sorted_idx])
    y_eff0 = np.array([float(data[i]['y_eff0']) for i in sorted_idx])
    y_eff1 = np.array([float(data[i]['y_eff1']) for i in sorted_idx])
    new_coil = np.array([float(data[i]['new_coil']) for i in sorted_idx])

    params = [movement_start, movement_end, speeds, y_eff0, y_eff1, new_coil]

    if len(params_prev[0]) > len(movement_start):
        missed_start = sorted(set(params_prev[0]) - set(movement_start))[0]
        missed_idx = np.argwhere(params_prev[0] == missed_start)[0,0]
        return params, missed_idx
    
    elif len(params_prev[0]) < len(movement_start):
        return params, -1
    else:
        return params, None


def get_starts_from_start_data(data, start_prev):
    starts = []
    for d in data:
        inp = d['start']
        try:
            start = int(y2roi(float(inp)))
            starts.append(start)
        except:
            print(f'start {inp} cannot be parsed into roi')
            pass
    
    if len(start_prev) > len(starts):
        missed_start = sorted(set(start_prev) - set(starts))[0]
        missed_idx = np.argwhere(np.array(start_prev) == missed_start)[0,0]
        return starts, missed_idx
    
    elif  len(start_prev) < len(starts):
        return starts, -1

    else:
        return starts, None


def params_2_metric_data(params, init_time):
    data = []
    m_start, m_end, speeds, y_eff0, y_eff1, new_coil = params
    for s, e, v, y0, y1, n_c in zip(m_start, m_end,speeds, y_eff0, y_eff1, new_coil):
        entry = {
            't_start' : get_time_data(s, init_time).strftime('%H:%M:%S'),
            't_end' : get_time_data(e, init_time).strftime('%H:%M:%S'),
            'speed' :  v*FPS,
            'y_eff0' : y0,
            'y_eff1' : y1,
            'new_coil' : n_c, 

        }
        data.append(entry)
    return data


def fetch_new_configuration(global_time):
    global heatmap, params, starts, times_starts
    global HEIGHT, WIDTH, offset
    global layout
    global done_trajs, trajectories, trajs_annot
    global x_v, y_v, x_t, y_t
    global trajectories_state, done_trajs_state, params_state, starts_state
    global trajectories_prop

    heatmap, params, starts, times_starts = load_data(global_time)

    HEIGHT, WIDTH = heatmap.shape
    offset = WIDTH
    layout = get_layout(heatmap)
    (x_v, x_t), (y_v, y_t) = get_ticks(heatmap)

    # starts = get_starts_beginning(heatmap)
    trajectories = get_trajectories(params, starts)
    done_trajs = np.zeros(len(trajectories))
    trajs_annot = {}

    trajectories_state.append(copy.deepcopy(trajectories))
    done_trajs_state.append(copy.deepcopy(done_trajs))
    starts_state.append(copy.deepcopy(starts))
    params_state.append(copy.deepcopy(params))
    trajs_annot_state.append(copy.deepcopy(trajs_annot))

    start_data = [{'start': roi2y(i)} for i in starts]
    params_data = params_2_metric_data(params, global_time)

    fig = get_figure_layout_only(layout)
    
    fig.update_layout(xaxis_range=[0,WIDTH])
    fig.update_layout(yaxis_range=[0,HEIGHT])
    
    return fig, "NOT YET SAVED", params_data, start_data

# endregion

# region figures and plotting routines

def get_ticks(heatmap):

    h, w = heatmap.shape
    
    y_v = np.arange(h)[::70]
    x_v = np.arange(w)[::int(FPS*200)]
    dates = [global_time + timedelta(seconds=i/FPS) for i in x_v]
    x_t = np.array([d.strftime('%H:%M') for d in dates])
    y_t = roi2y(y_v).astype(int)
    return (x_v, x_t), (y_v, y_t)

def get_layout(heatmap):

    heatmap[-20:,:] = 0
    fig = go.Figure(data=go.Heatmap(
        z=np.flip(heatmap, axis=0)))
    fig.update_layout(yaxis_range=[0, heatmap.shape[0]])
    fig.update_layout(xaxis_range=[0, heatmap.shape[1]])
    # get rid of all the stuff around the main figure
    fig.update_traces(dict(showscale=False,
                           coloraxis=None,
                           colorscale='viridis'), selector={'type': 'heatmap'})
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    # get rid of the blank white space around
    fig.update_layout(
        autosize=False,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )

    img_bytes = pio.to_image(fig,
                             format='png', width=heatmap.shape[1], height=heatmap.shape[0])
    img = Image.open(io.BytesIO(img_bytes))
    # img.save("heamtap.png")

    layout = dict(
        source=img,
        xref="x",
        yref="y",
        x=0,
        y=heatmap.shape[0],
        sizex=heatmap.shape[1],
        sizey=heatmap.shape[0],
        sizing="stretch",
        opacity=1,
        layer="below")

    return layout

def get_figure_layout_only(layout):

    fig = go.Figure()
    fig.update_layout(
        title=dict(text=f"{global_time}", font=dict(size=30))
    )
        
    fig.update_yaxes(
    tickmode="array",
    tickvals=y_v,
    ticktext=y_t
    )

    fig.update_xaxes(
    tickmode="array",
    tickvals=x_v,
    ticktext=x_t,
    )

    fig.update_layout(showlegend=False)
    fig.add_layout_image(layout)
    
    return fig

def hover_info_shift(frames):
    dates = [global_time + timedelta(seconds=j/FPS) for j in frames]
    text_t = [d.strftime("%H:%M:%S") for d in dates]
    return text_t


def get_figure(layout, params, trajs_y, done_trajs):
    
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=f"{global_time}", font=dict(size=30))
    )

    trajs_roi = get_trajs_roi(trajs_y)
    for i, traj_roi in enumerate(trajs_roi):
        color = colors[i % 3]
        linetype = ['dash', None][int(done_trajs[i])]

        line_dict = dict(width=3, dash=linetype, color=color)

        x_f = np.arange(len(traj_roi))[::int(20*FPS)]
        y_f = traj_roi[::int(20*FPS)]

        fig.add_trace(go.Scatter(
            x=x_f[y_f!=0],
            y=y_f[y_f!=0],
            hovertext = hover_text(i, y_f, x_f),
            hoverinfo="text",
            mode='lines',
            opacity=0.98,
            # marker=dict(size=5, color=color),
            line=line_dict
            ))
        
        fig.update_yaxes(
        tickmode="array",
        tickvals=y_v,
        ticktext=y_t
        )

        fig.update_xaxes(
        tickmode="array",
        tickvals=x_v,
        ticktext=x_t,
        )

    for i, (m_s, m_e) in enumerate(zip(params[0], params[1])):
        fig.add_trace(go.Scatter(
            y = [5,5], 
            x = [m_s, m_e],
            line = dict(width=4, color= colors_shifts[i%len(colors_shifts)]),
            hoverinfo="text",
            hovertext=hover_info_shift([m_s, m_e])
        ))

    fig.update_layout(showlegend=False)
    fig.add_layout_image(layout)

    return fig

def get_figure_prop(layout, params, trajs_prop):
    
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=f"{global_time}", font=dict(size=30))
    )

    for id, traj_roi in trajs_prop.items():
        color = colors[id%3]
        linetype = 'dash'

        # mark lines that end up with a pick-up
        if traj_roi[-1, 1] < WIDTH * 0.95:
            last_point_marker = dict(size=10, symbol='circle-open', color=color)
        else:
            last_point_marker = None

        line_dict = dict(width=3, dash=linetype, color=color)

        fig.add_trace(go.Scatter(
            x= traj_roi[:,1],
            y= Nrois - traj_roi[:,0],
            hovertext = hover_text(id, Nrois - traj_roi[:,0], traj_roi[:,1]),
            hoverinfo="text",
            mode='lines',
            opacity=0.98,
            # marker=dict(size=5, color=color),
            line=line_dict
            ))

        if last_point_marker:
            last_point = go.Scatter(
                x=[traj_roi[-1, 1]],
                y=[Nrois - traj_roi[-1, 0]],
                mode='markers',
                marker=last_point_marker,
                hovertemplate='Suggested pickup at x: %{x}<br>y: %{y}<extra></extra>'
            )
            fig.add_trace(last_point)
        
        fig.update_yaxes(
        tickmode="array",
        tickvals=y_v,
        ticktext=y_t
        )

        fig.update_xaxes(
        tickmode="array",
        tickvals=x_v,
        ticktext=x_t,
        )

    for i, (m_s, m_e) in enumerate(zip(params[0], params[1])):
        fig.add_trace(go.Scatter(
            y = [5,5], 
            x = [m_s, m_e],
            line = dict(width=4, color= colors_shifts[i%len(colors_shifts)]),
            hoverinfo="text",
            hovertext=hover_info_shift([m_s, m_e])
        ))

    fig.update_layout(showlegend=False)
    fig.add_layout_image(layout)

    return fig

def hover_text(coil, y, t):
    
    y_real = roi2y(y)
    dates = [global_time + timedelta(seconds=j/FPS) for j in t]
    text_t = [d.strftime("%H:%M:%S") for d in dates]
    text_part = [f'<b>Coil ID</b> : {coil}<br><b>t</b> : {t}<br><b>y</b> : {y:.2f}<br>'
                for t, y in zip(text_t,y_real)]
    return text_part

# endregion 

# region layout

app = dash.Dash(__name__, external_scripts=[
                "assets/script.js"], external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "conveyor tracking trajectories fix"
app.layout = html.Div([
    html.Button('Previous', id='previous-page', n_clicks=0),
    html.Button('Next', id='next-page', n_clicks=0),
    html.Button('Save', id='save', n_clicks=0),
    html.Button('Edit region', id='edit-region', n_clicks=0),
    html.Button('Annotate pickups', id='edit-ends', n_clicks=0),
    html.Button('Undo (Z)', id='undo-state', n_clicks=0),
    dcc.Slider(0, 80, 4,
               value=0,
               id='slider_stops'
               ),
    html.Div([
        dcc.DatePickerSingle(
            id='date',
            display_format='YYYY MM DD',
            date=date(2023,8,15), #date.today(),
            style={'marginRight': '10px'}),
        dcc.Input(id='hour', placeholder="0"),
        html.Button('Load', id='load_button', n_clicks=0),

        html.Div(children=[
            html.Span("You need to load something")
            ], 
        id='text_save'),

        html.Div(
            children=[daq.ToggleSwitch(
                    id='toggle-show-trajs',
                    label='Show trajs',
                    labelPosition='right',
                    value=False)
                ],
            style={'width': '20%', 'textAlign': 'center'}
            ),

        ],style={'display': 'flex'}
        )
        
,
    html.Div(),
    html.Div(children=[
        html.Div([
            DataTable(
                id="tbl_start",
                columns=([{'id': 'start', 'name': 'start'}]),
                data=[],
                editable=True,
                row_deletable=True),
            html.Button('Add Row', id='editing-rows-starts', n_clicks=0),
            ],
           
            style={'width': '5%', 'textAlign': 'center'}),

        html.Div([
            dcc.Graph(id='graph_working',
                    figure=go.Figure(data=go.Heatmap(z=np.eye(10))),
                    style=dict(height=f"{HEIGHT*2.5}px")
                    ),
            ],
            style={'width': '75%', 'display': 'inline-block'}),

        html.Div([
            DataTable(
                id="tbl_params",
                columns=([{'id': p, 'name': p} for p in cols_params]),
                data=[],
                editable=True,
                row_deletable=True),
            html.Button('Add Row', id='editing-rows-params', n_clicks=0),
            ],
            style={'width': '20%', 'textAlign': 'center'}),

        ],style={'display': 'flex'}
    ),
])

# endregion layout

# region app callbacks

@app.callback(
    [Output('graph_working', 'figure'),
     Output('text_save', 'children'),
     Output('tbl_params', 'data'),
     Output('tbl_start', 'data')
     ],
    [Input('next-page', 'n_clicks'),
     Input('previous-page', 'n_clicks'),
     Input('edit-region', 'n_clicks'),
     Input('slider_stops', 'value'),
     Input('edit-ends', 'n_clicks'),
     Input('graph_working', 'clickData'),
     Input('graph_working', 'relayoutData'),

     Input('save', 'n_clicks'),
     Input("load_button", "n_clicks"),

     Input('tbl_params', 'data'),
     Input('tbl_params', 'columns'),

     Input('tbl_start', 'data'),
     Input('tbl_start', 'columns'),
     Input('undo-state', 'n_clicks'),

     Input('editing-rows-params', 'n_clicks'),
     Input('editing-rows-starts', 'n_clicks'),

     Input('toggle-show-trajs', 'value'),
     ],
    [State('date', 'date'),
     State('hour', 'value'),
     State('graph_working', 'figure')]
)
def click_to_graph(next, previous, edit_region_clck, slider_value, edit_ends_clck, clickData, relayoutData,
                   save, load,
                   params_data, params_cols,
                   start_data, start_cols, undo_clck,
                   add_rows_params_clck, add_rows_starts_clck,
                   toggle_show_trajs,
                   date_value, hour, graph_state,
                   ):

    global HEIGHT, WIDTH
    global global_time, layout, heatmap
    global params, starts, trajectories, trajs_annot, done_trajs, times_starts
    global x_v, y_v, x_t, y_t
    global trajectories_state, done_trajs_state, params_state, starts_state

    triggerId = dash.callback_context.triggered[0]['prop_id']

    if "next" in triggerId:
        global_time += timedelta(hours=1)
        return fetch_new_configuration(global_time)


    if "previous" in triggerId:
        global_time -= timedelta(hours=1)
        return fetch_new_configuration(global_time)

    if "graph_working.clickData" == triggerId:
        curvenumber = clickData['points'][0]['curveNumber']
        end = clickData['points'][0]['x']
        traj = trajectories[curvenumber]
        start = max(0, np.argwhere(traj == traj[traj > y_carousel][0])[0, 0]-1)

        if done_trajs[curvenumber] == 0:
            traj[end+1:] = 0
            trajs_annot[curvenumber] = {
                'traj': traj[start:end+1],
                'start_frame': start,
                't_start': None,
                'end_frame': end,
            }
            done_trajs[curvenumber] = 1
        else:
            trajs_annot.pop(curvenumber, None)
            done_trajs[curvenumber] = 0

        trajectories_state.append(copy.deepcopy(trajectories))
        done_trajs_state.append(copy.deepcopy(done_trajs))
        trajs_annot_state.append(copy.deepcopy(trajs_annot))

        fig = get_figure(layout, params, trajectories, done_trajs)
        fig.update_layout(xaxis_range=graph_state['layout']['xaxis']['range'])
        fig.update_layout(yaxis_range=graph_state['layout']['yaxis']['range'])
        return fig, dash.no_update, dash.no_update, dash.no_update

            
    if "save" in triggerId:
        pass
        #TODO

    elif "load_button" in triggerId:
        global_time = datetime.strptime(
            date_value, '%Y-%m-%d') + timedelta(hours=int(0 if hour is None else hour))
        return fetch_new_configuration(global_time)


    if "tbl_params" in triggerId:
        #TODO reorder 
        params, missed_idx = params_from_transdata(params_data, params_prev=params)
        if missed_idx is not None:
            if missed_idx != -1:
                idx_glob = len(starts) + missed_idx
                done_trajs = done_trajs[np.arange(len(done_trajs)) != idx_glob]
                trajectories.pop(len(starts) + missed_idx)
            else:
                done_trajs = np.append(done_trajs,[0])

        trajectories = get_trajectories(params, starts, done_trajs, trajectories)
        print(f'** \n Missed idx: {missed_idx}, len(trajs) :{len(trajectories)}, len(done): {len(done_trajs)} \n')
        if len(trajectories) != len(done_trajs):
            print('ERROR')

        trajectories_state.append(copy.deepcopy(trajectories))
        done_trajs_state.append(copy.deepcopy(done_trajs))
        params_state.append(copy.deepcopy(params))

        fig = get_figure(layout, params, trajectories, done_trajs)
        fig.update_layout(xaxis_range=graph_state['layout']['xaxis']['range'])
        fig.update_layout(yaxis_range=graph_state['layout']['yaxis']['range'])

        return fig, dash.no_update, dash.no_update, dash.no_update


    if "tbl_start" in triggerId:
        starts, missed_idx = get_starts_from_start_data(start_data, start_prev=starts)
        if missed_idx is not None:
            if missed_idx != -1:
                done_trajs = done_trajs[np.arange(len(done_trajs)) != missed_idx]
                trajectories.pop(missed_idx)
                times_starts.pop(missed_idx)
            else: 
                # insert into corect order 
                done_trajs_now = np.zeros(len(done_trajs) + 1)
                if len(done_trajs) != 0:
                    paste = np.zeros(len(done_trajs)+1)
                    paste[:len(done_trajs)] = done_trajs
                    done_trajs_now[:len(starts)] = paste[:len(starts)]
                    done_trajs_now[len(starts)+1:] = done_trajs[len(starts):]
                done_trajs = done_trajs_now

                times_starts.append(get_carousel_time_lvl2(global_time, roi2y(starts[-1])))
                # trajectories.insert(np.zeros(WIDTH), len(starts))

        trajectories = get_trajectories(params, starts, done_trajs, trajectories)

        print(f'** \n Missed idx: {missed_idx}, len(trajs) :{len(trajectories)}, len(done): {len(done_trajs)} \n')
        if len(trajectories) != len(done_trajs):
            print('ERROR')

        trajectories_state.append(copy.deepcopy(trajectories))
        done_trajs_state.append(copy.deepcopy(done_trajs))
        starts_state.append(copy.deepcopy(starts))

        fig = get_figure(layout, params, trajectories, done_trajs)
        fig.update_layout(xaxis_range=graph_state['layout']['xaxis']['range'])
        fig.update_layout(yaxis_range=graph_state['layout']['yaxis']['range'])
        return fig, dash.no_update,  dash.no_update, dash.no_update
    
     
    if 'editing-rows-params' in triggerId:
        params_data.append({c['id']: '' for c in params_cols})
        return dash.no_update, dash.no_update, params_data, dash.no_update


    if 'editing-rows-starts' in triggerId:
        start_data.append({c['id']: '' for c in start_cols})
        return dash.no_update, dash.no_update, dash.no_update, start_data
    

    if 'toggle-show-trajs' in triggerId:
        if toggle_show_trajs:
            fig = get_figure(layout, params, trajectories, done_trajs)
        else:
            fig = get_figure_layout_only(layout)
        fig.update_layout(xaxis_range=graph_state['layout']['xaxis']['range'])
        fig.update_layout(yaxis_range=graph_state['layout']['yaxis']['range'])
        return fig, dash.no_update, dash.no_update, dash.no_update


    if "undo-state" in triggerId:
        if len(trajectories_state) > 1:
            trajectories_state.pop()
            trajectories = trajectories_state[-1]
        
        if len(done_trajs_state) > 1:
            done_trajs_state.pop()
            done_trajs = done_trajs_state[-1]
        
        if len(params_state) > 1:
            params_state.pop()
            params = params_state[-1]
        
        if len(starts_state) > 1:
            starts_state.pop()
            starts = starts_state[-1]
        
        if len(trajs_annot_state) > 1:
            trajs_annot_state.pop()
            trajs_annot = trajs_annot_state[-1]

        figure = get_figure(layout, params, trajectories, done_trajs)

        start_data = [{'start': roi2y(i)} for i in starts]
        params_data = params_2_metric_data(params, global_time)

        figure.update_layout(xaxis_range=graph_state['layout']['xaxis']['range'])
        figure.update_layout(yaxis_range=graph_state['layout']['yaxis']['range'])
        
        return figure, dash.no_update, params_data, start_data


    return dash.no_update

# endregion

def run_dashboard():
    app.run_server(debug=False, host="0.0.0.0")


if __name__ == '__main__':
    run_dashboard()
