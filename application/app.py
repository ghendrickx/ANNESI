"""
Online application to implement the internally trained neural network.

Author: Gijs G. Hendrickx
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc
from dash import html
import dash.dependencies as dep

from application.components import AppEstuary, input_check, EstuaryType
from machine_learning.neural_network import NeuralNetwork

# initiate dash-app
APP = dash.Dash(__name__)

# neural network
NN = NeuralNetwork()
NN.output = NN.get_output_vars()

# data processing
_UNITS_VARS = {
    'tidal_range': 'm',
    'surge_level': 'm',
    'river_discharge': 'm3 s-1',
    'channel_depth': 'm',
    'channel_width': 'm',
    'channel_friction': 's m-1/3',
    'convergence': 'm-1',
    'flat_depth_ratio': '-',
    'flat_width': 'm',
    'flat_friction': 's m-1/3',
    'bottom_curvature': 'm-1',
    'meander_amplitude': 'm',
    'meander_length': 'm',
}
_INPUT_VARS = list(_UNITS_VARS.keys())
_SLIDER_DEFAULT_SETTINGS = {
    'updatemode': 'drag',
    'tooltip': {
        'placement': 'top',
        'always_visible': True,
    },
}
_SLIDER_SETTINGS = {
    'tidal_range': {
        'min': 1,
        'max': 5,
        'step': .1,
        'value': 3,
    },
    'surge_level': {
        'min': 0,
        'max': 2,
        'step': .1,
        'value': 0,
    },
    'river_discharge': {
        'min': 100,
        'max': 16000,
        'step': 100,
        'value': 1000,
    },
    'channel_depth': {
        'min': 5,
        'max': 25,
        'step': .1,
        'value': 10,
    },
    'channel_width': {
        'min': 500,
        'max': 3000,
        'step': 100,
        'value': 1000,
    },
    'channel_friction': {
        'min': .02,
        'max': .05,
        'step': .001,
        'value': .023,
    },
    'convergence': {
        'min': 2.5e-5,
        'max': 4e-4,
        'step': 1e-6,
        'value': 1e-4,
    },
    'flat_depth_ratio': {
        'min': -1,
        'max': 1,
        'step': .1,
        'value': 0,
    },
    'flat_width': {
        'min': 0,
        'max': 3000,
        'step': 100,
        'value': 0,
    },
    'flat_friction': {
        'min': .02,
        'max': .05,
        'step': .001,
        'value': .023,
    },
    'bottom_curvature': {
        'min': 0,
        'max': 6e-5,
        'step': 1e-6,
        'value': 0,
    },
    'meander_amplitude': {
        'min': 0,
        'max': 10000,
        'step': 10,
        'value': 0,
    },
    'meander_length': {
        'min': 0,
        'max': 80000,
        'step': 100,
        'value': 0,
    },
}

# app layout
_SLIDERS = []
for p in _INPUT_VARS:
    _SLIDERS.extend([
        html.Div(
            html.P(p.replace('_', ' ') + f' [{_UNITS_VARS[p]}]'),
            style={'width': '7%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),
        html.Div(
            dcc.Slider(id=p, **_SLIDER_SETTINGS[p], **_SLIDER_DEFAULT_SETTINGS),
            style={'width': '12%', 'display': 'inline-block'}
        )
    ])

APP.layout = html.Div([
    html.H1('Estuarine dynamics: Neural network'),

    html.H2('Input'),
    *_SLIDERS,

    html.Div(
        id='warning',
        children=[],
        style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top', 'color': 'red'}
    ),

    html.H2('Estimates'),
    html.Div(id='output', children=[]),

    html.H2('Visualisation'),
    dcc.Dropdown(
        id='figure-type',
        options=[
            {'label': 'land boundaries', 'value': 'banks'},
            {'label': 'geomorphology', 'value': 'geom'},
            {'label': 'salinity', 'value': 'salt'},
        ],
        value='banks',
        multi=False,
        clearable=False,
    ),

    dcc.Graph(id='fig-estuary', figure={}),
])


@APP.callback(
    [
        dep.Output(component_id='output', component_property='children'),
        dep.Output(component_id='warning', component_property='children')
    ],
    [dep.Input(component_id=p, component_property='value') for p in _INPUT_VARS]
)
def nn_output(
        tidal_range, surge_level, river_discharge,
        channel_depth, channel_width, channel_friction, convergence,
        flat_depth_ratio, flat_width, flat_friction,
        bottom_curvature, meander_amplitude, meander_length
):
    """Estimate estuary output, based on neural network, and estuary type from input parameters.

    :param tidal_range: tidal range
    :param surge_level: storm surge level
    :param river_discharge: river discharge
    :param channel_depth: channel depth
    :param channel_width: channel width
    :param channel_friction: channel friction
    :param convergence: convergence
    :param flat_depth_ratio: flat depth ratio
    :param flat_width: flat width
    :param flat_friction: flat friction
    :param bottom_curvature: bottom curvature
    :param meander_amplitude: meander amplitude
    :param meander_length: meander wave length

    :type tidal_range: float
    :type surge_level: float
    :type river_discharge: float
    :type channel_depth: float
    :type channel_width: float
    :type channel_friction: float
    :type convergence: float
    :type flat_depth_ratio: float
    :type flat_width: float
    :type flat_friction: float
    :type bottom_curvature: float
    :type meander_amplitude: float
    :type meander_length: float

    :return: HTML-output, and warning
    :rtype: tuple
    """
    # verify input parameters
    args = locals().copy()
    try:
        input_check(**args)
    except ValueError as e:
        msg = [html.P(arg) for arg in list(*e.args)]
        return None, msg

    # estuary classification
    estuary = AppEstuary(
        tidal_range=tidal_range, river_discharge=river_discharge,
        channel_depth=channel_depth, channel_width=channel_width, channel_friction=channel_friction,
        convergence=convergence, flat_depth_ratio=flat_depth_ratio, flat_width_ratio=1 + flat_width / channel_width,
        flat_friction=flat_friction, bottom_curvature=bottom_curvature, meander_amplitude=meander_amplitude,
        meander_length=meander_length,
        step_size=100
    )
    classification = EstuaryType(estuary)

    # neural network prediction
    length, variability = NN.single_predict(**args)[['L', 'V']].values[0]

    # define output
    output = [
        html.P(f'Salt intrusion length: {1e-3 * length:.1f} [km]'),
        # html.P(f'Salt variability: {variability:.2f}'),
        html.P(
            f'Estuary type: {classification} '
            f'(M = {classification.mixing:.2f}, '
            f'Fr = {classification.froude:.2E})'
        )
    ]
    return output, None


@APP.callback(
    dep.Output(component_id='fig-estuary', component_property='figure'),
    [dep.Input(component_id=arg, component_property='value') for arg in ['figure-type'] + _INPUT_VARS]
)
def update_figure(
        fig_option,
        tidal_range, surge_level, river_discharge,
        channel_depth, channel_width, channel_friction, convergence,
        flat_depth_ratio, flat_width, flat_friction,
        bottom_curvature, meander_amplitude, meander_length
):
    """Create figure of idealised estuary layout.

    :param fig_option: type of data to be presented in figure
    :param tidal_range: tidal range
    :param surge_level: storm surge level
    :param river_discharge: river discharge
    :param channel_depth: channel depth
    :param channel_width: channel width
    :param channel_friction: channel friction
    :param convergence: convergence
    :param flat_depth_ratio: flat depth ratio
    :param flat_width: flat width
    :param flat_friction: flat friction
    :param bottom_curvature: bottom curvature
    :param meander_amplitude: meander amplitude
    :param meander_length: meander wave length

    :type fig_option: str
    :type tidal_range: float
    :type surge_level: float
    :type river_discharge: float
    :type channel_depth: float
    :type channel_width: float
    :type channel_friction: float
    :type convergence: float
    :type flat_depth_ratio: float
    :type flat_width: float
    :type flat_friction: float
    :type bottom_curvature: float
    :type meander_amplitude: float
    :type meander_length: float

    :return: figure
    :rtype: plotly.graph_objects.Figure
    """
    # initiation
    estuary = AppEstuary(
        tidal_range=tidal_range, river_discharge=river_discharge,
        channel_depth=channel_depth, channel_width=channel_width, channel_friction=channel_friction,
        convergence=convergence, flat_depth_ratio=flat_depth_ratio, flat_width_ratio=1 + flat_width / channel_width,
        flat_friction=flat_friction, bottom_curvature=bottom_curvature, meander_amplitude=meander_amplitude,
        meander_length=meander_length,
        step_size=100
    )
    fig = go.Figure()

    # figure-options
    if fig_option == 'banks':
        # estuarine banks
        banks = estuary.land_boundaries
        for bank in ('SOUTH_BANK', 'NORTH_BANK'):
            fig.add_trace(
                go.Scatter(
                    x=banks['x'],
                    y=banks[bank],
                    hovertemplate='x: %{text:.2f} [km]<br>y: %{y:.2f} [m]',
                    text=banks['x'] / 1e3,
                    mode='lines',
                    line=dict(color='black', width=4),
                    name=bank.replace('_', ' ').capitalize(),
                    showlegend=False,
                )
            )

    elif fig_option == 'geom':
        # estuarine geomorphology/bathymetry
        bathymetry = estuary.bathymetry
        d0 = -_SLIDER_SETTINGS['channel_depth']['max']
        d1 = -.5 * _SLIDER_SETTINGS['flat_depth_ratio']['max'] * _SLIDER_SETTINGS['tidal_range']['max']
        d2 = 0
        d3 = -.5 * _SLIDER_SETTINGS['flat_depth_ratio']['min'] * _SLIDER_SETTINGS['tidal_range']['max']
        def color(depth): return (depth - d0) / (d3 - d0)
        fig.add_trace(
            go.Scatter(
                x=bathymetry['x'],
                y=bathymetry['y'],
                marker=dict(
                    color=bathymetry['z'],
                    colorbar=dict(title='Depth [m]', thickness=20, tickvals=[d0, d1, d2, d3]),
                    colorscale=[
                        (color(d0), 'rgb(0, 0, 0)'),
                        (color(d1), 'rgb(124, 124, 255)'),
                        (color(d2), 'rgb(200, 200, 200)'),
                        (color(d3), 'rgb(255, 124, 124)'),
                    ],
                    cmin=d0, cmax=d3,
                ),
                hovertemplate='<b>depth: %{marker.color:.2f} [m]</b><br>x: %{text:.2f} [km]<br>y: %{y:.2f} [m]',
                text=bathymetry['x'] / 1e3,
                mode='markers',
                name='',
                showlegend=False,
            )
        )

    elif fig_option == 'salt':
        # estuarine salt intrusion
        salt = NN.single_predict(
            tidal_range=tidal_range, surge_level=surge_level, river_discharge=river_discharge,
            channel_depth=channel_depth, channel_width=channel_width, channel_friction=channel_friction,
            convergence=convergence, flat_depth_ratio=flat_depth_ratio, flat_width=flat_width,
            flat_friction=flat_friction, bottom_curvature=bottom_curvature, meander_amplitude=meander_amplitude,
            meander_length=meander_length
        )['L'].values[0]

        # salt concentration: colored scatter plot
        grid = estuary.bathymetry
        grid['s'] = 30 ** (1 - (grid['x'] + estuary.ocean_extent) / (salt + estuary.ocean_extent))
        fig.add_trace(
            go.Scatter(
                x=grid['x'],
                y=grid['y'],
                marker=dict(
                    color=grid['s'],
                    colorbar=dict(title='Salinity [psu]', thickness=20),
                    colorscale=[
                        (0, 'rgb(200, 200, 255)'),
                        (1, 'rgb(0, 0, 124)')
                    ],
                    cmin=0, cmax=30
                ),
                hovertemplate='<b>salinity: %{marker.color:.2f} [psu]</b><br>x: %{text:.2f} [km]<br>y: %{y:.2f} [m]',
                text=grid['x'] / 1e3,
                mode='markers',
                name='',
                showlegend=False,
            )
        )

        # salt intrusion length: 1 psu-line
        y = [-.5 * (channel_width + flat_width), .5 * (channel_width + flat_width)]
        si_line = pd.DataFrame(
            data=np.transpose(estuary.lateral_reformat(salt * np.ones_like(y), y)),
            columns=['x', 'y']
        )

        fig.add_trace(
            go.Scatter(
                x=si_line['x'],
                y=si_line['y'],
                mode='lines',
                line=dict(color='black', width=4),
                name='Salt intrusion length [m]',
                showlegend=False,
            )
        )
        fig.add_annotation(
            text=f'<b><i>L</i>(<i>s</i>=1 psu)={salt / 1e3:.1f} km</b>',
            x=si_line['x'].mean() + 250, y=si_line['y'].mean(),
            xref='x', yref='y',
            xanchor='left', yanchor='middle',
            showarrow=False,
        )

    else:
        msg = f'Figure option not included: {fig_option}.'
        raise NotImplementedError(msg)

    # prettify figure
    fig.update_xaxes(
        range=[-estuary.ocean_extent, estuary.length],
        title_text='longitudinal axis, <i>x</i> [m]',
    )
    fig.update_yaxes(
        title_text='lateral axis, <i>y</i> [m]',
    )
    fig.update_layout(
        template='none',
    )

    # return figure
    return fig


if __name__ == '__main__':
    APP.run_server(debug=True)
