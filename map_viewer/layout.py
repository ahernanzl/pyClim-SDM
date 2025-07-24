from dash import html, dcc
import dash_bootstrap_components as dbc
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config')))
from settings import targetVars
from advanced_settings import model_list, scene_list, season_dict

def create_layout():
    return html.Div([
        # MENU BAR (top row)
        html.Div([
            html.Div([
                html.Label("Method"),
                dcc.Dropdown(id="method-select", value=None, clearable=False),
            ], style={"width": "12%", "padding": "0 5px"}),

            html.Div([
                html.Label("Model"),
                dcc.Dropdown(
                    id="model-select",
                    options=[{"label": m, "value": m} for m in ["ENSEMBLE MEAN"] + model_list],
                    value=None,
                    clearable=False
                )
            ], style={"width": "12%", "padding": "0 5px"}),

            html.Div([
                html.Label("Variable"),
                dcc.Dropdown(
                    id="var-select",
                    options=[{"label": v, "value": v} for v in targetVars],
                    value=targetVars[0] if targetVars else None,
                    clearable=False
                )
            ], style={"width": "12%", "padding": "0 5px"}),

            html.Div([
                html.Label("Climate Index"),
                dcc.Dropdown(id="climdex-select", value=None, clearable=False),
            ], style={"width": "12%", "padding": "0 5px"}),

            html.Div([
                html.Label("Scenario"),
                dcc.Dropdown(
                    id="scene-select",
                    options=[{"label": s, "value": s} for s in scene_list],
                    value=scene_list[0] if scene_list else None,
                    clearable=False
                )
            ], style={"width": "12%", "padding": "0 5px"}),

            html.Div([
                html.Label("Season"),
                dcc.Dropdown(id="season-select", value=None, clearable=False),
            ], style={"width": "12%", "padding": "0 5px"}),

            html.Div([
                html.Label("Display Mode"),
                dcc.Dropdown(
                    id="display-mode-select",
                    options=[
                        {"label": "Value", "value": "Value"},
                        {"label": "Change (1961-1990)", "value": "Change_1961_1990"},
                        {"label": "Change (1971-2000)", "value": "Change_1971_2000"},
                        {"label": "Change (1991-2010)", "value": "Change_1991_2010"},
                    ],
                    value="Value",
                    clearable=False
                )
            ], style={"width": "12%", "padding": "0 5px"}),

        ], style={
            "height": "14%",
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "space-between",
            "padding": "10px",
            "color": "black",
            "backgroundColor": "lightblue",
            "borderBottom": "2px solid #ccc",
        }),

        # Slider
        html.Div([
            dcc.Slider(id="period-slider", min=0, max=0, value=0, marks={}),
            html.Div(id="message-container", style={
                "color": "white", "fontWeight": "bold", "marginTop": "5px", "textAlign": "center"
            }
            )
        ], style={
            "left": "0%",
            "width": "100%",
            "height": "7%",
            "zIndex": 1,
            "color": "orange",
            "backgroundColor": "blue",
            "padding": "5px 20px",
            "borderRadius": "8px",
            "overflow": "hidden",
            "whiteSpace": "nowrap",
            "textOverflow": "ellipsis",
        }),

        # MAP FULLSCREEN CONTAINER
        html.Div([
            # Map
            dcc.Graph(id="map-graph",
                config={
                    "scrollZoom": True,
                    "displayModeBar": False,
                },
                style={
                    "width": "100%",
                    "height": "100vh",
                    "margin": "0",
                    "padding": "0",
                    "zIndex": 0,
                }
            ),

            # Edit colorbar inputs
            html.Div([
                dbc.Button(html.I(className="bi bi-pencil"), id="edit-colorbar-btn", size="sm", color="secondary"),
                html.Div([
                    dbc.Input(id="vmax-input", type="number", value=None, step=0.1, placeholder="vmax"),
                    dbc.Input(id="vmin-input", type="number", value=None, step=0.1, placeholder="vmin"),
                ], id="vmin-vmax-container", style={"display": "none"})
            ], style={
                "position": "absolute",
                "top": "10%",
                "left": "4.5%",
                "width": "100px",
                "zIndex": 2,
                # "backgroundColor": "rgba(255,255,255,0.9)",
                "padding": "5px",
                "borderRadius": "8px"
            }),

            # Evolution graph at top left
            html.Div([
                dcc.Graph(id="evolution-graph", style={"height": "100%", "width": "100%"}, config={"displayModeBar": False})
                ], style={
                    "position": "absolute",
                    "bottom": "25%",
                    "left": "20px",
                    "height": "30%",
                    "zIndex": 1,
                    "backgroundColor": "rgba(255,255,255,0.5)",
                    "borderRadius": "8px",
                    "overflow": "hidden",
                    "aspectRatio": "3 / 2",
            }),

        ], style={
            "position": "relative",
            "height": "100vh",
            "margin": "0",
            "padding": "0",
            "overflow": "hidden"
        })
    ], style={"overflow": "hidden", "height": "100vh",
  })
