import os
import sys
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import dash
from dash import Input, Output, State
from dash import callback_context as ctx


# Add config paths for settings import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config')))
from settings import *
from advanced_settings import *
from utils import *
# Determine which data path to use depending on file availability
path1D = "../results/PROJECTIONS"+bc_sufix+"/"
path2D = "../results/PROJECTIONS"+bc_sufix+"_2D/"
data_path = path2D if os.path.exists(path2D) else path1D
is_grid = os.path.exists(path2D)



def register_callbacks(app):

    # Update method options based on selected variable
    @app.callback(
        Output("method-select", "options"),
        Output("method-select", "value"),
        Input("var-select", "value"),
        State("method-select", "value")
    )
    def update_method_options(target_var, current_value):
        """
        Update method dropdown options and preserve current selection if valid.
        """
        options = [{"label": m['methodName'], "value": m['methodName']} for m in methods if m['var'] == target_var]
        if bc_sufix != '':
            options = [{"label": x['label'] + ' (bias corrected)', "value": x['value'] + bc_sufix} for x in options]
        valid_values = [opt["value"] for opt in options]
        value = current_value if current_value in valid_values else (valid_values[0] if valid_values else None)
        return options, value

    # Update model options independently of var and method, preserve current selection if valid
    @app.callback(
        Output("model-select", "options"),
        Output("model-select", "value"),
        Input("method-select", "value"),
        State("model-select", "value")
    )
    def update_model_options(_, current_value):
        """
        Update model dropdown options, always including ENSEMBLE MEAN first
        and defaulting to it if current selection is invalid or None.
        """
        models = ["ENSEMBLE MEAN"] + model_list
        options = [{"label": m, "value": m} for m in models]
        valid_values = [opt["value"] for opt in options]

        # Default to ENSEMBLE MEAN if nothing is selected or current value is invalid
        value = current_value if current_value in valid_values else "ENSEMBLE MEAN"
        return options, value

    # Update climdex and season options when variable changes, preserve current selections
    @app.callback(
        Output("climdex-select", "options"),
        Output("climdex-select", "value"),
        Output("season-select", "options"),
        Output("season-select", "value"),
        Input("var-select", "value"),
        State("climdex-select", "value"),
        State("season-select", "value")
    )
    def update_climdex_and_season_options(target_var, current_climdex, current_season):
        """
        Update climdex and season dropdown options based on variable,
        preserving current valid selections if possible.
        """
        climdex_options = climdex_names.get(target_var, [])
        season_options = list(season_dict.keys())

        climdex_valid = current_climdex if current_climdex in climdex_options else (climdex_options[0] if climdex_options else None)
        season_valid = current_season if current_season in season_options else (season_options[0] if season_options else None)

        return (
            [{"label": c, "value": c} for c in climdex_options], climdex_valid,
            [{"label": s, "value": s} for s in season_options], season_valid
        )

    # Update scene options, preserve current selection if valid
    @app.callback(
        Output("scene-select", "options"),
        Output("scene-select", "value"),
        Input("scene-select", "value"),
        State("scene-select", "value")
    )
    def update_scene_options(_, current_value):
        """
        Update scene dropdown options and preserve selection.
        """
        options = [{"label": s, "value": s} for s in scene_list]
        valid_values = [opt["value"] for opt in options]
        value = current_value if current_value in valid_values else (valid_values[0] if valid_values else None)
        return options, value

    # Update the period slider marks and range based on selected parameters
    @app.callback(
        Output("period-slider", "min"),
        Output("period-slider", "max"),
        Output("period-slider", "marks"),
        Output("period-slider", "value"),
        Input("var-select", "value"),
        Input("method-select", "value"),
        Input("climdex-select", "value"),
        Input("model-select", "value"),
        Input("scene-select", "value"),
        Input("season-select", "value"),
    )
    def update_period_slider(var, method, climdex_name, model, scene, season):
        """
        Load dataset and generate dynamic 30-year periods for the slider.
        """
        if not all([var, method, climdex_name, model, scene, season]):
            return 0, 0, {}, 0

        model_to_check = model_list[0] if model == "ENSEMBLE MEAN" else model
        base = os.path.join(data_path, var.upper(), method, "climdex")
        fname = f"{var}_{climdex_name}_{scene}_{model_to_check}_{season}.nc"
        fpath = os.path.join(base, fname)

        if not os.path.exists(fpath):
            return 0, 0, {}, 0

        ds = xr.open_dataset(fpath)
        years = ds['time'].dt.year.values
        periods, labels = generate_dynamic_periods(years.min(), years.max())
        marks = {i: label for i, label in enumerate(labels)}

        return 0, len(periods) - 1, marks, len(periods) - 1

    # Update evolution graph based on all selection inputs
    @app.callback(
        Output("evolution-graph", "figure"),
        Input("var-select", "value"),
        Input("method-select", "value"),
        Input("climdex-select", "value"),
        Input("model-select", "value"),
        Input("scene-select", "value"),
        Input("season-select", "value"),
        Input("display-mode-select", "value"),
    )
    def update_evolution_graph(var, method, climdex_name, model, scene, season, display_mode):
        """
        Plot evolution with change relative to historical baseline for any scenario.
        Supports absolute or relative changes based on climdex settings.
        """
        if not all([var, method, climdex_name, scene, model, season]):
            return go.Figure()
        try:
            base = os.path.join(data_path, var.upper(), method, "climdex")

            # Load scene data and years
            data_scene = load_models(base, var, climdex_name, model, scene, season)
            years_scene = get_years(base, var, climdex_name, model, scene, season)

            # Load historical reference if needed
            if display_mode.startswith('Change_'):
                data_hist = load_models(base, var, climdex_name, model, 'historical', season)
                years_hist = get_years(base, var, climdex_name, model, 'historical', season)

            # Get bias_mode
            bias_mode = 'abs'
            if var+'_'+climdex_name in bias_units_and_palette:
                bias_mode = bias_units_and_palette[var+'_'+climdex_name]['biasMode']

            # Get units
            units_and_palette_dict = bias_units_and_palette if display_mode.startswith(
                "Change_") else absolute_units_and_palette
            units = units_and_palette_dict.get(var + '_' + climdex_name, {}).get("units", "")

            # Add model data to traces and all_anomalies
            traces = []
            all_anomalies = []
            for model in data_scene:
                data_scene_model = data_scene[model]
                if display_mode.startswith("Change_"):
                    data_hist_model = data_hist[model]
                    ref_start, ref_end = map(int, display_mode.replace("Change_", "").split("_"))
                    ref_mask = (years_hist >= ref_start) & (years_hist <= ref_end)
                    ref_mask_xr = xr.DataArray(ref_mask, coords={"time": data_hist_model["time"]}, dims="time")
                    ref_mask_expanded = ref_mask_xr.broadcast_like(data_hist_model)
                    ref_data = data_hist_model.where(ref_mask_expanded)
                    ref_mean = ref_data.mean(dim="time")
                    anomaly = apply_change_mode(data_scene_model, ref_mean, bias_mode)
                else:
                    anomaly = data_scene_model

                # Compute spatial mean
                anomaly_spatial_mean = anomaly.mean(dim=[d for d in anomaly.dims if d != "time"])
                all_anomalies.append(anomaly_spatial_mean)
                traces.append(go.Scatter(x=years_scene, y=anomaly_spatial_mean.values, mode='lines', name=model))

            # Ensemble mean, min and max
            if len(all_anomalies) > 1:
                combined = xr.concat(all_anomalies, dim="model")
                mean = combined.mean(dim="model")
                min_ = combined.min(dim="model")
                max_ = combined.max(dim="model")
                traces.append(go.Scatter(x=years_scene, y=mean.values, mode='lines', name="ENSEMBLE MEAN",
                                         line=dict(width=4, color='black')))
                traces.append(go.Scatter(x=years_scene, y=min_.values, mode='lines', name="Min",
                                         line=dict(width=0), showlegend=False, hoverinfo='skip'))
                traces.append(go.Scatter(x=years_scene, y=max_.values, mode='lines', name="Max",
                                         fill='tonexty', line=dict(width=0),
                                         fillcolor='rgba(0,0,0,0.2)', showlegend=False, hoverinfo='skip'))

            # Define y_title
            if display_mode == 'Value':
                y_title = var + '_' + climdex_name + ' (' + units + ')'
            else:
                y_title = var + '_' + climdex_name + ' change (' + units + ')'

            # Create figure
            fig = go.Figure(data=traces)
            fig.update_layout(
                            # title="Spatial average evolution",
                            xaxis_title="Year", yaxis_title=y_title,
                              # legend=dict(
                              #     x=0.01,
                              #     y=0.99,
                              #     bgcolor='rgba(255,255,255,0.7)',
                              #     bordercolor='gray',
                              #     borderwidth=1
                              # ),
                              showlegend=False,
                              margin=dict(t=30, r=10, b=30, l=40)
                              )
            return fig
        except:
            return go.Figure()


    @app.callback(
        Output("vmin-vmax-container", "style"),
        Output("vmin-input", "value"),
        Output("vmax-input", "value"),
        State("vmin-input", "value"),
        State("vmax-input", "value"),
        Input("edit-colorbar-btn", "n_clicks"),
        Input("method-select", "value"),
        Input("model-select", "value"),
        Input("var-select", "value"),
        Input("climdex-select", "value"),
        Input("scene-select", "value"),
        Input("season-select", "value"),
        Input("display-mode-select", "value"),
        Input("period-slider", "value"),
        prevent_initial_call=True
    )
    def toggle_inputs(vmin, vmax, n_clicks, method, model, var, climdex_name, scene, season, display_mode, period):

        triggered = ctx.triggered_id

        # If new var, climdex or display-mode, set vmin/vmax to default
        if triggered in ["var-select", "climdex-select", "display-mode-select", ] or (vmin==None and vmax==None):
            units_and_palette_dict = bias_units_and_palette if display_mode.startswith(
                "Change_") else absolute_units_and_palette
            defaults = units_and_palette_dict.get(f"{var}_{climdex_name}", {})
            vmin = defaults.get("vmin", None)
            vmax = defaults.get("vmax", None)

        # Only displays vmin vmax selector if triggered by edit-colorbar-btn
        if triggered == "edit-colorbar-btn":
            return {"display": "block"}, vmin, vmax
        else:
            return {"display": "none"}, vmin, vmax


    # Update map figure callback
    @app.callback(
        Output("map-graph", "figure"),
        Input("var-select", "value"),
        Input("method-select", "value"),
        Input("climdex-select", "value"),
        Input("model-select", "value"),
        Input("scene-select", "value"),
        Input("season-select", "value"),
        Input("display-mode-select", "value"),
        Input("period-slider", "value"),
        Input("vmin-input", "value"),
        Input("vmax-input", "value"),
        prevent_initial_call=True
    )
    def update_map_figure(var, method, climdex_name, model, scene, season, display_mode, period_index, vmin, vmax):
        """
        Generate a map of the selected variable for the chosen time period and model/scenario.
        If display_mode is a Change mode, it applies absolute or relative change based on climdex settings.
        Supports both grid (lat/lon) and point datasets.
        """
        if not all([var, method, climdex_name, model, scene, season]) or period_index is None:
            return go.Figure()
        try:
            base = os.path.join(data_path, var.upper(), method, "climdex")

            # Load scene data, years and periods
            data_scene = load_models(base, var, climdex_name, model, scene, season)
            years_scene = get_years(base, var, climdex_name, model, scene, season)
            periods, _ = generate_dynamic_periods(years_scene.min(), years_scene.max())
            lats, lons = get_lat_lon(base, var, climdex_name, model, scene, season)

            # Load historical reference if needed
            if display_mode.startswith('Change_'):
                data_hist = load_models(base, var, climdex_name, model, 'historical', season)
                years_hist = get_years(base, var, climdex_name, model, 'historical', season)

            # Get bias_mode
            bias_mode = 'abs'
            if var+'_'+climdex_name in bias_units_and_palette:
                bias_mode = bias_units_and_palette[var+'_'+climdex_name]['biasMode']


            # Add model data all_anomalies
            all_anomalies = []
            for model in data_scene:
                data_scene_model = data_scene[model]
                sce_start, sce_end = periods[period_index]
                sce_mask = (years_scene >= sce_start) & (years_scene <= sce_end)
                sce_mask_xr = xr.DataArray(sce_mask, coords={"time": data_scene_model["time"]}, dims="time")
                sce_mask_expanded = sce_mask_xr.broadcast_like(data_scene_model)
                data_scene_model = data_scene_model.where(sce_mask_expanded)

                if display_mode.startswith("Change_"):
                    data_hist_model = data_hist[model]
                    ref_start, ref_end = map(int, display_mode.replace("Change_", "").split("_"))
                    ref_mask = (years_hist >= ref_start) & (years_hist <= ref_end)
                    ref_mask_xr = xr.DataArray(ref_mask, coords={"time": data_hist_model["time"]}, dims="time")
                    ref_mask_expanded = ref_mask_xr.broadcast_like(data_hist_model)
                    ref_data = data_hist_model.where(ref_mask_expanded)
                    ref_mean = ref_data.mean(dim="time")
                    anomaly = apply_change_mode(data_scene_model, ref_mean, bias_mode)
                else:
                    anomaly = data_scene_model

                # Compute temporal mean
                anomaly_temporal_mean = anomaly.mean(dim="time")
                all_anomalies.append(anomaly_temporal_mean)

            # Convert all_anomalies list to field xarray
            combined = xr.concat(all_anomalies, dim="model")
            field_mean = combined.mean(dim="model")
            field_min = combined.min(dim="model")
            field_max = combined.max(dim="model")

            # Reassign lat/lon as coordinates if working with point data
            if is_grid==False:
                field_mean = field_mean.assign_coords(
                    lat=("point", lats),
                    lon=("point", lons)
                )
                field_min = field_min.assign_coords(
                    lat=("point", lats),
                    lon=("point", lons)
                )
                field_max = field_max.assign_coords(
                    lat=("point", lats),
                    lon=("point", lons)
                )

            # Define custom_data for hovertemplate
            custom_data = np.stack([field_min.values.flatten(), field_max.values.flatten()], axis=-1)

            # Define limits
            lon_min = float(np.min(field_mean["lon"])) - 12
            lon_max = float(np.max(field_mean["lon"])) + 5
            lat_min = float(np.min(field_mean["lat"])) - 3
            lat_max = float(np.max(field_mean["lat"])) + 1

            # Get units, cmap, vmin and vmax
            units_and_palette_dict = bias_units_and_palette if display_mode.startswith(
                "Change_") else absolute_units_and_palette
            cbar_title={"text": f"{units_and_palette_dict.get(var + '_' + climdex_name, {}).get('units', '')}"}
            cmap=units_and_palette_dict.get(var + '_' + climdex_name, {}).get('cmap', '')
            if vmin is None and vmax is None:
                vmin=units_and_palette_dict.get(var + '_' + climdex_name, {}).get('vmin', '')
                vmax=units_and_palette_dict.get(var + '_' + climdex_name, {}).get('vmax', '')


            # Define projection_type
            projection_type = 'equirectangular'
            # projection_type = 'mercator'

            # Plot depending on grid or points
            if is_grid==True:
                fig = go.Figure(data=go.Heatmap(
                    z=field_mean.values,
                    x=field_mean["lon"].values,
                    y=field_mean["lat"].values,
                    colorscale=cmap,
                    colorbar=dict(title=cbar_title, len=0.35, y=0.75, x=0.01),
                    zmin=vmin,
                    zmax=vmax,
                    customdata=custom_data.reshape(field_mean.values.shape + (2,)),
                    hovertemplate=(
                        'lon: %{x}<br>'
                        'lat: %{y}<br>'
                        'max: %{customdata[1]:.2f}<br>'
                        'mean: %{z:.2f}<br>'
                        'min: %{customdata[0]:.2f}<extra></extra>')
                ))
                fig.update_layout(
                    geo=dict(
                        projection=dict(type=projection_type),
                        showcountries=True,
                        showcoastlines=True,
                        showland=True,
                        showocean=True,
                        landcolor="rgb(240, 240, 240)",
                        oceancolor="rgb(200, 230, 255)",
                        lonaxis=dict(range=[lon_min, lon_max]),
                        lataxis=dict(range=[lat_min, lat_max]),
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(range=[lon_min, lon_max], visible=False),
                    yaxis=dict(range=[lat_min, lat_max], visible=False),
                    dragmode="pan",
                )
                return fig
            # For points, we must have coords 'lat' and 'lon'
            else:
                fig = go.Figure(data=go.Scattergeo(
                    lon=field_mean["lon"].values,
                    lat=field_mean["lat"].values,
                    text=np.round(field_mean.values, 2),
                    customdata=custom_data.reshape(field_mean.values.shape + (2,)),
                    hovertemplate=(
                        'lon: %{lon}<br>'
                        'lat: %{lat}<br>'
                        'max: %{customdata[1]:.2f}<br>'
                        'mean: %{text:.2f}<br>'
                        'min: %{customdata[0]:.2f}<extra></extra>'),
                    marker=dict(
                        size=8,
                        color=field_mean.values,
                        colorscale=cmap,
                        cmin=vmin,
                        cmax=vmax,
                        colorbar=dict(title=cbar_title, len=0.35, y=0.75, x=0.01),
                    ),
                    mode="markers"
                ))
                fig.update_layout(
                    geo=dict(
                        projection=dict(type=projection_type),
                        showcountries=True,
                        showcoastlines=True,
                        showland=True,
                        showocean=True,
                        landcolor="rgb(240, 240, 240)",
                        oceancolor="rgb(200, 230, 255)",
                        lonaxis=dict(range=[lon_min, lon_max]),
                        lataxis=dict(range=[lat_min, lat_max]),
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(range=[lon_min, lon_max], visible=False),
                    yaxis=dict(range=[lat_min, lat_max], visible=False),
                    dragmode="pan",
                )
                return fig
        except:
            return go.Figure()

