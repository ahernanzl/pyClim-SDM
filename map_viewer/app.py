from dash import Dash
import layout, callbacks
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
])
app.title = "pyClim-SDM map viewer"
app._favicon = 'favicon.ico'
app.layout = layout.create_layout()
callbacks.register_callbacks(app)

# This is for the period-slider text to be white
from dash import Input, Output
app.clientside_callback(
    """
    function() {
        const sliderLabels = document.querySelectorAll('.rc-slider-mark-text');
        sliderLabels.forEach(label => label.style.color = 'white');
        return '';
    }
    """,
    Output('message-container', 'children'),
    Input('period-slider', 'value'),
)

if __name__ == "__main__":
    # app.run_server(debug=True, port=8050)
    app.run_server(debug=False, port=8050)
