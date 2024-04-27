from dash import html
import dash_bootstrap_components as dbc
import dash

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "231px",
    "height": "100vh", # Set full height of viewport
    "padding": "20px",
    "background-color": "#333", # Set background color
    "color": "#fff",       # Set text color  
    }

CONTENT_STYLE = {
    "margin-left": "4rem",
    "margin-right": "2rem",
    "padding": "10px",
    "background-color": "#333",
    }
_nav = dbc.Container([
	dbc.Row([
        dbc.Col([
            html.Div([
                html.I(className="fa-solid fa-chart-simple fa-2x")],
		    className='logo')
        ], width = 4),
        dbc.Col([html.H1(['Speed Wind'], className='app-brand')], width = 8)
	]),
	dbc.Row([
        dbc.Nav(
	        [dbc.NavLink(page["name"], active='exact', href=page["path"]) for page in dash.page_registry.values()],
	        vertical=True, pills=True, class_name='my-nav')
    ])
],fluid=True,  # Hace que el contenido ocupe todo el ancho de la ventana
style=SIDEBAR_STYLE,)