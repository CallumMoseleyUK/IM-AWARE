## Dam break model
### dambreak_lib.py
Handles data output from dam break MATLAB model. Example script is given at the bottom of the file. See Google drive IM_AWARE/data/data_dambreak for sample data in csv format.
### run_parallel.m
Reads site data from siteData.csv and runs analyses on each in parallel.
### dam_break.py <IP (optional)>
Starts a Flask app hosted by default on localHost:5000. Optionally specify an IP (e.g. 0.0.0.0 hosts on the local network).
Depends on "static" and "templates" folders.

Note: requires mutually compatible versions of Python/Matlab in order to run the Matlab engine in the Flask app (see https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf).
### trajectory2csv.m
Writes an input model output .mat file to a .csv readable by dambreak_lib.dbSimData objects (see dambreak_lib.py).
### model
Contains the Matlab dam break model.
#### main_func.m
Wrapper function containing tunable model parameters. This is where the main sim_dambreak_stochastic.m function is called from.
Additionally saves renders and exports relevant ones to "static" folder to be referenced by the Flask app.
Callable from Python (see dam_break.py).
#### sim_dambreak_stochastic.m
Calculates raw model output.
### get_map.m
Automatically retrieves digital elevation maps from the JAXA servers based on input latitude and longitude.
