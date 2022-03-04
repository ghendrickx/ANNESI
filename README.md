# Neural network: Estuarine dynamics
As a by-product of the sensitivity analysis, a neural network has been trained to the elaborate data set created, 
consisting of approximately 2,000 simulations with 
[Delft3D Flexible Mesh](https://www.deltares.nl/en/software/delft3d-flexible-mesh-suite/) (specifically the 
[D-Flow module](https://www.deltares.nl/en/software/module/d-flow-flexible-mesh/)).

The neural network is accessible via a web-API that can be locally hosted by running [`api.py`](api.py). Efforts are 
made to host this web-API publicly.

## Requirements
This sub-repository has the following requirements (see also [`requirements.txt`](requirements.txt)):
*   `dash==2.0.0`
*   `numpy==1.19.4`
*   `pandas==1.1.4`
*   `joblib==1.0.1`
*   `Shapely==1.8.0`
*   `torch==1.9.0`
*   `plotly==5.5.0`
*   `scikit_learn==0.24.2`

For the installation of `torch`, please look at there [installation guide](https://pytorch.org/get-started/locally/);
the installation of `torch` is slightly different from other Python-packages for which a `pip install` suffices. Also
note that `torch` is only supported for Python 3.7-3.9, and not for Python 2.x (or 3.10).

In addition, the [`utils`](../utils)-folder is required for the functioning of the `neural_network` (i.e. the web-API).

## Usage
For the use of the web-API, [`api.py`](api.py) must be executed with Python. This provides a link to a local-host, 
which allows to use the neural network locally in a web-browser.

To use the web-API, the following steps are required:
1.  Install all the requirements (see [*Requirements*](#Requirements)).
1.  Open a Python IDE (e.g. PyCharm) or the command line.
1.  Run `api.py` (or `/neural_network/api.py`).

When running from the command line, you can either (1) first change the directory (`cd`) to the repository and 
subsequently run `api.py`; or (2) run `api.py` from any location by including the full directory to the file. Option 1
and option 2 are written-out below, respectively:
```commandline
cd path/to/repository
python neural_network/api.py
```
```commandline
python path/to/repository/neural_network/api.py
```
This will return the following message, including the link to the locally hosted web-page:
```commandline
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'neural_network.application.app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)
```

There is also the possibility to use the neural network without the web-API - as stand-alone. For more information on
this approach, see [`machine_learning`](machine_learning).

## Structure
The neural network and web-API are located in the folders [`machine_learning`](machine_learning) and 
[`application`](application), respectively:
```
+-- application/
|   +-- __init__.py
|   +-- app.py
|   +-- components.py
+-- machine_learning/
|   +-- _data/
|   |   +-- __init__.py
|   |   +-- nn_default.pkl
|   |   +-- nn_scaler.gz
|   +-- __init__.py
|   +-- _backend.py
|   +-- neural_network.py
+-- __init__.py
+-- api.py
+-- README.md
+-- requirements.txt
```

## Author
Gijs G. Hendrickx 
[![alt text](https://camo.githubusercontent.com/e1ec0e2167b22db46b0a5d60525c3e4a4f879590a04c370fef77e6a7e00eb234/68747470733a2f2f696e666f2e6f726369642e6f72672f77702d636f6e74656e742f75706c6f6164732f323031392f31312f6f726369645f31367831362e706e67) 0000-0001-9523-7657](https://orcid.org/0000-0001-9523-7657)
(*Delft University of Technology*)

## References
When using this repository, please cite accordingly:
> Hendrickx, G.G. (2022). SALTISolutions: A neural network for estuarine salt dynamics. 4TU.ResearchData. Software.
[doi:10.4121/19161752](https://doi.org/10.4121/19161752).

### Version-control
The neural network, and so the web-API, are subject to updates. These updates are reflected by different versions of the
repository.

### Releases
The releases that affected this sub-repository are:
1.  Pre-release: [`v0.1-beta`](https://doi.org/10.4121/19161752.v1)

For more information on the different releases, see the [main-repository](../README.md#Releases).
