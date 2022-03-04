# ANNESI: Artificial neural network for estuarine salt intrusion
This repository is part of a PhD research on developing nature-based solutions to mitigate salt intrusion (for more 
information, see the [central repository](https://github.com/ghendrickx/SALTISolutions)).

Part of this research is a sensitivity analysis of estuarine salt intrusion to estuarine characteristics. As a 
by-product of the sensitivity analysis, a neural network has been trained to the elaborate data set created, consisting 
of approximately 2,000 simulations with 
[Delft3D Flexible Mesh](https://www.deltares.nl/en/software/delft3d-flexible-mesh-suite/) (specifically the 
[D-Flow module](https://www.deltares.nl/en/software/module/d-flow-flexible-mesh/)).

The neural network is accessible via a web-API that can be locally hosted by running [`api.py`](api.py). Efforts are 
made to host this web-API publicly. The neural network is also available without the web-API (see
[`machine_learning`](machine_learning)).

## Requirements
This repository has the following requirements (see also [`requirements.txt`](requirements.txt)):
*   `numpy==1.19.4`
*   `pandas==1.1.4`
*   `torch==1.9.0`
*   `scikit_learn==0.24.2`
*   `joblib==1.0.1`
*   `dash==2.0.0`*
*   `plotly==5.5.0`*
*   `Shapely==1.8.0`*

*\* Only required for the web-API.*

For the installation of `torch`, please look at their [installation guide](https://pytorch.org/get-started/locally/);
the installation of `torch` is slightly different from other Python-packages for which a `pip install` suffices. Also
note that `torch` is (currently) only supported for `python 3.7`-`3.9`, and not for `python 2.x` (or `3.10`); see the
[official documentation](https://pytorch.org/get-started/locally/#windows-python) of `torch` for the latest updates.

## Usage
For the use of the web-API, [`api.py`](api.py) must be executed with Python. This provides a link to a local-host, 
which allows to use the neural network locally in a web-browser.

To use the web-API, the following steps are required:
1.  Install all the requirements (see [*Requirements*](#Requirements)).
1.  Open a Python IDE (e.g. PyCharm) or the command line.
1.  Run `api.py` (or `/api.py`).

When running from the command line, you can either (1) first change the directory (`cd`) to the repository and 
subsequently run `api.py`; or (2) run `api.py` from any location by including the full directory to the file. Both 
options are written-out below, respectively:
```commandline
cd path/to/repository
python api.py
```
```commandline
python path/to/repository/api.py
```
This will return the following message, including the link to the locally hosted web-page:
```commandline
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'application.app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)
```

There is also the possibility to use the neural network without the web-API, i.e. as stand-alone. For more information 
on this approach, see [`machine_learning`](machine_learning).

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
+-- utils/
|   +-- __init__.py
|   +-- data_conv.py
|   +-- files_dirs.py
+-- .gitignore
+-- __init__.py
+-- api.py
+-- LICENSE
+-- README.md
+-- requirements.txt
+-- setup.py
```

## Author
Gijs G. Hendrickx 
[![alt text](https://camo.githubusercontent.com/e1ec0e2167b22db46b0a5d60525c3e4a4f879590a04c370fef77e6a7e00eb234/68747470733a2f2f696e666f2e6f726369642e6f72672f77702d636f6e74656e742f75706c6f6164732f323031392f31312f6f726369645f31367831362e706e67) 0000-0001-9523-7657](https://orcid.org/0000-0001-9523-7657)
(*Delft University of Technology*)

## References
When using this repository, please cite accordingly:
> Hendrickx, G.G. (2022). ANNESI: An open-source artificial neural network for estuarine salt intrusion. 
4TU.ResearchData. Software. [doi:10.4121/19307693](https://doi.org/10.4121/19307693).

### Version-control
The neural network, and so the web-API, are subject to updates. These updates are reflected by different versions of the
repository.
