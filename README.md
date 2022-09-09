# ANNESI: Artificial neural network for estuarine salt intrusion
This repository is part of a PhD research on developing nature-based solutions to mitigate salt intrusion (for more 
information, see the [central repository](https://github.com/ghendrickx/SALTISolutions)).

Part of this research is a sensitivity analysis of estuarine salt intrusion to estuarine characteristics. As a 
by-product of the sensitivity analysis, a neural network has been trained to the elaborate data set created, consisting 
of 2,000 simulations with [Delft3D Flexible Mesh](https://www.deltares.nl/en/software/delft3d-flexible-mesh-suite/) 
(specifically the [D-Flow module](https://www.deltares.nl/en/software/module/d-flow-flexible-mesh/)).

The neural network is contained in this repository but is accessible via a web-API on the website of 
[SALTISolutions]() of which the `GitHub`-repository is called [`ANNESI-web`](https://github.com/ghendrickx/ANNESI-web). 
The neural network can also be used without the web-API (see [`src`](./src)).

## Requirements
This repository has the following requirements (see also [`requirements.txt`](./requirements.txt)):
*   `numpy==1.19.4`
*   `pandas==1.1.4`
*   `torch==1.9.0`
*   `scikit_learn==0.24.2`
*   `joblib==1.0.1`

For the installation of `torch`, please look at their [installation guide](https://pytorch.org/get-started/locally/);
the installation of `torch` is slightly different from other Python-packages for which a `pip install` suffices. Also
note that `torch` is (currently) only supported for `python 3.7`-`3.9`, and not for `python 2.x` (or `3.10`); see the
[official documentation](https://pytorch.org/get-started/locally/#windows-python) of `torch` for the latest updates.

## Usage
The (basic) usage of the neural network requires importing and initialising the `NeuralNetwork` in a straightforward 
manner:
```python
from src.neural_network import NeuralNetwork

nn = NeuralNetwork()
```
The most basic usage of the neural network encompasses a single prediction, using the `single_predict()`-method:
```python
from src.neural_network import NeuralNetwork

# initialise neural network
nn = NeuralNetwork()

# single prediction
prediction = nn.single_predict(
    tidal_range=2.25,
    surge_level=0,
    river_discharge=10000,
    channel_depth=20,
    channel_width=1000,
    channel_friction=.023,
    convergence=1e-4,
    flat_depth_ratio=0,
    flat_width=500,
    flat_friction=.05,
    bottom_curvature=1e-5,
    meander_amplitude=1000,
    meander_length=20000
)

# print prediction
print(prediction)
```
This will return the salt intrusion length (in metres):
```
10934.607982635498
```

In addition to the above basic usage of the neural network, some more elaborate use-cases are supported by the
neural network. These are demonstrated in the [`src`](./src)-folder.

## Structure
The neural network is stored in the [`src`](./src)-folder:
```
+-- src/
|   +-- _data/
|   |   +-- __init__.py
|   |   +-- annesi.gz
|   |   +-- annesi.onnz
|   |   +-- annesi.pkl
|   +-- __init__.py
|   +-- _backend.py
|   +-- neural_network.py
|   +-- README.md
+-- tests/
|   +-- __init__.py
|   +-- test_nn.py
|   +-- test_utils.py
+-- utils/
|   +-- __init__.py
|   +-- check.py
|   +-- data_conv.py
|   +-- files_dirs.py
+-- .gitignore
+-- __init__.py
+-- LICENSE
+-- README.md
+-- requirements.txt
+-- setup.py
```

## Author
Gijs G. Hendrickx 
[![alt text](https://camo.githubusercontent.com/e1ec0e2167b22db46b0a5d60525c3e4a4f879590a04c370fef77e6a7e00eb234/68747470733a2f2f696e666f2e6f726369642e6f72672f77702d636f6e74656e742f75706c6f6164732f323031392f31312f6f726369645f31367831362e706e67) 0000-0001-9523-7657](https://orcid.org/0000-0001-9523-7657)
(Delft University of Technology).

Contact: [G.G.Hendrickx@tudelft.nl](mailto:G.G.Hendrickx@tudelft.nl?subject=[GitHub]%20ANNESI).

## References
When using this repository, please cite accordingly:
> Hendrickx, G.G. (2022). ANNESI: An open-source artificial neural network for estuarine salt intrusion. 
4TU.ResearchData. Software. [doi:10.4121/19307693](https://doi.org/10.4121/19307693).

### Version-control
The neural network, and so [`ANNESI-web`](https://github.com/ghendrickx/ANNESI-web), are subject to updates. These 
updates are reflected by different versions of the repository.

### Related references
The standalone neural network is used as part of the following peer-reviewed article:
*   [Hendrickx, G.G.](https://orcid.org/0000-0001-9523-7657), 
    [Antol&iacute;nez, J.A.&Aacute;](https://orcid.org/0000-0002-0694-4817), and 
    [Herman, P.M.J.](https://orcid.org/0000-0003-2188-6341). 
    (2022). Machine learning techniques to aid in the development of nature-based solutions. *Journal*, V(I):pp-pp.
    doi:[TBD]().

The addition of the web-API has also been part of a presentation at the following conference (*presenter in bold-face*):
*   [**Hendrickx, G.G.**](https://orcid.org/0000-0001-9523-7657), 
    [Antol&iacute;nez, J.A.&Aacute;](https://orcid.org/0000-0002-0694-4817), 
    [Aarninkhof, S.G.J.](https://orcid.org/0000-0002-4591-0257), 
    [Huismans, Y.](https://orcid.org/0000-0001-6537-6111), 
    [Kranenburg, W.M.](https://orcid.org/0000-0002-4736-7913), and 
    [Herman, P.M.J.](https://orcid.org/0000-0003-2188-6341). 
    March 4, 2022.
    Combining machine learning and process-based models to enhance the understanding of estuarine salt intrusion and
    development of estuary-scale nature-based solutions. 
    *Ocean Sciences Meeting 2022*. Online.

## License
This repository is licensed under [`Apache License 2.0`](LICENSE).
