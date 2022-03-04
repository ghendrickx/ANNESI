# ANNESI: Artificial neural network for estuarine salt intrusion
The neural network can also be used separately from the web-API. This allows for evaluating a wide range of estuarine
configurations as well as a more stochastic approach.

For this ''limited'' approach, certain requirements from [`requirements.txt`](../requirements.txt) are redundant (see
[*Requirements*](../README.md#Requirements)).

## Basic usage
The usage of the neural network requires importing and initialising the `NeuralNetwork` in a straightforward manner:
```python
from machine_learning.neural_network import NeuralNetwork

nn = NeuralNetwork()
```
The most basic usage of the neural network encompasses a single prediction, using the `single_predict()`-method:
```python
from machine_learning.neural_network import NeuralNetwork

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
```commandline
10934.607982635498
```

## Data set prediction
Multiple predictions can be provided without `for`-looping the `single_predict()`-method. Instead, the input data should
be provided in a `pandas.DataFrame` containing all relevant data, i.e.:
```python
columns = [
    'tidal_range', 
    'surge_level', 
    'river_discharge', 
    'channel_depth', 
    'channel_width', 
    'channel_friction',
    'convergence', 
    'flat_depth_ratio', 
    'flat_width', 
    'flat_friction', 
    'bottom_curvature', 
    'meander_amplitude',
    'meander_length',
]
```
In such a case, the `python`-file will look similar to the following MWE:
```python
import pandas as pd
from machine_learning.neural_network import NeuralNetwork

# define the data set (dummy data, provide real data)
columns = [
    'tidal_range', 
    'surge_level', 
    'river_discharge', 
    'channel_depth', 
    'channel_width', 
    'channel_friction',
    'convergence', 
    'flat_depth_ratio', 
    'flat_width', 
    'flat_friction', 
    'bottom_curvature', 
    'meander_amplitude',
    'meander_length',
]
df = pd.DataFrame(data=1, columns=columns)

# run neural network
nn = NeuralNetwork()
predictions = nn.predict(df)

# print predictions
print(predictions)
```
When the data is stored in a file, the neural network is also able to read it directly from the file, as long as the 
headers in the file correspond to the input parameters, i.e. the above defined `columns`:
```python
from machine_learning.neural_network import NeuralNetwork

# run neural network
nn = NeuralNetwork()
predictions = nn.predict_from_file(file_name='file_name.csv', directory='directory/to/file')

# print predictions
print(predictions)
```

## Stochastic approach
At last, there is the option for a stochastic approach. This approach can become computationally more demanding than the
other approaches because it performs many predictions. Nevertheless, in case there is uncertainty about one or more of 
the input parameters, the `estimate()`-method can be used.

In the `estimate()`-method, the input parameters with uncertainty are provided as (1) ranges (using a `list` or `tuple`)
when a range of values is known; or (2) `None` when nothing is known, which results in the neural network using the
range of the training data set:
```python
from machine_learning.neural_network import NeuralNetwork

nn = NeuralNetwork()

# estimate
estimation = nn.estimate(
    tidal_range=2.25,
    surge_level=0,
    river_discharge=[7750, 20000],
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

# print estimation
print(estimation)
```
This will return some basic statistics:
```
count        3.000000
mean      9945.273995
std       4593.710587
min       6172.037721
25%       7387.638986
50%       8603.240252
75%      11831.892133
max      15060.544014
Name: L, dtype: float64
```
In addition, the above example returns a `warning` because the provided range exceeds the training data:
```
Defined range exceeds training data; "river_discharge" range used: (7750, 15999.456290763)
```
Note that the above statistics are based on a sample size of only three samples (`count        3.000000`). This is the
default value, which can be changed, e.g. to 100 samples:
```python
from machine_learning.neural_network import NeuralNetwork

nn = NeuralNetwork()

# estimate
estimation = nn.estimate(
    tidal_range=2.25,
    surge_level=0,
    river_discharge=[7750, 20000],
    channel_depth=20,
    channel_width=1000,
    channel_friction=.023,
    convergence=1e-4,
    flat_depth_ratio=0,
    flat_width=500,
    flat_friction=.05,
    bottom_curvature=1e-5,
    meander_amplitude=1000,
    meander_length=20000,
    parameter_samples=100  # defaults to 3
)

# print estimation
print(estimation)
```
This returns some more reliable statistics but also increases the computational costs, especially when there are 
multiple unknowns in the input space:
```
count      100.000000
mean      9283.223510
std       2617.902980
min       6172.035933
25%       6961.381435
50%       8603.241146
75%      11131.941676
max      15060.544014
Name: L, dtype: float64
```

## Other options
There are a few other options available when using the neural network as stand-alone, i.e. separate from the web-API:
*   Predict multiple output variables/change the output variable(s): 
    ```python
    from machine_learning.neural_network import NeuralNetwork

    nn = NeuralNetwork()
    nn.output = 'L', 'V'
    ```
    The possible output variables can be retrieved from the neural network using the following command: 
    `NeuralNetwork.get_output_vars()`. (An overview of the input parameters can be extracted with 
    `NeuralNetwork.get_input_vars()`.)
    
*   The `estimate()`-method can also return the (statistics of) the input parameters used for the stochastic approach:
    ```python
    from machine_learning.neural_network import NeuralNetwork

    nn = NeuralNetwork()
    
    # estimate
    estimation = nn.estimate(
        tidal_range=2.25,
        surge_level=0,
        river_discharge=[7750, 20000],
        channel_depth=20,
        channel_width=1000,
        channel_friction=.023,
        convergence=1e-4,
        flat_depth_ratio=0,
        flat_width=500,
        flat_friction=.05,
        bottom_curvature=1e-5,
        meander_amplitude=1000,
        meander_length=20000,
        include_input=True  # defaults to False
    )
    ```
    
*   The `estimate()`-method can also return the full data set that is used to determine the statistics:
    ```python
    from machine_learning.neural_network import NeuralNetwork

    nn = NeuralNetwork()
    
    # estimate
    estimation = nn.estimate(
        tidal_range=2.25,
        surge_level=0,
        river_discharge=[7750, 20000],
        channel_depth=20,
        channel_width=1000,
        channel_friction=.023,
        convergence=1e-4,
        flat_depth_ratio=0,
        flat_width=500,
        flat_friction=.05,
        bottom_curvature=1e-5,
        meander_amplitude=1000,
        meander_length=20000,
        statistics=False  # defaults to True
    )
    ```

For more information and options, see the [source code](neural_network.py).
