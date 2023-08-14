# TIF - The Trajectory Interval Forest Classifier for Trajectory Classification

GPS devices generate spatio-temporal trajectories for different types of moving objects.
Scientists can exploit them to analyze migration patterns, manage city traffic, monitor the spread of diseases, etc. 
The wide variety of mobility data available allows the creation of machine learning models that can identify different types of movements. 
However, many current state-of-the-art models that use this data type require a not negligible running time to be trained. 
To overcome this issue, inspired by time series analytical approaches, we propose the Trajectory Interval Forest (TIF) classifier, an efficient model designed to have a high throughput.
TIF works by calculating various mobility-related statistics over a set of randomly selected intervals. 
These statistics are used to create a simplified representation of the data, i.e., as features, which becomes then the input of a Random Forest classifier.
We test the effectiveness of TIF by comparing it against state-of-the-art competitors on real world datasets. 
Our results show that TIF is comparable to or better than state-of-art in terms of accuracy but is orders of magnitude faster.

## Setup

### Using PyPI

```bash
  pip install tif (coming soon)
```

### Manual Setup

```bash
git clone https://github.com/USERNAME/T-CIF
cd TCIF
pip install -e .
```

Dependencies are listed in `requirements.txt`.
Dependencies for the experimental part are listed in `requirements_optional.txt`.


## Running the code

You can run TIF using the following code snippet.

```python
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from TCIF.algorithms.utils import prepare
from TCIF.classes.T_CIF_observation import T_CIF_observations
from TCIF.classes.T_CIF_space import T_CIF_space
from TCIF.classes.T_CIF_time import T_CIF_time

# read the dataset and sort the record by trajectory identifier and timestamp
df = pd.read_csv("dataset").sort_values(by=["tid", "t"])

# select from the dataset only the relevant features: trajectory id, target class, lat, lon, timestamp
df = df[["tid", "class", "c1", "c2", "t"]]

# split the data in train and test using the trajectory id
tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                             df.groupby(by=["tid"]).max().reset_index()["class"],
                                             test_size=.3,
                                             stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                             random_state=3)

# convert the dataset in TIF data structure
id_train, y_train, lat_train, lon_train, time_train = prepare(df, tid_train)
id_test, y_test, lat_test, lon_test, time_test = prepare(df, tid_test)
X_train = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_train, lon_train, time_train)]
X_test = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_test, lon_test, time_test)]

# Train and test the model (T_CIF_time() or T_CIF_space())
tif = T_CIF_observations(n_trees=1000, n_interval=50, min_length=10, interval_type=None).fit(X_train, y=y_train) 
y_pred = tif.predict(X_test)
print(classification_report(y_test, y_pred))
```

In alternative, you can use only the feature extraction function by doing: `tif.transform(X)`.

Code with examples on real datasets can be found in the `Experiments/` directory.


## Docs and reference


You can (soon) find the software documentation in the `/docs/` folder and 
a powerpoint presentation on TIF can be found [here](http://example.org).
You can cite this work with
```
SOON
```
