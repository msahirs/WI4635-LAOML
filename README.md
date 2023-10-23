# WI4635-LAOML
Repository containing code for the assignments in the course WI4635 - Linear Algebra and Optimisation for Machine Learning - at TU Delft

## Directory Guide
- Project description under `/assets`
- IRIS and URL datasets under `/data`
    - **IRIS**: The dataset contains a set of 150 records under five attributes: sepal length, sepal width, petal length, petal width and species.

    - **URL**: The file url.mat contains variables which we describe as follows:

        - **`FeatureTypes`**: A list of column indices for the data matrices that are real-valued features.
        - **`DayX`** (where X is an integer from 0 to 120): A struct containing the data for day X.
            - **`DayX.data`**: an N x D data matrix where N is the number of URLs (rows), and D is the number of features (columns).
            - **`DayX.labels`**: an N x 1 label vector where 1 indicates a malicious URL and 0 indicates a benign URL

- All code and routines under `/src`

## Notes
- All code written in **Python**
- Only basic Linear Algebra libs, so *Numpy* and *Scipy*, allowed. Maybe other plotting libraries for data visualisation, i.e. *Matplotlib* or *Plotly*
- We gotta be super explicit about everything we do. Like on performance, stability, convergence progression, expectations, etc.
- Report is informal! :D



![Deep Learning lolxd](/assets/meme_deep_learning.jpg)
