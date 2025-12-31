# Iris Species Classification Challenge 🌸

## Overview

### Description

The Iris flower dataset is one of the most famous datasets in machine learning and statistics. This challenge requires
you to build a classification model that can accurately predict the species of Iris flowers based on their physical
measurements.

The dataset contains measurements of 150 Iris flowers from three different species: Iris setosa, Iris versicolor, and
Iris virginica. Each flower is characterized by four features:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Your task is to train a machine learning model on the training set and predict the species for flowers in the test set.

### Acknowledgements

This classic dataset was introduced by British statistician and biologist Ronald Fisher in his 1936 paper "The use of
multiple measurements in taxonomic problems". The dataset is widely used for testing machine learning algorithms and is
available in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

If you use this dataset in publication, please cite:
*Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936)*

## Data

### Dataset Description

The competition dataset consists of 150 samples split into:

- **Training set**: 120 samples (80%)
- **Test set**: 30 samples (20%)

Each species is equally represented with balanced class distribution.

#### Files

- **train.csv** - the training set containing features and target labels
- **test.csv** - the test set containing only features (species labels withheld)
- **sample_submission.csv** - a sample submission file in the correct format
- **test_answer.csv** - ground truth labels for evaluation (hidden in real competition)

#### Data Fields

**Training Set (train.csv)**

- `sepal_length`: Sepal length in cm (float)
- `sepal_width`: Sepal width in cm (float)
- `petal_length`: Petal length in cm (float)
- `petal_width`: Petal width in cm (float)
- `species`: Target variable - Iris species (string: setosa / versicolor / virginica)

**Test Set (test.csv)**

- `id`: Unique identifier for each sample (integer)
- `sepal_length`: Sepal length in cm (float)
- `sepal_width`: Sepal width in cm (float)
- `petal_length`: Petal length in cm (float)
- `petal_width`: Petal width in cm (float)

### Sample Data

| sepal_length | sepal_width | petal_length | petal_width | species    |
|--------------|-------------|--------------|-------------|------------|
| 5.1          | 3.5         | 1.4          | 0.2         | setosa     |
| 7.0          | 3.2         | 4.7          | 1.4         | versicolor |
| 6.3          | 3.3         | 6.0          | 2.5         | virginica  |

## Evaluation

### Evaluation Metric

Submissions are evaluated using **Classification Accuracy**:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

The accuracy score ranges from 0.0 to 1.0, where 1.0 represents perfect classification.

### Submission Format

Submissions must be in CSV format with the following columns:

- `id`: The test sample identifier (must match test.csv)
- `species`: Your predicted Iris species (must be one of: setosa, versicolor, virginica)

**Example submission file:**

```csv
id,species
1,versicolor
2,setosa
3,virginica
4,versicolor
5,virginica
...
```

**Important Notes:**

- Ensure your submission includes predictions for all test samples
- The `id` column must match the IDs in test.csv
- Species names must be lowercase and exactly match: `setosa`, `versicolor`, or `virginica`
- Missing predictions or incorrect format will result in evaluation errors

### Submission Instructions

1. Train your model using `train.csv`
2. Generate predictions for `test.csv`
3. Format your predictions according to `sample_submission.csv`