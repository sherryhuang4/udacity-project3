# Data Science Pipeline for Fashion Forward Forecasting

In this project scenario, we have recently joined StyleSense, a rapidly growing women's clothing retailer as a data scientist. We are tasked with leveraging existing data on customer reviews to build a predictive model of whether or not our customers would recommend the product.

The data includes numerical, categorical, and text data, so our pipeline has preprocessing, feature engineering, and model prediction steps.

## Getting Started

### Dependencies

My code requires the following Python packages:

```
spacy
pandas
matplotlib
seaborn
scikit-learn
numpy
```

### Installation

You will need to install Jupyter Notebook if you haven't already. Install spacy with the following code:

```
! python -m spacy download en_core_web_sm
```

Install any packages listed under the Dependencies section you don't already have with `pip install`.

## Project Instructions

Open the `starter.ipynb` file in Jupyter Notebook and run each cell sequentially.

### Load Data

This part of the code loads the data in, prepares the feature and target variables, and splits the data into training and test sets.

### Data Exploration

This part of the code checks for NULL data and creates a correlation matrix.

### Building Pipeline

#### Numerical Data

For the numerical data (Clothing ID, Age, Positive Feedback Count), the numerical pipeline scales the data with MinMaxScaler.

#### Categorical Data

For the categorical data (Class Name, Division Name, Department Name), the categorical pipeline one hot encodes the data with OneHotEncoder.

#### Text Data

For the text data (Title, Review Text), the text pipeline includes both a character counts pipeline and a TFIDF pipeline. The character counts pipeline counts the number of times spaces, exclamation points, and question marks appear in the text. The TFIDF pipeline vectorizes and applies lemmatization to the text.

### Training Pipeline

This part of the code fits and trains a random forest classifier model to our data and looks at the accuracy of our model.

### Fine-Tuning Pipeline

This part of the code uses RandomizedSearchCV to find the best parameters and estimators for our model.