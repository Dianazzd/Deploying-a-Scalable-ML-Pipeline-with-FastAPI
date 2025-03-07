# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
I used Random Forest classifier from sklearn, and the dataset used in this project is a publicly available Census Bureau data

## Intended Use
This project shows the process of deploying a scalable ML pipeline with FastAPI, and uses the model to predict the salary label on a group of features.

## Training Data
The dataset is from the Census Bureau data. The dataset has 15 features. 80% of the dataset was used for training the model.

## Evaluation Data
20% of the original dataset was used for testing the model.

## Metrics
Model performance is measured based on three metrics: f1 score, precision score and recall score.
The values achieved are: Precision: 0.7493 | Recall: 0.6372 | F1: 0.6887

## Ethical Considerations
Data bias may occur during data collection, annotation and preprocessing Preventives steps need to be taken to mitigate bias and to ensure fairness in machine learning models.

## Caveats and Recommendations
dataset used in this project comes from a 1994 Census Bureau data. Machine learning models should be trained on the latest dataset available to ensure a true representation of the sectors the data is about