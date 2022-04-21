# Disaster Response Pipeline Project


# Installations
- Python version 3.8.5
- All packages were installed with the Anaconda distribution
- Packages used:
	- pandas
	- pickle
	- nltk
	- sklearn
	- re

# Project Motivation
Detecting emotions is difficult since it is personal, constantly changing, and nuanced.  This goal of this project is to read in a document of text data and classify the underlying emotion of the text.  Here, exploratory data analysis, text processing, and machine-learning pieplines were 
applied to classify the text corpus for the Udacity Data Science Nanodegree program's second project.  A custom from scratch web application was built where a new message can be input and the dashboard displays
the emotion, the sentiment, and an image associated with the emotion.  

The data was taken from a [Kaggle competition](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) in the NLP category.  emotions data sent during different disasters and the associated distinct response categories.  The project also includes
a web application where a new message can be input and the dashboard will display multi-label classification results.  


# File descriptions

| Name| Description |
| ----------- | ----------- |
| eda.py|  Script to perform exploratory data analysis |
| emotions_model.py| Script perform text processing, and apply, evaluate, and save a multi-class classification model. |
| pipeML.pkl| Serialized classification model.  This is generated in the emotions_model.py file.|


# How to interact with this project
The .py files within the repository were designed for others to replicate the analysis if desired.    


## Instructions
1. Run the following commands in the project's root directory to set up your database and model.



2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

## Data preparation summary  


## Results summary


In looking at the distribution of emotions, joy occurs the most, and surprise occurs the least.  However, grouping joy, love, and surprise together as positive and sadness, anger, and fear together as negative, the sentiment 


# Licensing, Authors, Acknowledgements
Thank you to Kaggle for making the data accessible.  Thank you to Udacity for the learnings used from previous projects and applied here.  Thank you to [Zolzaya Luvsandorj](https://zluvsand.github.io/) whose articles covered more advanced
NLP concepts that were applied here and the [Charming Data Channel](https://www.youtube.com/channel/UCqBFsuAz41sqWcFjZkqmJqQ) whose videos covered more advanced web-application topics.    