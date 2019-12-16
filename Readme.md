##Description
This project consists of two main modules:

1- **Text Classification:** This module classifies news articles into 5 categories (politics, sports, entertainment, business,tech) and uses them to learn how to classify an unseen news article to one of these categories. 

It uses the BBC News Articles dataset and utilizes a supervised learning technique (Logistic Regression). The implementation of this project was guided by this blog post https://towardsdatascience.com/text-classification-in-python-dd95d264c802


2- **Text Summarization:**  This module performs Extractive Summarization on the news articles. It was part of my graduation project at university (Github Link: https://github.com/rana1afifi/Text-Simplification-and-Summarization) and is based on the paper Barrios, F., López, F., Argerich, L., & Wachenchauzer, R. (2016). *Variations of the similarity function of textrank for automated summarization*. arXiv preprint arXiv:1602.03606.

##User Guide: 

Run Main.py and it will prompt the user to choose what module to run and read the required parameters for each module. 


####There are 5 python files in this project: 

Main.py --> contains the main function

Classification.py --> contains the classification pipeline 

Summarization.py --> contains the summarization pipeline

Utilities.py --> cotains utilities functions for the other files 


####The project requires the following packages: 

scikit-learn

scipy

numpy 

nltk

pickle

