# DS8001 Course Project – Recommendation Systems  
**Empirical Analysis of Recommendation Algorithms**

## Course Information
- **Course**: DS8001 – Design of Algorithms and Programming for Massive Data  
- **Project Type**: Empirical Analysis  
- **Dataset**: The Movies Dataset (Kaggle) (https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

## Team Members
- Devanshu Prajapati  
- Taran Veer Singh  


## Project Overview
This project performs a comprehensive empirical comparison of multiple recommendation algorithms using a real-world movie dataset. The goal is to evaluate the behavior, accuracy, and scalability of different recommendation strategies under identical preprocessing and evaluation conditions.

The following algorithms are implemented and analyzed:

1. Content-Based Filtering (CountVectorizer & TF-IDF)
2. Collaborative Filtering (User–User and Item–Item)
3. Popularity-Based Recommendation
4. Hybrid Recommendation Model
5. Latent Factor Model using Singular Value Decomposition (SVD)

## Repository Structure:
- archive folder (here you have all datasets that is downloaded form the Kaggle Dataset)
- 1.1.Content based.py
- 1.2.Content based Recommender.py
- Collaborative Filtering.py
- Popularity Based.py
- Hybrid Based Recommender.py
- 5.1.Latent Model.py
- 5.2.Latent Recommender.py
- README.md


---

## Dataset
The project uses **The Movies Dataset** from Kaggle:
- movies_metadata.csv
- credits.csv
- keywords.csv
- links.csv
- links_small.csv
- ratings.csv
- ratings_small.csv

**Source**: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

**Note**: 
- First Download the dataset zip from the above link.
- Unzip the folder.
- That folder name should be archive and it must contain the above mentioned datasets.
- Put the archive folder to this zip folder so that you can access data as in all the pyton files it's in this format (i.e. read_csv("archive/moives_metadata.csv")).



---

## Installation and Requirements

### Python

### Required Libraries
Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk surprise
