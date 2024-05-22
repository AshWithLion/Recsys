# Movie Recommendation System using Two-Tower Model

This repository contains the implementation of a movie recommendation system using a Two-Tower Model. The project is based on the MovieLens 25M dataset and was developed in Google Colab.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to build a robust movie recommendation system that predicts user ratings for movies and recommends top N movies to users that they haven't watched yet. The Two-Tower Model consists of separate neural networks (towers) for users and items, trained simultaneously to capture user-item interactions effectively.

## Dataset

The MovieLens 25M dataset is used in this project. You can download the dataset manually from [MovieLens](https://grouplens.org/datasets/movielens/25m/).

After downloading, upload the dataset files (`movies.csv`, `ratings.csv`, `tags.csv`) to your Google Colab environment.

## Requirements

- Python 3.x
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Polars
- Implicit

You can install the required libraries using the following command:

```bash
pip install torch pandas numpy scikit-learn matplotlib polars implicit
```

## Models

Two-Tower Model

User Tower: Embeds user IDs into dense vectors and processes them through fully connected layers.

Item Tower: Embeds item (movie) IDs into dense vectors and processes them through fully connected layers.

Candidate Retrieval: Computes the dot product between user and item embeddings to generate relevance scores.

Ranking Model: Uses the user, item, and tag embeddings to predict ratings or ranking scores.

## Additional Models

TransformerRecommender: A transformer-based model for recommendations.

MatrixFactorizationNN: Matrix factorization with neural networks.

DeepNCF: Deep Neural Collaborative Filtering.

CFNN: Collaborative Filtering with Neural Networks.

## Evaluation Metrics

The models are evaluated using the following metrics:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Normalized Discounted Cumulative Gain (NDCG)

Precision

Recall
