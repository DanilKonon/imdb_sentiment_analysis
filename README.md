# IMDB reviews sentiment analysis

This project is done for ODS NLP course. 
The new IMDb reviews dataset was collected in a similar way to the original one. Several models are trained to test the performance. 

## Data 
New IMDb review dataset was collected
Tnree scripts:
1. data/imdb_download.py downloads all movies from 2012 year to 2024 in a directory
2. data/imdb_reviews_download.py for every movie downloads all its reviews 
3. Finally, in jupyter notebook the dataset is created. Labels are assigned, neutral reviews are removed, undersampling is performed for those movies that have too many reviews, and balancing of the dataset is done so that there is an equal number of positive and negative reviews.  

## Training
There are two main models in this repository: tf-idf logistic regression and huggingface based one. 
To train model one should create a config.

An example config for tf-idf model: [an example config](baseline_tfidf_logreg/configs_best/config_trainnew.json)

To launch training:

```python train_tfidf.py --config config.json```

Similarly, huggingface model can be trained. [an example config](baseline_transformers/configs_best/best_config19_trainnew.json)

To launch training:

```CUDA_VISIBLE_DEVICES=0 python trainer_transformer.py --config config.json```


## Code
The code is based on this repository: https://github.com/e0xextazy/nlp_huawei_new2_task


## IMDb review
Original IMDb movie reviews dataset: https://ai.stanford.edu/~amaas/data/sentiment/
