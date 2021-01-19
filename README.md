# RPR

This is our experiment codes for the paper:

Review Polarity-wise Recommender.

## Environment settings
* Python 3.7
* Tensorflow-GPU 1.14.0
* Numpy 1.19.5
* Pandas 1.1.3

## File specification
* data_load.py : loads the raw json data in path `./raw_data`.
* data_pro.py : processes the loaded data for model training, and the results are saved in path `./pro_data`.
* word2vec_glove.py : processes the pre-trained word embeddings in path `./glove_embeddings` for model training, and the results are saved in path `./pro_data`.
* Model.py : implements the model framework of RPR.
* Model_train.py : integrates the training and testing process of RPR model.

## Usage
* Execution sequence

  The execution sequence of codes is as follows : data_load.py--->data_pro.py--->word2vec_glove.py--->Model_train.py
  
* Execution results

  During the execution, the RPR performance on both training and testing sets will be printed after each optimization epoch:
  
  When the execution finished, the best performance on testing set will be printed:
