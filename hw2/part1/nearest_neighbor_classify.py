from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''

    test_predicts = []

    dist = distance.cdist(test_image_feats, train_image_feats, 'cityblock')

    # Difference distance comparison
    # cityblock = 0.39
    # cosine = 3.28
    # correlation = 3.42
    # jensenshannon = 0.388
    # canberra = 0.382
    # braycurtis = 0.39

    K = 1
    for d in dist:
        top = np.argsort(d)[:K]

        labels = []
        for idx in top:
            labels.append(train_labels[idx])

        # find the most frequent label
        count = 0
        prediction = ''
        for label in labels:
            freq = labels.count(label)
            if freq > count:
                prediction = label

        test_predicts.append(prediction)


    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
