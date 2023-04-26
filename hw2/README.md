# Homework 2:

Results and finding of this homework could be found [here](https://github.com/herjanice/computer-vision-course/blob/main/hw2/report.pdf)

 Basics
- What the pipeline of bag of sifts is.
- How to build a KNN classifier.
- Some useful function in opencv and cyvlfeat.
- How to build convolutional layers, fully-connected layers and residual blocks in CNN-based model.
- How to train a model under pytorch framework. • Advanced learning (optional)
- How to perform simple data augmentation method in pytorch to gain accuracy.
- How to perform data cleaning method on some dirty data.
- How to supply semi-supervised to unlabeled data.

Advanced learning (optional)
- How to perform simple data augmentation method in pytorch to gain accuracy.
- How to perform data cleaning method on some dirty data.
- How to supply semi-supervised to unlabeled data.

## Part 1: Scene Recognition
Task:  Use SIFT in OpenCV to extract features and apply K-Nearest Neighbor algorithm as a weak classifier.

**part1/p1.py**
- Read image, construct feature representations, classify features, etc. 
**part1/get_tiny_images.py ## TO DO ##**
- Build tiny images features.
**part1/build_vocabulary.py ## TO DO ##**
- Sample SIFT descriptors from training images, cluster them with k-means and return centroids.
**part1/get_bags_of_sifts.py ## TO DO ##**
- Construct SIFT and build a histogram indicating how many times each centroid was used.
**part1/nearest_neighbor_classify.py ## TO DO ##**
- Predict the category for each test image. (USE sklearn.neighbors.KNeighborsClassifier)

## Part 2: Image Classification
Task: Use convolutional neural network as a feature extractor and perform image classification.

**part2/main.py**
- Top. Start training and some basics settings, etc. 
**part2/cfg.py**
- Some hyperparameters, seeds setting for certain mode. 
**part2/myDatasets.py**
- Define your customized Datasets for training process. ## TO DO ##
**part2/tool.py ## TO DO ##**
- Functions/tools for saving/loading model parameters, Training/validation process, and some other useful function.
**part2/myModels.py ## TO DO ##**
- DefineyourownmodelzooincludingatleastmyResnetandmyLenet
**part2/eval.py**
- Predict the labels for public datasets. ## TO DO ##
Note : In part 2, feel free to modify all the files as long as it’s reasonable and reproducible.

