# Test script for learning how tto use the evaluate for 

import preprocess
from preprocess import visualize_map
import evaluate
import numpy as np
import pdb

def test1():
    train_dataset, val_dataset, test_dataset= preprocess.retrieve_heart_dataset()


    metric = evaluate.load("mean_iou")
    predicted = np.ones(train_dataset[0][1].shape, dtype= np.int32)


    label = train_dataset[0][1].numpy()
    label = label.astype(np.int32)


    image_np = train_dataset[0][2]
    visualize_map(image_np, label)

    # these should both have the same size and datatype.
    
    print(np.unique(label))
    print(np.unique(predicted))



    metric.add(predictions = predicted, references = label)

    metrics = metric.compute(num_labels = 2, ignore_index = 0, reduce_labels = False) 
    # ignore_index useful for specifying which class is a background
    # how does num_labels relate to ignore index? do I need to have 2 labels representing bg?
    # The idea here is that the background seems to be completely disregarded when it comes to creating classes is this a problem in cost calculations?

    print(metrics["mean_iou"])


import numpy as np

def example_test():
    
    mean_iou = evaluate.load("mean_iou")
    # suppose one has 3 different segmentation maps predicted
    predicted_1 = np.array([[1, 2], [3, 4], [5, 255]])
    actual_1 = np.array([[0, 3], [5, 4], [6, 255]])
    predicted_2 = np.array([[2, 7], [9, 2], [3, 6]])
    actual_2 = np.array([[1, 7], [9, 2], [3, 6]])
    predicted_3 = np.array([[2, 2, 3], [8, 2, 4], [3, 255, 2]])
    actual_3 = np.array([[1, 2, 2], [8, 2, 1], [3, 255, 1]])
    predictions = [predicted_1, predicted_2, predicted_3]
    references = [actual_1, actual_2, actual_3]

    mean_iou.add(predictions = predicted_1, references = actual_1)
    results = mean_iou.compute(num_labels=2, ignore_index=255, reduce_labels=False)
    print(results) # doctest: +NORMALIZE_WHITESPACE


    

def simplified_example(): 

    predicted_1 = np.array([[0, 0], [1, 1], [1, 1]])
    actual_1 = np.array([[0, 0], [1, 1], [0, 0]])

    # the iou in this case is 1 because we actually don't care
    
    mean_iou = evaluate.load("mean_iou")

    mean_iou.add(predictions=predicted_1, references=actual_1)
    results = mean_iou.compute( num_labels=2, ignore_index=255, reduce_labels=False)

    print(results)
if __name__ == "__main__": 
    simplified_example()