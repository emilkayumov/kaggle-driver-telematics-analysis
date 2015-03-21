# kaggle-driver-telematics-analysis
Kaggle competition "Driver Telematics Analysis".

- makefeatures.py - make a numpy array of 77 features of all routes.
- merge.py - combine results in csv from 2 different algorithm.
- randomforest.py - use random forest to predict from numpy array of 77 features
- svm.py - use svm to predict from numpy array of 77 features

my best result is 0.91051 from randomforest without merge

svm and merge rf with svm made worse result (0.87060 and 0.90004)
