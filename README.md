# HeartbeatClassification

In the data_preprocessing folder there are two files, one with the explanation of the preprocessing phase on the four datasets merged, and the other file contains the same phases, without explanation, on the dataset MIT-BIHArrhythmiaDatabase

In the models folder, there are 3 folder and two files:

                                                      -merged_datasets: classification phase on the entire dataset, using the functions from utils.py
                                                      
                                                      -MIT-BIHArrhythmiaDB: classification phase on the quoted dataset, with more pipelines, the functions present in utils.py are also visible on the files on this folder
                                                      
                                                      -prove: just a test with undersampling and KNN on one of the four dataset
                                                      
                                                      -compare_dataset_KNN: in the previous folder are present the comparison between different pipelines and different classifiers on the same dataset. Here we compare the results of KNN on each of the four dataset.  
                                                      
                                                      -utils.py: it contains the most used functions with an explanation above each of them
