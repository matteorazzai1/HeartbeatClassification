import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
from sklearn.preprocessing import OrdinalEncoder

#on this function I handle the warning generated when the precision or the recall are 0/0.
#When we have undefined recall or precision, also the f1-score is undefined and we can decide to set these metrics to 0 or 1 using zero_division field 
# of the classification report.
# This situation could increase (if I set zero_division=1) or decrease (if I set zero_division=0) our results improperly. 
# The following function avoid this behaviour excluding the 0/0 metrics from the list, in this way at the end the mean value of a metric is done only on defined values.

def handle_warning(confusion_matrix,cr):
    
    df=pd.DataFrame()
    for k,v in cr.items():
        if(k!='accuracy' and k!='macro avg' and k!= 'weighted avg'):
            new_v={}
            
            recall=0
            precision=0
            
            #handle precision
            if(k=='0.0'):
                #classe N
                
                for i in range(confusion_matrix.shape[0]):
                    if(confusion_matrix[i][0]!=0):
                        new_v['precision']=v['precision']
                        precision=1
            if(k=='1.0'):
                #classe SVEB
                for i in range(confusion_matrix.shape[0]):
                    if(confusion_matrix[i][1]!=0):
                        new_v['precision']=v['precision']
                        precision=1
            if(k=='2.0'):
                #classe VEB
                for i in range(confusion_matrix.shape[0]):
                    if(confusion_matrix[i][confusion_matrix.shape[0]-1]!=0):  #potrebbe essere di dimensione 2x2 la confusion, e se questa classe c'è, in ogni caso è in posizione dimensione_matrice-1
                        new_v['precision']=v['precision']
                        precision=1
            #handle recall            
            if(v['support']!=0):
                new_v['recall']=v['recall']
                recall=1

            #handle f1_score
            if(recall==1 and precision==1):
                new_v['f1-score']=v['f1-score']
                
            #insert support
            new_v['support']=v['support']
            
            cr[k]=new_v
    print(pd.DataFrame(cr))
    return cr
        

# function for the cross validation using LeaveOneGroupOut (for each fold only the records of one patient are in the test set)
def cross_valid(pipeline, skf, X, y,groups):
    list_df = []
    list_accuracy = []
    list_f1score={}
    labels={0.0:'N',1.0:'SVEB',2.0:'VEB'}
    confusion_matrix_sum = np.zeros((3, 3))

    k = 1
    for i, (train, val) in enumerate(skf.split(X, y, groups)):
        print(f"Fold {k}:")
        print(f"  Test:  patient="+str(i))
        # fit and predict using pipeline
        X_tr = X.to_numpy()[train]
        y_tr = y.to_numpy()[train]
        X_val = X.to_numpy()[val]
        y_val = y.to_numpy()[val]
        
        
        pipeline.fit(X_tr,y_tr)
        y_pred = pipeline.predict(X_val)
        # compute classification report
        # I want two exclude the labels which the precision AND the recall are 0/0
        labels_true=np.unique(y_val)
        labels_pred=np.unique(y_pred)
        index_true = [i for i in range(len(labels_true))]
        index_pred = [i for i in range(len(labels_pred))]
        labels_concatenate=np.concatenate((labels_true[index_true],labels_pred[index_pred]))
        labels_un=np.unique(labels_concatenate)
        cr = classification_report(y_val,y_pred,labels=labels_un,output_dict = True) 
        print(classification_report(y_val,y_pred,labels=labels_un))
        
        # store per-class metrics as a dataframe
        classes = sorted(set(y_val) | set(y_pred))
        display_labels = [labels[cls] for cls in classes]
        
        simple_labels=[cls for cls in classes]
        confusion_matrix_fold = confusion_matrix(y_val, y_pred,labels=simple_labels)

        
        cr_new=handle_warning(confusion_matrix_fold,cr)
        
        df = pd.DataFrame({k:v for k,v in cr_new.items() if k!='accuracy'})
        list_df.append(df)
        
        list_f1score["patient"+str(i)]=({k:v['f1-score'] for k,v in cr_new.items() if k!='accuracy' and v.get('f1-score') is not None})
        
        classes_present = np.unique(np.concatenate((y_val, y_pred)))
        
        num_classes_present = len(classes_present)
        
        class_indices = {classes_present[i]: i for i in range(num_classes_present)}
        
        
        for i in range(num_classes_present):
            for j in range(num_classes_present):
                class_i = classes_present[i]
                class_j = classes_present[j]
                i_fold = class_indices[class_i]
                j_fold = class_indices[class_j]
                confusion_matrix_sum[i_fold, j_fold] += confusion_matrix_fold[i, j]

        ConfusionMatrixDisplay.from_predictions(y_val, 
                                        y_pred,display_labels = display_labels )
        plt.show()
        
        k+=1


    # compute average per-class metrics    
    df_concat = pd.concat(list_df)
    grouped_by_row_index = df_concat.groupby(df_concat.index)
    df_avg = grouped_by_row_index.mean()
    
    # compute average accuracy
    accuracy_avg = np.mean(list_accuracy)
    return df_avg, accuracy_avg,confusion_matrix_sum,list_f1score

# plots that represent the f1-score values for each patient for each class for a specific classifier results
def print_plot(list_f1score):
    f, axarr = plt.subplots(3, sharex=True, figsize=(8,8))
    for i in list(['0.0','1.0','2.0']):
        #print(i)
        list_fscore=({k:list_f1score[k][i] for k in list_f1score if list_f1score[k].get(i) is not None})
        #print(list_fscore)
        axarr[int(float(i))].scatter(list_fscore.keys(),list_fscore.values())
        yticks = np.arange(0, 0.5, 1)
        axarr[int(float(i))].set_yticks(np.arange(0, 1.1, 0.1))
        axarr[int(float(i))].set_ylabel('f1_score')

    axarr[0].set_title("Normal beat")

    axarr[1].set_title("Supraventricular beat")

    axarr[2].set_title("Ventricular beat")

    axarr[2].set_xticklabels([])



    plt.xlabel('patients')




    plt.tight_layout()
    plt.show()


# compare the plots of two different classifier pipelines put on the same figure the two graphics
def compare_plots(list_f,list_f1score):
    f, axarr = plt.subplots(3, sharex=True,figsize=(8,8))
    list_fscore=[]
    list_fscore1=[]
    for i in list(['0.0','1.0','2.0']):
        #print(i)
        list_fscore=({k:list_f1score[k][i] for k in list_f1score if list_f1score[k].get(i) is not None})
        #print(list_fscore)
        axarr[int(float(i))].scatter(list_fscore.keys(),list_fscore.values(),s=10)
        yticks = np.arange(0, 0.5, 1)
        axarr[int(float(i))].set_yticks(np.arange(0, 1.1, 0.1))
        axarr[int(float(i))].set_ylabel('f1_score')

    axarr[0].set_title("Normal beat")

    axarr[1].set_title("Supraventricular beat")

    axarr[2].set_title("Ventricular beat")

    axarr[2].set_xticklabels([])
    
    f2, axarr2 = plt.subplots(3, sharex=True,figsize=(2,2))
    for i in list(['0.0','1.0','2.0']):
        #print(i)
        list_fscore1=({k:list_f[k][i] for k in list_f if list_f[k].get(i) is not None})
        #print(list_fscore1)
        axarr2[int(float(i))]=axarr[int(float(i))].twinx()
        axarr2[int(float(i))].scatter(list_fscore1.keys(),list_fscore1.values(),color="red",s=10)
        yticks = np.arange(0, 0.5, 1)
        axarr2[int(float(i))].set_yticks(np.arange(0, 1.1, 0.1))
        axarr2[int(float(i))].set_yticklabels([])



    plt.xlabel('patients')



    f2.set_visible(False)
    plt.tight_layout()
    plt.show()

# see the data_preprocessing.ipynb file for the explanation of each phase
def preprocessed_dataset(dataset):

    df_cardio=dataset

    df_cardio=df_cardio.dropna(subset=['0_pre-RR','0_post-RR','0_pPeak','0_tPeak','0_rPeak','0_sPeak','0_qPeak','0_qrs_interval','0_pq_interval','0_qt_interval','0_st_interval','0_qrs_morph0','0_qrs_morph1','0_qrs_morph2','0_qrs_morph3','0_qrs_morph4'])
    df_cardio=df_cardio.dropna(subset=['1_pre-RR','1_post-RR','1_pPeak','1_tPeak','1_rPeak','1_sPeak','1_qPeak','1_qrs_interval','1_pq_interval','1_qt_interval','1_st_interval','1_qrs_morph0','1_qrs_morph1','1_qrs_morph2','1_qrs_morph3','1_qrs_morph4'])

    df_cardio=df_cardio.rename(columns={'record':'patient'})

    df_cardio['patient']=df_cardio['patient'].astype('category').cat.codes

    df_cardio=df_cardio[df_cardio['type']!= 'Q']
    df_cardio=df_cardio[df_cardio['type']!= 'F']

    enc=OrdinalEncoder()
    X=df_cardio.drop('type',axis=1)
    y=df_cardio['type']
    encoded_class=enc.fit_transform(y.values.reshape(-1,1))

    df_cardio['type']=encoded_class

    groups=df_cardio.groupby('patient')
    
    for name,patient in groups:
    #print(patient)
        df_patient=pd.DataFrame(patient)
        
        RR_pre0=df_patient['0_pre-RR']
        RR_pre1=df_patient['1_pre-RR']
        RR_post0=df_patient['0_post-RR']
        RR_post1=df_patient['1_post-RR']
        
        RR_pre0=RR_pre0.to_numpy()
        RR_pre1=RR_pre1.to_numpy()
        RR_post0=RR_post0.to_numpy()
        RR_post1=RR_post1.to_numpy()
        
        meanRR_pre0=np.mean(RR_pre0)
        meanRR_pre1=np.mean(RR_pre1)
        meanRR_post0=np.mean(RR_post0)
        meanRR_post1=np.mean(RR_post1)
        
        std0= np.std(RR_pre0)
        std1=np.std(RR_pre1)
        std2=np.std(RR_post0)
        std3=np.std(RR_post1)
        
        RR_pre0_norm=(RR_pre0-meanRR_pre0)/std0
        RR_pre1_norm=(RR_pre1-meanRR_pre1)/std1
        RR_post0_norm=(RR_post0-meanRR_post0)/std2
        RR_post1_norm=(RR_post1-meanRR_post1)/std3
        
        
        df_patient['0_pre-RR']=RR_pre0_norm
        df_patient['1_pre-RR']=RR_pre1_norm
        df_patient['0_post-RR']=RR_post0_norm
        df_patient['1_post-RR']=RR_post1_norm
        
        df_cardio[df_cardio['patient']==name]=df_patient

    return df_cardio

# utility function for retrieve the list of the macro avg f1-score from the entire list with all the metrics for each patient
def retrieve_fscore_avg(list_f):
    f1score_list=[]
    for patient in list_f:
        #print(list_f[patient])
        f1score_list.append(list_f[patient]['macro avg'])
    return f1score_list