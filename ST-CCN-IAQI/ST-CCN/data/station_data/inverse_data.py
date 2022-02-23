import numpy as np
import pandas as pd

def inverse_MinMaxScaler(list_original,list_pred,list_test):
    min=10000
    max=0
    sum=0
    l_pred=[]
    l_test=[]
    #x_std=(x-xmin)/xmax-xmin
    #x_scaled=x_std*(max-min)+min
    #((x_scaled+1)/2)*(xmax-xmin)+xmin
    for i in range(len(list_original)):
        if list_original[i]<min:
            min=list_original[i]
        if list_original[i]>max:
            max=list_original[i]
        sum+=list_original[i]
    avg = sum/len(list_original)
    max_min = max-min
    print(avg)
    print(max_min)

    for i in range(len(list_pred)):
        x=((list_pred[i][0]+1)/2)*(max_min)+min
        l_pred.append(x)
        y=((list_test[i][0]+1)/2)*(max_min)+min
        l_test.append(y)

    return l_pred,l_test

def inverse_StandardScaler(list_original,list_pred,list_test):
    sum=0
    l_pred=[]
    l_test=[]
    for i in range(len(list_original)):
        sum+=list_original[i]
    avg=sum/len(list_original)

    sigma2=0
    for i in range(len(list_original)):
        sigma2 += (list_original[i]-avg)**2
    sigma=np.sqrt(sigma2/len(list_original))

    for i in range(len(list_pred)):
        x=list_pred[i][0]*sigma+avg
        l_pred.append(x)
        y = list_test[i][0] * sigma + avg
        l_test.append(y)
    return l_pred,l_test
