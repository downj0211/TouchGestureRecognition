#pGesture_DataProcessor.py
# the function to handle the dataset

import math
import numpy as np
import pandas as pd
from scripts import DataFiltering

def pGesture_DataProcessor(**args):
    data = args['data']
    
    # variable range setting
    for i in range(1 + 1 + 7 + 6 + 7):
        data[:,i] -= data[0,i]
        
    df = pd.DataFrame({'time':data[:,0] - data[0,0]})
    
    insert_dataframe(dataframe = df, data = data)

    reduced_sample_number = data.shape[0]

    # command list
    for command, value in args.items():
        #sample reduction
        if command == 'SAMPLE_REDUCTION':
            reduced_sample_number = value
            
        # high pass filtering
        if command == 'HIGHPASS':
            insert_dataframe(dataframe = df, data = data, value = value, name = '_highpass', function = DataFiltering.highpass_filtering)

        # low pass filtering
        if command == 'LOWPASS':
            insert_dataframe(dataframe = df, data = data, value = value, name = '_lowpass', function = DataFiltering.lowpass_filtering)
            
        if command == 'NORMALIZATION':
            if value:
                insert_dataframe(dataframe = df, data = data, name = '_normalization', function = DataFiltering.normalization)
            
            
    # return the reduced data
    original_sample_number = df.shape[0]            
    df_value = df.values
    df_reduced = np.empty((reduced_sample_number, df.shape[1]),float)
    
    for i in range(reduced_sample_number):
        df_reduced[i,:] = df_value[math.ceil((original_sample_number-1)*i/(reduced_sample_number-1)),:]
    
    return df_reduced




def insert_dataframe(dataframe, data, value = 0, name = '', function = None):
    
    if function ==  DataFiltering.highpass_filtering or function ==  DataFiltering.lowpass_filtering :
        dataframe['energy'+name] = function(data[:,0], data[:,1], value)
        
        for joint_num in range(1,8):
            dataframe['momentum_j'+str(joint_num)+name] = function(data[:,0], data[:,1+joint_num], value)
            
        for force_num in range(1,7):
            dataframe['fext_j'+str(force_num)+name]     = function(data[:,0], data[:,8+force_num], value)
            
        for joint_num in range(1,8):
            dataframe['text_j'+str(joint_num)+name]     = function(data[:,0], data[:,14+joint_num], value)
       
    elif function ==  DataFiltering.normalization:
        dataframe['energy'+name] = function(data[:,1])
        
        for joint_num in range(1,8):
            dataframe['momentum_j'+str(joint_num)+name] = function(data[:,1+joint_num])
            
        for force_num in range(1,7):
            dataframe['fext_j'+str(force_num)+name]     = function(data[:,8+force_num])
            
        for joint_num in range(1,8):
            dataframe['text_j'+str(joint_num)+name]     = function(data[:,14+joint_num])
        
        
    else:
        dataframe['energy'+name] = data[:,1]
        
        for joint_num in range(1,8):
            dataframe['momentum_j'+str(joint_num)+name] = data[:,1+joint_num]
            
        for force_num in range(1,7):
            dataframe['fext_j'+str(force_num)+name]     = data[:,8+force_num]
            
        for joint_num in range(1,8):
            dataframe['text_j'+str(joint_num)+name]     = data[:,14+joint_num]
        
        
    
    
    
    
