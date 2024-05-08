# -*- coding: utf-8 -*-
import numpy as np

def highpass_filtering(t, data, highpass_gain):
    data_out = np.zeros(data.shape)
    data_integral = 0
    
    for i in range(data.shape[0]):
        data_out[i] += highpass_gain*(data[i] - data_integral)
        data_integral += data_out[i]*(t[i] - t[i-1])
        
    return data_out
    

def lowpass_filtering(t, data, lowpass_gain):
    data_out = np.zeros(data.shape)
    
    return data_out


def normalization(data):
    data_max = max(data)
    data_min = min(data)
    
    if data_max - data_min == 0:
        return data
    else:
        return (data - data_min)/(data_max - data_min)
    
    
    
