#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:20:21 2018

@author: tang
"""
from datetime import datetime
import pandas as pd 

def save_record(output_file_path,
                record_list):                                 

    header_name = ['epoch','train_time','train_acc',
                  'test_time','test_acc',
                  'dropped_samples','total_time']             
    df1 = pd.DataFrame(record_list)

    writer = pd.ExcelWriter(output_file_path)
    df1.to_excel(writer,'Sheet1',header = header_name)
    writer.save() 
    writer.close()

        
        