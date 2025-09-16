# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 08:04:39 2022

@author: SFerneyhough
"""

import serial
import time
import pandas as pd
from io import StringIO 
import datetime

ser = serial.Serial(port='COM7',
                    baudrate=9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS)


if(ser.isOpen() == False):
    ser.open()


# # Zero scale
# ser.write(b'Z\r\n')
date = datetime.datetime.now()
df_date = pd.DataFrame(['date'])
ser.write(b'SI\r\n')

out = b''
time.sleep(1)
while ser.inWaiting() > 0:
    out += ser.read(1)
    
print(out)

out2 = out.decode('utf-8')

data = StringIO(out2)

df = pd.read_csv(data,sep=';')

df2 = pd.merge(df,date)

ser.close()

