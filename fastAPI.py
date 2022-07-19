#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  18 20:30:11 2022

@author: venky
"""

import uvicorn
from fastapi import FastAPI
import pickle


app = FastAPI()

@app.get('/')
def home():
    return{'text' :'Price prediction of Houses'}

@app.get('/predict')
def predict(
    Median_Income: int,
    Median_Age: int,
    Tot_Rooms: int,
    Tot_Bedrooms: int,
    Population: int,
    Households: int,
    Distance_to_coast: int,
    Distance_to_LA:int,
    Distance_to_SanDiego: int,
    Distance_to_SanJose: int,
    Distance_to_SanFrancisco:int):
    
    model = pickle.load(open("reg_model.pkl","rb"))
    
    makeprediction = model.predict([[Median_Income, Median_Age, Tot_Rooms,
                Tot_Bedrooms, Population, Households, Distance_to_coast, Distance_to_LA,
                Distance_to_SanDiego, Distance_to_SanJose,  Distance_to_SanFrancisco]])
    output = round(makeprediction[0])
    return {'Price prediction of Houses: {}'.format(output) + ' USD'} 

if __name__ == '__main__':
    uvicorn.run(app)





    



















    
    
    
    
    
    
    
    
    
