#!/usr/bin/env python
# coding: utf-8

# In[1]:


#To run a notebook in voila, replace the word "notebooks" in the url with "voila/render"
from fastai import * 
from fastai.vision.all import *
from fastai.vision.widgets import *


# In[2]:


#Carregando o modelo
try:
    learn_inf = load_learner('export.pkl')
except FileNotFoundError:
    print("Modelo não encontrado")


# In[3]:


#Defining widgets 
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()
btn_run = widgets.Button(description='Classify')


# In[4]:


#Definindo evento onClick para "Classify"
def on_click_classify(change):
#Load the selected image    
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))

#Get the predictions and show them        
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[5]:


#Putting widgets in a box
VBox([widgets.Label('Select your gem!'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# In[ ]:




