from fastai import *
from fastai.vision.all import *
from fastai.vision.widgets import *


#Loading the model

try:
    learn_inf = load_learner('export.pkl')
except FileNotFoundError:
    print("Modelo não encontrado")
    
#defining widgets
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()
btn_run = widgets.Button(description='Classificar')

#Definindo evento onClick para "Classify"
def on_click_classify(change):
#Load the selected image    
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))

#Get the predictions and show them        
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Predição: {pred}; Probabilidade: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)

#Putting widgets in a box
VBox([widgets.Label('Select your gem!'), 
          btn_upload, btn_run, out_pl, lbl_pred])


