import gradio as gr
from transformers import pipeline
from fastai.vision.all import *

def label_func(x):
    return x.parent.name
learn=load_learner("model.pkl")
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
# gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3)).launch(share=True)
title = "Indian Food Classifier"
description = "A food classifier trained on the Indian Food Images dataset on Kaggle with fastai. Created by Krithik Ravindran."
examples = ['gulabjamun.jpeg','Rasgulla.jpg']
interpretation = 'default'
gr.Interface(fn=predict,inputs="image",outputs="label",title=title,description=description, examples=examples).launch(share=True)


