from sentence_transformers import SentenceTransformer, util
from PIL import Image
import matplotlib.pyplot as plt
import glob
import torch
import os
import pickle


model= SentenceTransformer('Clip-ViT-B-32')
"""save the model"""
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
"""encoding the image"""
img_names=list(glob.glob('photos/*.jpg'))[:5000]
img_embed = model.encode([Image.open(img) for img in img_names], batch_size=32, convert_to_tensor=True, show_progress_bar=True)
"""save the img embed"""
with open("img_embed", "wb") as fp:   #Pickling
    pickle.dump(img_embed, fp)

with open("img_names", "wb") as fp:   #Pickling
    pickle.dump(img_names, fp)

with open("img_names.txt", 'w') as f:
    for s in img_names:
        f.write(str(s) + '\n')
