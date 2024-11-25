from sentence_transformers import SentenceTransformer
import pandas as pd
from pypdf import PdfReader
from sklearn.cluster import KMeans
import os
import shutil

# %%
import matplotlib.pylab as plt
import numpy as np
# %% 

reader1 = PdfReader("jailbrokenllm.pdf") 
page1 = reader1.pages[0]
reader2 = PdfReader("backdoorattacks.pdf")
page2 = reader2.pages[0]
reader3 = PdfReader("efficientdata.pdf")
page3 = reader3.pages[0]
reader4 = PdfReader("taskspecificdata.pdf")
page4 = reader4.pages[0]

page1emb =page1.extract_text()
page2emb = page2.extract_text()
page3emb = page3.extract_text()
page4emb = page4.extract_text()

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

abstract_files = ["jailbrokenllm.pdf","backdoorattacks.pdf","efficientdata.pdf","taskspecificdata.pdf"]
abstract_list = [page1emb,page2emb,page3emb,page4emb]

embeddings = model.encode(abstract_list) # calculate embeddings

# Do KMeans for the abstracts
km = KMeans(n_clusters = 3, random_state=50)
y_predicted = km.fit_predict(embeddings)

# Make a new folder for organized files
organized_folder = "organized_abstracts"
os.makedirs(organized_folder, exist_ok=True)


for cluster in np.unique(y_predicted):
    cluster_folder = os.path.join(organized_folder,f"Cluster_{cluster}")
    os.makedirs(cluster_folder,exist_ok=True)
    
for i,cluster, in enumerate(y_predicted):
    moving_file = abstract_files[i]
    destination_folder = os.path.join(organized_folder,f"Cluster_{cluster}")
    shutil.move(moving_file,os.path.join(destination_folder,os.path.basename(moving_file)))

