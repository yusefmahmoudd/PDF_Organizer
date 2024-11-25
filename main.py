from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os,shutil

# get the list of PDF file names in the folder
messy_folder_path = 'C:\\Users\\yusef\\OneDrive\\Desktop\\Unorganized PDFs\\'
pdf_files = []
for file in os.listdir(messy_folder_path):
    if file.endswith('.pdf'):
        pdf_files.append(file)  

# turn the first page of all the PDFs into text embeddings
def text_to_emb(pdf_path):
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    return page.extract_text()

abstract_list = []
for pdf in pdf_files:
    full_path = os.path.join(messy_folder_path, pdf)  
    text_emb = text_to_emb(full_path) 
    abstract_list.append(text_emb)

# convert text embeddings and run kmeans
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(abstract_list) 
km = KMeans(n_clusters = 3, random_state=50)
y_predicted = km.fit_predict(embeddings)

print(y_predicted)

#create required amount of cluster folders
organized_folder_path = 'C:\\Users\\yusef\\OneDrive\\Desktop\\Organized PDFs\\'
os.makedirs(organized_folder_path, exist_ok=True)

for cluster in np.unique(y_predicted):
    cluster_folder = os.path.join(organized_folder_path,f"Cluster_{cluster}")
    os.makedirs(cluster_folder,exist_ok=True)

#move pdf files from jumbled folder to organized folder inside of respected subfolder
for i,cluster, in enumerate(y_predicted):
    moving_file = os.path.join(messy_folder_path,pdf_files[i])
    destination_folder = os.path.join(organized_folder_path,f"Cluster_{cluster}")
    shutil.move(moving_file,os.path.join(destination_folder,os.path.basename(moving_file)))