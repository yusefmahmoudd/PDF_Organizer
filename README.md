This project basically takes a set of unorganized PDFs and organizes them based on the text they contain. By grabbing the first page of each pdf (which normally contains the abstract and introduction)
we are able to convert these texts into text embeddings which we can then use to organize them by k means clustering. By this we then move the unorganized pdfs to their respected cluster value. 

Main things that still need to be implemented is some code to tell us the amount of clusters we need due to the variable amount of pdfs someone may need to organize. 
As of right now this project also needs a litle bit of edge cases to make it work better for possible situations.
