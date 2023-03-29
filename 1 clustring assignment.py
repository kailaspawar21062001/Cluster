# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:19:11 2023

@author: kailas
"""
################################################################################################
1] HIERARCHICAL CLUSTERING


1]Problem:=(crime_data.csv)

BUSINESS OBJECTIVE:-Perform the clusters for Crime data using HIERARCHICAL CLUSTERING.





#Import Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


#Dataset
data=pd.read_csv("D:/data science assignment/Assignments/cluster/crime_data.csv")

#EDA
data.info()
data.describe()
data=data.rename(columns={'Murder':'murder','Assault':'assault','UrbanPop':'urbanpop','Rape':'rape'})
data.drop(['Unnamed: 0'],axis=1,inplace=True)
data.shape
data.tail()
data.head()

#Normalization Function
def norm_func(i):
    x = (i - i.min()) /(i.max() - i.min())
    return(x)

#Normalized the DataFrame(data)
data1=norm_func(data.iloc[:,:])
data1.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(data1,method='complete',metric='euclidean')


#Dendrogram           
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  
    leaf_font_size = 10 
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram

from sklearn.cluster import AgglomerativeClustering
h=AgglomerativeClustering(n_clusters=5,linkage='complete').fit(data1)
h.labels_

cluster_labels=pd.Series(h.labels_)

data1['clust'] =cluster_labels # creating a new column and assigning it to new column 

#Rearrange the attributes,in which 1st columns is of clust attributes
data1=data1.iloc[:,[4,0,1,2,3]]
data1.head()

# Aggregate mean of each cluster
data1.iloc[:,:].groupby(data1.clust).mean()

# creating a csv file 
data1.to_csv("crime.csv", encoding = "utf-8")







2]Problem:- EastWestAirlines.csv

BUSINESS OBJECTIVE:-Perform clustering (Hierarchical) for the airlines data to obtain optimum number of clusters.    



# Dataset
data=pd.read_excel("D:/data science assignment/Assignments/cluster/EastWestAirlines .xlsx",sheet_name='data')

#EDA
data.info()
data.shape
data.head()
data.tail()
data.describe()

#Normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)

#Normalized the DataFrame(df)
data1=norm_func(data.iloc[:,:])
data1.describe()

#creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(data1,method='complete',metric='euclidean')

#Dendrogram
plt.figure(figsize=(15,8));plt.title('hierarchy clustering dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
               leaf_rotation=0,
               leaf_font_size=15,
)
plt.show()


#Now applying the Agglomerative Clustering to choose 5 clusters.
from sklearn.cluster import AgglomerativeClustering 
h=AgglomerativeClustering(n_clusters=5,linkage='complete').fit(data1)
h.labels_

cluster_labels=pd.Series(h.labels_)

#Creating a new column and assign it to this new column.
data['clust']=cluster_labels


#Rearrange the features,and get Clust features as 1st.
data=data.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
data.head()

#mean of each cluster
data.iloc[:, :].groupby(data.clust).mean()

#Creating a csv file
data.to_csv('eastwest.csv',encoding='utf-8')
import os
os.getcwd()


###################################################################################################

2] K-MEANS CLUSTERING


1]Problem:-crime.csv

BUSINESS OBJECTIVE:-Perform the clusters for Crime data using K-Means CLUSTERING.



#Import Liabrary
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


#Dataset
data=pd.read_csv("D:/data science assignment/Assignments/cluster/crime_data.csv")

#EDA
data.info()
data.shape
data.head()
data.tail()
data.describe()
#Remove the unwanted column
data1=data.drop(['Unnamed: 0'],axis=1)
data1.head()

#Normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return (x)

#Normalized the dataframe(data)
data1=norm_func(data1.iloc[:,:])
data1.describe()


#Scree Plot or elbow curve
TWSS=[]
k = list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data1)
    TWSS.append(kmeans.inertia_)
    
TWSS
 
#Scree PLot
plt.plot(k,TWSS,'ro-');plt.title('scree plot');plt.xlabel('no of clusters');plt.ylabel('total within sum of square')   

#Select 4 clusters from above scree plot ,which is optimum no of clusters.
model=KMeans(n_clusters=4)
model.fit(data1)

# getting the labels of clusters assigned to each row
model.labels_ 

# creating a  new column and assigning it to new column
data['clust']=pd.Series(model.labels_) 
data.head()


#Put the 'clust' features at 1st position in given dataset.
data=data.iloc[:,[5,0,1,2,3,4]]
data.head()

data.loc


#Creatng a CSV file
data.to_csv('crimekmeans.csv',encoding='utf-8')
import os
os.getcwd()






2]Problem:-  EastWestAirlines.csv

    
BUSINESS OBJECTIVE:-Perform clustering (K-Means) for the airlines data to obtain optimum number of clusters.    


from sklearn.cluster import KMeans

#dataset
data=pd.read_excel("D:/data science assignment/Assignments/cluster/EastWestAirlines .xlsx",sheet_name='data')

#EDA
data.info()
data.shape
data.head()
data.tail()
data.describe()

#Normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)
    
#Normalizing the DataFrame(data)
data1=norm_func(data.iloc[:,:])    
data1.describe()

#Scree Plot/Elbow Curve

TWSS=[]
k = list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data1)
    TWSS.append(kmeans.inertia_)
    
TWSS    

#Plot Scree plot
plt.plot(k,TWSS,'ro-');plt.title('scree plot');plt.xlabel('no of clusters');plt.ylabel('total within sum of clusters')

#Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(data1)

# getting the labels of clusters assigned to each row 
model.labels_
# creating a  new column and assigning it to new column 
data['clust']=pd.Series(model.labels_)

data.head()

#Put the 'clust' features at 1st position in given datset.
data=data.iloc[:,[12,11,10,9,8,7,6,5,4,3,2,1,0]]

#Creating a CSV file
data.to_csv('airlineskmean.csv',encoding='utf-8')
import os
os.getcwd()

###################################################################################################################### 

3]DBSCAN CLUSTERING
            
                     
            
1]Problem:-crime.csv

    
BUSINESS OBJECTIVE:-Perform the clusters for Crime data using DBSCAN CLUSTERING.
            
        
#Import Liabrary            
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


#dataset
data=pd.read_csv("D:/data science assignment/Assignments/cluster/crime_data.csv")

#EDA
data.info()
data.head()            
data.tail()
data.shape
data.describe()

#Removing Unwanted Column.
data1=data.iloc[:,1:4]
data1.values

#Standardized the data
s=StandardScaler()
data1=s.fit_transform(data1)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters
dbscan=DBSCAN(eps=2,min_samples=5)
dbscan.fit(data1)

# getting the labels of clusters assigned to each row 
dbscan.labels_
# creating a  new column and assigning it to new column 
data['clust']=pd.Series(dbscan.labels_)


#Put the 'clust' features at 1st position in given datset.
data=data.iloc[:,[5,4,3,2,1,0]]

#Creating a CSV file
data.to_csv("dbscancrime.csv", encoding = "utf-8")

import os
os.getcwd()






2]Problem:-  EastWestAirlines.csv

    
BUSINESS OBJECTIVE:-Perform clustering (DBSCAN) for the airlines data to obtain optimum number of clusters.    



    
#Dataset
data=pd.read_excel("D:/data science assignment/Assignments/cluster/EastWestAirlines .xlsx",sheet_name='data')

#EDA
data.info()
data.shape
data.head()
data.tail()
data.describe()

#Standardized the data
s=StandardScaler()
data1=s.fit_transform(data)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters
dbscan=DBSCAN(eps=2,min_samples=10)
dbscan.fit(data1)

# getting the labels of clusters assigned to each row 
dbscan.labels_
# creating a  new column and assigning it to new column 
data['clust']=pd.Series(dbscan.labels_)

#Put the 'clust' features at 1st position in given datset.
data=data.iloc[:,[12,11,10,9,8,7,6,5,4,3,2,1,0]]

#creating the CSV file
data.to_csv("dbscanairlines.csv", encoding = "utf-8")
import os
os.getcwd()
