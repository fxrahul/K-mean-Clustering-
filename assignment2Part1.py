# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:43:02 2019

@author: Rahul
"""
#---------------------------------------Importing Packages----------------------------------------------------- 
import pandas as pd
import random
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
from statistics import mean
import numpy as np
import collections
from datetime import datetime
import os
from pathlib import Path
#--------------------------------------------------------------------------------------------------------------

#----------------------------------------------Reading CSV File------------------------------------------------
data = pd.read_csv('data.csv')
df = pd.DataFrame(data).astype(float)
#print(df.shape)
#print(dataFrame.iloc[])
#---------------------------------------------------------------------------------------------------------------

medoidVisited = []

#-----------------------------------------Generating Medoids at Start----------------------------------------------------
def generateRandomAtStart(n):
    index = random.sample(range(0, df.shape[0]), n)
    return index
    
#---------------------------------------------------------------------------------------------------------------

#------------------------------------------Generating Medoids From the Clusters---------------------------------
def generateRandomFromClusters(k,previousOrderedClusters):
#    print(previousOrderedClusters)
    index = []
    for i in range(k):
        res = defaultdict(list) 
        for key, val in sorted(previousOrderedClusters.items()): 
            res[val].append(key)
        
        index.append( random.choice(res.get(i+1)) )
    
    if index in medoidVisited:
        generateRandomFromClusters(k,previousOrderedClusters)
    else:
        return index
    
#---------------------------------------------------------------------------------------------------------------

#----------------------------------------------K-Medoid Clustering----------------------------------------------    
def clustering(k):
    print("For K = ",k)
    averageSilhouette = []
    iterations = 0
    clusterChange = True
    previousOrderedClusters = dict()
    while(clusterChange == True and iterations <= 99):
        medoids = []
        
        clusters = dict()
        previousClusters = dict()
        
    #    cost = 0
        if iterations == 0:
            medoidIndex = generateRandomAtStart(k)
        else:
            medoidIndex = generateRandomFromClusters(k,previousOrderedClusters)
        
        
        previousOrderedClusters = dict()
        
        for i in range(k):
            medoids.append(medoidIndex[i])
        medoidVisited.append(medoidIndex)
        
        
        dissimilarityMatrix = []
        
        dataToConsider = df[~(df.index.isin(medoids))]
    
        
      
        
        for j in range(k):
            dissimilarityMatrix.append(euclidean_distances(dataToConsider,[ df.iloc[medoids[j]] ]))
        
        for i in range(dataToConsider.shape[0]):
            minimum = []
            for j in range(len(dissimilarityMatrix)):
                minimum.append(dissimilarityMatrix[j][i])
            minIndex = minimum.index(min(minimum))
            elementOfConsideredData = dataToConsider.iloc[i]
            indexOfElement = df.index.get_loc(elementOfConsideredData.name)
    #--------------------------clusters dictionary structure, key is element index from original dataframe----------
    #--------------------------and value is cluster number
            
            clusters[indexOfElement] = minIndex+1
        for i in range(k):
            clusters[medoids[i]] = i+1
        
        if clusters == previousClusters:
            clusterChange = False
        else:   
            previousClusters = clusters
            previousOrderedClusters = previousClusters
        
        
        
    
        iterations+=1
        
#-------------------------------------------Adding cluster id column--------------------------------------------      
    orderClusters = collections.OrderedDict(sorted(clusters.items()))
    copyDF = df
    clusterNo =  []
    for i in range( len(orderClusters) ):
        clusterNo.append(orderClusters.get(i) )
    
    copyDF['ClusterId'] = clusterNo
#--------------------------------------------------------------------------------------------------------------- 

#--------------------------------------------Generating Cluster Result File------------------------------------   
    dirName = 'clusters'+ ' '+'k= '+str(k) +' '+str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
     
    try:
        # Create target Directory
        os.mkdir(dirName)
        
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    #    print(df)
    p = Path(dirName)
    fileName = 'clusters_' + str(k) + '.csv' 
    copyDF.to_csv(Path(p, fileName))
#--------------------------------------------------------------------------------------------------------------
#------------------------------------------------making clusters----------------------------------------------        
    res = defaultdict(list) 
    for key, val in sorted(clusters.items()): 
        res[val].append(key)
    clusters = res


#--------------------------------------------------------------------------------------------------------------

#---------------------silhouette width calculation-------------------------------------------------------------
    for i in range(k):
        silhouette = 0;
        items = clusters.get(i+1)
        dataForDistance = []
        for j in range(len(items)):
           dataForDistance.append(df.iloc[items[j]])
        for d in range(len(items)):
            silhouetteScore = []
            distance = euclidean_distances(dataForDistance,[dataForDistance[d]])
            a = np.mean(distance)
            #finding nearest cluster to calculate b
            
            distancesOfCluster = []
            for n in range(k):
                dataset =[]
                if n!=i:
                    clusterElements = clusters.get(n+1)
                    for m in range(len(clusterElements)):
                        dataset.append(df.iloc[clusterElements[m]])
                    distanceCalculation = euclidean_distances(dataset,[dataForDistance[d]])
                    avgDistance = np.mean(distanceCalculation)
                    distancesOfCluster.append(avgDistance)
            b = min(distancesOfCluster)
            silhouetteScore.append( (b-a)/max(a,b) )
        silhouette = mean(silhouetteScore)

        print("Silhouette Width for Cluster No",i+1,"is ",silhouette)
        print()
        averageSilhouette.append(silhouette)
    print("Average Silhouette Width for Clustering Solution: ",np.mean(averageSilhouette))
                    
  
#--------------------------------------------------------------------------------------------------------------  
     
#----------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------Start---------------------------------------------------
if __name__ == "__main__":
    clustering(2)
    medoidVisited =[]
    clustering(3)
    
#-------------------------------------------------------End---------------------------------------------------


        
        
        



