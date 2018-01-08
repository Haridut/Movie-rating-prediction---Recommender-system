#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:10:55 2017

@author: Haridut
"""

import pandas as pd
import numpy as np
from math import sqrt
import sys
#importing the datasets
train_df=pd.read_csv('/Users/Haridut/Documents/UIUC/DataMining /project/train.txt')
movie_df=pd.read_csv('/Users/Haridut/Documents/UIUC/DataMining /project/movie.txt')
#test_df=pd.read_csv('/Users/Haridut/Documents/UIUC/DataMining /project/test.txt')
user_df=pd.read_csv('/Users/Haridut/Documents/UIUC/DataMining /project/user.txt')
test_df=pd.read_csv('/Users/Haridut/Documents/UIUC/DataMining /project/test.txt')
#renaming the datasets to perform a merge of dataframes
user_df=user_df.rename(index=str, columns={'ID':'user-Id'})
movie_df=movie_df.rename(index=str, columns={'Id':'movie-Id'})
'''Creating a nested dictionary such that the keys are the users and each of the values are in turn 
dictionaries that have the movies they watched as keys and ratings as values.'''
df=train_df.groupby('user-Id').apply(lambda x: x.set_index('movie-Id')['rating'].to_dict()).to_dict()
#Calculating Pearson's coefficient between two users that are passed into this function'''
def Pearson_coeff(user1,user2):
    both_rated={}
    for movie in df[user1]:
        if movie in df[user2]:
            both_rated[movie]=1
    if len(both_rated)==0:
        return 0
    user1_sum=sum([df[user1][i] for i in both_rated])
    user2_sum=sum([df[user2][i] for i in both_rated])
    user1_sum_sq=sum([pow(df[user1][i],2) for i in both_rated])
    user2_sum_sq=sum([pow(df[user2][i],2) for i in both_rated])
    product_sum = sum([df[user1][i] * df[user2][i] for i in both_rated])
    numerator= product_sum -(user1_sum*user2_sum/len(both_rated))
    denominator=sqrt((user1_sum_sq-pow(user1_sum,2)/len(both_rated))*(user2_sum_sq-pow(user2_sum,2)/len(both_rated)))
    if denominator==0:
        return 0
    else:
        r_value=numerator/denominator
        return r_value
#Making a Matrix that contains Pearson's coefficient between each user.This may take a long time to execute
Pearson_mat=[]
for i in df:
    Pearson_mat.append([0]*len(df))
for i in df:
    for j in range(i,len(df)):
            temp=Pearson_coeff(i,j)
            Pearson_mat[i][j] =temp
            Pearson_mat[j][i]=temp
#Matrix converted to array to facilitate sorting
Pearson_mat=np.array([np.array(x) for x in Pearson_mat])
#Creating a dataframe that has the median rating of each user across all movies
average_df=train_df[['user-Id','rating']].groupby(['user-Id'],as_index=True).median()
#Sorting each row in the matrix in descending order of Pearson's score
neigh=[]
for i in df:
    temp=list(Pearson_mat[i].argsort()[::-1])
    neigh.append(temp)   
#Function returns the k closest neighbor for a given user based on the movie:
def get_neighbors(user,movie,k):
    nearest_neighbors=[]
    for i in neigh[user]:
        if len(nearest_neighbors)==k:
            return nearest_neighbors
        elif movie in df[i]:
            nearest_neighbors.append(i)
    return nearest_neighbors
#This function retruns the predicted rating based on majority vote
def get_rating(user,movie):
    total_dic={}
    neighbors_list=get_neighbors(user,movie,175)
    #Making a dictionary that contains the rating as key and count as value:
    if len(neighbors_list)!=0:
        for i in neighbors_list:
            if Pearson_mat[i][user]<=0:
                continue
            if df[i][movie] in total_dic:
                total_dic[df[i][movie]]+=1
            else:
                total_dic[df[i][movie]]=1
        #Sorting the dictionary in descending order of count based on rating.
        if total_dic!={}:
            sorted_list=sorted(total_dic,key=total_dic.get, reverse=True)
            #Return the rating with majority votes among k neighbors
            return sorted_list[0]
        else:
            rating=average_df.loc[user]['rating']
            return (int(round(rating)))
        
    #If no user has watched the movie, return the user's median rating
    else:
        rating=average_df.loc[user]['rating']
        return (int(round(rating)))
#Writing the predicted ratings for test data onto a CSV file
f=open('/Users/Haridut/Documents/UIUC/DataMining /project/output12.txt','w')
f.write('Id,rating') 
for i in range(len(test_df['Id'])):  
    f.write('\n'+str(test_df['Id'][i])+','+str(get_rating(test_df['user-Id'][i],test_df['movie-Id'][i])))     
        
