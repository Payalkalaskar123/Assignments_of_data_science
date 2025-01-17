# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 08:22:13 2024

@author: SAINATH
"""
'''1. You are given a dataset of movies with various attributes like genres, 
keywords, and descriptions. Your task is to build a content-based 
recommendation engine that recommends movies similar to a given movie 
based on these attributes.
Steps:
 Preprocess the Data: Extract relevant features (e.g., genres, overview).
 Vectorize the Text Data: Use TF-IDF on the overview field.
 Compute Similarity: Use cosine similarity to find similar movies.
 Recommend: Given a movie, recommend the top 10 most similar movies based on 
content.
Note: Use IMDB dataset'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data=pd.read_csv("C:/10-recommendation_engine/IMDb_Movie_Reviews.csv")

#step 1:remove the stop words
tfidf=TfidfVectorizer(stop_words="english")
tfidf_matrix=tfidf.fit_transform(data['Review_Text'])#fit and transform the 'overview'data
#step2 find the cosine similarity between the moview
cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
def get_recommendations(title,cosine_sim=cosine_sim):
    #get the index of the title that matches the input title
    idx=data[data['Titles']==title].index=[0]
    #get the pair wise similarity score of all titles with that title
    sim_scores=list(enumerate(cosine_sim[idx]))
    #sort the titles based on the similarity scores the descending orader
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    #get the indices of most similar title
    sim_indices=[i[0] for i in sim_scores[1:11]] #exclude the first as its the title itself
    #return the top 10 most similar title
    return data['Titles'].iloc[sim_indices]

#test the recommendation system with an example title
example_title='20th Century'
recommended_titles=get_recommendations(example_title)

#print recommendation
print(f"recommendations for '{example_title}':")
for title in cosine_sim:
    print(title)
    

'''Q.2'''
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pnadas as pd
transaction=[['Apple','Banana'],['Apple','Orange'],['Banana','Orange'],['Apple','Banana','Orange']]
te=TransactionEncoder()
transformed_data=te.fit(transaction).transform(transaction)
df=pd.DataFrame(transformed_data,columns=te.columns_)
frequent_itemsets=fpgrowth(df,min_support=0.5,use_colnames=True)
print(frequent_itemsets)
'''O/p:-support         itemsets
0     0.75         (Banana)
1     0.75          (Apple)
2     0.50         (Orange)
3     0.50  (Apple, Banana)
4     0.50  (Apple, Orange)'''





'''3. Build an item-based collaborative filtering recommendation engine. 
Instead of recommending items based on similar users, recommend items 
that are similar to those that a user has already interacted with.
Steps:
 Preprocess the Data: Create a user-item matrix where rows are users and columns are 
items (movies).
 Compute Item Similarity: Calculate similarity between items based on user 
interactions.
 Recommend Items: For a given user, recommend items that are similar to those the 
user has already rated highly.'''
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (Assuming 'user_id', 'movie_id', 'rating' columns)
# You can replace 'movie_ratings.csv' with your actual dataset path
data = pd.read_csv('movie_ratings.csv')

# Create the user-item matrix (rows are users, columns are movies)
user_item_matrix = data.pivot_table(index='user_id', columns='movie_id', values='rating')

# Fill NaN values with 0 (if unrated items should be treated as 0)
user_item_matrix = user_item_matrix.fillna(0)

# Display the user-item matrix
print(user_item_matrix.head())
# Compute item-item similarity using cosine similarity
item_similarity_matrix = cosine_similarity(user_item_matrix.T)

# Convert it to a DataFrame for better understanding
item_similarity_df = pd.DataFrame(item_similarity_matrix, 
                                  index=user_item_matrix.columns, 
                                  columns=user_item_matrix.columns)

# Display the item-item similarity matrix
print(item_similarity_df.head())
def recommend_items(user_id, user_item_matrix, item_similarity_df, num_recommendations=5):
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Get the movies the user has rated
    rated_movies = user_ratings[user_ratings > 0].index.tolist()
    
    # Compute the weighted sum of similarities for each movie
    movie_scores = pd.Series()
    
    for movie in rated_movies:
        # For each movie rated by the user, get the similar movies
        similar_movies = item_similarity_df[movie].copy()
        
        # Weight the similarity by the user's rating for that movie
        similar_movies = similar_movies * user_ratings[movie]
        
        # Append the scores
        movie_scores = movie_scores.append(similar_movies)
    
    # Remove the movies already rated by the user from the recommendations
    movie_scores = movie_scores.groupby(movie_scores.index).sum()
    movie_scores = movie_scores[~movie_scores.index.isin(rated_movies)]
    
    # Sort and get top N recommendations
    recommended_movies = movie_scores.sort_values(ascending=False).head(num_recommendations)
    
    return recommended_movies

# Example: Recommend 5 movies for user with ID 1
user_id = 1
recommended_movies = recommend_items(user_id, user_item_matrix, item_similarity_df, num_recommendations=5)

print("Recommended movies for user", user_id)
print(recommended_movies)



'''4. Using the mlxtend library, write a Python program to generate association 
rules from a dataset of transactions. The program should allow setting a 
minimum support threshold and minimum confidence threshold for rule 
generation.
transactions = [['Tea', 'Bun'], ['Tea', 'Bread'], ['Bread', 'Bun'], ['Tea', 'Bun', 
'Bread']'''
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
import pandas as pd
transactions =[['Tea', 'Bun'], ['Tea', 'Bread'], ['Bread', 'Bun'], ['Tea', 'Bun', 'Bread']]
te=TransactionEncoder()
transformed_data=te.fit(transactions).transform(transactions)

#step2:apply apriori algorithm to find frequent itemsets
frequent_itemsets=apriori(data,min_support=0.1,use_colnames=True)
#step3:generate association rules from the frequent itemsets
rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.5)

#step4:output the results
print("frequent Itemsets:",frequent_itemsets)

print("\nAssociation Rules:",rules[['antecedents','consequents','support','confidence','lift']])
#print(rules[['antecedents','consequents','support','confidence','lift']])

'''Q.5. Build a popularity-based recommendation system. The system should 
recommend movies based on their overall popularity (e.g., number of ratings 
or average rating).
Steps:
 Preprocess the Data: Calculate the total number of ratings and average rating for each 
movie.
 Rank the Movies: Rank movies based on the chosen popularity metric.
 Recommend Movies: Recommend the top N most popular movies to any user'''           
import pnadas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data             

