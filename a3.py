# coding: utf-8


# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    tokenColumn =[]
    for each in movies['genres']:
        tokenColumn.append(tokenize_string(each))
    movies['tokens'] = tokenColumn
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
<<<<<<< HEAD
      
     - The movies DataFrame, which has been modified to include a column named 'features'.
     - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
=======
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
>>>>>>> template/master
    """
    
    
    movieGenres = movies['tokens'].tolist()
    
    N = len(movieGenres)
    vocabT = defaultdict(int)
    uniqueWords = []
    csrMatrices = []
    
    #Create Vocab
    for eachList in movieGenres:
        for eachGenre in eachList:
            if eachGenre not in uniqueWords:
                uniqueWords.append(eachGenre)
    
    uniqueWords = sorted(uniqueWords)
    i = 0
    for word in uniqueWords:
        vocabT[word] = i
        i += 1
    vocabT = sorted(vocabT)
    
    
    vocab = {}
    for k, v in enumerate(vocabT):
        vocab[v] = k
    
    
    # Here I have a VOCAB with values as index.
    
    # Get the list of genres from movies['tokens']
    for eachList in movieGenres:
        #Numpy arrays for csr- matrix
        data = []
        
        col = [] #np.array([i for i in range(len(eachList))])
        
        # A dictionary object to temporaily hold values to calculate tf-idf
        newDict = {}
        
        c = Counter()
        c.update(eachList)
     
        #X = csr_matrix((1, len(eachList)))
        
        newDict['maxTermFrequency'] = sorted(c.items(), key = lambda x:x[::-1][1])[len(c.items())-1][1]
    
        
        for eachGenre in eachList:
            newDict['termFrequency'] = c[eachGenre]
    
            termCountInDocs = 0
            for eachList in movieGenres:
                if eachGenre in eachList:
                    termCountInDocs += 1
            newDict['termCountInDocs'] = termCountInDocs
    
            
            tfidf = newDict['termFrequency'] / newDict['maxTermFrequency'] * math.log10(N/newDict['termCountInDocs'])
            
            data.append(tfidf)
            
            col.append(vocab[eachGenre])
            #if eachGenre not in uniqueWords:
                #uniqueWords.append(eachGenre)
        #print()
        
        data = np.array(data)
        row = np.zeros(len(data))
        col = np.array(col)
        #print(data)
        #csr-matrix creation
        csr = csr_matrix( (data, (row,col)), shape = (1, len(vocab)) )
        #print(csr)
        csrMatrices.append(csr)
        
    movies['features'] = csrMatrices
        
    
    
    return (movies, vocab)


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    X = a.data.tolist()
    Y = b.data.tolist()
    
    csr_X = np.array([i if j !=0 else 0 for i,j in zip(X, Y)])
    csr_Y = np.array([j if i!=0 else 0 for i,j in zip (X, Y)])
    
    cos_sim = float(np.dot(csr_X, csr_Y)) / np.sqrt(np.dot(csr_X, csr_X) * np.dot( csr_Y, csr_Y))
    
    return cos_sim


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
    
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """

    #This list stores predictions. Note: convert to numpy array while returning.
    myPredictions = [] 
    
    #For each row in rating_test, calculate rating and append to list
    for index1, row1 in ratings_test.iterrows():
        
        Numerator = 0
        Denominator = 0
        similarity_list = []

        #Get current user
        current_user = row1['userId']

        #Get target movie
        current_movie = row1['movieId']
        current_user_data = ratings_train.loc[ratings_train['userId'] == current_user]
        target_movie_dataframe = movies.loc[movies['movieId'] == current_movie]
        movie_Y = target_movie_dataframe.iloc[0]['features']              
         
        #Using rating_train find out movie_id and calculate similarity
        for index2, row2 in current_user_data.iterrows():
            
            movie_id = row2['movieId']
            movie_rating = row2['rating']
            movie_dataframe = movies.loc[movies['movieId'] == movie_id]
            movie_X = movie_dataframe.iloc[0]['features']           
            sim = cosine_sim(movie_X, movie_Y)           
            similarity_list.append((movie_rating, sim))                
        
    #Filter similarity list where similarity > 0
        similarity_list = [item for item in similarity_list if item[1] > 0]
        
    #Calculate numerator and denominator values.
        for rating_sim in similarity_list:  
            sim_value = rating_sim[1]
            if sim_value > 0:   
                rating = rating_sim[0]
                Numerator += (sim_value * rating)                
                Denominator += sim_value

    #If len of similarity list < 0, using average rating
        if len(similarity_list) > 0:
            myPredictions.append(Numerator / Denominator)   
        else:                     
            myPredictions.append(sum([rating_sim[0] for rating_sim in similarity_list]) / len(similarity_list))
    
    return np.array(myPredictions)


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
