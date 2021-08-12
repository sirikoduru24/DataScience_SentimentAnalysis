import re

#Preprocessing the data for cleaning of tweets to remove hash tags
def preprocess_tweets(tweet):
    tweet.lower()
    tweet = re.sub('((www.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet

a="@V_DEL_ROSSI: Me, #dragging myself? to the gym. https://t.co/cOjM0mBVeY"
a = preprocess_tweets(a)
a

#AT_USER Me, dragging myself? to the gym. URL

#For uploading files from drive
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

#remove stop words from the data
stopWords = pd.read_csv('stop-words.txt').values
stopWords = np.append(stopWords, ["ATUSER","URL"])

#divide the words to tokens
def word_tokenizer(tweet):
  tweet = re.sub('['+string.punctuation+']', '', tweet)
  tokenized_tweets = [word for word in word_tokenize(tweet) if word not in stopWords] 
  return tokenized_tweets
  
tokenized_tweets = word_tokenizer(a)
tokenized_tweets

#Example output : ['Me', 'dragging', 'gym']


from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.decomposition import PCA

data = pd.read_csv('twitter_train.csv',encoding='latin').values
train_data = data[:80000]
val_data = data[80000:]

#vector size to generate feature vectors
vector_size=300

#getting the vector for each tweet
def getAvgVector(sentence,model):
  sentence_vector = np.zeros(vector_size,dtype="float32")
  nwords = 0
  for word in sentence:
    if word in model.wv.vocab:
      nwords += 1
      sentence_vector = np.add(sentence_vector,model[word])
      sentence_vector = np.divide(sentence_vector,nwords) 
  return sentence_vector
 
def vectorizeTweets(sentences,model):
  feature_vec =[]
  for sentence in sentences:
    feature_vec.append(getAvgVector(sentence,model))
  return feature_vec
  
#getting the eigen vectors and pca features for dimensionality reduction
def extract_eigenvectors(X_train,k):
    pca = PCA(n_components=k)
    #pca = PCA(n_components=None)
    pca.fit(X_train)
    eigen_vectors = pca.components_
    return eigen_vectors, pca

def make_pca_features(eigen_vectors, X):
    return np.transpose(np.dot(eigen_vectors, np.transpose(X)))
	
#training model with word2vec in order to generate a feature vector
def train_w2v(data):
  tokenised_data = []
  for row in data:
    tokenised_data += [word_tokenizer(preprocess_tweets(row[2]))]

  model = Word2Vec(tokenised_data,size=vector_size,window=10,min_count=1,workers=4)
  model.train(tokenised_data, total_examples=len(tokenised_data), epochs=10)
  model.save("Tweets_Word2Vec.model")
  
train_w2v(data)

def getfeaturevector(data):
  tokenised_data = []
  label = []
  features_vec = []
  for row in data:
    tokenised_data += [word_tokenizer(preprocess_tweets(row[2]))]
    label.append(row[1])
    
  model = Word2Vec.load("Tweets_Word2Vec.model")
  features_vec = vectorizeTweets(tokenised_data,model)
    
  return features_vec,label
  
tr_vec,train_label = getfeaturevector(train_data)

eigen_vectors, pca_object = extract_eigenvectors(tr_vec,13)
train_vec = make_pca_features(eigen_vectors, tr_vec)
#r=np.cumsum(pca_object.explained_variance_ratio_)


#Classification Algorithms

#using SVM
clf = SVC(kernel='linear')
clf.fit(train_vec,train_label)

#using MLP
from sklearn.neural_network import MLPClassifier
clMLP = MLPClassifier(solver='sgd',hidden_layer_sizes=(100,50,20)).fit(tr_vec,train_label)

#using Random Forest
from sklearn.ensemble import RandomForestClassifier
clRandom = RandomForestClassifier(n_estimators=400, random_state=11).fit(tr_vec,train_label) 

vl_vec, val_label = getfeaturevector(val_data)
val_vec = make_pca_features(eigen_vectors, vl_vec)

#Predictions 
print(clf.score(val_vec,val_label)) #Percentage prediction using algorithms 
#change clf to clMLP and ClRandom to see % for those 2 algorithms

#Testing with Twitter data

#keys to connect with twitter

consumer_key = 'uuL2iBcXOVZcCTe1Z1REk59Xp'
consumer_secret = 'iXlVTYyaYc35AYQtLrdAZV6AegCcXtLyp6GOAHF3c2ijxd2HNZ'
access_token = '1057312876141002753-njI7cYGKs6nU9Qo5YafCXAahQTJXAx'
access_secret = 'XsgaZFSH38pjWxxRSlieNa662tUa9yGtHOKc1AsbVW0G6'

import tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth) 

test_data = []
#change #tag accordingly to see outputs
#outputs attached for #olympics and #angry
for i in tweepy.Cursor(api.search, q='#angry', lang = 'en', full_text=True).items(10):
    test_data.append(i._json['text'])
	
	
tst_vec, _ = getfeaturevector(test_data)
test_vec = make_pca_features(eigen_vectors, tst_vec)                              
predict = clf.predict(test_vec) #change the algorithm accordingly to see outputs to clMLP and clRandom

sentiment = ["Negative", "Neutral", "Positive"]
for i in range(0, len(test_data)):
  print("Tweet: ", test_data[i])
  print("Sentiment", sentiment[predict[i]])


#we have got the count of positive, neutral and negative tweets and then plotted the graphs based on these counts