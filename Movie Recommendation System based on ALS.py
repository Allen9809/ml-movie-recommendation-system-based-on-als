# Databricks notebook source
# MAGIC %md
# MAGIC ##Movie recommendation system project
# MAGIC ####In this project, I will use an Alternating Least Squares (ALS) algorithm with Spark APIs to predict the ratings for the movies in MovieLens small dataset

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Part 0: Data ETL and Data Exploration

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# COMMAND ----------

movies_df = spark.read.load("/FileStore/tables/movies.csv", format='csv', header = True)
ratings_df = spark.read.load("/FileStore/tables/ratings.csv", format='csv', header = True)
links_df = spark.read.load("/FileStore/tables/links.csv", format='csv', header = True)
tags_df = spark.read.load("/FileStore/tables/tags.csv", format='csv', header = True)
movies_df.show(5)

# COMMAND ----------

ratings_df.show(5)

# COMMAND ----------

links_df.show(5)

# COMMAND ----------

tags_df.show(5)

# COMMAND ----------

tmp1 = ratings_df.groupBy("userID").count().toPandas()['count'].min()
tmp2 = ratings_df.groupBy("movieId").count().toPandas()['count'].min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}'.format(tmp1))
print('Minimum number of ratings per movie is {}'.format(tmp2))

# COMMAND ----------

tmp1 = sum(ratings_df.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings_df.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Part 1: Spark SQL and OLAP 

# COMMAND ----------

movies_df.registerTempTable("movies")
ratings_df.registerTempTable("ratings")
links_df.registerTempTable("links")
tags_df.registerTempTable("tags")

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.1 The Number of Users

# COMMAND ----------

# Dataframe: 
# ratings_df.select("userId").distinct().count()

users_amount = spark.sql("SELECT count(distinct userID) FROM ratings")
users_amount.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.2 The Number of Movies

# COMMAND ----------

movies_amount = spark.sql("SELECT count(distinct movieID) FROM movies")
movies_amount.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.3 The Number of movies are rated by users & Movies not Rated Before

# COMMAND ----------

rated_movies = spark.sql("SELECT count(distinct movieID) FROM ratings")
rated_movies.show()

# COMMAND ----------

Nrated_movies = spark.sql("""SELECT distinct title, genres  
                             FROM movies where movieID not in 
                             (SELECT distinct movieID FROM ratings)
                          """)
display(Nrated_movies)

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.4 All Movie Genres

# COMMAND ----------

genres_split = spark.sql("SELECT distinct explode(split(genres,'[|]')) as genres FROM movies")
genres_split.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.5 Nmuber of Movies for Each Genre

# COMMAND ----------

res_1_5 = spark.sql("""SELECT genre, sum(num) as number_of_movies
                    FROM (SELECT explode(split(genres,'[|]')) as genre, count(movieID) as num FROM movies GROUP BY 1)
                    GROUP BY 1 ORDER BY 1 ASC
                    """)
res_1_5.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Part 2: Spark ALS based Approach for Training Model
# MAGIC We will use an Spark ML to predict the ratings, so let's reload "ratings.csv" using sc.textFile and then convert it to the form of (user, item, rating) tuples.

# COMMAND ----------

ratings_df.show(5)

# COMMAND ----------

movie_ratings=ratings_df.drop('timestamp')
movie_ratings.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ####2.1 Data Type Convert

# COMMAND ----------

# Data type convert
from pyspark.sql.types import IntegerType, FloatType
movie_ratings = movie_ratings.withColumn("userId", movie_ratings["userId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("movieId", movie_ratings["movieId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("rating", movie_ratings["rating"].cast(FloatType()))
movie_ratings.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.2 ALS Model Selection and Evaluation
# MAGIC With the ALS model, a grid search is used to find the optimal hyperparameters.

# COMMAND ----------

# import package
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder

# COMMAND ----------

#Create test and train set
(training,test)=movie_ratings.randomSplit([0.8,0.2],seed = 1234)
als = ALS()

# COMMAND ----------

#Create ALS model
als = ALS(userCol = 'userId', itemCol = 'movieId', ratingCol = 'rating',coldStartStrategy = 'drop', nonnegative = True, implicitPrefs = False)

# COMMAND ----------

#Tune model using ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(als.regParam, [0.2])
             .addGrid(als.rank, [10])
             .addGrid(als.maxIter, [20])
             .build())

# COMMAND ----------

# Define evaluator as RMSE
RMSE_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# COMMAND ----------

# Build Cross validation 
cv_setting = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=RMSE_evaluator, numFolds=3)

# COMMAND ----------

#Fit ALS model to training data
cv_model = cv_setting.fit(training)

# COMMAND ----------

#Extract best model from the tuning exercise using ParamGridBuilder
best_model = cv_model.bestModel
best_model

# COMMAND ----------

print("cross validation RMSE of the best model: {}".format(min(cv_model.avgMetrics)))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.3 Model Testing
# MAGIC And finally, we make a prediction and check the testing error.

# COMMAND ----------

#Generate predictions and evaluate using RMSE
predictions=best_model.transform(test)
RSME = RMSE_evaluator.evaluate(predictions)

# COMMAND ----------

#Print evaluation metrics and model parameters
print ("RMSE = "+str(RSME))
print ("**Best Model**")
print (" Rank:", best_model.rank), 
print (" MaxIter:", best_model._java_obj.parent().getMaxIter()), 
print (" RegParam:" ,best_model._java_obj.parent().getRegParam()), 

# COMMAND ----------

predictions.sort('userID',ascending=True).show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Part 3: Recommendation Systems based on the Model

# COMMAND ----------

# MAGIC %md 
# MAGIC ####3.1 Prediction of All Data

# COMMAND ----------

alldata = best_model.transform(movie_ratings)
RMSE_all = RMSE_evaluator.evaluate(alldata)
print ("RMSE = "+str(RMSE_all))

# COMMAND ----------

alldata.registerTempTable("alldata")
spark.sql("Select movies.movieId,title,userId,rating,prediction From movies join alldata on movies.movieID = alldata.movieID").show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Part 3.2: Recommend Moives to Users according to the userID
# MAGIC So we can recommend the moives to specific users.
# MAGIC Suppose we want to recommend moive to users with id: 575, 232 here.

# COMMAND ----------

#recommend 10 movies for each users
user_recs = best_model.recommendForAllUsers(10)
display(user_recs)

# COMMAND ----------

user_recs.registerTempTable("als_recs_temp")

# COMMAND ----------

# explode the recommendation column into two categories and make it stuctured
recommendation_each = spark.sql("""
                                SELECT userID, t1.movieid as MovieID, t1.rating as rating
                                FROM als_recs_temp
                                LATERAL VIEW explode(recommendations) exploded_table AS t1 
                                """)
recommendation_each.show()

# COMMAND ----------

recommendation_each.registerTempTable("recommendation_each")
movies_df.registerTempTable("movies_df")

# COMMAND ----------

# MAGIC %md
# MAGIC #### For user with id 575, top 10 movies that we will recommend:

# COMMAND ----------

# DBTITLE 0,For user with id 575:
res_user575 = spark.sql("""
                        select userId,title
                        FROM recommendation_each t1
                        LEFT JOIN movies_df t2
                        ON t1.movieId = t2.movieId
                        WHERE t1.userId = 575
                        """)
res_user575.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### For user with id 232, top 10 movies that we will recommend:

# COMMAND ----------

res_user232 = spark.sql("""
                        select userId,title
                        FROM recommendation_each t1
                        LEFT JOIN movies_df t2
                        ON t1.movieId = t2.movieId
                        WHERE t1.userId = 232
                        """)
res_user232.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Part 3.3: Find the Similar Moives according to the MovieID
# MAGIC Also based on ALS results above.
# MAGIC Suppose we want to find similar moives for the movies with id: 463, 471.

# COMMAND ----------

movie_factors=best_model.itemFactors
movie_factors.createOrReplaceTempView('movie_factors')
display(movie_factors)

# COMMAND ----------

# access the movie factor matrix by string method
comd=["movie_factors.selectExpr('id as movieId',"]
for i in range(best_model.rank):
    comd.append("'features["+str(i)+"] as feature"+str(i)+"',")
comd.append(')')
movie_factors=eval(''.join(comd))
movie_factors.createOrReplaceTempView('movie_factors')
display(movie_factors)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Method 1: Euclidean Distance based Similarity
# MAGIC 
# MAGIC  e.g. Movie A with factor [1,2,3] and movie B with factor [2,4,6].
# MAGIC The distance between them is sqrt(1^2+2^2+3^2).

# COMMAND ----------

movie_info=spark.sql('select * from movie_factors where movieid='+str(471))
movie_info.show()

# COMMAND ----------

movie_info=spark.sql('select * from movie_factors where movieid=' + str(463)).toPandas()
movie_info.shape

# COMMAND ----------

def dist_similar(movieid):
  '''
  number of similar movies to find is 10 here, we can also change it into a input value, which is easy to do
  movieid: id of the movie to find similarities
  '''  
  movie_info=spark.sql('select * from movie_factors where movieid=' + str(movieid)).toPandas()
    
  if movie_info.shape[0] == 0:
    print('No movie with id '+str(movieid)+' is found in the data.')
    return None, None

  temp=['select movieid,']
  for i in range(best_model.rank):
    val=movie_info.iloc[0,i+1]
    if val>0:
      comd='feature'+str(i)+'-'+str(val)
    else:
      comd='feature'+str(i)+'+'+str(-val)

    if i<best_model.rank-1:
      temp.append('('+comd+')*('+comd+') as sd'+str(i)+',')
    else:
      temp.append('('+comd+')*('+comd+') as sd'+str(i))    
  temp.append('from movie_factors where movieId!='+str(movieid))
  ssd=spark.sql(' '.join(temp)).toPandas()
  ssd['ssd']= ssd.apply(lambda x: x['sd0']**2+x['sd1']**2+x['sd2']**2+x['sd3']**2+x['sd4']**2+x['sd5']**2+x['sd6']**2+x['sd7']**2+x['sd8']**2+x['sd9']**2, axis=1)
  ssd = ssd.sort_values(by=['ssd'],ascending=[True]).head(10)

  out = None  
  for i in ssd['movieid']:
    if not out:
      out=movies_df.where(movies_df.movieId==str(i))
    else:
      out=out.union(movies_df.where(movies_df.movieId==str(i)))
  out=out.toPandas()
  out.index=range(1,11)
  return out, ssd

  

# COMMAND ----------

# MAGIC %md
# MAGIC #### MovieId 463, Top 10 Similar:

# COMMAND ----------

res,ssd1=dist_similar(463)
res

# COMMAND ----------

# MAGIC %md
# MAGIC #### MovieId 471, Top 10 Similar:

# COMMAND ----------

res,ssd2=dist_similar(471)
res

# COMMAND ----------

# MAGIC %md
# MAGIC #### Method 2: Cosine Distance based Similarity
# MAGIC 
# MAGIC e.g. Movie A with factor [1,2,3] and movie B with factor [2,4,6]. The distance between them is 0. 
# MAGIC Because cosine similarity only considers the directions of two vectors.

# COMMAND ----------

def cos_similar(movieid_):
  '''
  number of similar movies to find is 10 here, we can also change it into a input value, which is easy to do
  movieid_: id of the movie to find similarities
  '''
  movie_info=spark.sql('select * from movie_factors where movieId='+str(movieid_)).toPandas()
  if movie_info.shape[0]<=0:
    print('No movie with id '+str(movieid_)+' is found in the data.')
    return None, None
  norm_m=sum(movie_info.iloc[0,1:].values**2)**0.5
  temp=['select movieId,']
  norm_str=['sqrt(']
  for i in range(best_model.rank):
    comd='feature'+str(i)+'*'+str(movie_info.iloc[0,i+1])
    temp.append(comd+' as inner'+str(i)+',')
    if i<best_model.rank-1:      
      norm_str.append('feature'+str(i)+'*feature'+str(i)+'+')
    else:
      norm_str.append('feature'+str(i)+'*feature'+str(i))
  norm_str.append(') as norm')
  temp.append(''.join(norm_str))
  temp.append(' from movie_factors where movieId!='+str(movieid_))  
  inner=spark.sql(' '.join(temp))
  inner=inner.selectExpr('movieId',\
                         '(inner0+inner1+inner2+inner3+inner4+inner5+inner6+inner7+inner8+inner9)/norm/'+str(norm_m)+' as innerP').\
                         orderBy('innerP',ascending=False).limit(10).toPandas()
  
  out=None
  for i in inner['movieId']:
    if not out:
      out=movies_df.where(movies_df.movieId==str(i))
    else:
      out=out.union(movies_df.where(movies_df.movieId==str(i)))
  out=out.toPandas()
  out.index=range(1,11)
  return out, inner

# COMMAND ----------

# MAGIC %md
# MAGIC #### MovieId 463, Top 10 Similar:

# COMMAND ----------

res,inner1=cos_similar(463)
res

# COMMAND ----------

# MAGIC %md
# MAGIC #### MovieId 471, Top 10 Similar:

# COMMAND ----------

res,inner2=cos_similar(471)
res

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 4: Overall Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1 Motivation:

# COMMAND ----------

No only for a movie website company, but all ecommerce platforms/companies, it is really profitable to build a good recommendation system. So that they can attract more customoers by offering better service than the competitors. Based on this idea, I tried myself to build a simple recommendation system using the data from GroupLens (https://grouplens.org/datasets/movielens/latest/) to understand how a recommendation system is constructed. In this way, I will be prepared when I do similar job as a data scientist in the future.

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2 Steps:

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Data ETL and Data Exploration: obtain the general information about the data, including the total number of users, total number of movies and the list of movies .etc; OLAP analysis to find the genres, all movie categories and number of movies in each category.
# MAGIC 2. Spark ALS based Approach for Training Model: drop useless columns; change the data type from string to numeric; separate the data to 80% training and 20% testing; fit the ALS model on the training data; select the hyper-parameters through grid search and 3-fold cross validation; evaluate the tuned model on testing data.
# MAGIC 3. Model apply and the Performance: use the model to make recommendations to users with given userIds; find top similar movies for given movieIds; both of them are based on ALS model's result of item features.

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.3 Output and Conclusions:

# COMMAND ----------

# MAGIC %md
# MAGIC The best model in this project for ALS has the parameters to be: maxIter=20, regParam=0.2, rank=10. The rooted mean squared error (RMSE) on the testing data is 0.88 and on the whole dataset is 0.74, which both are acceptable.

# COMMAND ----------

# MAGIC %md
# MAGIC As mentioned in the steps, the ALS model is not only able to provide recommendations, but also able to mine latent information, which is the latent variable in matrix factorization. In this project, that is the 10-feature matrix that mined from the movie-related data. This latent information is helpful that it can help us gain some deeper insight. For example, this information was used to measure the difference between any two movies so that we are able to find similar movies. 

# COMMAND ----------

# MAGIC %md
# MAGIC Lastly, there are two ways to find similar items in my project: the Euclidean distance based function and cosine distance based function. It is hard to tell which one is better. However, for the movie recommendation system I highly recommend the cosine distance based one. Because cosine distance cares only about the 'direction' of the movie, or we can say, the theme, which is exactly the key factor for the audience to decide whether to watch or not.
