import os
from time import time
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math

sc = SparkContext()

dataset_path = os.path.join('..', 'Datasets')
ratings_file = os.path.join(dataset_path , 'ml-latest'  ,'ratings.csv')
movies_file = os.path.join(dataset_path, 'ml-latest' ,'movies.csv')

ratings_raw_data = sc.textFile(ratings_file)
movies_raw_data = sc.textFile(movies_file)

ratings_raw_data_header = ratings_raw_data.take(1)[0]
movies_raw_data_header = movies_raw_data.take(1)[0]

# Now we can pass raw data to RDD
ratings_data = ratings_raw_data.filter(lambda x : x!= ratings_raw_data_header).map(lambda x : x.split(",")).map(lambda tokens: (tokens[0] , tokens[1] , tokens[2])).cache()
movies_data = movies_raw_data.filter(lambda x: x!= movies_raw_data_header).map(lambda x : x.split(",")).map(lambda tokens: (tokens[0] , tokens[1])).cache()

# Selecting the ALS parametes using datasets
# Spliting the data in train , test and validation to determine the best ALS

training_RDD  , test_RDD  = ratings_data.randomSplit([7,3] , seed = 0)

# Parameters
seed = 5
best_rank = 8
iterations = 10
regularization_parameters = 0.01

# Training
model = ALS.train(training_RDD , rank=best_rank , seed= seed , iterations= iterations ,lambda_ = regularization_parameters)

# Testing
test_predict_RDD = test_RDD.map(lambda x: (x[0] , x[1]))
predictions = model.predictAll(test_predict_RDD).map(lambda x: ((x[0] , x[1]), x[2]))
ratings_AND_predictions = test_RDD.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)
error = math.sqrt(ratings_AND_predictions.map(lambda x: (x[1][0] - x[1][1])**2).mean())

print('For testing data the RMSE is %s' % (error))


def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

movie_ID_with_ratings_RDD = (ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

## Recommendations for new users

new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,9), # Star Wars (1977)
     (0,1,8), # Toy Story (1995)
     (0,16,7), # Casino (1995)
     (0,25,8), # Leaving Las Vegas (1995)
     (0,32,9), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,4), # Flintstones, The (1994)
     (0,379,3), # Timecop (1994)
     (0,296,7), # Pulp Fiction (1994)
     (0,858,10) , # Godfather, The (1972)
     (0,50,8) # Usual Suspects, The (1995)
    ]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)

data_with_new_ratings_RDD = ratings_data.union(new_user_ratings_RDD)

t0 = time()
new_ratings_model = ALS.train(data_with_new_ratings_RDD, best_rank, seed=seed, 
                              iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0


## Getting top recommendations
new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs

# keep just those not on the ID list
new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)


# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)

new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])

model_path = os.path.join('..', 'models', 'movie_lens_als')

# Save and load model
model.save(sc, model_path)
print('All Done !!!')
