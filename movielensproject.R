##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()

download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)



ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))



movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)



colnames(movies) <- c("movieId", "title", "genres")



#if using R 4.0 or later:

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))




movielens <- left_join(ratings, movies, by = "movieId")

head(movielens)



# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
# if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)

edx <- movielens[-test_index,]

temp <- movielens[test_index,]



# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
#rm(dl, ratings, movies, test_index, temp, movielens, removed) 



edx %>% summarize(n_users = n_distinct(userId),n_movies = n_distinct(movieId))



#####################################################################
## Simple recommendation model
#####################################################################
mu <- mean(edx$rating)
mu
naive_rmse <- RMSE(validation$rating, mu)
cat("naive_rmse: ", naive_rmse)
###################################################################


mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))




qplot(b_i, data = movie_avgs, bins = 20, color = I("blue"))

#####################################################################
### Method 2
## effects or bi . Bias ,b notation  movie effects 
#####################################################################


predicted_ratings <- mu + validation  %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effects_rmse <- RMSE(predicted_ratings, validation$rating)

cat("movie_effects_rmse: ", movie_effects_rmse)



#####################################################################
### Method 3 : User  effects

###################################################################


edx %>% 
  group_by(userId) %>% 
  filter(n()>=100) %>%
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "green")



user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))  


  
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
  
user_specific_rmse <- RMSE(predicted_ratings, validation$rating)

cat("user_specific_rmse: ", user_specific_rmse)


#############################################################
###  Regulariztion
#############################################################

validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%  
  slice(1:10) %>% 
  pull(title)


movie_titles <- movies %>% 
  select(movieId, title) %>%
  distinct()


movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10)  %>% 
  pull(title)


movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10)  %>% 
  pull(title)


edx %>% count(movieId) %>% 
  left_join(movie_avgs, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull





edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(n)




#####################################################################
## Method 4
#Let's compute these regularized estimates of  bi  using  = 3 . Later, we will see why we picked 3.

##Regularization,Penalized  least  squares estimates with full  cross validaton.

#####################################################################

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})


penalized_least_squares_lambda <- lambdas[which.min(rmses)]
penalized_least_squares_lambda

penalized_least_squares_lambda_rmse = min(rmses)

cat("penalized_least_squares_lambda_rmse: ", penalized_least_squares_lambda_rmse)


Method <- c("naive_rmse", "movie_effects_rmse", "user_specific_rmse", "penalized_least_squares_lambda_rmse") 
RMSE <- c(naive_rmse,movie_effects_rmse,user_specific_rmse,penalized_least_squares_lambda_rmse) 
df<- class.df<- data.frame(Method,RMSE) 
df



