#### SET UP

# clear memory
rm(list=ls())

# Import libraries
library(lspline)
library(sandwich)
library(modelsummary)
library(caret)
library(ranger)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(RColorBrewer)
library(viridis)
library(ggthemes)
library(pander)

# source("code/gabor_textbook/da_helper_functions.R")

data_in <- "data/"

output <- "output(s)/"

# import data -------------------------------------------------------------

data <- read.csv(paste0(data_in,"hotels-vienna.csv"))

# goal is a prediction of price to find the best deals (through residuals)
# check distribution of target variable

summary(data$price)

data %>% 
  ggplot(aes(x= price))+
  geom_histogram(fill= "navyblue", col= "black", alpha = 0.5)+
  theme_bw()+
  scale_fill_wsj()+
  labs(x = "Price in euro",y = "Count of Hotels")

# filter out extreme values in price to bring distribution closer to normal
data <- data %>% filter(price <= 400)

# check city_actual values
table(data$city_actual)

# keep only Vienna
data <- data %>% filter(city_actual == 'Vienna')

# check distribution of neighbourhoods; looks fine --> no changes made
table(data$neighbourhood)

# check distribution of offers; 1 extreme values --> remove deal
table(data$offer_cat)

data <- data %>% filter(!offer_cat == '75%+ offer ')

# check accommodation type distribution
table(data$accommodation_type)

data <- data %>% mutate(accommodation_type = trimws(accommodation_type))

to_filter <- data %>% 
  group_by(accommodation_type) %>% 
  summarise(n = n()) %>% 
  filter(n < 10)

data <- data %>% mutate(accommodation_type = 
                          ifelse(accommodation_type %in% to_filter$accommodation_type, 
                                 "Other Accommodation", accommodation_type))

# check neighbourhood distribution
table(data$neighbourhood)

to_filter <- data %>% 
  group_by(neighbourhood) %>% 
  summarise(n = n()) %>% 
  filter(n < 10)

data <- data %>% mutate(neighbourhood = ifelse(neighbourhood %in% to_filter$neighbourhood, 
                                 "Other Neighbourhood", neighbourhood))

# check for missing values
to_filter <- sapply(data, function(x) sum(is.na(x)))
to_filter[to_filter > 0]

# handle missing values by deleting columns (+ redundant columns)
data <- data %>% select(-c(ratingta, ratingta_count, 
                           country, city, city_actual, center1label, center2label, offer))

# handle missing values by imputation & flags
data <- data %>% mutate(
  flag_missing_rating = as.numeric(is.na(rating)),
  rating = ifelse(is.na(rating), median(rating, na.rm = T), rating), # most properties start from about 4.0 rating
  # flag_missing_rating_count = as.numeric(is.na(rating_count)), # no need for a flag as there are no other 0 values
  rating_count = ifelse(is.na(rating_count), 0, rating_count) 
)

# right-skewed count of ratings
ggplot(data = data, aes(x = rating_count))+
  geom_histogram()

# Pool num of reviews to 5 categories (of fairly comparable sizes)
data <- data %>%
  mutate(f_rating_count = cut(rating_count, c(0,1,20,70,200,max(data$rating_count)), labels=c('None','1-19','20-69','70-199','200+'), right = F))
data %>%
  group_by(f_rating_count) %>%
  summarise(median_price = median(price) ,mean_price = mean(price) ,  n=n())
# drop the extremely large value for count of ratings
data <- data %>% filter(!is.na(f_rating_count))

# dropping variables with no variation
variances<- data %>%
  apply(2, var, na.rm = TRUE) == 0

data <- data %>%
  select(-one_of(names(variances)[variances]))

# drop unused factor levels
data <- data %>%
  mutate_at(vars(colnames(data)[sapply(data, is.factor)]), funs(fct_drop))

# set factor variables
data <- data %>% mutate(
  neighbourhood = as.factor(neighbourhood),
  offer_cat = as.factor(offer_cat),
  f_rating_count = as.factor(f_rating_count),
  accommodation_type = as.factor(accommodation_type)
)

# variable sets for model selection
basic_vars <- c('accommodation_type', 'stars')
location_vars <- c('distance' ,'distance_alter', 'neighbourhood')
rating_features <- c('rating', 'flag_missing_rating', 'f_rating_count')
accommodation_features <- c('scarce_room','offer_cat')

# interactions
interact <- c("accommodation_type*neighbourhood", "accommodation_type*scarce_room", "accommodation_type*offer_cat", "rating*f_rating_count", "distance*rating", "distance*stars")

# Predictor sets for linear regression
X1 <- basic_vars
X2 <- c(basic_vars, location_vars)
X3 <- c(basic_vars, location_vars, rating_features)
X4 <- c(basic_vars, location_vars, rating_features, accommodation_features)
# last model is created to check overfitting results
X5 <- c(basic_vars, location_vars, rating_features, accommodation_features, interact)

# create model formulae
modellev1 <- paste0(" ~ ",paste(X1, collapse = " + "))
modellev2 <- paste0(" ~ ",paste(X2, collapse = " + "))
modellev3 <- paste0(" ~ ",paste(X3, collapse = " + "))
modellev4 <- paste0(" ~ ",paste(X4, collapse = " + "))
modellev5 <- paste0(" ~ ",paste(X5, collapse = " + "))


#################################
# Separate hold-out set #
#################################

# cannot separate hold-out set due to extremely small sample size


# cross validation OLS ----------------------------------------------------

# function to compute MSE
mse_lev <- function(pred, y) {
  # Mean Squared Error for log models
  (mean((pred - y)^2, na.rm=T))
}

n_folds=3
# Create the folds
set.seed(20210214)

folds_i <- sample(rep(1:n_folds, length.out = nrow(data) ))
# Create results
model_results_cv <- list()

for (i in (1:5)){
  model_name <-  paste0("modellev",i)
  model_pretty_name <- paste0("(",i,")")
  
  yvar <- "price"
  xvars <- eval(parse(text = model_name))
  formula <- formula(paste0(yvar,xvars))
  
  # Initialize values
  rmse_train <- c()
  rmse_test <- c()
  
  model_work_data <- lm(formula,data = data)
  BIC <- BIC(model_work_data)
  nvars <- model_work_data$rank -1
  r2 <- summary(model_work_data)$r.squared
  
  # Do the k-fold estimation
  for (k in 1:n_folds) {
    test_i <- which(folds_i == k)
    # Train sample: all except test_i
    data_train <- data[-test_i, ]
    # Test sample
    data_test <- data[test_i, ]
    # Estimation and prediction
    model <- lm(formula,data = data_train)
    prediction_train <- predict(model, newdata = data_train)
    prediction_test <- predict(model, newdata = data_test)

    # Criteria evaluation
    rmse_train[k] <- mse_lev(prediction_train, data_train$price)**(1/2)
    rmse_test[k] <- mse_lev(prediction_test, data_test$price)**(1/2)
    
  }
  
  model_results_cv[[model_name]] <- list(yvar=yvar,xvars=xvars,formula=formula,model_work_data=model_work_data,
                                         rmse_train = rmse_train,rmse_test = rmse_test,BIC = BIC,
                                         model_name = model_pretty_name, nvars = nvars, r2 = r2)
}

t1 <- imap(model_results_cv,  ~{
  as.data.frame(.x[c("rmse_test", "rmse_train")]) %>%
    dplyr::summarise_all(.funs = mean) %>%
    mutate("model_name" = .y , "model_pretty_name" = .x[["model_name"]] ,
           "nvars" = .x[["nvars"]], "r2" = .x[["r2"]], "BIC" = .x[["BIC"]])
}) %>%
  bind_rows()
t1

# Model 3 best in terms of test RMSE and provides a good BIC.
# Model of choice is model 3 - good trade-off between number of variables and performance --> Occam's Razor!


# CART --------------------------------------------------------------------

# set parameters
train_control <- trainControl(method = "cv", number = n_folds)
model_cart <- formula(paste0(yvar,modellev3))


# CART with rpart & using cross-validation
set.seed(20210214)
system.time({
  cart1 <- train(model_cart,
                      data = data,
                      method = "rpart",
                      tuneLength = 10,
                      trControl = train_control
  )
})

cart1
summary(cart1)
pred_cart1 <- predict(cart1, data)
rmse_cart1 <- sqrt(mean((pred_cart1 - data$price)^2))

# Tree graph
rpart.plot(cart1$finalModel, tweak=1.2, digits=-1, extra=1)



# CART with rpart2
# setting the total number of splits
set.seed(20210214)
cart2 <- train(
  model_cart, data = data, method = "rpart2",
  trControl = train_control,
  tuneGrid= data.frame(maxdepth=10))

summary(cart2)
pred_cart2 <- predict(cart2, data)
rmse_cart2 <- sqrt(mean((pred_cart2 - data$price)^2))

# Tree graph
rpart.plot(cart2$finalModel, tweak=1.2, digits=-1, extra=1)



# CART with rpart & using cross-validation
# setting the minimum number of observations in a node be equal to at least 10
# a split must decrease the overall lack of fit by a factor of 0.01
set.seed(20210214)
system.time({
  cart3 <- train(model_cart,
                 data = data,
                 method = "rpart",
                 tuneGrid= expand.grid(cp = 0.01),
                 control = rpart.control(minsplit = 10),
                 na.action = na.pass,
                 trControl = train_control
  )
})

cart3
summary(cart3)
pred_cart3 <- predict(cart3, data)
rmse_cart3 <- sqrt(mean((pred_cart3 - data$price)^2))

# Tree graph
rpart.plot(cart3$finalModel, tweak=1.2, digits=-1, extra=1)

# Compare the three models

cart_compare <- data.frame(
  "Model" = c("CART Model 1", "CART Model 2", "CART Model 3"),
  "RMSE" = c(rmse_cart1, rmse_cart2, rmse_cart3)
)
cart_compare

# Random Forest -----------------------------------------------------------


## Random forest
# do 3-fold CV
train_control <- trainControl(method = "cv",
                              number = 3,
                              verboseIter = T)
# set tuning
tune_grid <- expand.grid(
  .mtry = c(3, 5, 7),
  .splitrule = "variance",
  .min.node.size = c(5, 10)
)

# Model RF with pre-set tuning parameters
set.seed(20210214)
system.time({
  rf_model1 <- train(model_cart,
                    data = data,
                    method = "ranger",
                    trControl = train_control,
                    tuneGrid = tune_grid,
                    importance = "impurity"
  )
})
rf_model1
rmse_rf1 <- mean(rf_model1$resample$RMSE)

# Model RF with auto-tuning function
set.seed(1234)
system.time({
  rf_model2 <- train(
    model_cart,
    data = data,
    method = "ranger",
    trControl = train_control,
    importance = "impurity"
  )
})
rf_model2
rmse_rf2 <- mean(rf_model2$resample$RMSE)

results <- resamples(
  list(
    model_1  = rf_model1,
    model_2  = rf_model2
  )
) 

# Compare the five models

CARTandRF_compare <- data.frame(
  "Model" = c("CART Model 1", "CART Model 2", "CART Model 3", "RF Model 1", "RF Model 2"),
  "RMSE" = c(rmse_cart1, rmse_cart2, rmse_cart3, rmse_rf1, rmse_rf2)
)
CARTandRF_compare

# Residual Analysis CART -------------------------------------------------------

## So the model with the best RMSE is the advanced CART (3rd)

set.seed(20210214)
system.time({
  final_CART <- train(model_cart,
                 data = data,
                 method = "rpart",
                 tuneGrid= expand.grid(cp = 0.01),
                 control = rpart.control(minsplit = 10),
                 na.action = na.pass,
                 trControl = trainControl(method = "none")
  )
})

final_CART

data$CART_predicted_price <- predict(final_CART, newdata = data, type = "raw")


# Calculate residuals

data$CART_prediction_res <- data$price - data$CART_predicted_price

data %>% select(price, CART_prediction_res, CART_predicted_price)

# List of 5 best deals
bestdeals <- data %>%
  select(hotel_id, price, CART_prediction_res, distance, stars, rating) %>%
  arrange(CART_prediction_res) %>%
  .[1:5,] %>%
  as.data.frame() 

rownames(bestdeals) <- NULL

pander(bestdeals)

# designate the best deals in the data as well
data$bestdeals <- ifelse(data$CART_prediction_res %in% tail(sort(data$CART_prediction_res, decreasing=TRUE),5),TRUE,FALSE)

original_hotels <- data[data$hotel_id %in% c(21912, 21975, 22080, 22184, 22344),]
best_original_deals <- original_hotels  %>%
  select(hotel_id, price, CART_prediction_res, distance, stars, rating) %>%
  .[1:5,] %>%
  as.data.frame() 

rownames(best_original_deals) <- NULL

pander(best_original_deals)


# y - yhat graph
y_yhat_hotels <- ggplot(data = data, aes(y=price, x=CART_predicted_price)) +
  geom_point(aes(color=bestdeals,shape=bestdeals), size = 1.5, fill='blue', alpha = 0.8, show.legend=F, na.rm = TRUE) +
  geom_segment(aes(x = 0, y = 0, xend = 400, yend = 400), size=0.5, color='red', linetype=2) +
  coord_cartesian(xlim = c(0, 400), ylim = c(0, 400)) +
  scale_x_continuous(expand = c(0.01,0.01),limits=c(0, 400), breaks=seq(0, 400, by=50)) +
  scale_y_continuous(expand = c(0.01,0.01),limits=c(0, 400), breaks=seq(0, 400, by=50)) +
  labs(y = "Price (EURO)", x = "Predicted price  (EURO)")+
  theme_bw()+
  scale_fill_wsj()
y_yhat_hotels

# Residual Analysis OLS -------------------------------------------------------
final_OLS <- lm(paste0('price',modellev3), data = data)

data$OLS_predicted_price <- predict(final_OLS, data)

# Calculate residuals

data$OLS_prediction_res <- data$price - data$OLS_predicted_price

data %>% select(price, OLS_prediction_res, OLS_predicted_price)

# List of 5 best deals
bestdeals_OLS <- data %>%
  select(hotel_id, price, OLS_prediction_res, distance, stars, rating) %>%
  arrange(OLS_prediction_res) %>%
  .[1:5,] %>%
  as.data.frame() 

rownames(bestdeals_OLS) <- NULL

pander(bestdeals_OLS)

# Residual Analysis RF -------------------------------------------------------
train_control <- trainControl(method = "none",
                              verboseIter = F)

tune_grid <- expand.grid(
  .mtry = c(10),
  .splitrule = "variance",
  .min.node.size = c(5)
)

set.seed(20210214)
system.time({
  final_RF <- train(model_cart,
                     data = data,
                     method = "ranger",
                     trControl = train_control,
                     tuneGrid = tune_grid,
                     importance = "impurity"
  )
})

data$RF_predicted_price <- predict(final_RF, data)

# Calculate residuals

data$RF_prediction_res <- data$price - data$RF_predicted_price

data %>% select(price, RF_prediction_res, RF_predicted_price)

# List of 5 best deals
bestdeals_RF <- data %>%
  select(hotel_id, price, RF_prediction_res, distance, stars, rating) %>%
  arrange(RF_prediction_res) %>%
  .[1:5,] %>%
  as.data.frame() 

rownames(bestdeals_RF) <- NULL

pander(bestdeals_RF)