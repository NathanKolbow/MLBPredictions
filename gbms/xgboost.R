library(xgboost)
library(tidyverse)
library(caret)
library(fastDummies)
library(tidymodels)
library(magrittr)
library(lme4)
library(tictoc)

#reading in data
pitches <- read_csv("pitches.csv")

#remove id column
pitches %<>% select(-c("X1"))

# change description to numeric. 0 = strike, 1 = contact, 2 = ball
pitches %<>%
  mutate(description = case_when(
    description == "strike" ~ 0,
    description == "hit" ~ 1,
    description == "ball" ~ 2))

pitches <- pitches %>%
  mutate(stand_r = ifelse(stand == "R", 1, 0)) %>%
  mutate(p_throws_r = ifelse(p_throws == "R", 1, 0)) %>%
  select(-c("stand", "p_throws"))

# set seed for reproducibility -------------------------------------------
set.seed(123)

# create indecies for splitting (need to use a specific column from df for it to run)
train_index <- createDataPartition(y = pitches$description, p = .7, list = FALSE) %>% as.vector()

# split into train and test
train <- pitches[train_index,]
test <- pitches[-train_index,]

vec_label_train <- train$description
train_data <- as.matrix(train %>% select(-c("pitch_type", "game_date", "zone", "inning", "inning_topbot", "bat_score", "post_bat_score",
                                            "pitcher_name", "was_0_0", "was_0_1", "was_0_2", "was_1_0", "was_1_1", "was_1_2", 
                                            "was_2_0", "was_2_1", "was_2_2", "was_3_0", "was_3_1", "was_3_2", "wOBA_cur_count", 
                                            "catcher_name", "umpire_name", "batter_name", "description"))) 

vec_label_test <- test$description
test_data <- as.matrix(test %>% select(-c("pitch_type", "game_date", "zone", "inning", "inning_topbot", "bat_score", "post_bat_score",
                                          "pitcher_name", "was_0_0", "was_0_1", "was_0_2", "was_1_0", "was_1_1", "was_1_2", 
                                          "was_2_0", "was_2_1", "was_2_2", "was_3_0", "was_3_1", "was_3_2", "wOBA_cur_count", 
                                          "catcher_name", "umpire_name", "batter_name", "description"))) 

# convert the train and test into xgb.DMatrix
# using model.matrix will handle converting factors to dummy columns
x_train = xgb.DMatrix(data = train_data, label = vec_label_train)
x_test = xgb.DMatrix(data = test_data, label = vec_label_test)


vec_label_all <- pitches$description
train_all <- as.matrix(pitches %>% select(-c("pitch_type", "game_date", "zone", "inning", "inning_topbot", "bat_score", 
                                                     "post_bat_score",
                                                     "pitcher_name", "was_0_0", "was_0_1", "was_0_2", "was_1_0", "was_1_1", "was_1_2", 
                                                     "was_2_0", "was_2_1", "was_2_2", "was_3_0", "was_3_1", "was_3_2", "wOBA_cur_count", 
                                                     "catcher_name", "umpire_name", "batter_name", "description"))) #need to remove description
full_train = xgb.DMatrix(data = train_all, label = vec_label_all)

#################### baseline #######################
nrounds = 5000
tic()
baseline_model <- xgboost::xgboost(params = list(objective = "multi:softmax", eval_metric = c("merror"), num_class = 3), 
                                   data = x_train, 
                                   nrounds = 250, early_stopping_rounds = 10, verbose = 2)
toc()
# make prediction
baseline_model_test_predict <- predict(baseline_model, x_test)

# add predictions to all the data
baseline_model_test_predict_data <- data.frame("description" = test$description, "des_pred" = baseline_model_test_predict)

# Accuracy 
sum(baseline_model_test_predict_data$description == baseline_model_test_predict_data$des_pred) / nrow(baseline_model_test_predict_data)

#################### cv #######################
tic()
baseline_model_cv <- xgboost::xgb.cv(params = list(objective = "multi:softmax", eval_metric = c("merror"), num_class = 3), data = x_train, 
                                     nrounds = nrounds, nfold = 5, showsd = T, stratified = T, 
                                     print_every_n = 1, early_stopping_rounds = 10)
toc()

best_rounds_cv <- baseline_model_cv$best_iteration # 239
tic()
cv_model <- xgboost::xgboost(params = list(objective = "multi:softmax", eval_metric = c("merror"), num_class = 3), 
                             data = x_train, 
                             nrounds = best_rounds_cv, verbose = 2)
toc()
cv_model_test_predict <- predict(cv_model, x_test)

cv_model_test_predict <- data.frame("description" = test$description, "des_pred" = cv_model_test_predict)

# Accuracy 
sum(cv_model_test_predict$description == cv_model_test_predict$des_pred) / nrow(cv_model_test_predict)

# Hyperparameter tuning
grid_train <- grid_latin_hypercube(
  finalize(mtry(), train_data), 
  min_n(),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(),
  size = 20 
)

grid_train_2 <- grid_train %>%
  mutate(
    learn_rate = .255 + .1 * ((1 : nrow(grid_train)) / nrow(grid_train)), 
    # has to be between 0 and 1
    mtry = mtry / nrow(train_data),
    min_n = min_n - 2,
    sample_size = ifelse(sample_size < .5, sample_size + .3, sample_size)
  )

get_metrics <- function(df, row = 1) {
  
  params <-
    list(
      booster = "gbtree",
      objective = "multi:softmax",
      eval_metric = c("merror"),
      num_class = 3,
      eta = df$learn_rate,
      gamma = df$loss_reduction,
      subsample= df$sample_size,
      colsample_bytree= df$mtry,
      max_depth = df$tree_depth,
      min_child_weight = df$min_n
    )
  
  # tuning with cv
  fd_model <- xgboost::xgb.cv(data = x_train, params = params, nrounds = nrounds, nfold = 5, stratified = T, 
                              metrics = list("merror"), early_stopping_rounds = 10, print_every_n = 10)
  
  output <- params
  output$iter = fd_model$best_iteration
  output$error = fd_model$evaluation_log[output$iter]$test_merror_mean
  
  this_param <- bind_rows(output)
  
  if (row == 1) {
    saveRDS(this_param, "data/modeling.rds")
  } else {
    prev <- readRDS("data/modeling.rds")
    for_save <- bind_rows(prev, this_param)
    saveRDS(for_save, "data/modeling.rds")
  }
  return(this_param)
}

tic()
results_train <- map_df(1 : nrow(grid_train_2), function(x) {
  message(glue::glue("Row {x}"))
  get_metrics(grid_train_2 %>% dplyr::slice(x), row = x)
})
toc()

saveRDS(results_train, file = "results_train.rds")

results_train %>%
  select(error, eta, gamma, subsample, colsample_bytree, max_depth, min_child_weight) %>% #
  pivot_longer(eta:min_child_weight,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, error, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE, size = 3) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "error") +
  theme_minimal()


tuned_rounds <- 431
tuned_params <- list(
  booster = "gbtree",
  objective = "multi:softmax",
  eval_metric = c("merror"),
  num_class = 3,
  eta = 0.340, 
  gamma = 8.508545e+00,
  max_depth = 8,
  min_child_weight = 18, 
  subsample = 0.5188943,
  colsample_bytree = 2.227221e-05,
  nthread = 0
)
tic()
final_model_proj <- xgboost::xgboost(params = tuned_params, data = x_train, nrounds = tuned_rounds, early_stopping_rounds = 10, verbose = 2)
toc()
# plot importance of variables
importance_final_mod <- xgb.importance(feature_names = colnames(final_model_proj), model = final_model_proj)

importance_final_mod

importance_plot_final_mod <- xgb.ggplot.importance(importance_matrix = importance_final_mod)

importance_plot_final_mod

# make prediction
xgb_mod_full_pred <- predict(final_model_proj, x_test)

# add predictions to all the data
full_pred_data <- data.frame("description" = test$description, "des_pred" = xgb_mod_full_pred)

# Accuracy 
sum(full_pred_data$description == full_pred_data$des_pred) / nrow(full_pred_data)
