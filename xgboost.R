library(xgboost)
library(tidyverse)
library(caret)
library(fastDummies)
library(tidymodels)
library(magrittr)
library(lme4)
library(tictoc)
library(baseballr)
library(recipes)
library(kernlab)

#Inspiration/guidance from: https://www.opensourcefootball.com/posts/2020-09-07-estimating-runpass-tendencies-with-tidymodels-and-nflfastr/#model-evaluation

#reading in data
pitches_2019 <- read_csv("pitches.csv")

#preparing the data
set.seed(123)
pitches_split <- initial_split(pitches_2019, strata = description)

pitches_train <- training(pitches_split)
pitches_train %<>% select(-c("game_date", "zone", "pitcher_name", "catcher_name", "umpire_name", "batter_name", "pitch_type", 
                             "stand", "p_throws", "balls", "strikes", "outs_when_up", "inning", "inning_topbot", "on_3b_yes_no", 
                             "on_2b_yes_no", "on_1b_yes_no"))

pitches_test <- testing(pitches_split)
pitches_test %<>% select(-c("game_date", "zone", "pitcher_name", "catcher_name", "umpire_name", "batter_name", "pitch_type", 
                            "stand", "p_throws", "balls", "strikes", "outs_when_up", "inning", "inning_topbot", "on_3b_yes_no", 
                            "on_2b_yes_no", "on_1b_yes_no"))

pitches_folds <- vfold_cv(pitches_train, strata = description)

#prepping the model
pitches_recipe <- recipe(description ~., data = pitches_train)

pitches_model <- 
  boost_tree(
    mtry = tune(),
    trees = 1000, 
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune(),                    
    sample_size = tune(),         
    stop_iter = 100
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

pitches_workflow <- workflow() %>%
  add_recipe(pitches_recipe) %>%
  add_model(pitches_model)

xgb_grid <- grid_latin_hypercube(
  finalize(mtry(), pitches_train),
  min_n(),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(),
  size = 30
)

#THIS IS GONNA TAKE A VERY LONG TIME - took my machine 23hrs for 378517 rows. This is 708138 rows.
tic()
xgb_res <- tune_grid(
  pitches_workflow,
  resamples = pitches_folds,
 grid = xgb_grid,
 control = control_grid(save_pred = TRUE)
)
toc()
saveRDS(xgb_res, file = "xgb_res.rds")

#results from hyperparameter tuning
xgb_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, mtry:sample_size) %>%
  pivot_longer(mtry:sample_size,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC") +
  theme_minimal()

best_auc <- select_best(xgb_res, "roc_auc")

pitches_xgb <- finalize_workflow(
  pitches_workflow,
  parameters = best_auc
)

tic()
final_mod <- last_fit(pitches_xgb, pitches_split)
saveRDS(final_mod, file = "final_mod.rds")
toc()

#model results
collect_metrics(final_mod)

final_mod %>%
  collect_predictions() %>%
  roc_curve(is_strike, .pred_0) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(size = 1.5, color = "midnightblue") +
  xlab('1 - Specificity') +
  ylab('Sensitivity') +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  ) +
  ggtitle('ROC Curve') +
  theme_minimal()

final_mod %>%
  collect_predictions() %>%
  mutate(pred_rounded = round(.pred_1,1)) %>%
  group_by(pred_rounded) %>%
  summarise(mean_prediction = mean(.pred_1),
            mean_actual = mean(as.numeric(is_strike) - 1),
            n = n(),
            se = sd(as.numeric(is_strike) - 1 - .pred_1)/sqrt(n)) %>%
  ggplot(aes(x = pred_rounded, y = mean_actual)) +
  geom_abline() +
  geom_point(aes(size = n)) +
  theme_minimal() +
  xlab('Predicted Probability') +
  ylab('Actual Probability') +
  ggtitle('Calibration Plot, Test Data') +
  ylim(0,1) +
  xlim(0,1)



