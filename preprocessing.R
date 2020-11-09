library(xgboost)
library(tidyverse)
library(caret)
library(fastDummies)
library(tidymodels)
library(magrittr)
library(lme4)
library(tictoc)
library(baseballr)

# reading in data
pitches_2019 <- read_csv("project_data.csv")
get_player_id <- baseballr::playerid_lookup("Baez", "Javier")
# getting rid of columns that the model won't need.
pitches_2019 %<>%
  select(-c("spin_dir", "spin_rate_deprecated", "break_angle_deprecated", "break_length_deprecated", "game_type", "type",
            "hit_location", "bb_type", "game_year", "hc_x", "hc_y", "tfs_deprecated", "tfs_zulu_deprecated", "umpire", "sv_id",
            "fielder_2", "hit_distance_sc", "launch_speed", "launch_angle", "pitcher_1", "fielder_3", "fielder_4", "fielder_5",
            "fielder_6", "fielder_7", "fielder_8", "fielder_9", "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
            "woba_value", "woba_denom", "babip_value", "iso_value", "launch_speed_angle", "at_bat_number", "pitch_number",
            "pitch_name", "home_score", "away_score", "fld_score", "post_away_score", "post_home_score",
            "post_fld_score", "if_fielding_alignment", "of_fielding_alignment", "barrel", "pitcher", "batter",
            "events", "des", "home_team", "away_team"))


# Convert the descriptions to strike/ball/hit/hit by pitch factors
pitches_2019$description <- factor(pitches_2019$description,
                                   levels = unique(pitches_2019$description),
                                   labels = c("hit", "strike", "strike", "ball", "hit", "strike", "hit", "hit",
                                              "hit", "strike", "hit_by_pitch", "strike", "strike", "strike", "strike"))

# Filter out hit-by-pitch incidents
pitches_2019 <- pitches_2019 %>%
  filter(description != "hit_by_pitch")

# turn on_3b/2b/1b into binary (yes/no)
pitches_2019 <- pitches_2019 %>%
  mutate(on_3b_yes_no = ifelse(is.na(on_3b), 0, 1)) %>%
  mutate(on_2b_yes_no = ifelse(is.na(on_2b), 0, 1)) %>%
  mutate(on_1b_yes_no = ifelse(is.na(on_1b), 0, 1)) %>%
  select(-c("on_3b", "on_2b", "on_1b"))

# remove pitch_type: KN, "", EP, FO. and make KC = CU
pitches_2019$pitch_type[pitches_2019$pitch_type == 'KC'] <- 'CU'

pitches_2019 <- pitches_2019[!(pitches_2019$pitch_type =="KN"  |
                                 pitches_2019$pitch_type =="" |
                                 pitches_2019$pitch_type =="EP" |
                                 pitches_2019$pitch_type =="FO"),]

# player look up by mlbam key
chadiwck_reduced <- chadwick_player_lu_table %>% select (key_mlbam, name_last, name_first)
# remove NA to make df smaller
chadiwck_reduced <- chadiwck_reduced[!is.na(chadiwck_reduced$key_mlbam),]
# getting each catcher's name
pitches_2019 <- merge(x = pitches_2019, y = chadiwck_reduced, by.x = "fielder_2_1", by.y = "key_mlbam")
pitches_2019 <- mutate(pitches_2019, catcher_name = paste(name_first, name_last, sep = " "))
pitches_2019 <- subset(pitches_2019, select = -c(name_last, name_first))

#can now remove fielder_2 b/c we have catcher name
pitches_2019 <- pitches_2019 %>%
  select(-c("fielder_2_1"))

#umpires
umpire_ids_game_pk <- read_csv("umpires_ids_game_pk.csv")
names(umpire_ids_game_pk) = c("id", "position", "umpire_name", "game_pk", "game_date")
hp_umpires <- umpire_ids_game_pk %>%
  filter(position == "HP") %>%
  filter(game_date < "2019-9-30" & game_date > "2019-03-22") %>%
  select(c("game_pk", "umpire_name"))

pitches_2019 <- merge(pitches_2019, hp_umpires, by = c("game_pk"))
pitches_2019 %<>% select(-c("game_pk"))
#nrow(pitches_2019) = 380841 before umps
#nrow(pitches_2019) = 378517 - after umps

pitches_2019 %<>%
  mutate(batter_name = player_name) %>%
  select(-c("player_name"))

# turning strings into factors
pitches_2019 %<>%
  mutate(pitch_type = as.factor(pitch_type)) %>%
  mutate(batter_name = as.factor(batter_name)) %>%
  mutate(stand = as.factor(stand)) %>%
  mutate(p_throws = as.factor(p_throws)) %>%
  mutate(balls = as.factor(balls)) %>%
  mutate(strikes = as.factor(strikes)) %>%
  mutate(outs_when_up = as.factor(outs_when_up)) %>%
  mutate(inning = as.factor(inning)) %>%
  mutate(inning_topbot = as.factor(inning_topbot)) %>%
  mutate(pitcher_name = as.factor(pitcher_name)) %>%
  mutate(catcher_name = as.factor(catcher_name)) %>%
  mutate(umpire_name = as.factor(umpire_name))

# turning pitch type variable into dummy variables
pitches_2019 <- fastDummies::dummy_cols(pitches_2019,select_columns = c("pitch_type"))
pitches_2019 <- pitches_2019[complete.cases(pitches_2019),]
write.csv(pitches_2019, "pitches.csv")
