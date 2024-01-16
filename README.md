# PSTAT 131
Predicting the Third Down Offensive Play: Run or Pass?
PSTAT 131 Final Project
Grace Wu

2023-03-19

Introduction
The purpose of this project is to develop a machine learning model that will predict whether a NFL team will decide to run or pass the ball on a third down play.
What is a 3rd Down?
In the NFL (National Football League), a third down refers to the team’s third attempt to advance the ball at least 10 yards down the field. Each team is given four downs, in other words, four attempts, to advance the ball 10 yards further down. However, the third down is a critical down since if they fail to the reach the necessary yardage to advance further on this down, then they either risk turning the ball over to the other team on fourth down if they choose to go for it, punting the ball, or attempting a field goal if possible.

Third Down GIF

Why is this model relevant?
Because of such high stakes on the third down play, this makes the team’s third down game plan that much more important. Viewers, including me, are often curious about whether the team will run or pass the ball to reach the first down marker. With this model, we will be able to more accurately predict which play the team chooses to make under certain circumstances such as yardage from the first down marker, wind speed, etc. Recently, NFL teams have been able to make more statistically accurate decisions due to advancements made in technology and machine learning models. For example, Amazon worked with the NFL to generate a “Next Gen Stats Decision Guide” utilizing their Amazon Web Services technology, which employs statistical probabilities and data to assist teams in their decision-making process. Likewise, this model we will be building will use machine learning techniques to create the best model to predict whether the team should run or pass the ball on a third down.

Project Guideline
To build this model, we will first need to tidy and clean the data. Our main objective is to use other predictor variables to forecast a binary class “third run or pass”, which outputs whether the team will run, pass, or other (punt, no play, field goal) on the third down. After splitting the training and testing data sets, we will create a recipe and set folds for the 10-fold cross validation. We will utilize Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, Elastic Net Regression, Lasso Regression, Ridge Regression, Decision Tree, Random Forest, and K-Nearest Neighbor models to model the training data. Then, we will select the model that performs the best and fit it into our testing data set to assess its effectiveness.

Exploratory Data Analysis
Before we begin our modeling, let’s take a look at our data to see what we’re working with. This is an important step as the raw data set may contain variables with missing values or wrong types. In such cases, we will need to remove and clean the missing values as well as convert the variables to be factors.

Loading and Exploring the Data
First, we will load in the packages and set up the environment.

library(corrplot)  
## corrplot 0.92 loaded
library(discrim) 
## Loading required package: parsnip
library(corrr)   
library(knitr)   
library(MASS) 
library(tidyverse)  
## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2
## ──
## ✔ ggplot2 3.4.0     ✔ purrr   1.0.1
## ✔ tibble  3.1.8     ✔ dplyr   1.1.0
## ✔ tidyr   1.3.0     ✔ stringr 1.5.0
## ✔ readr   2.1.3     ✔ forcats 1.0.0
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ dplyr::filter() masks stats::filter()
## ✖ dplyr::lag()    masks stats::lag()
## ✖ dplyr::select() masks MASS::select()
library(tidymodels)
## ── Attaching packages ────────────────────────────────────── tidymodels 1.0.0 ──
## ✔ broom        1.0.3     ✔ rsample      1.1.1
## ✔ dials        1.1.0     ✔ tune         1.0.1
## ✔ infer        1.0.4     ✔ workflows    1.1.2
## ✔ modeldata    1.1.0     ✔ workflowsets 1.0.0
## ✔ recipes      1.0.4     ✔ yardstick    1.1.0
## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
## ✖ scales::discard() masks purrr::discard()
## ✖ dplyr::filter()   masks stats::filter()
## ✖ recipes::fixed()  masks stringr::fixed()
## ✖ dplyr::lag()      masks stats::lag()
## ✖ dplyr::select()   masks MASS::select()
## ✖ yardstick::spec() masks readr::spec()
## ✖ recipes::step()   masks stats::step()
## • Use suppressPackageStartupMessages() to eliminate package startup messages
library(ggplot2)
library(ggrepel)
library(ggimage)
library(rpart.plot)  
## Loading required package: rpart
## 
## Attaching package: 'rpart'
## 
## The following object is masked from 'package:dials':
## 
##     prune
library(ranger)
library(vip)       
## 
## Attaching package: 'vip'
## 
## The following object is masked from 'package:utils':
## 
##     vi
library(vembedr)     
library(janitor)     
## 
## Attaching package: 'janitor'
## 
## The following objects are masked from 'package:stats':
## 
##     chisq.test, fisher.test
library(randomForest) 
## randomForest 4.7-1.1
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:ranger':
## 
##     importance
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
## 
## The following object is masked from 'package:ggplot2':
## 
##     margin
library(stringr) 
library("dplyr")    
library("yardstick")
tidymodels_prefer()
We will then retrieve our data through two R packages called nflfastR and “nflreadr”, which is a package that contains play-by-play data for almost every NFL play from 1999 till the most recent season, 2022. In this project, we will be manipulating and cleaning the data from seasons 2010 to 2022 to prepare for modeling in the form of a csv file.

library(nflfastR)
options(scipen = 9999)
options(nflreadr.verbose = FALSE)
nflpbp_data <- nflreadr::load_pbp(2010:2022) 
nflpbp_data %>%
  head()
## ── nflverse play by play data ──────────────────────────────────────────────────
## ℹ Data updated: 2023-02-28 01:24:38 PST
## # A tibble: 6 × 372
##   play_id game_id  old_g…¹ home_…² away_…³ seaso…⁴  week posteam poste…⁵ defteam
##     <dbl> <chr>    <chr>   <chr>   <chr>   <chr>   <int> <chr>   <chr>   <chr>  
## 1       1 2010_01… 201009… LA      ARI     REG         1 <NA>    <NA>    <NA>   
## 2      36 2010_01… 201009… LA      ARI     REG         1 ARI     away    LA     
## 3      58 2010_01… 201009… LA      ARI     REG         1 ARI     away    LA     
## 4      82 2010_01… 201009… LA      ARI     REG         1 ARI     away    LA     
## 5     103 2010_01… 201009… LA      ARI     REG         1 ARI     away    LA     
## 6     132 2010_01… 201009… LA      ARI     REG         1 ARI     away    LA     
## # … with 362 more variables: side_of_field <chr>, yardline_100 <dbl>,
## #   game_date <chr>, quarter_seconds_remaining <dbl>,
## #   half_seconds_remaining <dbl>, game_seconds_remaining <dbl>,
## #   game_half <chr>, quarter_end <dbl>, drive <dbl>, sp <dbl>, qtr <dbl>,
## #   down <dbl>, goal_to_go <dbl>, time <chr>, yrdln <chr>, ydstogo <dbl>,
## #   ydsnet <dbl>, desc <chr>, play_type <chr>, yards_gained <dbl>,
## #   shotgun <dbl>, no_huddle <dbl>, qb_dropback <dbl>, qb_kneel <dbl>, …
The NFL data set we are using includes 632374 rows and 372 columns. To build our model and have it run successfully, we will definitely need to narrow down this data set as we don’t need all 372 predictor variables.

dim(nflpbp_data)
## [1] 632374    372
Morphing the Data
For starters, we will filter our data to include only third down offensive plays and exclude all other types of plays such as special team plays or kickoffs as we won’t be needing these other types of plays to build our model.

# 3rd down 
nflpbp_3rd_down <- nflpbp_data %>%  # getting just 3rd down data
  filter(!is.na(down))%>% # offense only 
  filter(down == 3) # third down only 
Now we will need to create our response variable— the variable we will be observing in our project. We will mutate our data set “nflpbp_3rd_down” to add this response variable, which we will call “third_run_or_pass”. This variable returns a 1 if the third down offensive play is a run, a 2 if the third down offensive play is a pass, and a 0 otherwise, such as other offensive plays like a punt, field goal, or no play at all.

nflpbp_3rd_down_runp <- nflpbp_3rd_down %>%
  mutate(third_run_or_pass = case_when(play_type == "punt" ~ 0,
                               play_type == "no_play" ~ 0,
                               play_type == "field_goal" ~ 0,
                               play_type == "run" ~ 1,
                               play_type == "pass" ~ 2))
I then wrote and read the data into a csv file and selected 21 variables out of the 372 to use to build my model. I chose the following 20 predictor variables and 1 response variable, respectively:

posteam: team playing offense
defteam: team playing defense
season: specific year to identify each NFL season in which each play occurred in
season_type: type of season that is being analyzed— REG for regular season, POST for postseason
week: week of the NFL regular season in which a particular play occurred
yardline_100: location of the play on the 100-yard field f.e. 62 on the team’s own 38
quarter_seconds_remaining: number of seconds left in the current quarter
half_seconds_remaining: number of seconds left in the current half
game_seconds_remaining: number of seconds left in the game
qtr: quarter the play is taking place in
goal_to_go: returns a 1 if the offensive team has 10 yards or less to go before they reach the opponent’s goal line and a 0 otherwise
ydstogo: how many yards left to go before getting a first down
posteam_score: offensive team’s score
defteam_score: defensive team’s score
score_differential: score difference between the offense and defense team
wind: direction and speed of the wind in miles per hour
temp: temperature in degrees Fahrenheit during the game
incomplete_pass: includes details about each incomplete pass, such as the quarterback’s name, the intended receiver’s name, and why the pass was ruled incomplete
interception: includes details about each interception, such as the name of the player who made the interception, the name of the quarterback who threw the interception, and possible yardage gained in the return
end_clock_time: time remaining on the clock at the end of a play
third_run_or_pass: returns a 1 if the third down play is a run, a 2 if the third down play is a pass, and a 0 otherwise
I selected these variables because each one holds significant importance in impacting an offensive play before it happens, so such variables will be helpful in predicting whether the team will decide to run or pass the ball on the third down.

Tidying the Data
We now want to remove any missing values in the 21 variables to prevent any potential errors in the future, and after doing so, I wrote it into a CSV file called ‘nfl_3rd_down_runpass’.

thirdrunpass <- as.data.frame(nflpbp_3rd_down_runp)
write.csv(thirdrunpass,"thirdrunpass.csv",row.names=FALSE)
firstrp <- read.csv("thirdrunpass.csv")

nfl_3rd_down_runpass <- 
  firstrp[,c("posteam","defteam","season","season_type",
              "week","yardline_100","quarter_seconds_remaining",
              "half_seconds_remaining","game_seconds_remaining","qtr","goal_to_go",
              "ydstogo", "posteam_score","defteam_score","score_differential","wind","temp",
              "third_run_or_pass","incomplete_pass","interception","end_clock_time")] %>% 
  tibble()

nfl_3rd_down_runpass <- nfl_3rd_down_runpass %>% 
  drop_na()
write.csv(nfl_3rd_down_runpass,"thirddownrp.csv",row.names=FALSE)
# checking that all the columns (variables) do not have any missing values 
colSums(is.na(nfl_3rd_down_runpass))
##                   posteam                   defteam                    season 
##                         0                         0                         0 
##               season_type                      week              yardline_100 
##                         0                         0                         0 
## quarter_seconds_remaining    half_seconds_remaining    game_seconds_remaining 
##                         0                         0                         0 
##                       qtr                goal_to_go                   ydstogo 
##                         0                         0                         0 
##             posteam_score             defteam_score        score_differential 
##                         0                         0                         0 
##                      wind                      temp         third_run_or_pass 
##                         0                         0                         0 
##           incomplete_pass              interception            end_clock_time 
##                         0                         0                         0
nfl_3rd_down_runpass %>%
  head()
## # A tibble: 6 × 21
##   posteam defteam season season_type  week yardl…¹ quart…² half_…³ game_…⁴   qtr
##   <chr>   <chr>    <int> <chr>       <int>   <int>   <int>   <int>   <int> <int>
## 1 ATL     PIT       2010 REG             1      31       4       4    1804     2
## 2 BAL     NYJ       2010 REG             1       1      11      11    1811     2
## 3 NYG     CAR       2010 REG             1      53     572     572    2372     2
## 4 NYG     CAR       2010 REG             1      19      50      50    1850     2
## 5 NYG     CAR       2010 REG             1       4     701     701     701     4
## 6 CAR     NYG       2010 REG             1       4     520     520     520     4
## # … with 11 more variables: goal_to_go <int>, ydstogo <int>,
## #   posteam_score <int>, defteam_score <int>, score_differential <int>,
## #   wind <int>, temp <int>, third_run_or_pass <int>, incomplete_pass <int>,
## #   interception <int>, end_clock_time <chr>, and abbreviated variable names
## #   ¹​yardline_100, ²​quarter_seconds_remaining, ³​half_seconds_remaining,
## #   ⁴​game_seconds_remaining
We see that the trimmed down data set we will be working with contains 11,567 observations and 21 variables, which is a fairly large data set.

dim(nfl_3rd_down_runpass)
## [1] 11567    21
We want our response variable to be of factor type, so we will factorize third_run_or_pass along with the following variables: goal_to_go, season, and season_type.

nfl_3rd_down_runpass$goal_to_go <- as.factor(nfl_3rd_down_runpass$goal_to_go) 
nfl_3rd_down_runpass$season <- as.factor(nfl_3rd_down_runpass$season)
nfl_3rd_down_runpass$season_type <- as.factor(nfl_3rd_down_runpass$season_type)
nfl_3rd_down_runpass$third_run_or_pass <- as.factor(nfl_3rd_down_runpass$third_run_or_pass)
Visual EDA
Now, we will visually take a look at how our predictor variables impact our response variable using a variable correlation plot and bar plots.

Variable Correlation Plot
We will need to select only the continuous, numeric variables when making the correlation heat map.

library(corrplot)
nfl_3rd_down_runpass_numeric <- nfl_3rd_down_runpass %>%
  select_if(is.numeric) # selecting only numeric values
thirddown_cor <- cor(nfl_3rd_down_runpass_numeric)  
thirddown_cor_plt <- corrplot(thirddown_cor, method = "circle", addCoef.col=1, number.cex=0.4) 
 As observed in the plot above, the distribution of the correlations between each variable is relatively spread out as we see slight and extreme negative and positive correlations. For the most part, there was little correlation between many of the predictor variables, which surprised me. However, after examining further, it makes sense as many of the predictor variables have no correlation between each other. For instance, the specific quarter of the game doesn’t affect what the wind speed will be. It’s important to note that the predictors with the greatest positive correlation is quarter and posteam_score & defteam_score, which makes sense as the posteam_score and defteam_score is going to be higher in the fourth quarter compared to the first quarter as the teams will be scoring more as the quarters go on.

Bar-plot
It is now time to examine the relationship between our response variable, third_run_or_pass, and many of our predictors. To do so, let’s create bar plots to see which predictor variables will affect our model most.

Season
We observe that the ratio between run and pass remains relatively the same each season from 2010 to 2022, with most teams passing on the third down.

ggplot(nfl_3rd_down_runpass, aes(season)) + 
  geom_bar(aes(fill = third_run_or_pass)) +
  scale_fill_manual(values = c("purple", "blue","magenta"))


# 1 (blue) represents run, 2 (magenta) represents pass
Yardline
From this plot, we notice that most of the third down plays occur on the opponent’s side of the field, especially in the red zone. Likewise, just from having watched football for many years and being an avid fan, I’ve noticed many times it comes down to the third down when in the red zone because the risk of getting a first down there means the offensive team will score a touchdown, so the defensive team oftentimes pulls out all their game play stops to prevent this from happening. Additionally, we see a significant jump in run plays when near the goal line. This makes sense as passing the ball runs the risk of an interception and a missed pass means zero yardage gain and consequently no first down or score, whereas running the ball has less of a risk when trying to get the first down, many quarterbacks doing a quarterback sneak.

ggplot(nfl_3rd_down_runpass, aes(yardline_100)) + 
  geom_bar(aes(fill = third_run_or_pass)) +
  scale_fill_manual(values = c("purple", "blue","magenta"))


Game Seconds Remaining
Before the very end of the game, there doesn’t appear to be much of a relationship between the time left in the game and our response variable, third_run_or_pass. However, towards the last 100 seconds of the game, we mostly just see pink meaning most teams pass the ball on the third down. This can be explained because since the offensive team doesn’t have much time left to score, they oftentimes will choose to pass the ball as it is the quickest way to gain a large amount of yardage.

ggplot(nfl_3rd_down_runpass, aes(game_seconds_remaining)) + 
  geom_bar(aes(fill = third_run_or_pass)) +
  scale_fill_manual(values = c("purple", "blue","magenta"))


Score Differential
The score differential appears to be normally distribution with an average of 0. It is noteworthy that there is slightly more pink when the score differential is negative, meaning that the losing team is more likely to pass the ball on a third down. Again, this can be explained by the fact that passing the ball allows for potentially greater yardage gain and a faster way to score more points compared to running the ball, so the losing team is more likely to pass the ball.

ggplot(nfl_3rd_down_runpass, aes(score_differential)) + 
  geom_bar(aes(fill = third_run_or_pass)) +
  scale_fill_manual(values = c("purple", "blue","magenta"))


Goal to Go
When the team on offense is in a goal to go (inside the 10 yard line) position, which is represented by 1 on the graph, we see the offensive team running the ball more than when they are not in a goal to go situation (represented by 0). Similarly explained in yardline_100, passing the ball runs the risk of an interception and a missed pass, so running the ball is a safer game play when within 10 yards of the goal line on the third play— pretty much the team’s last chance at scoring a touchdown.

ggplot(nfl_3rd_down_runpass, aes(goal_to_go)) + 
  geom_bar(aes(fill = third_run_or_pass)) +
  scale_fill_manual(values = c("purple", "blue","magenta"))


Yards to Go
In this graph, we see that when the team has only 2 yards or under to go until scoring a touchdown, there is a sharp increase in the type of play used by the offensive team. We notice a drastic jump in run plays, which makes sense since as usually, on third and short plays, either the running back or quarterback will sneak the ball in or jump over the crowd of players to reach the goal line.

ggplot(nfl_3rd_down_runpass, aes(ydstogo)) + 
  geom_bar(aes(fill = third_run_or_pass)) +
  scale_fill_manual(values = c("purple", "blue","magenta"))


Wind
Surprisingly, wind didn’t impact the type of play used on offensive third downs. I expected with greater wind speeds, there would be more run plays. However, it appears that the distribution of run and pass plays remains roughly similar regardless of wind, as observed in the skewed distribution shown below.

ggplot(nfl_3rd_down_runpass, aes(wind)) + 
  geom_bar(aes(fill = third_run_or_pass)) +
  scale_fill_manual(values = c("purple", "blue","magenta"))


Temperature
We see the proportion of run pass plays remain roughly the same regardless of the temperature. I was taken aback by this at first, but later realized that when the temperature is extremely low, the success rate of run and pass plays will both likely decrease together. Thus, if the temperature is lower, this doesn’t necessarily equate to more pass plays proportional to run plays, vice versa.

ggplot(nfl_3rd_down_runpass, aes(temp)) + 
  geom_bar(aes(fill = third_run_or_pass)) +
  scale_fill_manual(values = c("purple","blue","magenta"))


Setting up Models
To set up our models, we will need to first train/test split the data, create a recipe, and then perform cross validation.

Train & Test Split
First, we set a random seed so that the training and testing split we will perform on our data will remain the same every time we run the later codes. I set the proportion to 0.75 so that our model would have more observations to train on than test on to eliminate any errors such as over-fitting. This way, in the future, we can use our testing data set to test the accuracy of our model. I also stratified the split on the response variable, “third_run_or_pass”.

set.seed(1234)
nfl_3rd_down_runpass_split <- nfl_3rd_down_runpass %>%
  initial_split(prop=0.75,strata="third_run_or_pass")
thirddown_train <- training(nfl_3rd_down_runpass_split)
thirddown_test <- testing(nfl_3rd_down_runpass_split)
After performing the splits, there are 8,675 observations in the training data set and 2,892 observations in the testing data set.

dim(thirddown_train)
## [1] 8675   21
dim(thirddown_test)
## [1] 2892   21
Recipe Building
As we build our model, we will be using pretty much the same predictors, response variable, and model conditions. So, we will create one universal recipe for all of our models to work with. Each model will use this one recipe but employ it under different methods specific to their respective model.

We will be using the following 13 predictors out of the 20 in our trimmed down data set: season, season_type, week, yardline_100, goal_to_go, quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining, qtr, ydstogo, score_differential, wind, and temp. We will exclude posteam, defteam, incomplete pass, interception, and end_clock_time as they do not affect the model’s ability; for example, the team playing offense and the team playing defense do not impact the model’s abilities. We will also convert season_type, goal_to_go, and season to be dummy variables as they are categorical variables (we converted them into factors earlier), center and scale our data, and prep and bake the trained model to pre-process the data before fitting the models and make predictions.

thirddown_recipe <- # building the recipe 
  recipe(third_run_or_pass ~ season + season_type + week + yardline_100 + goal_to_go +
           quarter_seconds_remaining + half_seconds_remaining + 
           game_seconds_remaining + qtr + ydstogo +
           score_differential + wind + temp, 
         data = thirddown_train) %>% 
  step_dummy(season_type) %>% 
  step_dummy(goal_to_go) %>%
  step_dummy(season) %>%
  step_center(all_predictors()) %>% # standardizing our predictors
  step_scale(all_predictors())

prep(thirddown_recipe) %>% bake(thirddown_train)
## # A tibble: 8,675 × 24
##     week yardlin…¹ quart…² half_…³ game_…⁴    qtr ydstogo score…⁵    wind   temp
##    <dbl>     <dbl>   <dbl>   <dbl>   <dbl>  <dbl>   <dbl>   <dbl>   <dbl>  <dbl>
##  1 -1.82    -1.53  -1.49    -1.48    0.107 -0.512 -1.21   -0.461   0.509  0.585 
##  2 -1.82    -1.42   1.08    -0.190  -0.970  1.27  -0.578   0.879   0.319  0.403 
##  3 -1.82    -1.05  -0.892   -1.18   -1.48   1.27   0.0532  1.36    0.319  0.403 
##  4 -1.82    -1.49  -0.992   -1.23   -1.51   1.27  -1.21    0.400  -0.443  1.80  
##  5 -1.82    -0.105 -1.06    -1.26   -1.53   1.27  -0.157   2.51    0.128  1.19  
##  6 -1.63    -1.53  -1.47     0.212   0.985 -1.40  -1.21    0.688  -0.823  0.0391
##  7 -1.63    -0.688  0.580    1.24   -0.228  0.378 -1.21   -1.32   -0.0621 1.13  
##  8 -1.63     1.54   1.50     1.70    1.76  -1.40  -0.368   0.113   0.319  0.463 
##  9 -1.63    -1.49   0.0211   0.960  -0.373  0.378 -0.999  -0.0783  0.319  0.463 
## 10 -1.63    -1.53   0.833   -0.315   0.712 -0.512 -1.21    0.783   0.128  2.04  
## # … with 8,665 more rows, 14 more variables: third_run_or_pass <fct>,
## #   season_type_REG <dbl>, goal_to_go_X1 <dbl>, season_X2011 <dbl>,
## #   season_X2012 <dbl>, season_X2013 <dbl>, season_X2014 <dbl>,
## #   season_X2015 <dbl>, season_X2016 <dbl>, season_X2017 <dbl>,
## #   season_X2018 <dbl>, season_X2019 <dbl>, season_X2020 <dbl>,
## #   season_X2022 <dbl>, and abbreviated variable names ¹​yardline_100,
## #   ²​quarter_seconds_remaining, ³​half_seconds_remaining, …
K-fold Cross Validation
To conduct k-fold (10-fold for this project) stratified cross validation, we will create 10 folds and stratify it on our response variable, third_run_or_pass, to ensure that there is no imbalance in each fold from the data set. K-fold cross validation is done by splitting the data into k folds, so in this case, we will be taking the training data set and assigning each observation to 1 of the 10 folds. With each fold, a testing set will be created and the remaining k-1 folds will represent the training set for that fold. We choose to use k-fold cross validation as it reduces the risk of over-fitting and results in a more representative estimate of the testing accuracy and overall model performance.

thirddown_folds <- vfold_cv(thirddown_train,v=10,strata=third_run_or_pass)
Model Building
Now, we will begin building our model. We will be testing out nine different machine learning techniques to see which one will generate the best model to predict whether a team will run or pass on the third down. These nine models consist of the following: logistic regression, LDA, QDA, k-nearest neighbor, elastic net regression, ridge regression, lasso regression, decision tree, and random forest. All together, this process took quite a while as our data set is relatively large. Tuning the models took the longest, and to avoid running each model every single time in the future, we will save our results into a RDA file.

Because this project is a classification model, I set my metric of performance to roc_auc as it will best measure the classification model’s performance by calculating the area under the curve of the receiver operating characteristic (ROC) curve. We know that the greater the roc_auc value is, the better the respective classification model will perform. To achieve the results, let’s start building our model using the following five steps!

Fitting the Models
To fit each model, we will follow the below steps.

Step 1: Set up the model by specifying the desired type of model and the parameters to be tuned, setting the mode (classification in this case), and setting the engine.

# Logistic Regression
log_model <- multinom_reg(penalty = 0) %>% # multinom because the response variable has three levels
  set_mode("classification") %>%
  set_engine("glmnet")

# LDA 
lda_model <- discrim_linear() %>%
  set_mode("classification") %>%
  set_engine("MASS")

# QDA
qda_model <- discrim_quad() %>%
  set_mode("classification") %>%
  set_engine("MASS")

# K-nearest Neighbor
knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

# Elastic Net Regression
elastic_spec <- multinom_reg(penalty=tune(),mixture=tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# Ridge Regression
ridge_spec <- multinom_reg(mixture=0, penalty = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# Lasso Regression
lasso_spec <- multinom_reg(mixture=1, penalty = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# Decision trees 
tree_spec <- decision_tree(cost_complexity=tune()) %>%
  set_mode("classification") %>%
  set_engine("rpart")

# Random Forest 
rf_spec <- rand_forest(mtry=tune(),trees=tune()) %>% # took out min_n as data set was too large 
  set_mode("classification") %>%
  set_engine("ranger")
Step 2: Set up the workflow for the model and add the model and recipe.

# Logistic Regression 
log_workflow <- workflow() %>% 
  add_model(log_model) %>% 
  add_recipe(thirddown_recipe)

# LDA
lda_workflow <- workflow() %>%
  add_model(lda_model) %>%
  add_recipe(thirddown_recipe)

# QDA
qda_workflow <- workflow() %>%
  add_model(qda_model) %>%
  add_recipe(thirddown_recipe)

# K-nearest Neighbor
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(thirddown_recipe)

# Elastic Net Regression
elastic_workflow <- workflow() %>% 
  add_recipe(thirddown_recipe) %>% 
  add_model(elastic_spec)

# Ridge Regression
ridge_workflow <- workflow() %>% 
  add_recipe(thirddown_recipe) %>% 
  add_model(ridge_spec)

# Lasso Regression 
lasso_workflow <- workflow() %>% 
  add_recipe(thirddown_recipe) %>% 
  add_model(lasso_spec)

# Decision Tree 
tree_wf <- workflow() %>%
  add_model(tree_spec) %>%
  add_recipe(thirddown_recipe)

# Random Forest
rf_workflow <- workflow() %>% 
  add_recipe(thirddown_recipe) %>% 
  add_model(rf_spec)
Step 3: Create a tuning grid, specifying the levels and ranges of the tuned parameters. (Note: We cannot tune the logistic regression, LDA, or QDA models.)

# K-Nearest Neighbor
knn_grid <- grid_regular(neighbors(range=c(10,50)),levels=10)

# Elastic Net Regression
elastic_grid <- grid_regular(penalty(range=c(0.1,4), trans = identity_trans()),mixture(range=c(0,1)),levels=10)

# Ridge Regression + Lasso Regression
penalty_grid <- grid_regular(penalty(range=c(1,5)),levels=20)

# Decision Tree
tree_grid <- grid_regular(cost_complexity(range=c(-3,-1)),levels=10)

# Random Forest 
rf_grid <- grid_regular(mtry(range = c(1, 3)), trees(range = c(1,200)), levels = 8)
Step 4: Tune the model and specify the workflow, k-fold cross validation folds, and tuning grid.

# K-Nearest Neighbor
knn_tune <- tune_grid(
    knn_workflow,
    resamples = thirddown_folds,
    grid = knn_grid
)

# Elastic Net Regression 
elastic_tune <- tune_grid(
  elastic_workflow,
  resamples = thirddown_folds,
  grid = elastic_grid
)

# Ridge Regression 
ridge_tune <- tune_grid(
  ridge_workflow,
  resamples = thirddown_folds,
  grid = penalty_grid
)

# Lasso Regression
lasso_tune <- tune_grid(
  lasso_workflow,
  resamples = thirddown_folds,
  grid = penalty_grid
)

# Decision Tree 
tree_tune <- tune_grid(
  tree_wf,resamples=thirddown_folds,
  grid = tree_grid
)

# Random Forest
rf_tune <- tune_grid(
  rf_workflow,
  resamples = thirddown_folds,
  grid = rf_grid
)
Step 5: Save the results with the tuned models into a RDA file.

save(knn_tune,file="knn_tune.rda")
save(elastic_tune,file="elastic_tune.rda")
save(ridge_tune,file="ridge_tune.rda")
save(lasso_tune,file="lasso_tune.rda")
save(tree_tune,file="tree_tune.rda")
save(rf_tune,file="rf_tune.rda")
Model Results
We will proceed to load the saved results, and now we finally have our model results!

load("knn_tune.rda")
load("elastic_tune.rda")
load("ridge_tune.rda")
load("lasso_tune.rda")
load("tree_tune.rda")
load("rf_tune.rda")
Model Autoplots
Using the autoplot function in r, we will visualize the tuned model results and the effects they hold on roc_auc, our metric of choice.

K-Nearest Neighbor Plot
We can see from the plot that the greater the number of nearest neighbors, the more accurate our model is. The highest ROC AUC value is roughly around 0.65.

autoplot(knn_tune, metric="roc_auc")


Elastic Net Plot
From the elastic net plot, it appears that lower penalty values did better as we see their respective ROC AUC values are larger. As the penalty value becomes larger, the model performs worse due to under-fitting and the coefficients of the predictors being reduced to too small of values. Thus, with greater penalty values, the model isn’t able to capture the complexity of the data and it becomes harder for the model to predict. Likewise, lower mixture values appear to perform better as when the value is 0.1, the ROC AUC value is much greater than when mixture is at 0.5, for instance. Compared this plot to the k-nearest neighbor plot above, we see that the elastic net regression model performs ever so slightly better.

autoplot(elastic_tune, metric="roc_auc") 


Decision Tree Plot
Similar to the elastic net model, we see that larger penalties resulted in a lower ROC AUC value and caused the accuracy of the decision tree to drop. This time, we observe that the largest ROC AUC value is roughly 0.725, which is fairly greater than the k-nearest neighbor and elastic net models.

autoplot(tree_tune, metric="roc_auc")


Random Forest Plot
Because of how large our data set is, I didn’t tune min_n and only tuned the following two parameters: mtry and trees. Mtry specifies the number of predictor variables to be randomly selected for each split when making its decision during tree building, and trees represents the number of trees to grow in the forest. It looks like the ROC AUC value increases as the number of trees increases and the accuracy increases as the number of predictors increases. This model seems to be the best model in predicting whether the offense team should run or pass the ball on a third down as it has the greatest ROC AUC value at roughly 0.74.

autoplot(rf_tune, metric="roc_auc")


Accuracy of Our Models
Let’s take a look at the results of all nine models to see which model had the greatest ROC AUC value, and subsequently, performed the best!

thirddown_log_reg_fit <- log_workflow %>%
  fit_resamples(thirddown_folds)
best_log <- collect_metrics(thirddown_log_reg_fit)[2,]

thirddown_lda_reg_fit <- lda_workflow %>%
  fit_resamples(thirddown_folds)
best_lda <- collect_metrics(thirddown_lda_reg_fit)[2,]

thirddown_qda_reg_fit <- qda_workflow %>%
  fit_resamples(thirddown_folds)
best_qda <- collect_metrics(thirddown_qda_reg_fit)[2,]

best_knn <- select_by_one_std_err(knn_tune,metric="roc_auc",desc(neighbors))
best_elastic <- select_by_one_std_err(elastic_tune,metric="roc_auc",penalty,mixture)
best_ridge <- select_by_one_std_err(ridge_tune,metric="roc_auc",penalty)
best_lasso <- select_by_one_std_err(ridge_tune,metric="roc_auc",penalty)
best_tree <- select_by_one_std_err(tree_tune,metric="roc_auc",cost_complexity)
best_rf <- select_by_one_std_err(rf_tune,metric="roc_auc",mtry,trees)

ROC_AUC <- c(best_log$mean, best_lda$mean, best_qda$mean, best_knn$mean, best_elastic$mean,best_ridge$mean, best_lasso$mean,best_tree$mean,best_rf$mean)
Model <- c("Logistic Regression", "LDA", "QDA", "K-Nearest Neighbors", "Elastic Net Regression", "Ridge Regression", "Lasso Regression", "Decision Tree","Random Forest")

# create a tibble of all nine models and their respective ROC AUC value 
thirddown_results <- tibble(Model,ROC_AUC) %>%
  arrange(ROC_AUC) # arrange by lowest ROC AUC value
thirddown_results
## # A tibble: 9 × 2
##   Model                  ROC_AUC
##   <chr>                    <dbl>
## 1 Ridge Regression         0.620
## 2 Lasso Regression         0.620
## 3 QDA                      0.631
## 4 K-Nearest Neighbors      0.648
## 5 Elastic Net Regression   0.663
## 6 LDA                      0.677
## 7 Logistic Regression      0.686
## 8 Decision Tree            0.725
## 9 Random Forest            0.739
It appears that the random forest model performed best with a ROC AUC value of 0.7421052!



Visualize the Results
To visualize the results of the model accuracy, we will create a bar plot and dot plot as seen below.

thirddown_bar_plot <- ggplot(thirddown_results, 
       aes(x = Model, y = ROC_AUC)) + 
  geom_bar(stat = "identity", width=0.2, fill = "blue", color = "black") + 
  labs(title = "Performance of Our Models") + 
  theme_minimal()
thirddown_bar_plot


thirddown_dot_plot <- ggplot(thirddown_results, aes(x = Model, y = ROC_AUC)) +
  geom_point(fill = "blue", col = "blue") + 
  geom_segment(aes(x = Model, 
                   xend = Model, 
                   y=min(ROC_AUC), 
                   yend = max(ROC_AUC)), 
               linetype = "dashed") + 
  labs(title = "Performance of Our Models") + 
  theme_minimal() +
  coord_flip()
thirddown_dot_plot


After viewing the results of the model in a visual matter, it becomes clear that the random forest model performed best. Thus, when fitting our model to the testing data to test our model’s performance, we will be using the random forest model.

Results From the Best Model
Performance on the Folds
Because we know our best model is the random forest model, we will now analyze the best random forest model and fit that specific model to our testing data. So, which tuned parameters resulted in the best random forest model?

show_best(rf_tune, metric = "roc_auc") %>% #showing the best random forest model
  select(-.estimator, .config) %>%
  slice(1)
## # A tibble: 1 × 7
##    mtry trees .metric  mean     n std_err .config              
##   <int> <int> <chr>   <dbl> <int>   <dbl> <chr>                
## 1     2   171 roc_auc 0.745    10 0.00677 Preprocessor1_Model20
It looks like random forest model #14 with 2 predictors and 114 trees performed the best with an ROC AUC value of 0.7451711!

ROC Curve
Looking at the ROC curve for our three levels in our response variable, third_run_or_pass, our model seems to have been built well as the results look fairly accurate! This is because ideally, we want a curve that follows an upward trajectory while touching the upper left as much as possible as we measure the ROC AUC value by taking the area under the ROC AUC curve.

# updating the random forest workflow to include importance = "impurity" to create a variable importance chart later 
rf_spec <- rand_forest(mtry=tune(),trees=tune()) %>% 
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")
rf_workflow <- workflow() %>% 
  add_recipe(thirddown_recipe) %>% 
  add_model(rf_spec)

# fitting to the training data set 
rf_final_workflow_train <- finalize_workflow(rf_workflow, best_rf)
thirddown_rf_fit <- fit(rf_final_workflow_train, data = thirddown_train)

# ROC curve
thirddown_roc_curve <- augment(thirddown_rf_fit, new_data = thirddown_test) %>%
  roc_curve(third_run_or_pass, estimate = .pred_0:.pred_2)  # computing the ROC curve for this model
## Warning: Returning more (or less) than 1 row per `summarise()` group was deprecated in
## dplyr 1.1.0.
## ℹ Please use `reframe()` instead.
## ℹ When switching from `summarise()` to `reframe()`, remember that `reframe()`
##   always returns an ungrouped data frame and adjust accordingly.
## ℹ The deprecated feature was likely used in the yardstick package.
##   Please report the issue at <]8;;https://github.com/tidymodels/yardstick/issueshttps://github.com/tidymodels/yardstick/issues]8;;>.
autoplot(thirddown_roc_curve)


Testing the Model + Final ROC AUC Results
As demonstrated in the code above, we took our best tuned random forest model and fit it to the training data to prepare it for testing. We can now test this model and collect its ROC AUC value to see how it performs on a data set that hasn’t been trained on yet (testing data set).

# collecting the roc_auc of the model on the testing data set 
thirddown_roc_auc <- augment(thirddown_rf_fit, new_data = thirddown_test) %>%
  roc_auc(third_run_or_pass, .pred_0:.pred_2) %>%
  select(.estimate)
thirddown_roc_auc
## # A tibble: 1 × 1
##   .estimate
##       <dbl>
## 1     0.737
Surprisingly, our random forest model performed slightly better on the testing data set with an ROC AUC value of 0.749052. This is a relatively solid value, showing overall, our model performed relatively well!

Variable Importance Chart
To see which variables played the most important role in predicting the outcome of the response variable, third_run_or_pass, we can create a variable important plot (VIP). We observe that the variable “ydstogo” affected our response variable most by far. This makes sense since depending how many yards there are left to go, this changes the team’s play book greatly. For example, if there are only say 2 yards left to go, we would notice a drastic jump in run plays as usually either the running back or quarterback will run the ball on third and short plays since the cost of an interception from a pass is much greater than it was if the team was say 30 yards away from the goal line.

thirddown_rf_fit %>%
  extract_fit_engine() %>%
  vip(aesthetics = list(fill = "blue", color = "black")) 


Conclusion
After fitting nine different models and testing and analyzing each one, the best model for predicting whether an NFL team will run or pass the ball on a third down offensive play is the random forest model. The model that performed the worst was the ridge regression model. I expected this as a ridge regression model typically doesn’t do as well with large data sets due to insufficient regularization and the over-fitting of the training data set.

Although the random forest model was our best model, it is still far from perfect. A possible improvement for this project would be to implement more models such as Support Vector Machines, which could have potentially performed better than the random forest model. However, I wasn’t able to run this model as my computer’s capability of performing on it failed. I also wasn’t able to incorporate min_n in my random forest model as my data set was too large, so an unresolved question I have is how to further improve my random forest model.

Moving forward, I would like to explore more predictor variables that could contribute to predicting whether a third down offensive play is more likely to be a run or pass play. I wish to also experiment with more visualization packages on the predictor variables and to overall increase the accuracy of my model. As an avid football fan, this project made me gain a deeper understanding of potential factors that could influence whether a team would run or pass the ball on a third down. For example, I previously thought the temperature would affect third down run pass plays greatly, but now I know that it doesn’t after using machine learning techniques to build this model. In conclusion, this project allowed me to apply my knowledge of machine learning and to take a deeper dive into the real world application of statistics with regards to football.
