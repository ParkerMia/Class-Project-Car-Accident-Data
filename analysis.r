# Load required libraries
library(tidyverse)   # data manipulation & visualization
library(GGally)      # ggpairs for EDA
library(caret)       # train/test split, cross-validation, treebag
library(rpart)       # classification trees
library(rpart.plot)  # plot trees
library(ipred)       # bagging
library(readr)       # read CSV files
library(class)       # knn
library(lubridate)   # handling date-time

# -----------------------------
# Load and prepare main dataset
# -----------------------------
US_Accidents <- read_csv("US_Accidents_March23_sampled_500k.csv") %>% drop_na()

# -----------------------------
# Load US cities data and merge
# -----------------------------
uscities <- read_csv("uscities.csv") %>%
  mutate(City = city, State = state_id)

US_Accidents <- US_Accidents %>%
  left_join(
    uscities %>% select(City, State, population, density),
    by = c("City", "State")
  ) %>%
  drop_na() %>%
  filter(population != 0)

# -----------------------------
# Normalize selected features
# -----------------------------
accidentsNorm <- data.frame(scale(US_Accidents[, c(10, 21, 22, 23, 24, 25, 27, 28, 47, 48)]),
                            Outcome = US_Accidents$Severity) %>%
  drop_na()

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
# Severity histogram
ggplot(US_Accidents, aes(Severity)) +
  geom_histogram(fill = '#a8c85f')

# GGPairs for selected numeric features
ggpairs(US_Accidents, columns = c(3, 21, 22, 23, 47, 48),
        lower = list(continuous = wrap("points", color = "#5badfbff", alpha = 0.3)),
        diag = list(continuous = wrap("densityDiag", fill = "#5badfbff", alpha = 0.3)))

# Accidents per month
US_Accidents$Start_Time <- as.POSIXct(US_Accidents$Start_Time)
daily_counts <- US_Accidents %>% mutate(date = as.Date(Start_Time)) %>% count(date)
monthly_avg <- daily_counts %>%
  mutate(year_month = format(date, "%Y-%m")) %>%
  group_by(year_month) %>%
  summarise(avg_accidents = mean(n)) %>%
  ungroup()
ggplot(monthly_avg, aes(x = as.Date(paste0(year_month, "-01")), y = avg_accidents)) +
  geom_line(color = "#ee5b61", linewidth = 1.2) +
  labs(title = "Average Daily Accidents per Month", x = "Month", y = "Average Daily Count")

# Accidents by hour
US_Accidents %>%
  mutate(hour = hour(Start_Time)) %>%
  ggplot(aes(x = hour)) +
  geom_histogram(binwidth = 1, position = "dodge", fill = '#ffe034', color = 'black') +
  labs(title = "Accidents by Hour", x = "Hour", y = "Count")

# Top 5 cities
top_cities <- c('Miami', 'Los Angeles', 'Orlando', 'Dallas', 'Houston')
US_Accidents %>%
  filter(City %in% top_cities) %>%
  ggplot(aes(x = City)) +
  geom_bar(fill = '#98c8f9', color = 'black')

# -----------------------------
# Milestone 1: KNN
# -----------------------------
# Stratified sampling by outcome
set.seed(1)
accidents1 <- accidentsNorm %>% filter(Outcome == 1)
accidents2 <- accidentsNorm %>% filter(Outcome == 2) %>% sample_n(7500)
accidents3 <- accidentsNorm %>% filter(Outcome == 3)
accidents4 <- accidentsNorm %>% filter(Outcome == 4)

# Training indices (70% sample)
sample1 <- sample(1:nrow(accidents1), nrow(accidents1) * 0.7)
sample2 <- sample(1:nrow(accidents2), nrow(accidents2) * 0.7)
sample3 <- sample(1:nrow(accidents3), nrow(accidents3) * 0.7)
sample4 <- sample(1:nrow(accidents4), nrow(accidents4) * 0.7)

# Training and testing sets
trainStrat <- rbind(accidents1[sample1, ], accidents2[sample2, ], accidents3[sample3, ], accidents4[sample4, ])
testStrat <- rbind(accidents1[-sample1, ], accidents2[-sample2, ], accidents3[-sample3, ], accidents4[-sample4, ])

trainFea <- trainStrat %>% select(-Outcome)
testFea <- testStrat %>% select(-Outcome)
trainOut <- trainStrat$Outcome
testOut <- testStrat$Outcome

# KNN prediction (example with k=12)
knn.pred12 <- knn(train = trainFea, test = testFea, cl = trainOut, k = 12)
table(knn.pred12, testOut)
mean(knn.pred12 == testOut)

# Determine best k
accuracy <- sapply(1:30, function(k) {
  mean(knn(train = trainFea, test = testFea, cl = trainOut, k = k) == testOut)
})
ggplot(data.frame(k = 1:30, accuracy = accuracy), aes(x = k, y = accuracy)) +
  geom_line(color = "#5badfbff") +
  xlab("Neighborhood Size") + ylab("Accuracy")
best_k <- which.max(accuracy)

# -----------------------------
# Milestone 2: Classification Trees
# -----------------------------
# Balance Outcome 2 by sampling
accidents2_sample <- accidentsNorm %>% filter(Outcome == 2) %>% sample_n(7500)
accidentsBalanced <- accidentsNorm %>% filter(Outcome != 2) %>% rbind(accidents2_sample)

set.seed(10)
trainIndex <- createDataPartition(accidentsBalanced$Outcome, p = 0.7, list = FALSE)
trainSet <- accidentsBalanced[trainIndex, ]
testSet <- accidentsBalanced[-trainIndex, ]

# Fit classification tree
classTree <- rpart(Outcome ~ ., data = trainSet, method = "class")
custom_colors <- c("#98c8f9", "#a8c85f", "#ffe034", "#ee5b61")
color_map <- setNames(custom_colors, as.character(1:4))
rpart.plot(classTree, box.col = color_map[as.character(classTree$frame$yval)], legend.cex = 0.7)

# Plot CP and prune
plotcp(classTree)
minCP <- classTree$cptable[which.min(classTree$cptable[, "xerror"]), "CP"]
prune_classTree <- prune(classTree, cp = minCP)
rpart.plot(prune_classTree)

# Predictions
predTree1 <- predict(classTree, testSet, type = "class")
mean(predTree1 == testSet$Outcome)
predTree2 <- predict(prune_classTree, testSet, type = "class")
mean(predTree2 == testSet$Outcome)

# -----------------------------
# Milestone 2: Bagging
# -----------------------------
# Bagging with ipred
set.seed(252)
pimaBag <- bagging(Outcome ~ ., data = trainSet, nbagg = 150, coob = TRUE,
                   control = rpart.control(minsplit = 2, cp = 0))
predBag <- predict(pimaBag, testSet, type = "class")
mean(predBag == testSet$Outcome)

# Bagging with caret
set.seed(10)
caretBag <- train(Outcome ~ ., data = trainSet, method = "treebag",
                  trControl = trainControl("cv", number = 10), importance = TRUE)
predCaretBag <- predict(caretBag, testSet)
mean(predCaretBag == testSet$Outcome)

vip(caretBag, geom = "col", aesthetics = list(fill = "#98c8f9"))

# -----------------------------
# Milestone 3: Severity Binary Modeling
# -----------------------------
M3 <- US_Accidents %>%
  mutate(SeverityBinary = ifelse(Severity >= 3, 1, 0)) %>%
  filter(!is.na(SeverityBinary))

set.seed(314)
caretSamp <- createDataPartition(M3$SeverityBinary, p = 0.7, list = FALSE)
trainCaret <- M3[caretSamp, ]
testCaret <- M3[-caretSamp, ]

# Simple logistic regression: population only
modSimp <- glm(SeverityBinary ~ population, data = trainCaret, family = "binomial")
pred1R <- predict(modSimp, newdata = testCaret, type = "response")
table(pred1R > 0.5, testCaret$SeverityBinary)
mean((pred1R > 0.5) == testCaret$SeverityBinary)

# Multi-variable logistic regression
clean_tc <- trainCaret %>%
  select(SeverityBinary, `Humidity(%)`, `Temperature(F)`, `Wind_Speed(mph)`,
         `Distance(mi)`, `Wind_Chill(F)`, population, density)
modMulti <- glm(SeverityBinary ~ ., data = clean_tc, family = "binomial")
summary(modMulti)

# Fit final best model
modBest <- glm(SeverityBinary ~ `Temperature(F)` + `Humidity(%)` + `Wind_Chill(F)` +
                 `Wind_Speed(mph)` + `Distance(mi)` + population + density,
               data = trainCaret, family = "binomial")
predBest <- predict(modBest, newdata = testCaret, type = "response")
table(predBest > 0.5, testCaret$SeverityBinary)
mean((predBest > 0.5) == testCaret$SeverityBinary)

# KNN with SeverityBinary
trainFea <- trainCaret %>% select(-SeverityBinary)
testFea <- testCaret %>% select(-SeverityBinary)
trainOut <- trainCaret$SeverityBinary
testOut <- testCaret$SeverityBinary
knn.pred12 <- knn(train = trainFea, test = testFea, cl = trainOut, k = 12)
table(knn.pred12, testOut)
mean(knn.pred12 == testOut)
