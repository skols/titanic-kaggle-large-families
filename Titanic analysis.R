# Load packages
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)

# Load data and setup factors
loaddata <- function(file) {
  data <- read.csv(file, header=TRUE, stringsAsFactors=FALSE)
  # Compute family size (including self)
  data$FamilySize <- data$SibSp + data$Parch + 1
  
  data
}

# Load training data
data <- loaddata("./input/train.csv")

# Load test data
titanic_test <- loaddata("./input/test.csv")

# Change Survived from integer to boolean
data$Survived <- as.logical(data$Survived)
levels(data$Survived) <- c("Not survived", "Survived")

# Make explicit factor levels for specific variables: 3=Pclass, 5=Sex, 12=Embarked
for (i in c(3, 5, 12)) {
  data[, i] <- as.factor(data[, i])
}
glimpse(data)

# Plot Age, Pclass, and Sex as survival factors
ggplot(data, aes(x=Age, y=Pclass, color=Survived)) +
  geom_jitter(position = position_jitter(height = .1)) +
  scale_color_manual(values=c("red", "blue")) +
  facet_grid(Sex ~ .) +
  ggtitle("Age, Sex, and Class as Survival Factors") +
  ylab("Pclass")

# Created adjusted family size variable for people sharing cabins but not registered
# as family members
# e.g. if two people are assigned to cabin A13 and familysize == 1, then bump up familysize to 2
# combine set of cabins from both test and training data
cabins <- data$Cabin # 891 rows
# cabins <- append(cabins, titanic_test$Cabin) # 1309 rows
n_occur <- data.frame(table(Var1=cabins)) # 148 rows
# Remove missing cabin and/or just use the cabin letter code
n_occur <- subset(n_occur, nchar(as.character(Var1)) > 1) # 145 rows

sharedCabins <- n_occur$Var1[n_occur$Freq > 1]
data$FamilySizeAdj <- data$FamilySize
print(table(data$FamilySize))

sharedInd <- data$FamilySizeAdj == 1 & data$Cabin %in% sharedCabins
data$FamilySizeAdj[sharedInd] <- 2
rowCount <- sum(sharedInd)
print(c("Adjusted rows", rowCount))

print(table(data$FamilySizeAdj))

# Break up training set into subset training and test set
set.seed(820)
inTrainingSet <- createDataPartition(data$Survived, p=0.5, list=FALSE)
train <- data[inTrainingSet,]
test <- data[-inTrainingSet,]

# Does adding more variables improve predictions?
modelaccuracy <- function(test, rpred) {
  result_1 <- test$Survived == rpred
  sum(result_1) / length(rpred)
}

checkaccuracy <- function(accuracy) {
  if (accuracy > bestaccuracy) {
    bestaccuracy <- accuracy
    assign("bestaccuracy", accuracy, envir = .GlobalEnv)
    label <- "better"
  } else if (accuracy < bestaccuracy) {
    label <- "worse"
  } else {
    label <- "no change"
  }
  label
}

# Starting with Age and Sex as indicators
fol <- formula(Survived ~ Age + Sex)
rmodel <- rpart(fol, method="class", data=train)
rpred <- predict(rmodel, newdata=test, type="class")
accuracy <- modelaccuracy(test, rpred)
bestaccuracy <- accuracy # initial base accuracy
print(c("accuracy1", accuracy))
