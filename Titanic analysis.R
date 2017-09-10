# Load packages
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)

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

# Add Pclass variable
fol <- formula(Survived ~ Age + Sex + Pclass)
rmodel <- rpart(fol, method="class", data=train)
rpred <- predict(rmodel, newdata=test, type="class")
accuracy <- modelaccuracy(test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy2", accuracy, accuracyLabel))

# Substitute Pclass with Fare variable
fol <- formula(Survived ~ Age + Sex + Fare)
rmodel <- rpart(fol, method="class", data=train)
rpred <- predict(rmodel, newdata=test, type="class")
accuracy <- modelaccuracy(test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy3", accuracy, accuracyLabel))

# Add back Pclass
fol <- formula(Survived ~ Age + Sex + Pclass + Fare)
rmodel <- rpart(fol, method="class", data=train)
rpred <- predict(rmodel, newdata=test, type="class")
accuracy <- modelaccuracy(test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy4", accuracy, accuracyLabel))

# Add SibSp + Parch
fol <- formula(Survived ~ Age + Sex + Pclass + Fare + SibSp + Parch)
rmodel <- rpart(fol, method="class", data=train)
rpred <- predict(rmodel, newdata=test, type="class")
print(rmodel)
accuracy <- modelaccuracy(test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy5", accuracy, accuracyLabel))

# Test if Deck letter helps predict Survivability
# Strip off cabin numbers
# Extract the deck number
# First letter: Deck (e.g. A31 -> A)

# Make sure Deck in both sets has same levels
# If Test has T but not Train different levels causes error in model
train$Deck <- substr(train$Cabin,1,1)
train$Deck[train$Deck==""] = NA
test$Deck <- substr(test$Cabin,1,1)
test$Deck[test$Deck==""] = NA

train$Deck <- as.factor(train$Deck)
test$Deck <- as.factor(test$Deck)

# Make Deck have the same levels
c <- union(levels(train$Deck), levels(test$Deck))
levels(test$Deck) <- c
levels(train$Deck) <- c

# Test if Deck letter improves the prediction
fol <- formula(Survived ~ Age + Sex + Pclass + Fare + SibSp + Parch + Deck)
rmodel <- rpart(fol, method="class", data=train)
rpred <- predict(rmodel, newdata=test, type="class")
accuracy <- modelaccuracy(test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy6", accuracy, accuracyLabel))

# Check if FamilySize is a useful predictor of Survival
fol <- formula(Survived ~ Age + Sex + Pclass + FamilySize)
rmodel <- rpart(fol, method="class", data=train)
rpred <- predict(rmodel, newdata=test, type="class")
print(rmodel)
accuracy <- modelaccuracy(test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy7", accuracy, accuracyLabel)) # best so far

p <- ggplot(aes(x=Pclass, y=factor(FamilySize), color=Survived), data=data) +
  geom_jitter() +
  facet_grid(Sex ~ .)
p + ggtitle("Large Family Size >=5 more likely to not survive") +
  theme_bw() +
  geom_hline(yintercept = 5) +
  ylab("Family Size")

# If a person had 5 family members or more (including themselves) then it is more likely
# the person didn’t survive especially in the third class which had entire families onboard.
mosaicplot(table(FamilySize=data$FamilySize, Survived=data$Survived),
           main="Passenger Survival by Family Size",
           color=c("#fb8072", "#8dd3c7"), cex.axis = 1.2)
# Mosaic plot above clearly shows the drop-off survival at family size of 5 or greater.

# Make explicit factor levels for specific variables: Sex and Pclass
titanic_test$Sex <- as.factor(titanic_test$Sex)
titanic_test$Pclass <- as.factor(titanic_test$Pclass)

# Now train on entire training set
fol <- formula(Survived ~ Age + Sex + Pclass + FamilySize)
model <- rpart(fol, method="class", data=data)
rpart.plot(model, branch=0, branch.type=2, type=1, extra=102, shadow.col="pink", box.col="grey",
           split.col="magenta", main="Decision tree for model")
