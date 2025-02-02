# Load required libraries
library(tidyverse)
library(caret)

# Load dataset
data <- read.csv("../data/Rank.csv")

# Drop unnecessary columns (SubjectID, UpdateDT)
data <- data %>% select(-SubjectID, -UpdateDT)

# Convert categorical variables to factors
categorical_columns <- paste0("Driver", 1:17)
data[categorical_columns] <- lapply(data[categorical_columns], factor)

# Split into training and test sets
set.seed(123)
trainIndex <- createDataPartition(data$Rank, p = 0.8, list = FALSE)
training_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Train a simple regression model
rank_model <- train(Rank ~ ., data = training_data, method = "lm")

# Save the model and factor levels
factor_levels <- lapply(training_data[categorical_columns], levels)
save(rank_model, factor_levels, file = "../model/rank_model.RData", version = 2)
