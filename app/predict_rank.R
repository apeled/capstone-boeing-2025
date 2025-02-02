# Load required libraries
library(tidyverse)
library(caret)
library(jsonlite)

# Load the trained model and factor levels
load("../model/rank_model.RData")

# Read new input data
new_data <- read.csv("new_input.csv")

# Apply factor levels from training data
categorical_columns <- paste0("Driver", 1:17)
for (col in categorical_columns) {
  new_data[[col]] <- factor(new_data[[col]], levels = factor_levels[[col]])
}

# Predict rank
predicted_rank <- predict(rank_model, newdata = new_data)

# Output result as JSON
output <- list(predicted_rank = predicted_rank)
cat(toJSON(output, auto_unbox = TRUE))
