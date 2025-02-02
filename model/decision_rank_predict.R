library(caret)
library(nnet)
library(jsonlite)

# Load the dataset
df <- read.csv("..data/Rank.csv")

# Convert categorical variables to factors
df[, 3:19] <- lapply(df[, 3:19], as.factor)

# One-hot encode categorical variables
dummy_vars <- dummyVars("Rank ~ .", data = df[, -c(1, 20)], fullRank = TRUE)
df_transformed <- as.data.frame(predict(dummy_vars, df[, -c(1, 20)]))

df_transformed$Rank <- df$Rank

# Fit a simple linear regression model
model <- lm(Rank ~ ., data = df_transformed)

# Save the model
dump(c("model"), file = "rank_model.RData")

# Function to make predictions
decision_rank_predict <- function(new_data) {
  load("rank_model.RData")
  
  # Convert input to dataframe
  new_data <- as.data.frame(new_data)
  
  # Ensure the same transformations
  new_data <- as.data.frame(predict(dummy_vars, new_data))
  
  # Predict rank
  prediction <- predict(model, new_data)
  
  return(toJSON(list(predicted_rank = prediction)))
}
