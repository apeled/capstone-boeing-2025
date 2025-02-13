library(dplyr)
library(lubridate)
library(ordinal)

# ============================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================

# Load dataset
df_long <- Rank

# Convert `UpdateDT` to Date-Time format if it exists
if("UpdateDT" %in% names(df_long)) {
  df_long <- df_long %>%
    mutate(UpdateDT = mdy_hm(UpdateDT),
           Year = factor(year(UpdateDT)),  # Ensure Year is a categorical factor
           Month = month(UpdateDT),
           Day = day(UpdateDT),
           Hour = hour(UpdateDT)) %>%
    select(-UpdateDT)  # Drop UpdateDT after extracting features
}

# Ensure SubjectID is a factor to avoid row indexing issues
df_long$SubjectID <- as.factor(df_long$SubjectID)

# Ensure Rank is an Ordered Factor with only observed levels
observed_levels <- unique(df_long$Rank)
df_long$Rank <- factor(df_long$Rank, levels = sort(observed_levels), ordered = TRUE)

# Encode Categorical Driver Variables Properly
driver_cols <- grep("Driver", names(df_long), value = TRUE)
df_long[driver_cols] <- lapply(df_long[driver_cols], function(x) as.integer(gsub("Factor", "", as.character(x))))


# ============================
# STEP 2: CHECK RESPONSE LEVELS
# ============================

# Ensure Rank has at least 2 response levels
if (length(unique(df_long$Rank)) < 2) {
  stop("Error: Rank must have at least two distinct levels for ordinal regression.")
}

# Ensure consistency in response levels
rank_levels <- levels(df_long$Rank)
num_responses <- length(unique(df_long$SubjectID))
if (length(rank_levels) > num_responses) {
  warning("Warning: More Rank levels than responses detected. Adjusting...")
  df_long$Rank <- droplevels(df_long$Rank)  # Drop unused levels
}
str(df_long)
formula <- as.formula(paste("Rank ~ ", paste(driver_cols, collapse = " + ")))

res_clm <- clm(formula, data = df_long, link = "logit")

# Print the model summary
summary(res_clm)
