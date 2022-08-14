library(tidyverse)
library(caret)
library(tm)
library(SnowballC)
library(ranger)
library(qdap)
library(textclean)

rm(list=ls())

fileName <- "train.csv"
target <- "tweet"
label <- "sentiment"

fileNameTrain <- "train.csv"
fileNameTest <- "test.csv"


dfTrain <- read.csv(fileNameTrain, stringsAsFactors = FALSE)
dfTest <- read.csv(fileNameTest, stringsAsFactors = FALSE)
dfTrain$id <- -1
dfTest$sentiment <- 0
print(nrow(dfTrain))
df <- rbind(dfTrain, dfTest)
corpus <- Corpus(VectorSource(df[, target]))

corpus <- tm_map(corpus, add_comma_space)

corpus <- tm_map(corpus, replace_url)

corpus <- tm_map(corpus, replace_tag)

corpus <- tm_map(corpus, replace_hash)

corpus <- tm_map(corpus, replace_html)

corpus <- tm_map(corpus, replace_white)

corpus <- tm_map(corpus, replace_emoticon)

corpus <- tm_map(corpus, replace_date)

corpus <- tm_map(corpus, replace_number)

corpus <- tm_map(corpus, replace_contraction)

corpus <- tm_map(corpus, replace_abbreviation)

corpus <- tm_map(corpus, replace_non_ascii)

# To lowercase
# corpus <- tm_map(corpus, function(x) iconv(enc2utf8(x), sub = "byte"))
# corpus <- tm_map(corpus, content_transformer(function(x)    iconv(enc2utf8(x), sub = "bytes")))
# corpus <- tm_map(corpus, content_transformer(tolower))


# To lowercase
corpus <- tm_map(corpus, function(x) iconv(enc2utf8(x), sub = "byte"))
corpus <- tm_map(corpus, content_transformer(function(x)    iconv(enc2utf8(x), sub = "bytes")))
corpus <- tm_map(corpus, content_transformer(tolower))
# Remove Stopwords
corpus <- tm_map(corpus,removeWords,stopwords("english"))

# Remove Punctuation
corpus <- tm_map(corpus,removePunctuation)

corpus <- tm_map(corpus, removeNumbers)

# Stemming
corpus <- tm_map(corpus,stemDocument)
print(as.character(corpus[[1]]))

# Convert to DTM
dtm <- DocumentTermMatrix(corpus)

# Remove Sparse Terms
dtmSparse <- removeSparseTerms(dtm, sparsity)

dtmSparse
# Convert to Data Frame
dtmSparseDF <- as.data.frame(as.matrix(dtmSparse))

# Attach ID and sentiment
dtmSparseDF <- cbind(df$id, dtmSparseDF, df$sentiment)
# Set Column Names
colnames(dtmSparseDF) <- make.names(colnames(dtmSparseDF))




## For Submission
set.seed(123)
str(dtmSparseDF$df.sentiment)
train.data <- subset(dtmSparseDF, df.id == -1)
train.data$df.sentiment <- as.factor(train.data$df.sentiment)
str(train.data$df.sentiment)
train.data <- select(train.data, -df.id)
test.data <- subset(dtmSparseDF, df.id != -1)
test.data <- select(test.data, -df.id, -df.sentiment)

# Fit the model
model500 <- ranger(df.sentiment~., data=train.data, num.trees=500, mtry=ceiling(sqrt(ncol(train.data)-1)))

# Summarize the model
# summary(model)
# Make predictions
predicted.classes <- predict(model500, data=test.data)
head(predicted.classes$predictions)
dfTest$predicted <- predicted.classes$predictions

table(predicted.classes$predictions)

submission <- dfTest
submission$sentiment <- dfTest$predicted
submission$id <- dfTest$id
str(submission)
submission <- select(submission, -tweet, -predicted)
str(submission)
submission$sentiment <- as.integer(submission$sentiment)
str(submission)
write.csv(submission, "submission3RandomForestClean.csv", row.names=F)

