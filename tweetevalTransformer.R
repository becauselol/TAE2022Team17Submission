## Code referenced from https://blogs.rstudio.com/ai/posts/2020-07-30-state-of-the-art-nlp-models-from-r/

install.packages(
  c(
    "dplyr",
    "reticulate",
    "tensorflow",
    "keras",
    "tfdatasets",
    "transformers",
    "plyr",
    "data.table"
  )
)

rm(list=ls())
library(reticulate)
py_install('tensorflow', pip=T)
py_install('transformers', pip=T)

library(dplyr)
library(tfdatasets)
library(tensorflow)
library(keras)
library(data.table)
library(plyr)

transformer <- reticulate::import('transformers')

# GPU can be added if available
physical_devices <- tf$config$list_physical_devices("GPU")
tf$config$experimental$set_memory_growth(physical_devices[[1]],TRUE)
# End of GPU optimisation

tf$keras$backend$set_floatx('float32')

df <- read.csv("train.csv")
df <- df %>% mutate(sentiment = sentiment - 1) %>%
  data.table::as.data.table()
idx_train <- sample.int(nrow(df)*0.8)
train <- df[idx_train,]
test <- df[!idx_train,]

max_len <- 50L
epochs <- 1L
batch_size <- 8

base_model <- "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer <- transformer$RobertaTokenizer$from_pretrained(base_model)
# inputs
text <- list()
# outputs
label <- list()
data_prep <- function(data) {
  for (i in 1:nrow(data)) {
    txt <- tokenizer$encode(data[['tweet']][i],max_length = max_len,
                            truncation=T, padding = 'max_length') %>%
      t() %>%
      as.matrix() %>% list()
    lbl <- data[['sentiment']][i] %>% t()
    text <- text %>% append(txt)
    label <- label %>% append(lbl)
  }
  list(do.call(plyr::rbind.fill.matrix,text), do.call(plyr::rbind.fill.matrix,label))
}
train_ <- data_prep(train)
test_ <- data_prep(test)

# slice dataset
tf_train <- tensor_slices_dataset(list(train_[[1]],train_[[2]])) %>%
  dataset_batch(batch_size = batch_size, drop_remainder = TRUE) %>%
  dataset_shuffle(128) %>% dataset_repeat(epochs)
tf_test <- tensor_slices_dataset(list(test_[[1]],test_[[2]])) %>%
  dataset_batch(batch_size = batch_size)

# get pretrained model
model_ <- transformer$TFRobertaModel$from_pretrained(base_model)

# create an input layer
input <- layer_input(shape=c(max_len), dtype='int32')
hidden_mean <- tf$reduce_mean(model_(input)[[1]], axis=1L) %>%
  layer_dense(128,activation = 'relu') %>% layer_dropout(rate = 0.1) %>%  layer_dense(64,activation = 'relu') %>% layer_dropout(rate = 0.4)

# create an output layer for multi-class classification
output <- hidden_mean %>% layer_dense(units = 3, activation='softmax')
model <- keras_model(inputs=input, outputs = output)
# compile with Adam, Categorical Cross-entropy Loss, and Accuracy metric
model <- model %>% compile(optimizer= tf$keras$optimizers$Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1.0),
                           loss = tf$losses$SparseCategoricalCrossentropy(from_logits=F),
                           metrics = 'accuracy')

resultingModel <- model$fit(tf_train, validation_data=tf_test, epochs=epochs)


# save model
resultingModel$model$save("tweetevalModel")


predict <- read.csv("test.csv")
predict <- predict %>%
  data.table::as.data.table()


data_prep <- function(data) {
  for (i in 1:nrow(data)) {
    txt <- tokenizer$encode(data[['tweet']][i],max_length = max_len,
                            truncation=T, padding = 'max_length') %>%
      t() %>%
      as.matrix() %>% list()
    text <- text %>% append(txt)
  }
  list(do.call(plyr::rbind.fill.matrix,text))
}
testing_ <- data_prep(predict)


pred <- resultingModel$model$predict(testing_[[1]])

pred_df <- as.data.frame(pred)

pred_ls <- apply(pred_df,1,function(x) which(x==max(x)))

table(pred_ls)

final_pred_df <- as.data.frame(pred_ls)

final_pred_df$id <- predict$id

str(final_pred_df)
final_pred_df$sentiment <-  final_pred_df$pred_ls
final_pred_df <- final_pred_df %>% select(id, sentiment)
write.csv(final_pred_df, "tweetevalModelPred.csv", row.names = FALSE) # DEPENDING ON RSTUDIO WORKING DIRECTORY
