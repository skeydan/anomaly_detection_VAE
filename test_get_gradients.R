library(keras)
K <- keras::backend()
library(tensorflow)

model <- keras_model_sequential() %>%
  layer_dense(units = 4, input_shape = 2) %>% 
  layer_dense(units = 1, activation='sigmoid')
model %>% compile(loss='binary_crossentropy', optimizer='adam')
model %>% summary()

outputTensor <- model$output 
outputTensor
listOfVariableTensors <- model$trainable_weights
listOfVariableTensors
gradients <- K$gradients(outputTensor, listOfVariableTensors)
gradients

trainingExample <- rnorm(2) %>% matrix(nrow=1, ncol=2)
sess <- tf$InteractiveSession()
sess$run(tf$global_variables_initializer())

input <- model$input
input
evaluated_gradients <- sess$run(gradients,feed_dict = dict(input=trainingExample))
evaluated_gradients

