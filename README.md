# Translation_w_transformers


## The model's architecture and some notes
input -> embedding 
we are taking the input sentence and converting each word of it into 
an input id (position in the vocabulary vector)
then it gets converted into the embedding vector of size 512
this would be the first layer



after the input embedding layer we get a vector for each word 
but we also need to know the position of that word in the sentence 
for this we are adding to each embedding another vector with the same size, that will encode it the position 
it includes special values ;)
only calculated once and used for every sentence during training and inference


