import torch
import torch.nn as nn
import math




################### IMPUT EMBEDING  ############################################
"""
input -> embading 
we are taking the input sentance and converting each word of it into 
a input id (position in the vacabulary vector)
then it gets converted into the embeding vector of size 512
this would be the first layer
"""

class InputEmbeddings(nn.Module):
# the constructor should rescive the dimention of the embeding space
# vocab_size is how many words there are in vocabulary
    def __init__(self, d_model:int, vocab_size:int ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
    # and now we will create the actual embedding layer 
    # there is already available version in pytorch so
        self.embedding = nn.Embedding(vocab_size, d_model)  #just a basic maping beetween words and numbers 

    # now we acctually do the maping by forwarding throug the emmbeding layer

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)   #the multiplication with the sqrt(dmodel) is specified in the paper
    
    
################### POSITIONAL ENCODING ###############################

"""
after the input embeding layer we are getting a vector for each word 
but we also need to know the position of that word in the sentance 
for this we are adding to each embading another vector with same size, that will encode in it the position 
it includes spetial values ;)
only calculated once and used for every sentance during training and inferance 

"""

# seq_length is the max lenght of the sentance (becouse we need to create one vector for each position)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

# we will build a matrix from seq_len to d_model,we need vectors of d_model size ,but we need seq_len of them 

        #create a matrix of size(d_model, seq_len)
        pe = torch.zeros(seq_len, d_model)
        # we create this based on the two formulas, PE(pos,2i+1) = cos{pos/10000^(2i/d_model)}
                                                #   PE(pos,21) = sin{pos/10000^(2i/d_model)}
        # so we will have a vector for each possible position(even and odd position)

        # we will use a simplified version of the calculation more numerically stable
    

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  #we have created tensor of shape (seq_len, 1)   #kotoraki verevi tiv
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))          #kotoraki taki tivy

# now we apply to sin and cos to this two numbers 

        pe[:, 0::2] = torch.sin(position*div_term) #for 2i
        pe[:, 1::2] = torch.cos(position*div_term) #for 2i + 1

        # add batch dimention so we can apply this to full sentances
        # now we have seq_len, 1 , but we are going to have batch of semtances 

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
# unsqueezeing on the first dimention 

        # finally we can register this tensor to the buffer of the model
        # buffer is when want to save tensor that we want to save not as a parameter but saved with the file of the model when it is saved
        # tensor is saved in a file when we save the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  #requires_grad_(False) will stop model from learning this, becouse its fixed
        return self.dropout(x)
    


#################### E N C O D I N G    L A Y E R ###########################################################


###########  ADD & NORM  ###################  NORMALIZATION LAYER   #########################################


# layer normalization means that if you have a batch of 3 items and each item have some features ,like 3 sentances made from many words
# we calculate for each item mean and varrience, independatlly from other items 
# then we calculate new values for each item based on their mean and varience 
# in this layer are also used parameters called alpha and beta, one is multiplicative (multiplied by each x)
# and the other is additive (added to each x)
# so the model will have the possibility to amplify when it needs to be amplified (modle will learn that)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 **-6):   #eps is a very small number 
        super().__init__()
        self.eps = eps   # becouse when sigma is close to 0 x normelizet will becoome too big

        self.alpha = nn.Parameter(torch.ones(1))   #Multiplied 
        self.bias = nn.Parameter(torch.zeros(1))   #Added


    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)  #keepdim is true becouse mean usually cancels the dimention to wich applied 
        std = x.std(dim = -1,keepdim =True)
        return self.alpha * (x-mean)/ (std + self.eps) + self.bias  #just the formula
    


############ FEED FORWARD laYER ###########################################################################


# fully conacted layer that the model uses for both the encoder and the decoder 
# its two matrixes W1 and W2 multipled by x , one after another with the relu in beetween and with the biase
# we will use linear layer of pytorch for this, 
# dimentions of this matrixes are W1 -> (d_model, d_ff), W2 -> (d_ff , d_model)
# d_ff is 2048

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff: int, dropout:float):
        super().__init__()
        self.linear_l = nn.Linear(d_model, d_ff) #the first matrix W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # the second matrix  W2 and B2
        # why b1 and b2, becouse the bies is true so its predifining bies matrix for us

    def forward(self, x):
        # we have an input sentance shich is a (batch, seq_len, d_model),
        # first we will convert it useing linear 1  into another tensor of (batch , seq_len, d_ff)
        # and then we will apply the linear tool which will converted back to the model

        return self.linear_2(self.dropout(torch.relu(self.linear_l(x))))
    


##################  MULTIHEAD ATTENTION  ################################################


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h: int, dropout:float)-> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # the d_model should be devisable by the h
        # so we can devide the embeding of the sentance into multiple h heads

        assert d_model % h ==0, "d_model is not divisiable by h"

        # as we can seein the paper d_mmoel/ h = d_k its the dimension of each head vector

        self.d_k = d_model // h
         
        #  lets also define the matixes for K Q V and also the W output matrix 
        self.w_q = nn.Linear(d_model,d_model) #Wq
        self.w_k = nn.Linear(d_model,d_model) #Wk
        self.w_v = nn.Linear(d_model,d_model) #Wv

        self.w_o = nn.Linear(d_model,d_model) #WO

        self.dropout = nn.Dropout(dropout)
        

    
    # attention caculation formula
    @staticmethod
    # makeing attentio() as a static method just lets to use the function 
    # without haveing an instance of this class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
# @ is the matrix multiplication sign in pytorch
        #(Batch, h, seq_len, d_k) --> (Batch, h, seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        # we have transposed the last two dimentions of the key matrix
        # NOW BEFORE APPLIMG THE SOFTMAX we will apply the mask
        # we will do that by replaceing the values of the attention scores with very small numbers
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            # for example this can be used with the padding values wich are used to fill in 
            # the seq length
        attention_scores =  attention_scores.softmax(dim = -1) # (Batch, h, seq_len, Seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

            # we will return the result as well as the attention scores 
            # latter is to be used in visualization
            
        return (attention_scores @ value), attention_scores




        # lets implement the forward method
    def forward(self, q, k, v, mask):
        # what is the mask for 
        # if we wont some words to not interact with other words we MASK them
        # Becouse Attention(Q,K,V)=  softmas{(QK^t)/sqrt(d_k)}V 
        # befor multipliing by V the result matrix QK^t will have the 
        # connection of the words with each other
        # that matrix contains the attention score for words with each other
        # so if we replace that score with a very small number 
        # then the result will will be 0 after appling the softmax
        # becouse in the softmax there is e^x and if x-> - inf => e^x->0

        query = self.w_q(q) # (Batch, Seq_len, d_mpdel) --> (Batch, Seq_len, d_omdel)
        key = self.w_k(k) # (Batch, Seq_len, d_mpdel) --> (Batch, Seq_len, d_omdel)
        value = self.w_v(v) # (Batch, Seq_len, d_mpdel) --> (Batch, Seq_len, d_omdel)

        # now we want to devide by h to smaller matrixes
        # we will devide by using the view method of pytorch , and keep the batch size
        # becouse we dont want to devide the sentancewe want to devide the embeddings
        # and we want to keep the second dim becouse its the Seq 
       
    #    (Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        # we trasposed becouse we want h dimention to be 2nd instead of third
        
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # we have the splited heads, now we need to calculate the attention by using the formula

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key,value, mask, self.dropout)
    # now we need to concat the small head matrixis and multiply by Wo to get the 
    # result og MultiHead
 
        # we transpose back 
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # the contiguous() was used to be able to concat in place

    # and now we multiply by Wo
    # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)


# now we will build a layer that handles the skip conection structure

class ResidualConnection(nn.Module):

    def __init__(self,  dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # the sublayer is the previous layer 
        return x + self.dropout(sublayer(self.norm(x)))
    


#  NOW WE WILL CREATE THE ENCODER BLOCK
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
# we define two residual conections 
# we use modulelist which is the way to orgaize list of modules (2 of them in this case)
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # src_mask is for the padding words
        
        # at first we send the input to multih att and then to norm but 
        # also we send it to the add and noerm separatlly too
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))

        # now the second one
        x = self.residual_connection[1](x, self.feed_forward_block)

        return x
    

# we can have N encoder heads so lets define it

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    



# ###################  DECODER ###################33


# for the decoder the output embeddings which are the inputs of it 
# are in the same format as for the encoder 
# and same applies to the Positional encodings of those so we will be using the same

# and becouse we already have the building blocks such as
# multihead attention layer, skip connections for the decoder
# we will be just building the Decoder block and then the Decodre itself
# with the containment of N decoder blocks


# fro the first attention layer of the decoder we will be using
# a basic self attention 
# while for the second multihead attention layer we will be using 
#  different inputs such as 
# Value coming from the preivous layer of the decoder
# and QUERY and KEY coming from the Encoder



class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # why do we have two kind of masks
        # becouse in this case we are making a translater and raughlly said
        # one mask the src one is for the english 
        # and the tgt one is for italian, so one mask is for the encoder 
        # and the oth rone is for the decoder(


# we calculateing the self attention first
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output,encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
    


# now we will define the Decoder, which is basically N times the decoder block

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
    

# now we will create the last linear layer that will project the result of decoder
# into the vocabulary
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, vocab_size)

        return torch.log_softmax(self.proj(x), dim = -1)
    


######################  tRANSFORMER ###############################333



class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    

    def decode(self,encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return tgt
    
    def project(self, x):
        return self.projection_layer(x)
    

# this function is for building a translater with use of the transformer
# it can be rewriten for any other task


# we pass the vocabulary size becosue we need to know how big is the vector
# that needs to be converted into a d_model sized embedding vector

# choose num of heads 8

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model:int =512, N:int = 6, h:int= 2, dropout: float = 0.1, dff:int = 2048) -> Transformer:
    # first we will create the embadding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)


    # Create the decoder blocks 

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)


    # Create the encoder and the decoder

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))


    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)


    # and then we build the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos,projection_layer)



    # INITIALIZEING THE PARAMETERS WITH FOR EXAMPLE XAVIER UNIFORM
    #  SO THE TRAINING WONT START WITH 0 PARAMETERS

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    
    return transformer

