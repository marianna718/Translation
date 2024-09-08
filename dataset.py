import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.seq_len = seq_len

        # we also need the special tokens such as the start, end of sentance
        # there is a method of tokenier that we can use for this

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype = torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype = torch.int64)



    # defineing the lenght method of this dataset
    # which defines the lenght of the dataset itself

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:

        # we extract the original pair from hugging face dataset
        src_target_pair = self.ds[index]

        # then we extract the source and target text
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # we convert each text into tokens and then into input ids

        # this means that tokenizer will split each sentance into words
        # and then will map each each word in its coresponding number in vocabulary

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # we also need to PAD our sentances to reach the seq lenght
        # we will be jsut filling the sentance with the PAD token untill
        # it reaches the seq_len

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 #-2 becouse we will add SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 #-1 when we do training we only add the SOS

        # we wanna make sure that the seq_len will be able to handle the len of the all sentance
        # so basically it would never become negative 

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # building two tensors for the encoder input and for the decoder input
        # and also one for the label , so 1 sentance will be sent to the 
        # input of the encoder, one sentence will be send as an input to the decoder
        # and one sentance is the output of the decoder
        # they are concationation of sentance parts
    


    #  this is add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]

        )

    # Add SOS to the decoder input
        decoder_input = torch.cat(

            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

    # Add EOS to the label (what we wxpect as output from the decoder)
        label = torch.cat(

            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)

            ]
        )

# for debagging lets check if we have reached seq lenght
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len


        return {
            "encoder_input": encoder_input, # {Seq_len}
            "decoder_input": decoder_input, # (Seq_len)
            # encoder mask , its used for the added PADs to 
            # them invisable for the self attention mechanism
            # we build it like "saying" all the tokens that are not PADs are ok
            # and all the tokens that are PADs are not ok
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  #(1, 1, Seq_len)
            # first unsqueez is for adding the seq dimmension
            # second unsqueeze is for adding the batch dimmension


            # we also need a causal mask
            # useage of which is to make sure that each word can only look at the 
            # previous word and make sure that thode are non PAD words

            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), #(1, Seq_len) & (1, Seq_len, Seq_len) 
            "label": label ,# (Seq_Len)
            "src_text": src_text,
            "tgt_text": tgt_text
             
        }
    


    # we wont to get masked out the connection on the dioganal and above it of the attention mechanism matrix
def casual_mask(size):
    # the torch.triu() function returns values above the diogonal
    # here we use a matrix made out of all ones
    mask = torch.triu(torch.ones(1, size, size),diagonal=1).type(torch.int)
    return mask == 0 
    # we returned the mask equal 0 so we can reverse
    # instead of getting everything that is above diaganal we mark them as false