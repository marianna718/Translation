import torch
import torch.nn as nn

import warnings
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
# this is for training the tokenizer given the list of sentances
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from  pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from tqdm import tqdm


# runing the encoder only once
def greedy_decode(model, source, source_mask, tokeizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
# Precompute the encoder output and reuse it fpr every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
# How do we do the inferanceing 
# we give the decoder the start of the token
# so that the decoder will output the first token of the translated sentance
# then at every itteration we add the previous token to the decoder input 
# so we will get the output of the decoder and use it for the next itteration

    # initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    2,27


# new method for validation
def run_validation(model, validation_ds, tokenier_src, tokenier_Tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
#    the first ting we do to rund the validation we put our model in evaluation mode
    model.eval()
# we will inferance two sentances an dsee
# what is the output of the model
    count = 0

    source_text = []
    espected = []
    predicted = []
    # sie of the control window (just use a default value)
    console_width = 80
    # we are desableing the gradient calculation
    with torch.no_grad():
        # for the validation we have batch size =1
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # lets verify that the size of batch is accrtually 1 
            assert encoder_input.size(0) ==1, "Batch size must be 1 for validation"





def get_all_sentences(ds, lang):
    for item in ds:
        # each item in our raw dataset is a pair of sentances one in english one in spanish
        # the item reprasanting the pair
        yield item['translation'][lang]





# make the method that builds the tokenizer
def get_or_build_tokenizer(config, ds, lang):
    # the path to the tokenizer file

    # config['tokenizer_file'] ='../tokenizers/tokenizer_file_{0}.json' given the lang it will replace 0 with en or esp
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # we are representing a word unk so if the word is
        # unknown in its vocabulary then it will replace it with
        # the UNK

        tokenizer= Tokenizer(WordLevel(unk_token='[UNK]'))

        # spliting by whitespace
        tokenizer.pre_tokenizer = Whitespace()
        # and we build trainer to train our tokenizer

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # sos = start og the sentance eos = end of the sentance
        # pad = padding 
        # min_frequency = for a word to apear in our vocabulary it shoud apear at least 2 

        # now we train our tokenizer
        # by passing the method that gets all the sentances of our dataset
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# now lets create a method for loading the dataset
def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split= 'train')
    # in this way we can dinamically choose the languagees for translation

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
# now we have created the tokens but for them we need to have our bilingual dataset
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'],config['lang_tgt'], config['seq_len'] )
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # we also would like to whatch what is max seq lenght of the source and the target
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentences: {max_len_src}')
    print(f'Max length of target sentences: {max_len_tgt}')


# now we will proceed to createing the data loaders

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # for the validation we will shuffle one , becouse we want to process
    # each sentance one by one
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


# this new method will according to our config vocabulary size build the model
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return model

# if the mode is too big for your gpu you can redusethe number of heads or the number of layers 
# thou it will of course impact your models preformance


# once we have the model we can start building the training loop

def train_model(config):
    # first we need to define on which device we need to put 
    # all the tensors 

    # if you have a cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # we make shure that the weights folder is created
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # load our dataaset

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    # we also creat the model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # We start the tensorboard
    # The Tensorboard allows to visualize the loss, the graffics and the charts
    writer = SummaryWriter(config['experiment_name'])

    # lets also create the optimizer, we will be using Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # since we also have configuration that allows us to resume the 
    # training in case if anything crashes, lets implament it
    # that will allow us to restore the state of the model and the state of the optimizer
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename) #we load the file
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    
    # the last function we will be using is the cross entropy loss
# for its use we want to tell it what is the ignore index, its for ignoreing the paddings
# and we also will be using label smoothing 
# label smoothing allosw our model to be less confident on its choses => less overfit
# what 0.1 label smoothing does is , it takes that much of the probability of 
# the predicted label and distributes that probability among other labels
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    # the final loop

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()

        # create batch iterator for the data loader using tqdm
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)  #(B, Seq_LEn)
            decoder_input = batch['decoder_input'].to(device)  #(B, Seq_LEn)
            encoder_mask = batch['encoder_mask'].to(device)   #(B, 1,1, Seq_Len)  becuse we are hideing paddings only
            decoder_mask = batch['decoder_mask'].to(device)    #(B,1, Seq_len, Seq_len)  becosue we are hideing all the subsequent words


            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  #(Batch, Seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  #(batch, seq_len, d_model)

            # and becouse we want to map it back to vocabulary , we need a projection
            proj_output = model.project(decoder_output)  #(Batch, Seq_len, tgt_vocab_size)

            # now once we have the output of our model
            # we want to compare it with our label
            label = batch['label'].to(device)  #(batch, seq_len)

            # view does this (Batch, seq_len, tgt_vocab_size) -->(Batch*Seq_len, tgt_vocab_size)
            # so we can compeare the projection with the real label
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # now we can update our progress bar with the loss we have calculated
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log loss on tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # now we can back propogate
            loss.backward()

            # and finally update the weights
            # global step is mostlly used by tensorboard to keep track of the loss

            optimizer.step()
            optimizer.zero_grad()

            global_step +=1

        # and we can save the model at the end of every Epoch

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            # its a good ideaa to also save the state of the optimizer
            # becouse it also keeps track of some statistics 
            # thou this information can be too big its still a good idea
            # to not start from 0 every time when training 
            'epoch':  epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step':global_step

        }, model_filename)

# NOW lets build the code to run this

if __name__ == '__main__':
    # lets filter out the warnings
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)


# WHAT WE EXPECT BY RUNING THE TRAINING
# the code shuld download the dataset the first time 
# then it suld create the tokenizer and saveit into its file
# and will strt training model for 30 epochs
