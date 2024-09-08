# here we will have two methods 
# one is get congif and the other is for getting the path
# where we will save the waights of the model

from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,   #its common to give a very high learning rate at the begining
        # and then to reduce it gradually with every epoch
        "seq_Len": 350,
        "d_model": 512,
        "lang_src":"en",
        "lang_tgt": "es", #espaniol, spanish
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None, #this can be used for restarting the training in case of
        # a crash
        "tokenizer_file": "tokenizer_{0}.json" ,#this is the tokenizer file
        # it will be used with en and es according to the language


        # here we will be saving the loses during the training
        "experiment_name": "runs/tmodel"

    }


def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)