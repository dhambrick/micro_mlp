import numpy as np  
import yaml

def sigmoid(X):
   return 1/(1+np.exp(-X))

def relu(X):
   return np.maximum(0,X)

fct_map = {"sigmoid":sigmoid , "relu":relu}

def LoadANNConfigFile(config_path):
    stream = open(config_path,"r")
    ann_config = yaml.load(stream)
    stream.close() 
    return ann_config

def InitializeMLP(cfg):
    for layer in cfg['Layers']:
        print(layer['name'])
        m = layer['InputDim']
        n = layer['OutputDim']
        layer['weights'] = np.random.rand(n,m)
    return cfg

cfg = LoadANNConfigFile("sine_mlp.yml")
cfg = InitializeMLP(cfg)
for layer in cfg['Layers']:
        print(layer['weights'])