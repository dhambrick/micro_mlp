import numpy as np  
import yaml

def sigmoid(X):
   return 1/(1+np.exp(-X))

def relu(X):
   return np.maximum(0,X)

def mse(A,B):
    return (np.square(A - B)).mean()


fct_map = {"sigmoid":sigmoid , "relu":relu}

def LoadANNConfigFile(config_path):
    stream = open(config_path,"r")
    ann_config = yaml.load(stream)
    stream.close() 
    return ann_config

def InitializeMLP(cfg):
    for layer in cfg['Layers']:
        m = layer['InputDim']
        n = layer['OutputDim']
        layer['weights'] = np.random.rand(n,m)
    return cfg

def ComputeLayerResponse(layer_input,layer):
    print('Layer name', layer['name'])
    print('Input: ',layer_input.shape)
    print('Weights', layer['weights'].shape)
    layer_output = np.matmul(layer['weights'],layer_input)
    layer_output = fct_map[layer["ActivationFunction"]](layer_output)
    return layer_output

def ComputeForwardPass(net_input,net_cfg):
    x0 = net_input
    for layer in net_cfg['Layers']:
        if layer["name"] == 'Input':
            x = x0
        else:
            x = ComputeLayerResponse(x,layer)
    return x        
def WeightPerturbationTrain(train_data_x,
                            train_data_labels,
                            net_cfg,
                            training_par={'etta':.00001,'gamma':.3}):
   for x,y in  zip(train_data_x,train_data_labels):
       A = ComputeForwardPass(np.asarray([x]),net_cfg)
       E = mse(A,y)
       print('x: ',x)
       print('_y: ',A)
       print('y: ',y)
       print('MSE: ',E)
       for layer in net_cfg['Layers']:
            if layer["name"] == 'Input':
                continue
            print('Layer: ',layer['name'])
            for w in np.nditer(layer['weights']):
                print('w',w)
                _w = w + training_par['etta']
                E_pert = ComputeForwardPass()
       
       
       
       
       


cfg = LoadANNConfigFile("sine_mlp.yml")
cfg = InitializeMLP(cfg)
t = np.linspace(0,np.pi/2)
y = np.sin(t)
WeightPerturbationTrain(t,y,cfg)