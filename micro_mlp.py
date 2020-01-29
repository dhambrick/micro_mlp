import numpy as np  
import yaml

def sigmoid(X):
   return 1/(1+np.exp(-X))

def relu(X):
   return np.maximum(0,X)

def mse(A,B):
    return (np.square(A - B)).mean()
def xavier(m,n):
    return (2*np.random.rand(m,n)-1)*np.sqrt(6./(m+n))

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
        layer['weights'] = xavier(n,m)
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
   beta = -1*(training_par['gamma']/training_par['etta'])
   for x,y in  zip(train_data_x,train_data_labels):
       A = ComputeForwardPass(x,net_cfg)
       E = mse(A,y)
       #print('x: ',x)
       #print('_y: ',A)
       #print('y: ',y)
       print('MSE: ',E)
       for layer in net_cfg['Layers']:
            if layer["name"] == 'Input':
                continue
            #print('Layer: ',layer['name'])
            m,n = layer['weights'].shape
            layer['delta_weights'] = np.zeros((m,n))
            for i in range(m):
                for j in range(n):
                    #print('w',layer['weights'][i][j])
                    layer['weights'][i][j] = layer['weights'][i][j] + training_par['etta']
                    #print('_w',layer['weights'][i][j])
                    _y = ComputeForwardPass(x,net_cfg)
                    E_pert = mse(_y,y)
                    #print('E_pert',E_pert)
                    delta_E = E_pert - E 
                    layer['weights'][i][j] = layer['weights'][i][j] - training_par['etta']
                    layer['delta_weights'][i][j] = beta*delta_E
            layer['weights'] = layer['weights'] + layer['delta_weights']
            #print('Perturbed Weights:')
            #print(layer['delta_weights'])
       
       


