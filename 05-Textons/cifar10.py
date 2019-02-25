def unpickle(file):
    import pickle
    import numpy as np
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='latin1')
        _dict['labels'] = np.array(_dict['labels'])
        _dict['data'] = _dict['data'].reshape(_dict['data'].shape[0], 3, 32, 32).transpose(0,2,3,1)

    return _dict

def get_data(data, sliced=1):
    from skimage import color
    import numpy as np
    data_x = data['data']
    data_x = color.rgb2gray(data_x)
    data_x = data_x[:int(data_x.shape[0]*sliced)]
    data_y = data['labels']
    data_y = data_y[:int(data_y.shape[0]*sliced)]
    return data_x, data_y

def merge_dict(dict1, dict2):
    import numpy as np
    if len(dict1.keys())==0: return dict2
    new_dict = {key: (value1, value2) for key, value1, value2 in zip(dict1.keys(), dict1.values(), dict2.values())}
    for key, value in new_dict.items():
        if key=='data':
            new_dict[key] = np.vstack((value[0], value[1]))
        if key=='labels':
            new_dict[key] = np.hstack((value[0], value[1]))            
        elif key=='batch_label':
            new_dict[key] = value[1]
        else:
            new_dict[key] = value[0] + value[1]
    return new_dict

def load_cifar10(meta='cifar-10-batches-py', mode=1):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(i+1))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict

    
