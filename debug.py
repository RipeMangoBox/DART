import pickle
import glob
import torch

dir_path = 'mld_denoiser/mlp_mnorm/mld_babel_smplx/samples/200000/15'

def compare_data(data1, data2):
    if set(data1.keys()) != set(data2.keys()):
        print("Keys differ!")
        return False

    for k in data1:
        v1, v2 = data1[k], data2[k]
        
        if isinstance(v1, torch.Tensor) or isinstance(v2, torch.Tensor):
            if not (isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor)):
                return False
            if not torch.equal(v1, v2):
                print(f"Tensor mismatch at '{k}'")
                return False
        elif isinstance(v1, dict) and isinstance(v2, dict):
            if not compare_data(v1, v2):
                return False
        else:
            if v1 != v2:
                print(f"Value mismatch at '{k}': {v1} != {v2}")
                return False
    return True     


all_data = []
for file_path in glob.glob(f"{dir_path}/*.pkl"):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
        all_data.append(dataset)
            
compare_data(all_data[0], all_data[1])
print(all_data)