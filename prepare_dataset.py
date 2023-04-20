import os
import h5py
import ujson as json
import numpy as np
from tqdm import tqdm


def numpy_to_python(cand_item):
    if not isinstance(cand_item,(np.bytes_)):
        return cand_item.item()
    else:
        return cand_item.decode('iso-8859-1').encode('utf-8').decode('utf-8')

def extract_fashiongen_json(data_root, split='train'):
    """
    regenerate info file for train
    """
    outfilename = os.path.join(data_root, split, 'info_data.json')
    infilename = os.path.join(data_root, 'fashiongen_256_256_{}.h5'.format(split))
    infile = h5py.File(infilename)
    keys = list(infile.keys())
    keys.remove('index_2')
    keys.remove('input_image')
    out_data = []
    with open(outfilename, mode='w', encoding='utf-8') as outfile:
        index_2_all = infile['index_2']
        for i in tqdm(range(len(index_2_all))):
            attrbute_dict = {key:numpy_to_python(infile[key][i][0]) for key in keys}
            attrbute_dict['index_2'] = numpy_to_python(infile['index_2'][i])
            out_data.append(attrbute_dict)
        json.dump(out_data, outfile, ensure_ascii=False,indent=2)


def extract_productid(data_root, split='train'):
    """
    extract the product id info from h5 file for next train step
    """
    infile = h5py.File(os.path.join(data_root, 'fashiongen_256_256_{}.h5'.format(split)), mode='r')
    prodcutid = -100
    pids = infile['input_productID']
    pids_list = []
    for i,pid in enumerate(tqdm(pids)):
        if pid[0] != prodcutid:
            pids_list.append([int(pid[0]),[i]])
            prodcutid = pid[0]
        else:
            pids_list[-1][1].append(i)
        
    
    json.dump(pids_list, open(os.path.join(data_root, split, 'productid_list.json'), mode='w', encoding='utf-8'), ensure_ascii=False)



if __name__ == '__main__':
    data_root = ''  #fill the new dir path there
    split = 'train' # or change to 'validation'
    # extract_productid(data_root, split)
    # extract_fashiongen_json(data_root, split)
    pass
    