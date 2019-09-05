import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import logging
from logging import debug as dbg

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - \n%(message)s')
#logging.disable(logging.CRITICAL)

def read_file_list(folder, var_name, comp_var):
    # open file and read the content in a list
    filename = folder+'/'+var_name+'.txt'
    var_list = []
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]

            # add item to the list
            var_list.append(currentPlace)

    #comp_filename = '../Reda-BG/MHC-I/generation-1/competition/MHCflurry/testing_no-MS/'+folder+'/'+comp_var+'.txt'
    comp_filename = '../Reda-BG/MHC-I/generation-1/competition/MHCflurry/HPV/'+folder+'/'+comp_var+'.txt'
    with open(comp_filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            break
    comp_val = currentPlace

    return var_list, comp_val

def make_boxplot(var_name, my_data, comp_data):
    print('var_name')
    print(var_name)

    print('my_data')
    print(my_data)
    
    print('comp_data')
    print(comp_data)


    # Create a figure instance
    fig = plt.figure(figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    bp = ax.boxplot(my_data, showfliers=False)
    #bp = ax.boxplot(auc_list_list)

    for (i, dset) in enumerate(my_data):
        y = dset
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i+1, 0.04, size=len(y))
        #plt.plot(x, y, 'ko', alpha=0.2)
        plt.plot(x, y, 'ko', mfc='none', alpha=0.3)

    for (i, dset) in enumerate(comp_data):
        y = dset
        # Add some random "jitter" to the x-axis
        #x = np.random.normal(i+1, 0.04, size=len(y))
        plt.plot(i+1, y, 'rD', mfc='none')

    ax.set_ylabel(var_name)
    ax.set_xticklabels(folders)
    ax.xaxis.grid(True)
    plt.xticks(rotation=45)

    #fig.savefig(var_name+'.png', bbox_inches='tight')

# define an empty list
auc_tmp = []
ppvn_tmp = []

auc_comp_list = []
ppvn_comp_list = []

folders = []

for (root,dirs,files) in os.walk('../Reda-BG/MHC-I/generation-1/competition/MHCflurry/HPV/'):
    dbg(root) 
    dbg(dirs) 
    dbg(files) 
    dbg('--------------------------------')
    break

for item in dirs:
    dbg('folder name')
    dbg(item)
    tmp_folder = './' + item + '/'
    dbg('tmp_folder')
    dbg(tmp_folder)
    for (subroot,subdirs,subfiles) in os.walk(tmp_folder):
        dbg('subfiles')
        dbg(subfiles)
        if ('auc.txt' in subfiles) and ('ppvn.txt' in subfiles):
            print("got a hit!! %s" % tmp_folder)
            folders.append(item)
        break
    #break

dbg('folders')
dbg(folders)

dbg('dirs - 2') 
dbg(dirs) 

#sys.exit()

auc_list_list = []
ppvn_list_list = []

for folder in folders:

    print(folder)

    auc_tmp, comp_auc = read_file_list(folder,'auc', 'AUC')
    ppvn_tmp, comp_ppvn = read_file_list(folder,'ppvn', 'PPV_n')

    print("auc_tmp")
    print(auc_tmp)

    print("comp_auc")
    print(comp_auc)

    print('ppvn_tmp')
    print(ppvn_tmp)
    
    print("comp_ppvn")
    print(comp_ppvn)

    auc_list_list.append(np.array(auc_tmp, dtype=np.float64))
    ppvn_list_list.append(np.array(ppvn_tmp, dtype=np.float64))

    auc_comp_list.append(np.array(comp_auc, dtype=np.float64))
    ppvn_comp_list.append(np.array(comp_ppvn, dtype=np.float64))

make_boxplot('AUC', auc_list_list, auc_comp_list)
make_boxplot('PPVN', ppvn_list_list, ppvn_comp_list)

