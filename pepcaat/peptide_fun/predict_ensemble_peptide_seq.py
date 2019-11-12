import logging 
from logging import debug as dbg
import numpy as np
import pandas as pd
from sklearn import preprocessing as skp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, precision_score
from catboost import CatBoostClassifier, Pool, cv, EFstrType
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import shap
import hyperopt
from numpy.random import RandomState
from joblib import Parallel, delayed
import argparse
import random


logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - \n%(message)s')
#logging.disable(logging.CRITICAL)

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#def read_seq_data():
def read_seq_data(data_path):
    #fi = open('../Reda-BG/MHC-I/generation-1/training_no-mass-spec/A_02-01/A_02-01_alignment.fasta','r')
    #fi = open('../Reda-BG/MHC-I/generation-1/training_MS/A_02-01/A_02-01_alignment.fasta','r')
    fi = open(data_path,'r')
    seq_str = ''
    seq_str_tot = ''
    seq_count=0
    seq_list=[]
    for line in fi:
        #dbg(line)

        ### Parse individual sequence read, if more than one
        fchar = line[0]
        #dbg(fchar)
        if fchar == '>':
            if(seq_count!=0):
                seq_list.append(seq_str)
            seq_str = ''
            seq_count += 1
            #dbg('start string found in fasta file')
        else:
            seq_str = seq_str + line.rstrip()
            seq_str_tot = seq_str_tot + line.rstrip()
            #dbg(seq_str)
    fi.close()

    seq_list.append(seq_str)
    seq_array = np.array(seq_list)
    return seq_array

def encode_seq_cat(seq_array):
    ### Encode the amino acids reads into integer types and save in numpy array
    seq_str_tot_list = ['*','-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    dbg(seq_str_tot_list)
    le = skp.LabelEncoder()
    le.fit(seq_str_tot_list) # I should fit on an explicitely stated list to ensure values are consistent from run to run
    dbg(list(le.classes_))
    #sys.exit()
    num_seq = len(seq_array)
    coded_seq = []
    for idx in range(num_seq):
        tmp_enc = le.transform(list(seq_array[idx]))
        coded_seq.append(tmp_enc)
        #dbg(tmp_enc)
        #dbg(tmp_enc.shape)
        #sys.exit()
        #dbg(len(tmp_enc))

    dbg(len(coded_seq))

    coded_seq_array = np.array(coded_seq)
    return coded_seq_array


#def read_binding_data():
def read_binding_data(data_path):
    ### Read in from binding file and convert to binary label

    #fi = open('../Reda-BG/MHC-I/generation-1/training_no-mass-spec/A_02-01/A_02-01_binding.txt','r')
    #fi = open('../Reda-BG/MHC-I/generation-1/training_MS/A_02-01/A_02-01_binding.txt','r')
    fi = open(data_path,'r')
    bind_count=0
    bind_list=[]
    for line in fi:
        #dbg(line)

        fint = int(line[0])
        #dbg(fint)
        #dbg(fint+1)
        bind_count += 1 
        if fint>=3:
            bind_list.append(0)
        else:
            bind_list.append(1)
    fi.close()
    dbg(bind_list)
    return bind_list

def train_grid_search(coded_seq_array, bind_list, parameters):
    df_seq = pd.DataFrame(coded_seq_array)
    df_seq_class = pd.DataFrame(bind_list)
    ### parameter search using GridSearchCV
    model_gscv = CatBoostClassifier(
        custom_loss=['Accuracy','AUC'],
        loss_function='Logloss',
        random_seed=1016,
        #iterations=2000,
        od_type='Iter',
        od_wait=20,
        verbose=100
        #logging_level='Verbose'
    )
    '''
    parameters = {
        'depth'         : [6,8,10],
        'learning_rate' : [0.01, 0.05, 0.1],
        'iterations'    : [30, 50, 100]
    }
    '''
    #grid = GridSearchCV(estimator=model_gscv, param_grid = parameters, cv = 2, n_jobs=2)
    grid = GridSearchCV(estimator=model_gscv, param_grid = parameters, cv = 3, n_jobs=-1)
    #grid = GridSearchCV(estimator=model_gscv, param_grid = parameters, cv = 6, n_jobs=6)
    grid.fit(df_seq, df_seq_class)
    best = grid.best_params_

    print('###############################################')
    print('###############################################')
    print('###############################################')

    dbg('grid')
    dbg(grid)

    dbg('best')
    dbg(best)

    dbg('grid.cv_results_')
    dbg(grid.cv_results_)

    df_cv_res = pd.DataFrame(grid.cv_results_)

    dbg('df_cv_res')
    dbg(df_cv_res)

    dbg('df_cv_res.shape[0]')
    dbg(df_cv_res.shape[0])

    df_cv_res.to_csv('grid_search_results.csv')

    return df_cv_res

def train_single_model(df_seq, df_seq_class, cat_idx, model_prefix, model_id=1, random_seed_val=None,  extra_params=None):
    
    if not random_seed_val:
        random_seed_val = random.randint(0,1016)
    

    model_tmp = CatBoostClassifier(
        custom_loss=['Accuracy','AUC'],
        loss_function='Logloss',
        random_seed=random_seed_val,
        #iterations=2000,
        **extra_params,
        verbose=100
    )

    model_tmp.fit(
        df_seq, df_seq_class,
        cat_features=cat_idx,
        logging_level='Verbose'  # you can uncomment this for text output
    );
    
    #tmp_str = 'catboost_model.allelle_._A_02-01.rank_._%d.bin' % (model_id)
    tmp_str = '.rank_._%d.bin' % (model_id)
    tmp_str = model_prefix + tmp_str
    
    #model_tmp.save_model('catboost_model.allelle_._A_02-01.rank_._1.bin')
    model_tmp.save_model(tmp_str)
    
    dbg('tmp_str')
    dbg(tmp_str)
    print("Model Train Complete")
    
def train_model_from_list(x_idx, df_cv_res, df_seq, df_seq_class, cat_idx, model_prefix):
    tmp_row = df_cv_res.iloc[x_idx]
    dbg('tmp_row')
    dbg(tmp_row)

    if tmp_row.rank_test_score > 20:
        dbg('model rank too low')
        return None

    dbg('tmp_row.params')
    dbg(tmp_row.params)

    dbg('type(tmp_row.params).__name__')
    dbg(type(tmp_row.params).__name__)
  
    param_type = type(tmp_row.params).__name__

    tmp_params = tmp_row.params
    
    if (param_type != 'dict'):
        dbg('not a dict')
        tmp_params = eval(tmp_params)

    dbg('type(tmp_params).__name__')
    dbg(type(tmp_params).__name__)
    
    
    #train_single_model(df_seq, df_seq_class, cat_idx, model_prefix, model_id=tmp_row.rank_test_score, extra_params=tmp_params)
    train_single_model(df_seq, df_seq_class, cat_idx, model_prefix, model_id=x_idx+1, extra_params=tmp_params)

def train_top_loop_model_grid_search(coded_seq_array, bind_list, df_cv_res, num_loops, model_prefix):
    df_seq = pd.DataFrame(coded_seq_array)
    df_seq_class = pd.DataFrame(bind_list)
    seq_len = coded_seq_array.shape[1]
    cat_idx = list(range(seq_len))

    top_df_cv_res = df_cv_res.sort_values(by=['rank_test_score'])

    top_df_cv_res = top_df_cv_res[0:1]

    dbg('top_df_cv_res')
    dbg(top_df_cv_res)

    tmp_row = top_df_cv_res.iloc[0]
    dbg('tmp_row')
    dbg(tmp_row)

    if tmp_row.rank_test_score > 20:
        dbg('model rank too low')
        return None

    dbg('tmp_row.params')
    dbg(tmp_row.params)

    dbg('type(tmp_row.params).__name__')
    dbg(type(tmp_row.params).__name__)
  
    param_type = type(tmp_row.params).__name__

    tmp_params = tmp_row.params
    
    if (param_type != 'dict'):
        dbg('not a dict')
        tmp_params = eval(tmp_params)

    dbg('type(tmp_params).__name__')
    dbg(type(tmp_params).__name__)


    # Parallelize by making a function containing everything in the for-loop and using joblib
    #Parallel(n_jobs=-1)(delayed(train_model_from_list)(x_idx, top_df_cv_res, df_seq, df_seq_class, cat_idx, model_prefix) for x_idx in range(0,top_df_cv_res.shape[0]))
    Parallel(n_jobs=-1)(delayed(train_single_model)(df_seq, df_seq_class, cat_idx, model_prefix, model_id=x_idx+1, random_seed_val=None, extra_params=tmp_params) for x_idx in range(num_loops))
        

def train_top_models_grid_search(coded_seq_array, bind_list, df_cv_res, model_prefix):
    df_seq = pd.DataFrame(coded_seq_array)
    df_seq_class = pd.DataFrame(bind_list)
    seq_len = coded_seq_array.shape[1]
    cat_idx = list(range(seq_len))

    top_df_cv_res = df_cv_res.sort_values(by=['rank_test_score'])
    top_df_cv_res = top_df_cv_res[0:20]

    # Parallelize by making a function containing everything in the for-loop and using joblib
    Parallel(n_jobs=-1)(delayed(train_model_from_list)(x_idx, top_df_cv_res, df_seq, df_seq_class, cat_idx, model_prefix) for x_idx in range(0,top_df_cv_res.shape[0]))

def predict_raw_probs_binding_catboost_single_model(coded_seq_array, model_prefix, model_rank=1):
    #tmp_str = 'catboost_model.allelle_._A_02-01.rank_._%d.bin' % (model_rank)
    tmp_str = '.rank_._%d.bin' % (model_rank)
    tmp_str = model_prefix + tmp_str
    model_tmp = CatBoostClassifier()
    model_tmp.load_model(tmp_str)
    df_seq_test = pd.DataFrame(coded_seq_array)

    raw_probs = model_tmp.predict(data=df_seq_test, prediction_type='RawFormulaVal')

    return raw_probs

def predict_binding_catboost_single_model(coded_seq_array, model_prefix, model_rank=1):
    #tmp_str = 'catboost_model.allelle_._A_02-01.rank_._%d.bin' % (model_rank)
    tmp_str = '.rank_._%d.bin' % (model_rank)
    tmp_str = model_prefix + tmp_str
    model_tmp = CatBoostClassifier()
    model_tmp.load_model(tmp_str)
    df_seq_test = pd.DataFrame(coded_seq_array)

    probs = model_tmp.predict_proba(data=df_seq_test)
    pred_label = model_tmp.predict(data=df_seq_test)

    return probs, pred_label

def predict_raw_probs_binding_catboost_ensemble_model(coded_seq_array, model_prefix, num_models=3):
    probs_cum = predict_raw_probs_binding_catboost_single_model(coded_seq_array, model_prefix, model_rank=1)
    for idx in range(1,num_models):
        probs_tmp = predict_raw_probs_binding_catboost_single_model(coded_seq_array, model_prefix, model_rank=idx+1)
        probs_cum = np.add(probs_cum, probs_tmp)
    probs_cum = probs_cum/num_models

    dbg('probs_cum')
    dbg(probs_cum)
    
    return probs_cum

def predict_binding_catboost_ensemble_model(coded_seq_array, model_prefix, num_models=3):
    probs_list = []
    probs_cum, _ = predict_binding_catboost_single_model(coded_seq_array, model_prefix, model_rank=1)
    probs_list.append(probs_cum)
    for idx in range(1,num_models):
        probs_tmp, _ = predict_binding_catboost_single_model(coded_seq_array, model_prefix, model_rank=idx+1)
        probs_cum = np.add(probs_cum, probs_tmp)
        probs_list.append(probs_tmp)
    probs_cum = probs_cum/num_models
    pred_cum = np.argmax(probs_cum, axis=1)

    dbg('probs_cum')
    dbg(probs_cum)
    
    dbg('pred_cum')
    dbg(pred_cum)

    #return probs_cum, pred_cum
    return np.array(probs_list)

def predict_probs_binding_catboost_ensemble_model(coded_seq_array, model_prefix, num_models=3):
    probs_cum, _ = predict_binding_catboost_ensemble_model(coded_seq_array, model_prefix, num_models)
    return probs_cum[:,1]

##############
def pivot_seq_shap_dataframe(df_seq_test, shap_values):
    list_seq_shap_tot = []
    
    num_allelle = shap_values.shape[0]

    for idx_allelle in range(num_allelle):
    
        df_seq_tmp = df_seq_test.loc[[idx_allelle]].transpose().copy()
        #df_seq_tmp.transpose()

        dbg("df_seq_tmp")
        dbg(df_seq_tmp)

        df_seq_tmp.rename(columns={idx_allelle:'AA'}, inplace=True)

        dbg("df_seq_tmp")
        dbg(df_seq_tmp)
    
        shaps = shap_values[idx_allelle,:]
        dbg("shaps")
        dbg(shaps)
    
        df_shap_tmp = pd.DataFrame(shaps, columns=['shap'])

        dbg('df_shap_tmp')
        dbg(df_shap_tmp)

        df_seq_shap_tmp = df_seq_tmp.join(df_shap_tmp)
        df_seq_shap_tmp['seq_pos'] = df_seq_shap_tmp.index

        dbg('df_seq_shap_tmp')
        dbg(df_seq_shap_tmp)

        df_pivot_tmp = pd.pivot_table(df_seq_shap_tmp, values ='shap', index='seq_pos', columns =['AA'], fill_value=0.0) 
        #df_pivot_tmp.reset_index(inplace=True)
        #df_pivot_tmp.set_index('seq_pos', inplace=True)
        #df_pivot_tmp.columns = df_pivot_tmp.columns.map('{0[0]}|{0[1]}'.format)

        #df_pivot_tmp[2]=0.0

        dbg('df_pivot_tmp')
        dbg(df_pivot_tmp)
    
        dbg('df_pivot_tmp[1]')
        dbg(df_pivot_tmp[1])

        df_seq_shap_tot = pd.DataFrame(np.zeros((15, 28)))

        dbg('df_seq_shap_tot')
        dbg(df_seq_shap_tot)

        df_seq_shap_tot = df_seq_shap_tot.add(df_pivot_tmp, fill_value=0.0)

        dbg('df_seq_shap_tot')
        dbg(df_seq_shap_tot)


        array_seq_shap_tmp = df_seq_shap_tot.to_numpy()

        dbg('array_seq_shap_tmp')
        dbg(array_seq_shap_tmp)

        dbg('array_seq_shap_tmp.shape')
        dbg(array_seq_shap_tmp.shape)

        list_seq_shap_tot.append(array_seq_shap_tmp)

        #dbg('list_seq_shap_tot')
        #dbg(list_seq_shap_tot)
    #array_seq_shap_tot = np.array(list_seq_shap_tot)
    return list_seq_shap_tot

def explain_shap_catboost_single_model_matrix(coded_seq_array, bind_list, model_prefix, allelle_name='unknown',  model_rank=1):
    ### This routine attempts to reshape the shap values into a format that allows to see average and standard deviation of shap values for protein sequences
    ### This approach should also extend to ensemble methods in more straightforward way

    # Load Model
    tmp_str = '.rank_._%d.bin' % (model_rank)
    tmp_str = model_prefix + tmp_str
    model_tmp = CatBoostClassifier()
    model_tmp.load_model(tmp_str)
    
    # Read sequences and binding lists
    df_seq_test = pd.DataFrame(coded_seq_array)
    bind_list_array = np.array(bind_list)

    # get model prediction for probability of binding
    probs = model_tmp.predict_proba(data=df_seq_test)
    dbg("probs")
    dbg(probs)

    # combine model prediction and ground truth into one array
    df_pred = pd.DataFrame({'pred':probs[:,1], 'true':bind_list_array})
    #df_pred_sort = df_pred.sort_values(by=['pred'], ascending=False)

    # Grab sequences that bind and sort them by prediction probability
    df_pred_bind_true_sort  = df_pred[df_pred['true']==1].sort_values(by=['pred'], ascending=False)
    # Grab sequences that DO NOT bind and sort them by prediction probability
    df_pred_bind_false_sort = df_pred[df_pred['true']==0].sort_values(by=['pred'], ascending=False)

    dbg('df_pred_bind_true_sort.head()')
    dbg(df_pred_bind_true_sort.head())

    dbg('df_pred_bind_false_sort.head()')
    dbg(df_pred_bind_false_sort.head())

    dbg('df_pred_bind_true_sort.tail()')
    dbg(df_pred_bind_true_sort.tail())

    dbg('df_pred_bind_false_sort.tail()')
    dbg(df_pred_bind_false_sort.tail())

    dbg('df_pred_bind_false_sort.iloc[0]')
    dbg(df_pred_bind_false_sort.iloc[0])
    
    dbg('df_pred_bind_false_sort.index')
    dbg(df_pred_bind_false_sort.index)

    dbg('df_pred_bind_false_sort.index[-1]')
    dbg(df_pred_bind_false_sort.index[-1])

    idx_top_fn = df_pred_bind_false_sort.index[0]
    idx_top_tn = df_pred_bind_false_sort.index[-1]

    idx_top_tp = df_pred_bind_true_sort.index[0]
    idx_top_fp = df_pred_bind_true_sort.index[-1]

    dbg('coded_seq_array[:10,:]')
    dbg(coded_seq_array[:10,:])

    seq_str_tot_list = ['*','-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    dbg(seq_str_tot_list)
    le = skp.LabelEncoder()
    le.fit(seq_str_tot_list)

    top_tp_seq = le.inverse_transform(coded_seq_array[idx_top_tp,:])
    top_tp_seq_str = ''.join(top_tp_seq).replace("-","")
    dbg('top_tp_seq_str')
    dbg(top_tp_seq_str)

    top_tn_seq = le.inverse_transform(coded_seq_array[idx_top_tn,:])
    top_tn_seq_str = ''.join(top_tn_seq).replace("-","")
    dbg('top_tn_seq_str')
    dbg(top_tn_seq_str)

    top_fp_seq = le.inverse_transform(coded_seq_array[idx_top_fp,:])
    top_fp_seq_str = ''.join(top_fp_seq).replace("-","")
    dbg('top_fp_seq_str')
    dbg(top_fp_seq_str)

    top_fn_seq = le.inverse_transform(coded_seq_array[idx_top_fn,:])
    top_fn_seq_str = ''.join(top_fn_seq).replace("-","")
    dbg('top_fn_seq_str')
    dbg(top_fn_seq_str)

    fo = open('seq_of_interest.txt','w')

    fo.write(allelle_name + ' tn ' + top_tn_seq_str + '\n')
    fo.write(allelle_name + ' tp ' + top_tp_seq_str + '\n')

    fo.write(allelle_name + ' fn ' + top_fn_seq_str + '\n')
    fo.write(allelle_name + ' fp ' + top_fp_seq_str + '\n')

    fo.close()

    cat_idx = list(range(coded_seq_array.shape[1]))

    train_pool = Pool(data=df_seq_test,  cat_features=cat_idx)
    dbg(train_pool.shape)
    feature_importances = model_tmp.get_feature_importance(train_pool, type=EFstrType.ShapValues, shap_mode='NoPreCalc')
    feature_names = df_seq_test.columns
    dbg(feature_importances.shape)
    #dbg(feature_names)

    dbg('feature_importances.shape')
    dbg(feature_importances.shape)

    expected_value = feature_importances[0,-1]
    shap_values = feature_importances[:,:-1]
    dbg("expected_value")
    dbg(expected_value)
    dbg("shap_values")
    dbg(shap_values)

    dbg("shap_values.shape")
    dbg(shap_values.shape)

    num_allelle = shap_values.shape[0]

    dbg("num_allelle")
    dbg(num_allelle)

    max_shap = np.max(shap_values)
    min_shap = np.min(shap_values)

    abs_max_shap = np.max([np.abs(min_shap), np.abs(max_shap)])

    dbg('max_shap')
    dbg(max_shap)
    
    dbg('min_shap')
    dbg(min_shap)

    dbg("df_seq_test.head()")
    dbg(df_seq_test.head())

    df_seq_test_copy = df_seq_test.copy()
    shap_values_copy = shap_values.copy()

    list_seq_shap_tot = pivot_seq_shap_dataframe(df_seq_test_copy, shap_values_copy)
    '''
    list_seq_shap_tot = []

    for idx_allelle in range(num_allelle):
    
        df_seq_tmp = df_seq_test.loc[[idx_allelle]].transpose().copy()
        #df_seq_tmp.transpose()

        dbg("df_seq_tmp")
        dbg(df_seq_tmp)

        df_seq_tmp.rename(columns={idx_allelle:'AA'}, inplace=True)

        dbg("df_seq_tmp")
        dbg(df_seq_tmp)
    
        shaps = shap_values[idx_allelle,:]
        dbg("shaps")
        dbg(shaps)
    
        df_shap_tmp = pd.DataFrame(shaps, columns=['shap'])

        dbg('df_shap_tmp')
        dbg(df_shap_tmp)

        df_seq_shap_tmp = df_seq_tmp.join(df_shap_tmp)
        df_seq_shap_tmp['seq_pos'] = df_seq_shap_tmp.index

        dbg('df_seq_shap_tmp')
        dbg(df_seq_shap_tmp)

        df_pivot_tmp = pd.pivot_table(df_seq_shap_tmp, values ='shap', index='seq_pos', columns =['AA'], fill_value=0.0) 
        #df_pivot_tmp.reset_index(inplace=True)
        #df_pivot_tmp.set_index('seq_pos', inplace=True)
        #df_pivot_tmp.columns = df_pivot_tmp.columns.map('{0[0]}|{0[1]}'.format)

        df_pivot_tmp[2]=0.0

        dbg('df_pivot_tmp')
        dbg(df_pivot_tmp)
    
        dbg('df_pivot_tmp[1]')
        dbg(df_pivot_tmp[1])

        df_seq_shap_tot = pd.DataFrame(np.zeros((15, 29)))

        dbg('df_seq_shap_tot')
        dbg(df_seq_shap_tot)

        df_seq_shap_tot = df_seq_shap_tot.add(df_pivot_tmp, fill_value=0.0)

        dbg('df_seq_shap_tot')
        dbg(df_seq_shap_tot)


        array_seq_shap_tmp = df_seq_shap_tot.to_numpy()

        dbg('array_seq_shap_tmp')
        dbg(array_seq_shap_tmp)

        dbg('array_seq_shap_tmp.shape')
        dbg(array_seq_shap_tmp.shape)

        list_seq_shap_tot.append(array_seq_shap_tmp)

        #dbg('list_seq_shap_tot')
        #dbg(list_seq_shap_tot)
    

    '''
    array_seq_shap_tot = np.array(list_seq_shap_tot)

    dbg('array_seq_shap_tot.shape')
    dbg(array_seq_shap_tot.shape)

    eps = 1e-6
    array_seq_shap_tot[(np.abs(array_seq_shap_tot) <= eps)] = np.NaN

    #mean_array_seq_shap_tot = np.mean(array_seq_shap_tot, axis=0)
    mean_array_seq_shap_tot = np.nanmean(array_seq_shap_tot, axis=0)
    
    dbg('mean_array_seq_shap_tot')
    dbg(mean_array_seq_shap_tot)
    
    dbg('mean_array_seq_shap_tot[:,0]')
    dbg(mean_array_seq_shap_tot[:,0])

    dbg('mean_array_seq_shap_tot.shape')
    dbg(mean_array_seq_shap_tot.shape)

    t_mean_array_seq_shap_tot = np.transpose(mean_array_seq_shap_tot)

    dbg('t_mean_array_seq_shap_tot.shape')
    dbg(t_mean_array_seq_shap_tot.shape)

    sys.exit()

    dummy_label = [idx for idx in range(15)]

    #tmp_seq_test = [ [itmp for idx in range(29)] for itmp in range(15)]
    tmp_seq_test = [ [itmp for idx in range(15)] for itmp in range(28)]
    #tmp_seq_test = [ dummy_label for itmp in range(29)]

    df_seq_test_tmp = pd.DataFrame(tmp_seq_test)

    dbg('df_seq_test_tmp')
    dbg(df_seq_test_tmp)

    tp_array_seq_shap = np.transpose(array_seq_shap_tot[idx_top_tp,:,:])
    fp_array_seq_shap = np.transpose(array_seq_shap_tot[idx_top_fp,:,:])

    tn_array_seq_shap = np.transpose(array_seq_shap_tot[idx_top_tn,:,:])
    fn_array_seq_shap = np.transpose(array_seq_shap_tot[idx_top_fn,:,:])

    dbg('tp_array_seq_shap.shape')
    dbg(tp_array_seq_shap.shape)
    
    make_peptide_shap_plot(df_seq_test_tmp, tp_array_seq_shap, plot_title=allelle_name+' - True Positive', plot_colormap="coolwarm_r", plot_filename="tp.png")
    make_peptide_shap_plot(df_seq_test_tmp, fp_array_seq_shap, plot_title=allelle_name+' - False Positive', plot_colormap="coolwarm_r", plot_filename="fp.png")
    make_peptide_shap_plot(df_seq_test_tmp, tn_array_seq_shap, plot_title=allelle_name+' - True Negative', plot_colormap="coolwarm_r", plot_filename="tn.png")
    make_peptide_shap_plot(df_seq_test_tmp, fn_array_seq_shap, plot_title=allelle_name+' - False Negative', plot_colormap="coolwarm_r", plot_filename="fn.png")

    ### Only keep top 2 AAs per position
    for idx in range(15):
        shaps = t_mean_array_seq_shap_tot[:,idx]
        dbg('shaps')
        dbg(shaps)
    
        shap_argsort = np.argsort(-1*shaps)
        dbg('shap_argsort')
        dbg(shap_argsort)

        shaps[shap_argsort[1:]] = np.NaN
        dbg('shaps')
        dbg(shaps)

    dbg('t_mean_array_seq_shap_tot')
    dbg(t_mean_array_seq_shap_tot)

    sum_t_mean_array_seq_shap_tot = np.nansum(t_mean_array_seq_shap_tot)

    dbg('sum_t_mean_array_seq_shap_tot')
    dbg(sum_t_mean_array_seq_shap_tot)

    odds_design = np.exp(sum_t_mean_array_seq_shap_tot)
    probs_design = odds_design / (1.0 + odds_design)

    dbg('probs_design')
    dbg(probs_design)

    ptitle = "Shap Values for Peptides vs Sequence Position - Probability of binding: %.4f" % (probs_design)

    dbg('ptitle')
    dbg(ptitle)

    #make_peptide_shap_plot(df_seq_test_tmp, t_mean_array_seq_shap_tot, plot_title="Shap Values for Peptides vs Sequence Position - Matrix", plot_colormap="winter")
    #make_peptide_shap_plot(df_seq_test_tmp, t_mean_array_seq_shap_tot, plot_title="Shap Values for Peptides vs Sequence Position - Matrix", plot_colormap="bwr")
    make_peptide_shap_plot(df_seq_test_tmp, t_mean_array_seq_shap_tot, plot_title=ptitle, plot_colormap="winter_r", plot_filename="flip_single.png")

def explain_shap_catboost_ensemble_model_matrix(coded_seq_array, model_prefix, num_models=2):
    ### This routine attempts to reshape the shap values into a format that allows to see average and standard deviation of shap values for protein sequences
    list_shap_values_tot = []
    
    df_seq_test = pd.DataFrame(coded_seq_array)
    feature_names = df_seq_test.columns

    cat_idx = list(range(coded_seq_array.shape[1]))
    
    train_pool = Pool(data=df_seq_test,  cat_features=cat_idx)
    #train_pool = Pool(data=df_seq_test.loc[[0]],  cat_features=cat_idx)
    dbg('train_pool.shape')
    dbg(train_pool.shape)

    for idx in range(num_models):
        tmp_str = '.rank_._%d.bin' % (idx+1)
        tmp_str = model_prefix + tmp_str
        model_tmp = CatBoostClassifier()
        model_tmp.load_model(tmp_str)

        probs = model_tmp.predict_proba(data=df_seq_test)
        dbg("probs")
        dbg(probs)

        feature_importances = model_tmp.get_feature_importance(train_pool, type=EFstrType.ShapValues, shap_mode='NoPreCalc')
        dbg(feature_importances.shape)
        #dbg(feature_names)

        dbg('feature_importances.shape')
        dbg(feature_importances.shape)

        expected_value = feature_importances[0,-1]
        shap_values = feature_importances[:,:-1]
        dbg("expected_value")
        dbg(expected_value)
        dbg("shap_values")
        dbg(shap_values)

        dbg("shap_values.shape")
        dbg(shap_values.shape)

        list_shap_values_tot.append(shap_values)

    array_shap_values_tot = np.array(list_shap_values_tot)

    dbg('array_shap_values_tot.shape')
    dbg(array_shap_values_tot.shape)

    list_seq_shap_tot = []

    for idx in range(num_models):
        dbg('array_shap_values_tot[idx].shape')
        dbg(array_shap_values_tot[idx].shape)

        df_seq_test_copy = df_seq_test.copy()
        shap_values_copy = array_shap_values_tot[idx].copy()

        list_seq_shap_tmp = pivot_seq_shap_dataframe(df_seq_test_copy, shap_values_copy)

        list_seq_shap_tot.append(list_seq_shap_tmp)

    array_seq_shap_tot = np.array(list_seq_shap_tot)

    dbg('array_seq_shap_tot.shape')
    dbg(array_seq_shap_tot.shape)


    eps = 1e-6
    array_seq_shap_tot[(np.abs(array_seq_shap_tot) <= eps)] = np.NaN

    #mean_array_seq_shap_tot = np.mean(array_seq_shap_tot, axis=0)
    mean_array_seq_shap_tot = np.nanmean(array_seq_shap_tot, axis=0)
    mean_mean_array_seq_shap_tot = np.nanmean(mean_array_seq_shap_tot, axis=0)

    t_mean_array_seq_shap_tot = np.transpose(mean_mean_array_seq_shap_tot)

    dbg('t_mean_array_seq_shap_tot.shape')
    dbg(t_mean_array_seq_shap_tot.shape)

    dummy_label = [idx for idx in range(15)]

    #tmp_seq_test = [ [itmp for idx in range(29)] for itmp in range(15)]
    tmp_seq_test = [ [itmp for idx in range(15)] for itmp in range(28)]
    #tmp_seq_test = [ dummy_label for itmp in range(29)]

    df_seq_test_tmp = pd.DataFrame(tmp_seq_test)

    dbg('df_seq_test_tmp')
    dbg(df_seq_test_tmp)

    '''
    ### Only keep top 2 AAs per position
    for idx in range(15):
        shaps = t_mean_array_seq_shap_tot[:,idx]
        dbg('shaps')
        dbg(shaps)
    
        shap_argsort = np.argsort(-1*shaps)
        dbg('shap_argsort')
        dbg(shap_argsort)

        shaps[shap_argsort[1:]] = np.NaN
        dbg('shaps')
        dbg(shaps)
    '''
    
    #only retain values greater than 0.0
    t_mean_array_seq_shap_tot[(t_mean_array_seq_shap_tot <= 0.0)] = np.NaN
    
    #only retain values less than 0.0
    #t_mean_array_seq_shap_tot[(t_mean_array_seq_shap_tot >= 0.0)] = np.NaN

    dbg('t_mean_array_seq_shap_tot')
    dbg(t_mean_array_seq_shap_tot)

    sum_t_mean_array_seq_shap_tot = np.nansum(t_mean_array_seq_shap_tot)

    dbg('sum_t_mean_array_seq_shap_tot')
    dbg(sum_t_mean_array_seq_shap_tot)

    odds_design = np.exp(sum_t_mean_array_seq_shap_tot)
    probs_design = odds_design / (1.0 + odds_design)

    dbg('probs_design')
    dbg(probs_design)

    ptitle = "Shap Values for Peptides vs Sequence Position - Ensemble - Probability of binding: %.4f" % (probs_design)

    dbg('ptitle')
    dbg(ptitle)

    make_peptide_shap_plot(df_seq_test_tmp, t_mean_array_seq_shap_tot, plot_title=ptitle, plot_colormap="winter_r", plot_filename="flip_ensemble_pos.png")
    #make_peptide_shap_plot(df_seq_test_tmp, t_mean_array_seq_shap_tot, plot_title=ptitle, plot_colormap="winter", plot_filename="flip_ensemble_neg.png")

def explain_shap_catboost_single_model(coded_seq_array, model_prefix, model_rank=1):
    tmp_str = '.rank_._%d.bin' % (model_rank)
    tmp_str = model_prefix + tmp_str
    model_tmp = CatBoostClassifier()
    model_tmp.load_model(tmp_str)
    df_seq_test = pd.DataFrame(coded_seq_array)

    probs = model_tmp.predict_proba(data=df_seq_test)
    dbg("probs")
    dbg(probs)
    # explain the model's predictions using SHAP values
    # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
    #explainer = shap.TreeExplainer(model_tmp)
    #shap_values = explainer.shap_values(df_seq_test)
    #shap.summary_plot(shap_values, df_seq_test, show=False)
    #plt.savefig('shap_binary.png')

    cat_idx = list(range(coded_seq_array.shape[1]))

    train_pool = Pool(data=df_seq_test,  cat_features=cat_idx)
    dbg(train_pool.shape)
    #feature_importances = model.get_feature_importance(x_train, type=EFstrType.ShapValues)
    feature_importances = model_tmp.get_feature_importance(train_pool, type=EFstrType.ShapValues, shap_mode='NoPreCalc')
    feature_names = df_seq_test.columns
    dbg(feature_importances.shape)
    #dbg(feature_names)

    expected_value = feature_importances[0,-1]
    shap_values = feature_importances[:,:-1]
    dbg("expected_value")
    dbg(expected_value)
    dbg("shap_values")
    dbg(shap_values)

    dbg("shap_values.shape")
    dbg(shap_values.shape)

    max_shap = np.max(shap_values)
    min_shap = np.min(shap_values)

    abs_max_shap = np.max([np.abs(min_shap), np.abs(max_shap)])

    dbg('max_shap')
    dbg(max_shap)
    
    dbg('min_shap')
    dbg(min_shap)

    #sys.exit()
    
    dbg("feature_importances[0,:]")
    dbg(feature_importances[0,:])

    #fig_fp, ax_fp = plt.subplots()
    #shap.force_plot(expected_value, shap_values[3,:], df_seq_test.iloc[3,:], show=False, matplotlib=True)

    dbg("shap_values[0,:]")
    dbg(shap_values[0,:])

    dbg("np.sum(shap_values[0,:])")
    dbg(np.sum(shap_values[0,:]))

    dbg("np.sum(shap_values[0,:]) + expected_value")
    dbg(np.sum(shap_values[0,:]) + expected_value)
    
    dbg("np.exp(np.sum(shap_values[0,:]))")
    dbg(np.exp(np.sum(shap_values[0,:])))
    
    dbg("1.0 / (1.0 + np.exp(-1.0*np.sum(shap_values[0,:])))")
    dbg(1.0 / (1.0 + np.exp(-1.0*np.sum(shap_values[0,:]))))
    
    dbg("1.0 / (1.0 + np.exp(-1.0*expected_value))")
    dbg(1.0 / (1.0 + np.exp(-1.0*expected_value)))

    dbg("np.log(expected_value)")
    dbg(np.log(expected_value))

    sum_c1 = np.sum(shap_values[0,:])
    prob_c1 = 1 / (1 + np.exp(-1*sum_c1))

    dbg("prob_c1")
    dbg(prob_c1)

    dbg('df_seq_test.iloc[0]')
    dbg(df_seq_test.iloc[0])

    dbg("model_tmp.predict_proba(data=[df_seq_test.iloc[0]])")
    dbg(model_tmp.predict_proba(data=[df_seq_test.iloc[0]]))

    dbg("model_tmp.predict([df_seq_test.iloc[0]], prediction_type='RawFormulaVal')")
    dbg(model_tmp.predict([df_seq_test.iloc[0]], prediction_type='RawFormulaVal'))

    #plt.savefig('shap_fp.png')

    #sys.exit()

    neg_shap_values = shap_values.copy()
    #only retain values less than 0.0
    neg_shap_values[(neg_shap_values >= 0.0)] = np.NaN

    pos_shap_values = shap_values.copy()
    #only retain values greater than or equal 0.0
    pos_shap_values[(pos_shap_values < 0.0)] = np.NaN

    make_peptide_shap_plot(df_seq_test, shap_values, plot_title="Shap Values for Peptides vs Sequence Position", plot_colormap="RdBu", plot_filename='all_test.png')
    sys.exit()
    make_peptide_shap_plot(df_seq_test, neg_shap_values, plot_title="Negative Shap Values for Peptides vs Sequence Position", plot_colormap="winter")
    make_peptide_shap_plot(df_seq_test, pos_shap_values, plot_title="Positive Shap Values for Peptides vs Sequence Position", plot_colormap="winter_r")
    
    sys.exit()


def explain_shap_catboost_ensemble_model(coded_seq_array, model_prefix, num_models=3):
    df_seq_test = pd.DataFrame(coded_seq_array)
    #df_seq_test = df_seq_test.iloc[:10] # Use for debugging - only uses first 10 data points to speed up shap calculation
    f = lambda df_seq: predict_raw_probs_binding_catboost_ensemble_model(df_seq, model_prefix, num_models)
    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(f, df_seq_test)
    shap_values = explainer.shap_values(df_seq_test)
    dbg("shap_values")
    dbg(shap_values)

    dbg("np.array(shap_values).shape")
    dbg(np.array(shap_values).shape)

    fig_ens, ax_ens = plt.subplots()
    #class1_shap_values = np.array(shap_values)[1,:,:]
    class1_shap_values = shap_values

    dbg("explainer.expected_value")
    dbg(explainer.expected_value)

    dbg("class1_shap_values[0]")
    dbg(class1_shap_values[0])

    sum_c1 = np.sum(class1_shap_values[0])
    prob_c1 = 1 / (1 + np.exp(-1*sum_c1))

    dbg("np.sum(class1_shap_values[0])")
    dbg(np.sum(class1_shap_values[0]))
    
    dbg("np.sum(class1_shap_values[0]) + explainer.expected_value")
    dbg(np.sum(class1_shap_values[0]) + explainer.expected_value)
    
    dbg("np.exp(np.sum(class1_shap_values[0]))")
    dbg(np.exp(np.sum(class1_shap_values[0])))

    dbg("prob_c1")
    dbg(prob_c1)

    dbg('df_seq_test.iloc[0]')
    dbg(df_seq_test.iloc[0])

    dbg("f([df_seq_test.iloc[0]])")
    dbg(f([df_seq_test.iloc[0]]))


    dbg("class1_shap_values.shape")
    dbg(class1_shap_values.shape)

    #sys.exit()

    shap.summary_plot(class1_shap_values, df_seq_test, plot_type='dot', show=False)
    fig_ens.savefig('shap_binary_ensemble_fig_ens.png')


def make_peptide_shap_plot(df_seq_test, shap_values, plot_filename="flip.png", plot_title="Shap Values at Each Peptide Position", plot_colormap="winter"):


    #eps = 0.3
    eps = 1e-6

    #only retain values less than 0.0
    #shap_values[(shap_values >= 0.0)] = np.NaN
    
    #only retain values greater than 0.0
    #shap_values[(shap_values <= 0.0)] = np.NaN
    
    #only retain values near 0.0
    #shap_values[(shap_values >= eps)] = np.NaN
    #shap_values[(shap_values <= -1*eps)] = np.NaN
    
    #only retain values away from 0.0
    shap_values[(np.abs(shap_values) <= eps)] = np.NaN
    #shap_values[(  -1*eps <= shap_values <= eps)] = np.NaN

    max_shap = np.nanmax(shap_values)
    min_shap = np.nanmin(shap_values)

    abs_max_shap = np.max([np.abs(min_shap), np.abs(max_shap)])

    dbg('max_shap')
    dbg(max_shap)
    
    dbg('min_shap')
    dbg(min_shap)

    #fig_sin, ax_sin = plt.subplots()
    #shap.summary_plot(shap_values, df_seq_test, show=False)
    #fig_sin.savefig('shap_binary.png')

    ### Need to remove ['B','J','O','U','X','Z']
    ### This stride was computed by hand to remove vertical columns from plot
    ### stride must be subtracted from x-coordinate
    col_stride = [0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3]
    #col_stride = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    dbg("df_seq_test.head()")
    dbg(df_seq_test.head())

    fig_sin_flip, ax_sin_flip = plt.subplots(figsize=(20,10))

    shap_colors = shap.plots.colors.red_blue

    ##################################################################################################
    for idx in range(15):
        ax_sin_flip.axhline(y=idx, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        df_seq_tmp = df_seq_test[idx]
        df_seq_tmp_stride = df_seq_tmp.copy()

        dbg("df_seq_tmp")
        dbg(df_seq_tmp)
    
        dbg("df_seq_tmp.shape[0]")
        dbg(df_seq_tmp.shape[0])

        for idx_df in range(df_seq_tmp.shape[0]):
            orig_col = df_seq_tmp[idx_df]
            dbg('idx_df')
            dbg(idx_df)

            dbg('orig_col')
            dbg(orig_col)
            df_seq_tmp_stride[idx_df] = orig_col - col_stride[orig_col]

        dbg("df_seq_tmp_stride")
        dbg(df_seq_tmp_stride)

        shaps = shap_values[:,idx]
        dbg("shaps")
        dbg(shaps)

        #only retain values less than 0.0
        #shaps[(shaps >= 0.0)] = np.NaN

        row_height = 0.3

        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        xs = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            xs[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        xs *= 0.9 * (row_height / np.max(xs + 1))



        #cs = ax_sin_flip.scatter(df_seq_tmp,idx+ys, cmap="coolwarm", c=shaps)
        #cs = ax_sin_flip.scatter(df_seq_tmp,idx+ys, cmap=shap_colors, c=shaps, vmin=min_shap, vmax=max_shap)
        #cs = ax_sin_flip.scatter(df_seq_tmp,idx+ys, cmap="winter_r", c=shaps, vmin=min_shap, vmax=max_shap) # Use for positive shap values
        #cs = ax_sin_flip.scatter(df_seq_tmp,idx+ys, cmap=plot_colormap, c=shaps, vmin=min_shap, vmax=max_shap) # Use for negative shap values
        cs = ax_sin_flip.scatter(df_seq_tmp_stride+xs,idx+ys+1, cmap=plot_colormap, c=shaps, vmin=min_shap, vmax=max_shap, norm=MidpointNormalize(midpoint=0.0,vmin=min_shap, vmax=max_shap)) # Use for negative shap values
        #cs = ax_sin_flip.scatter(df_seq_tmp+xs,idx+ys+1, cmap=plot_colormap, c=shaps, vmin=min_shap, vmax=max_shap, norm=MidpointNormalize(midpoint=0.0,vmin=min_shap, vmax=max_shap)) # Use for negative shap values
        #cs = ax_sin_flip.scatter(df_seq_tmp,idx+ys, cmap=shap_colors, c=abs(shaps), vmin=0, vmax=abs_max_shap ) 
        
        cs.set_facecolor("none")
    ##################################################################################################


    ax_sin_flip.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax_sin_flip.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    #ax_sin_flip.grid(which='major', alpha=0.2)
    ax_sin_flip.grid(which='major', alpha=0.75)
    #ax_sin_flip.axes.set_xlim([0,27])
    ax_sin_flip.axes.set_xlim([0,24])
    #ax_sin_flip.axes.set_xlim([-2,28])
    
    ax_sin_flip.axes.set_title(plot_title)
    ax_sin_flip.axes.set_xlabel("Amino Acid")
    ax_sin_flip.axes.set_ylabel("Position in Peptide Sequence")
    cbar = fig_sin_flip.colorbar(cs)
    cbar.set_label('Shap Value [log-odds]')

    #seq_str_tot_list = ['+','-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    seq_str_tot_list = [' ','-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','Y', ' ']
    
    ax_sin_flip.set_xticks(range(len(seq_str_tot_list)))
    ax_sin_flip.set_xticklabels(seq_str_tot_list)
    ax_sin_flip.xaxis.grid(True)

    plt.show()
    
    fig_sin_flip.savefig(plot_filename)

def compute_metrics(bind_list_test, pred_label, probs):
    #### Need to add correct funtion inputs and need to add ppv_n
    df_seq_class_test = pd.DataFrame(bind_list_test)
    bind_list_test_array = np.array(bind_list_test)
    pred_label_array = np.array(pred_label, dtype=np.int32)

    model_auc_score = roc_auc_score(df_seq_class_test,probs[:,1])
    print("model_auc_score")
    print(model_auc_score)
   
    model_precision_score = precision_score(bind_list_test_array,pred_label_array)
    print("model_precision_score")
    print(model_precision_score)

    df_pred = pd.DataFrame({'pred':probs[:,1], 'true':bind_list_test_array})
    df_pred_sort = df_pred.sort_values(by=['pred'], ascending=False)

    dbg('df_pred.head()')
    dbg(df_pred.head())

    dbg('df_pred_sort.head()')
    dbg(df_pred_sort.head())

    df_top_n_true=df_pred_sort['true'].iloc[:np.sum(bind_list_test_array)]
    #dbg("df_pred_sort['true'].iloc[:np.sum(bind_list_test_array)]")
    #dbg(df_pred_sort['true'].iloc[:np.sum(bind_list_test_array)])

    ppvn = np.sum(df_top_n_true) / np.sum(bind_list_test_array)
    dbg('ppvn')
    dbg(ppvn)

    return model_auc_score, model_precision_score, ppvn


def compute_metrics_multi_model(bind_list_test, probs_array):
    #### Need to add correct funtion inputs and need to add ppv_n

    num_models = probs_array.shape[0]

    auc_array = []
    precision_array = []
    ppvn_array = []
    
    auc_ens_array = []
    precision_ens_array = []
    ppvn_ens_array = []

    dbg('probs_array.shape')
    dbg(probs_array.shape)
    
    dbg('probs_array.shape[1]')
    dbg(probs_array.shape[1])
   
    probs_cum = np.zeros((probs_array.shape[1], probs_array.shape[2]))
    dbg('probs_cum')
    dbg(probs_cum)
    
    for idx in range(num_models):
        probs = probs_array[idx]
        pred_label = np.argmax(probs, axis=1)
        auc, precision, ppvn = compute_metrics(bind_list_test, pred_label, probs)
        auc_array.append(auc)
        precision_array.append(precision)
        ppvn_array.append(ppvn)

        probs_cum = np.add(probs_cum, probs)
        probs_tmp = probs_cum / (idx+1)
        pred_tmp_label = np.argmax(probs_tmp, axis=1)

        auc, precision, ppvn = compute_metrics(bind_list_test, pred_tmp_label, probs_tmp)
        auc_ens_array.append(auc)
        precision_ens_array.append(precision)
        ppvn_ens_array.append(ppvn)


    dbg("auc_array")
    dbg(auc_array)

    dbg("precision_array")
    dbg(precision_array)

    dbg("ppvn_array")
    dbg(ppvn_array)

    return np.array(auc_array), np.array(precision_array), np.array(ppvn_array), np.array(auc_ens_array), np.array(precision_ens_array), np.array(ppvn_ens_array)

    '''
    df_seq_class_test = pd.DataFrame(bind_list_test)
    bind_list_test_array = np.array(bind_list_test)
    pred_label_array = np.array(pred_label, dtype=np.int32)

    dbg("probs.shape")
    dbg(probs.shape)

    model_auc_score = roc_auc_score(df_seq_class_test,probs[:,1])
    print("model_auc_score")
    print(model_auc_score)
   
    model_precision_score = precision_score(bind_list_test_array,pred_label_array)
    print("model_precision_score")
    print(model_precision_score)

    df_pred = pd.DataFrame({'pred':probs[:,1], 'true':bind_list_test_array})
    df_pred_sort = df_pred.sort_values(by=['pred'], ascending=False)

    dbg('df_pred.head()')
    dbg(df_pred.head())

    dbg('df_pred_sort.head()')
    dbg(df_pred_sort.head())

    df_top_n_true=df_pred_sort['true'].iloc[:np.sum(bind_list_test_array)]
    #dbg("df_pred_sort['true'].iloc[:np.sum(bind_list_test_array)]")
    #dbg(df_pred_sort['true'].iloc[:np.sum(bind_list_test_array)])

    ppvn = np.sum(df_top_n_true) / np.sum(bind_list_test_array)
    dbg('ppvn')
    dbg(ppvn)

    return model_auc_score, model_precision_score, ppvn
    '''

if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to the directory containing fasta and binding.txt files")
    ap.add_argument("-a", "--allelle", required=True, help="Name of allelle")
    ap.add_argument("-l", "--alignment-file-name", required=True, help="Name of alignment file")
    ap.add_argument("-b", "--binding-file-name", required=True, help="Name of binding file")
    args = vars(ap.parse_args())

    dbg('args["path"]')
    dbg(args["path"])

    dbg('args["allelle"]')
    dbg(args["allelle"])

    data_path_align = args["path"] + args["alignment_file_name"]
    data_path_bind = args["path"] + args["binding_file_name"]

    dbg('data_path_align')
    dbg(data_path_align)

    dbg('data_path_bind')
    dbg(data_path_bind)

    #sys.exit()

    #data_path_align = '../Reda-BG/MHC-I/generation-1/training_MS/A_02-01/A_02-01_alignment.fasta'
    #data_path_bind = '../Reda-BG/MHC-I/generation-1/training_MS/A_02-01/A_02-01_binding.txt'
    #data_path_align = '../Reda-BG/MHC-I/generation-1/training_no-MS/A_02-01/A_02-01_alignment.fasta'
    #data_path_bind = '../Reda-BG/MHC-I/generation-1/training_no-MS/A_02-01/A_02-01_binding.txt'
    seq_array = read_seq_data(data_path = data_path_align)
    coded_seq_array = encode_seq_cat(seq_array)
    bind_list = read_binding_data(data_path = data_path_bind)
    
    model_prefix_str = 'TEST_catboost_model.allelle_._' + args["allelle"]

    parameters = {
        'depth'         : [6,8,10,12],
        'learning_rate' : [0.01, 0.05, 0.1]
        #'iterations'    : [30, 50, 100]
    }

    df_cv_res = train_grid_search(coded_seq_array, bind_list, parameters)
    #df_cv_res = pd.read_csv('grid_search_results.csv')

    dbg('df_cv_res')
    dbg(df_cv_res)


    train_top_models_grid_search(coded_seq_array, bind_list, df_cv_res, model_prefix=model_prefix_str)

    ##########
    data_path_align = '../Reda-BG/MHC-I/generation-1/testing_no-MS/A_02-01/A_02-01_alignment.fasta'
    data_path_bind = '../Reda-BG/MHC-I/generation-1/testing_no-MS/A_02-01/A_02-01_binding.txt'
    seq_array = read_seq_data(data_path = data_path_align)
    coded_seq_array = encode_seq_cat(seq_array)
    bind_list = read_binding_data(data_path = data_path_bind)

    model_prefix_str = 'TEST_catboost_model.allelle_._A_02-01'
    print('\n')
    print('----Single Model Results----')
    probs, pred_label = predict_binding_catboost_single_model(coded_seq_array, model_prefix = model_prefix_str)
    model_auc_score, model_precision_score, ppvn = compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')

    print('----Five Model Results----')
    probs, pred_label = predict_binding_catboost_ensemble_model(coded_seq_array, num_models=5, model_prefix = model_prefix_str)
    model_auc_score, model_precision_score, ppvn = compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')

    print('----Ten Model Results----')
    probs, pred_label = predict_binding_catboost_ensemble_model(coded_seq_array, num_models=10, model_prefix = model_prefix_str)
    model_auc_score, model_precision_score, ppvn = compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')
     


'''
### Read in fasta file from Reda's github repo
#fi = open('../bNAb-ReP/alignments/10-1074_IC50_50_alignment.fasta','r')
#fi = open('../bNAb-ReP/alignments/b12_IC50_50_alignment.fasta','r')
#fi = open('../bNAb-ReP/alignments/4E10_IC50_50_alignment.fasta','r')
#fi = open('../bNAb-ReP/alignments/PGT128_IC50_50_alignment.fasta','r')
fi = open('Reda-BG/MHC-I/generation-1/training_no-mass-spec/A_02-01/A_02-01_alignment.fasta','r')
fi2 = open('Reda-BG/MHC-I/generation-1/testing_no-mass-spec/A_02-01/A_02-01_alignment.fasta','r')
seq_str = ''
seq_str_tot = ''
seq_count=0
seq_list=[]
for line in fi:
   #dbg(line)

   ### Parse individual sequence read, if more than one
   fchar = line[0]
   #dbg(fchar)
   if fchar == '>':
      if(seq_count!=0):
         seq_list.append(seq_str)
      seq_str = ''
      seq_count += 1
      #dbg('start string found in fasta file')
   else:
      seq_str = seq_str + line.rstrip()
      seq_str_tot = seq_str_tot + line.rstrip()
      #dbg(seq_str)
   #break

seq_str_test = ''
seq_str_tot_test = ''
seq_count_test=0
seq_list_test=[]
for line in fi2:
   #dbg(line)

   ### Parse individual sequence read, if more than one
   fchar = line[0]
   #dbg(fchar)
   if fchar == '>':
      if(seq_count_test!=0):
         seq_list_test.append(seq_str_test)
      seq_str_test = ''
      seq_count_test += 1
      #dbg('start string found in fasta file')
   else:
      seq_str_test = seq_str_test + line.rstrip()
      seq_str_tot_test = seq_str_tot_test + line.rstrip()
      #dbg(seq_str_test)
   #break
fi.close()
fi2.close()

seq_list.append(seq_str)
seq_list_test.append(seq_str_test)

seq_array = np.array(seq_list)
seq_array_test = np.array(seq_list_test)

dbg(seq_array[4])
dbg(seq_str_tot)


### Encode the amino acids reads into integer types and save in numpy array
seq_str_tot_list = list(seq_str_tot)
seq_str_tot_list = ['*','-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
dbg(seq_str_tot_list)
le = skp.LabelEncoder()
le.fit(seq_str_tot_list) # I should fit on an explicitely stated list to ensure values are consistent from run to run
dbg(list(le.classes_))
#sys.exit()
num_seq = len(seq_array)
coded_seq = []
for idx in range(num_seq):
   tmp_enc = le.transform(list(seq_array[idx]))
   coded_seq.append(tmp_enc)
   #dbg(tmp_enc)
   #dbg(tmp_enc.shape)
   #sys.exit()
   #dbg(len(tmp_enc))

dbg(len(coded_seq))

coded_seq_array = np.array(coded_seq)

num_seq_test = len(seq_array_test)
coded_seq_test = []
for idx in range(num_seq_test):
   tmp_enc_test = le.transform(list(seq_array_test[idx]))
   coded_seq_test.append(tmp_enc_test)
   #dbg(tmp_enc_test)
   #dbg(tmp_enc_test.shape)
   #sys.exit()
   #dbg(len(tmp_enc_test))

dbg(len(coded_seq_test))

coded_seq_array_test = np.array(coded_seq_test)
'''
### Try PCA on sequence vectors
'''
pca = PCA(n_components=10) # 150 components captures 87% of variance
pca_seq = pca.fit_transform(coded_seq_array)
dbg(pca.explained_variance_ratio_)   
dbg(np.sum(pca.explained_variance_ratio_))   
'''

### Use sklearn to preform K-means clustering
'''
sum_of_squared_distances = []
k_range = range(1,50)
for k in k_range:
   km = KMeans(n_clusters=k)
   #km = km.fit(coded_seq_array)
   km = km.fit(pca_seq)
   sum_of_squared_distances.append(km.inertia_)
   dbg('computing k-means using %d clusters' % k)

plt.plot(k_range, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.savefig('test.png')

dbg(sum_of_squared_distances)
'''
'''
#km = KMeans(n_clusters=10)
km = KMeans(n_clusters=2)
km_seq = km.fit_predict(pca_seq)


### Use K-means cluster results as labels for catboost
seq_len = coded_seq_array.shape[1]
cat_idx = list(range(seq_len))
dbg(coded_seq_array)
dbg(coded_seq_array.shape)
dbg(cat_idx)

df_seq = pd.DataFrame(coded_seq_array)
df_seq_cluster = pd.DataFrame(km_seq)
dbg(df_seq.head())
dbg(df_seq_cluster.head())

#x_train, x_validation, y_train, y_validation = train_test_split(coded_seq_array, km_seq, train_size=0.75, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(df_seq, df_seq_cluster, train_size=0.75, random_state=42)

model = CatBoostClassifier(
   iterations=400,
   custom_loss=['Accuracy'],
   random_seed=42,
   logging_level='Silent'
)

model.fit(
   x_train, y_train,
   cat_features=cat_idx,
   eval_set=(x_validation, y_validation),
   logging_level='Verbose'  # you can uncomment this for text output
#   plot=True
);
'''
''''
### Read in from binding file and convert to binary label

fi = open('Reda-BG/MHC-I/generation-1/training_no-mass-spec/A_02-01/A_02-01_binding.txt','r')
bind_count=0
bind_list=[]
for line in fi:
   #dbg(line)

   fint = int(line[0])
   #dbg(fint)
   #dbg(fint+1)
   bind_count += 1 
   if fint>=3:
      bind_list.append(0)
   else:
      bind_list.append(1)
fi.close()

dbg(bind_list)

fi2 = open('Reda-BG/MHC-I/generation-1/testing_no-mass-spec/A_02-01/A_02-01_binding.txt','r')
bind_count_test=0
bind_list_test=[]
for line in fi2:
   #dbg(line)

   fint = int(line[0])
   #dbg(fint)
   #dbg(fint+1)
   bind_count_test += 1 
   if fint>=3:
      bind_list_test.append(0)
   else:
      bind_list_test.append(1)
fi2.close()

dbg(bind_list_test)
'''
### Run catboost
'''
seq_len = coded_seq_array.shape[1]
cat_idx = list(range(seq_len))

df_seq = pd.DataFrame(coded_seq_array)
df_seq_class = pd.DataFrame(bind_list)
dbg(df_seq.head())
dbg(df_seq_class.head())
'''

'''
#x_train, x_validation, y_train, y_validation = train_test_split(coded_seq_array, km_seq, train_size=0.75, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(df_seq, df_seq_class, train_size=0.75, random_state=42)

model = CatBoostClassifier(
   #iterations=3,
   #iterations=3000,
   iterations=30,
   custom_loss=['Accuracy', 'AUC'],
   #random_seed=42,
   random_seed=1016,
   #logging_level='Silent',
   logging_level='Verbose',
   loss_function='Logloss'
)

cv_params = model.get_params()
cv_params.update({
    'loss_function': 'Logloss'
})

cv_data = cv(
    Pool(df_seq, df_seq_class, cat_features=cat_idx),
    cv_params,
    fold_count=10
    #plot=True
)

dbg('cv_data')
dbg(cv_data)
'''


### parameter search using hyperopt
'''
def hyperopt_objective(params):
    model = CatBoostClassifier(
        l2_leaf_reg=int(params['l2_leaf_reg']),
        learning_rate=params['learning_rate'],
        iterations=500,
        custom_loss=['Accuracy','AUC'],
        loss_function='Logloss',
        random_seed=1016,
        od_type='Iter',
        od_wait=20,
        logging_level='Verbose'
    )

    cv_params = model.get_params()
    
    cv_data = cv(
        Pool(df_seq, df_seq_class, cat_features=cat_idx),
        cv_params
    )
    best_accuracy = np.max(cv_data['test-Accuracy-mean'])
    
    return 1 - best_accuracy # as hyperopt minimises

params_space = {
    'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 3, 1),
    'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 8e-1),
}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=RandomState(123)
)
'''

'''
### parameter search using GridSearchCV
model_gscv = CatBoostClassifier(
    custom_loss=['Accuracy','AUC'],
    loss_function='Logloss',
    random_seed=1016,
    od_type='Iter',
    od_wait=20
    #logging_level='Verbose'
)

parameters = {
    'depth'         : [6,8,10],
    'learning_rate' : [0.01, 0.05, 0.1],
    'iterations'    : [30, 50, 100]
}

#grid = GridSearchCV(estimator=model_gscv, param_grid = parameters, cv = 2, n_jobs=2)
grid = GridSearchCV(estimator=model_gscv, param_grid = parameters, cv = 2, n_jobs=-1)
grid.fit(df_seq, df_seq_class)
best = grid.best_params_

print('###############################################')
print('###############################################')
print('###############################################')

dbg('grid')
dbg(grid)

dbg('best')
dbg(best)

dbg('grid.cv_results_')
dbg(grid.cv_results_)

df_cv_res = pd.DataFrame(grid.cv_results_)

dbg('df_cv_res')
dbg(df_cv_res)

dbg('df_cv_res.shape[0]')
dbg(df_cv_res.shape[0])

for x_idx in range(0,df_cv_res.shape[0]):
    tmp_row = df_cv_res.iloc[x_idx]
    dbg('tmp_row')
    dbg(tmp_row)

    if tmp_row.rank_test_score > 10:
        continue

    dbg('tmp_row.params')
    dbg(tmp_row.params)

    tmp_params = tmp_row.params

    model_tmp = CatBoostClassifier(
        custom_loss=['Accuracy','AUC'],
        loss_function='Logloss',
        random_seed=1016,
        **tmp_params,
        logging_level='Verbose'
    )

    model_tmp.fit(
        df_seq, df_seq_class,
        cat_features=cat_idx,
        logging_level='Verbose'  # you can uncomment this for text output
    );

    tmp_str = 'catboost_model.allelle_._A_02-01.rank_._%d.bin' % (tmp_row.rank_test_score)

    #model_tmp.save_model('catboost_model.allelle_._A_02-01.rank_._1.bin')
    model_tmp.save_model(tmp_str)

    dbg('tmp_str')
    dbg(tmp_str)
    print("Model Train Complete")
##########
sys.exit()
##########

### Train model using the best parameters from hyperparameter search
model = CatBoostClassifier(
    l2_leaf_reg=int(best['l2_leaf_reg']),
    learning_rate=best['learning_rate'],
    iterations=500,
    custom_loss=['Accuracy','AUC'],
    loss_function='Logloss',
    random_seed=1016,
    logging_level='Verbose'
)
cv_data = cv(Pool(df_seq, df_seq_class, cat_features=cat_idx), model.get_params())

dbg('cv_data')
dbg(cv_data)

##########
sys.exit()
##########

model.fit(
   x_train, y_train,
   cat_features=cat_idx,
   eval_set=(x_validation, y_validation),
   logging_level='Verbose'  # you can uncomment this for text output
#   plot=True
);

print("Model Train Complete")

seq_len_test = coded_seq_array_test.shape[1]
cat_idx_test = list(range(seq_len_test))

df_seq_test = pd.DataFrame(coded_seq_array_test)
df_seq_class_test = pd.DataFrame(bind_list_test)

print("Starting Model Prediction on Test set")
probs = model.predict_proba(data=df_seq_test)
print("Completed Model Prediction on Test set")
pred_label = model.predict(data=df_seq_test)

bind_list_test_array = np.array(bind_list_test)
pred_label_array = np.array(pred_label, dtype=np.int32)
print("bind_list_test_array")
print(bind_list_test_array)
print("pred_label_array")
print(pred_label_array)
#dbg(probs)
#dbg(probs[:,1])
#dbg(probs[:,bind_list_test])

dbg("roc_curve(df_seq_class_test,probs[:,1])")
dbg(roc_curve(df_seq_class_test,probs[:,1]))
print("roc_auc_score(df_seq_class_test,probs[:,1])")
print(roc_auc_score(df_seq_class_test,probs[:,1]))
dbg("precision_score([1, 1, 1],[1, 1, 1])")
dbg(precision_score([1, 1, 1],[1, 1, 1]))
print("precision_score(bind_list_test_array,pred_label_array)")
print(precision_score(bind_list_test_array,pred_label_array))

sys.exit()
'''

'''
### Perform SHAP analysis on K-means

### Perform SHAP analysis on catboost
#dbg(x_train.head(1))
#dbg(y_train.iloc[0,:])
#train_pool = Pool(data=x_train.iloc[0,:], label=y_train.iloc[0,:], cat_features=cat_idx)
train_pool = Pool(data=x_train,  cat_features=cat_idx)
dbg(train_pool.shape)
#feature_importances = model.get_feature_importance(x_train, type=EFstrType.ShapValues)
feature_importances = model.get_feature_importance(train_pool, type=EFstrType.ShapValues)
feature_names = x_train.columns
dbg(feature_importances.shape)
#dbg(feature_names)

expected_value = feature_importances[0,-1]
shap_values = feature_importances[:,:-1]
dbg(shap_values.shape)

shap.summary_plot(shap_values, x_train, show=False)
plt.savefig('shap_binary.png')
'''
