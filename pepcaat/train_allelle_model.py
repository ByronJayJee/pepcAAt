import logging 
from logging import debug as dbg
import numpy as np
import pandas as pd
import peptide_fun.predict_ensemble_peptide_seq as pfun
import argparse


logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - \n%(message)s')
logging.disable(logging.CRITICAL)

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
    model_prefix_str = 'TEST_catboost_model.allelle_._' + args["allelle"]

    dbg('data_path_align')
    dbg(data_path_align)

    dbg('data_path_bind')
    dbg(data_path_bind)

    seq_array = pfun.read_seq_data(data_path = data_path_align)
    coded_seq_array = pfun.encode_seq_cat(seq_array)
    bind_list = pfun.read_binding_data(data_path = data_path_bind)

    parameters = {
        'iterations'             : [500, 1000, 2000],
        'learning_rate'          : [0.01, 0.03, 0.05, 0.1],
        'depth'                  : [6, 8, 10, 12],
        'l2_leaf_reg'            : [2.0, 3.0, 4.0],
        'rsm'                    : [0.25, 0.5, 0.75, 1.0]
        #'bootstrap_type'         : ['Bayesian', 'Bernoulli', 'MVS'],
        #'bagging_temperature'    : [0.25, 0.5, 1.0, 2.0, 3.0]
        #'leaf_estimation_method' : ['Newton', 'Gradient'],
        #'boosting_type'          : ['Ordered', 'Plain']
        #'iterations'             : [30, 50, 100]
    }

    df_cv_res = pfun.train_grid_search(coded_seq_array, bind_list, parameters)

    dbg('df_cv_res')
    dbg(df_cv_res)

    pfun.train_top_models_grid_search(coded_seq_array, bind_list, df_cv_res, model_prefix=model_prefix_str) #This should be in training script

    '''
    pfun.train_top_models_grid_search(coded_seq_array, bind_list, df_cv_res)

    print('\n')
    print('----Single Model Results----')
    probs, pred_label = pfun.predict_binding_catboost_single_model(coded_seq_array)
    model_auc_score, model_precision_score, ppvn = pfun.compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')

    print('----Five Model Results----')
    probs, pred_label = pfun.predict_binding_catboost_ensemble_model(coded_seq_array, num_models=5)
    model_auc_score, model_precision_score, ppvn = pfun.compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')

    print('----Ten Model Results----')
    probs, pred_label = pfun.predict_binding_catboost_ensemble_model(coded_seq_array, num_models=10)
    model_auc_score, model_precision_score, ppvn = pfun.compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')
    '''

