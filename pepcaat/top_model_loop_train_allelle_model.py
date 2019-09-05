import logging 
from logging import debug as dbg
import numpy as np
import pandas as pd
import peptide_fun.predict_ensemble_peptide_seq as pfun
import argparse


logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - \n%(message)s')
#logging.disable(logging.CRITICAL)

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

    #df_cv_res = pfun.train_grid_search(coded_seq_array, bind_list, parameters)
    df_cv_res = pd.read_csv('grid_search_results.csv')
    dbg('df_cv_res')
    dbg(df_cv_res)

    model_prefix_str = 'TOP_LOOP_catboost_model.allelle_._' + args["allelle"]
    #model_prefix_str = 'catboost_model.allelle_._' + args["allelle"]
    
    pfun.train_top_loop_model_grid_search(coded_seq_array, bind_list, df_cv_res, num_loops=20, model_prefix=model_prefix_str) # This should be in train script

    print('\n')
    print('----Single Model Results----')
    probs, pred_label = pfun.predict_binding_catboost_single_model(coded_seq_array, model_prefix=model_prefix_str)
    model_auc_score, model_precision_score, ppvn = pfun.compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')

    print('----Five Model Results----')
    probs, pred_label = pfun.predict_binding_catboost_ensemble_model(coded_seq_array, num_models=5, model_prefix=model_prefix_str)
    model_auc_score, model_precision_score, ppvn = pfun.compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')

    print('----Ten Model Results----')
    probs, pred_label = pfun.predict_binding_catboost_ensemble_model(coded_seq_array, num_models=10, model_prefix=model_prefix_str)
    model_auc_score, model_precision_score, ppvn = pfun.compute_metrics(bind_list, pred_label, probs)
    print('AUC: %f' % model_auc_score)
    print('PPV_n: %f' % ppvn)
    print('\n')
    

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

