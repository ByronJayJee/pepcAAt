```
###########################################################################################
###########################################################################################
### ooooooooo.                                        .o.             .o.           .   ###
### `888   `Y88.                                     .888.           .888.        .o8   ###
###  888   .d88'  .ooooo.  oo.ooooo.   .ooooo.      .8"888.         .8"888.     .o888oo ###
###  888ooo88P'  d88' `88b  888' `88b d88' `"Y8    .8' `888.       .8' `888.      888   ###
###  888         888ooo888  888   888 888         .88ooo8888.     .88ooo8888.     888   ###
###  888         888    .o  888   888 888   .o8  .8'     `888.   .8'     `888.    888 . ###
### o888o        `Y8bod8P'  888bod8P' `Y8bod8P' o88o     o8888o o88o     o8888o   "888" ###
###                         888                                                         ###
###                        o888o                                                        ###
###########################################################################################
###########################################################################################
```
# Description (Brief text on what the project is about)

PepcAAt is a peptide binding predictor using the machine learning technique, gradient boosting machine (GBM), implemented using the Catboost machine learning Python library. 

Using various independent test sets, PepcAAt achieved average area under the ROC curve (AUC) values that matched state-of-the-art MHC-I predictors performance while also provide insightful feature importance data. Further, PepcAAt’s feature importance scores enabled us to identify critical features that distinguish between binding and non-binding of peptides to MHC-I.

Developed by Byron Gaskin

# Dependencies (List of libs and tools the projects uses or need to run)
Conda should be used to handle all dependencies.

Create a conda environment using the prodvided `environment.yml` file with the following command:

    conda env create -f environment.yml

then activate the pepcAAt environment 

    conda activate pepcaat



# Getting Started (Steps to spin up the project)

## Training Model:

### Training with hyperparameter optimization

Training an allele binding prediction model with hyperparameter optimization can be done using the `train_allelle_model.py` script

    python train_allelle_model.py -p path_to_allele_folder -a allele_name -l name_of_fasta_file -b name_of_binding_file > training_log.out

## Computing test metrics for trained models:

After training model using hyperparameter optimization, `test_allelle_model.py` can be used to compute performance metrics (e.g., AUC, PPV\_n, precision)

    python test_allelle_model.py -p path_to_allele_folder -a allele_name -l name_of_fasta_file -b name_of_binding_file > testing_log.out


## Explain Model using Shap Values:

# Contribution (Section on how to contribute and what not)
# Troubleshooting (Gotchas that most folks come across when working with your project)
# Roadmap (Future changes you wish to add to your project)

- [ ] use biopython to handle fasta files

- [ ] change shapely plot to get rid of unecessary blank sequence positions or AAs

- [ ] remove hardcoded directories in plot\_auc\_ppvn.py


# License

