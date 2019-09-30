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
# Description

PepcAAt is a peptide binding predictor using the machine learning technique, gradient boosting machine (GBM), implemented using the Catboost machine learning Python library. 

Using various independent test sets, PepcAAt achieved average area under the ROC curve (AUC) values that matched state-of-the-art MHC-I predictors performance while also provide insightful feature importance data. Further, PepcAAt’s feature importance scores enabled us to identify critical features that distinguish between binding and non-binding of peptides to MHC-I.

Developed by Byron Gaskin

# Dependencies 

In addition to conda (can be installed as part of the Anaconda package found at https://www.anaconda.com/distribution/), the following dependencies are required:

```
  - _tflow_select=2.1.0
  - absl-py=0.7.1
  - astor=0.7.1
  - backcall=0.1.0
  - bzip2=1.0.6
  - c-ares=1.15.0
  - ca-certificates=2019.6.16
  - catboost=0.15.1
  - certifi=2019.6.16
  - cloudpickle=1.1.1
  - cudatoolkit=10.0.130
  - cudnn=7.6.0
  - cupti=10.0.130
  - cycler=0.10.0
  - cytoolz=0.9.0.1
  - dask-core=1.2.2
  - dbus=1.13.6
  - decorator=4.4.0
  - expat=2.2.5
  - fontconfig=2.13.1
  - freetype=2.10.0
  - future=0.17.1
  - gast=0.2.2
  - gettext=0.19.8.1
  - glib=2.58.3
  - grpcio=1.16.1
  - gst-plugins-base=1.14.4
  - gstreamer=1.14.4
  - h5py=2.9.0
  - hdf5=1.10.4
  - hyperopt=0.1.2
  - icu=58.2
  - imageio=2.5.0
  - ipython=7.5.0
  - ipython_genutils=0.2.0
  - jedi=0.13.3
  - joblib=0.13.2
  - jpeg=9c
  - keras-applications=1.0.7
  - keras-base=2.2.4
  - keras-gpu=2.2.4
  - keras-preprocessing=1.0.9
  - kiwisolver=1.1.0
  - libblas=3.8.0
  - libcblas=3.8.0
  - libffi=3.2.1
  - libgcc-ng=8.2.0
  - libgfortran-ng=7.3.0
  - libiconv=1.15
  - liblapack=3.8.0
  - libpng=1.6.37
  - libprotobuf=3.7.1
  - libstdcxx-ng=8.2.0
  - libtiff=4.0.10
  - libuuid=2.32.1
  - libxcb=1.13
  - libxml2=2.9.9
  - lz4-c=1.8.3
  - markdown=3.1
  - matplotlib=3.1.0
  - matplotlib-base=3.1.0
  - mock=3.0.5
  - ncurses=6.1
  - networkx=2.3
  - numpy=1.16.4
  - olefile=0.46
  - openblas=0.3.6
  - openssl=1.1.1b
  - pandas=0.24.2
  - parso=0.4.0
  - pcre=8.41
  - pexpect=4.7.0
  - pickleshare=0.7.5
  - pillow=6.0.0
  - pip=19.1.1
  - prompt_toolkit=2.0.9
  - protobuf=3.7.1
  - pthread-stubs=0.4
  - ptyprocess=0.6.0
  - pygments=2.4.2
  - pymongo=3.8.0
  - pyparsing=2.4.0
  - pyqt=5.9.2
  - python=3.7.3
  - python-dateutil=2.8.0
  - pytz=2019.1
  - pywavelets=1.0.3
  - pyyaml=5.1
  - qt=5.9.7
  - readline=7.0
  - scikit-image=0.15.0
  - scikit-learn=0.21.2
  - scipy=1.3.0
  - setuptools=41.0.1
  - setuptools=41.0.1
  - shap=0.28.5
  - sip=4.19.8
  - six=1.12.0
  - sqlite=3.28.0
  - tensorboard=1.13.1
  - tensorflow=1.13.1
  - tensorflow-base=1.13.1
  - tensorflow-estimator=1.13.0
  - tensorflow-gpu=1.13.1
  - termcolor=1.1.0
  - tk=8.6.9
  - toolz=0.9.0
  - tornado=6.0.2
  - tqdm=4.32.1
  - traitlets=4.3.2
  - wcwidth=0.1.7
  - werkzeug=0.15.2
  - wheel=0.33.4
  - xorg-libxau=1.0.9
  - xorg-libxdmcp=1.1.3
  - xz=5.2.4
  - yaml=0.1.7
  - zlib=1.2.11
  - zstd=1.4.0
```

Conda should be used to handle all dependencies.

Create a conda environment using the prodvided `environment.yml` file with the following command:

    conda env create -f environment.yml

then activate the pepcAAt environment 

    conda activate pepcaat

# Getting Started 

## Training Model:

### Training with hyperparameter optimization

Training an allele binding prediction model with hyperparameter optimization can be done using the `train_allelle_model.py` script

    python train_allelle_model.py -p $path_to_allele_folder -a $allele_name -l $name_of_fasta_file -b $name_of_binding_file > training_log.out

## Computing test metrics for trained models:

After training model using hyperparameter optimization, `test_allelle_model.py` can be used to compute performance metrics (e.g., AUC, PPV\_n, precision)

    python test_allelle_model.py -p $path_to_allele_folder -a $allele_name -l $name_of_fasta_file -b $name_of_binding_file > testing_log.out


## Explain Model using Shap Values:

Computing Shapely feature importance values for a trained model can be done using the `explain_allelle_model.py` script

    python explain_allelle_model.py -p $path_to_allele_folder -a $allele_name -l $name_of_fasta_file -b $name_of_binding_file > explain_log.out

# Contribution 
# Troubleshooting 
# Roadmap 

- [ ] use biopython to handle fasta files

- [ ] change shapely plot to get rid of unecessary blank sequence positions or AAs

- [ ] remove hardcoded directories in plot\_auc\_ppvn.py


# License

