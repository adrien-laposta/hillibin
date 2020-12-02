# Binned hillipop

## How to use

  ```scripts``` folder contains files used to run MCMC chains with a modifier version of Hillipop likelihood

  - compute_full_invcov.py : used to produce the full covariance matrix and the binning.dat file
  - hillipop.py : modification in the code (mainly in the ```select_spectra```method) to bin model and data.
  - Hillipop.yaml : add two entries (```binning_rte_file``` and ```bin_data```) to select a path to the ```binning.dat``` file, and a boolean value to use binned likelihood or not
  
  Step 1: Produce the covariances/binning with the ```compute_full_invcov.py``` script
  
  Step 2: Modifiy ```hillipop.py``` and ```Hillipop.yaml``` where the hillipop package is installed
  
  Step 3: Run chains with a yaml file
