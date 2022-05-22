# Microscope-Super-Resolution
**Super-Resolution for microscope images using Deep Learning**
## Data preparation

The raw data (unregistered images) is in *./Registration/Raw-Data*.

Run the following files to do image-registration:
* *./Registration/Match_PIPE_down.py* (for post-upsample model)
* *./Registration/Match_PIPE_up.py* (for pre-upsample model) 

The image-registration results are in *./Registration/Registration-down* and *./Registration/Registration-up*.

Delete the bad registration results in *./Registration/Registration-down* and *./Registration/Registration-up*.

Do the following steps to copy the images:
* Copy the low resolution images (*test\{...\}\_pic\_10x\_\{...\}\_\{...\}\_\{...\}\_\{...\}\_\{...\}divide.tif*) in *./Registration/Registration-up*  to *./Data-Pre-upsample/10x_origin*. Then rename them as *10x\{...\}.tif*.
* Copy the low resolution images (*test\{...\}\_pic\_10x\_\{...\}\_\{...\}\_\{...\}\_\{...\}\_\{...\}divide.tif*) in *./Registration/Registration-down* to *./Data-Post-upsample/10x_origin*. Then rename them as *10x\{...\}.tif*.
* Copy the high resolution images (*test\{...\}\_pic\_20x.tif*) in *./Registration/Registration-up*  to *./Data-Pre-upsample/20x_origin*. Then rename them as *20x\{...\}.tif*.
* Copy the high resolution images (*test\{...\}\_pic\_20x.tif*) in *./Registration/Registration-down* to *./Data-Post-upsample/20x_origin*. Then rename them as *20x\{...\}.tif*.


Run the following files to do data preparation.
* *./Data-Preparation/data_train_post.py* 
* *./Data-Preparation/data_train_pre.py*
* *./Data-Preparation/data_eval_post.py*
* *./Data-Preparation/data_eval_pre.py*
* *./Data-Preparation/data_predict_post.py*
* *./Data-Preparation/data_predict_post.py*

Congratulations! The data preparation for deep learning microscope super resolution is finished.
## Run deep learning Super-Resolution models
For each model (nn-EDSR/nn-SRCNN/nn-SRGAN/nn-VDSR), do the following 4 steps:
* run *prepare.py* to make .h5 data files.
* run *train.py* to train the model.
* run *eval.py* for network evaluation and select best weight files(.pth)
* select weight files(epoch_xx.pth) and run *test.py* to predict the HR images.

## Use synthetic data for network training
Run the following files to do synthetic data preparation
* *./Data-Preparation/pre_syn_data_eval.py*
* *./Data-Preparation/pre_syn_data_train.py*
* *./Data-Preparation/post_syn_data_eval.py*
* *./Data-Preparation/post_syn_data_train.py*
### The following files are used to create synthetic data for network preparation 
Input synthetic LR pics to compare the performance of the network on real pics and synthetic pics.
* *./Data-Preparation/data_predict_pre_syn.py*
* *./Data-Preparation/data_predict_post_syn.py*

### For each model (nn-EDSR/nn-SRCNN/nn-SRGAN/nn-VDSR)
* for network trained with synthetic data, input synthetic pics for prediction
    * run *prepare.py* to make .h5 data files.
    * run *train_syn.py* to train the model.
    * run *eval_syn.py* for network evaluation and select best weight files(epoch_xx.pth)
    * run *test_results_syn_syn.py* 

* for network trained with synthetic data, input real pics for prediction
    * run *prepare.py* to make .h5 data files.
    * run *train_syn.py* to train the model.
    * run *eval_syn.py* for network evaluation and select best weight files(epoch_xx.pth)
    * run *test_results_syn.py*

### For evaluate bicubic insertion algorithm performance
* run *./Bicubic/test.py*