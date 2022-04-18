# Microscope-Super-Resolution
**Super-Resolution for microscope images using Deep Learning**
## Data preparation

The raw data (unregistered images) is in *./Registration/Raw-Data*.

Use *./Registration/Match_PIPE_down.py* (for post-upsample model) or *./Registration/Match_PIPE_up.py* (for pre-upsample model) to do image-registration.

The image-registration results are in *./Registration/Registration-down* and *./Registration/Registration-up*.

Delete the bad registration results in *./Registration/Registration-down* and *./Registration/Registration-up*.

Copy the low resolution images (*test\{...\}\_pic\_10x\_\{...\}\_\{...\}\_\{...\}\_\{...\}\_\{...\}divide.tif*) in *./Registration/Registration-up* and *./Registration/Registration-down* to *./Data-Pre-upsample/10x_origin* and *./Data-Post-upsample/10x_origin*, respectively. Then rename them as *10x\{...\}.tif*.

Copy the high resolution images (*test\{...\}\_pic\_20x.tif*) in *./Registration/Registration-up* and *./Registration/Registration-down* to *./Data-Pre-upsample/20x_origin* and *./Data-Post-upsample/20x_origin*, respectively. Then rename them as *20x\{...\}.tif*.

Run the following files to do data augmentation.
* *./Data-Preparation/data_train_post.py* 
* *./Data-Preparation/data_train_pre.py*
* *./Data-Preparation/data_eval_post.py*
* *./Data-Preparation/data_eval_pre.py*
* *./Data-Preparation/data_predict_post.py*
* *./Data-Preparation/data_predict_post.py*

Congratulations! The data preparation for deep learning microscope super resolution is finished.
## Run deep learning Super-Resolution models
For each model (nn-EDSR/nn-SRCNN/nn-SRGAN/nn-VDSR), do the following 3 steps:
* run *prepare.py* to make .h5 data files.
* run *main.py* to train the model.
* run *test.py* to predict the HR images.