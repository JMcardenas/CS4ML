# -*- coding: utf-8 -*-
"""CS4ML_Appendix_D_v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XGkVt2sVtOAXe7bwhYgvGB_a6KP8tMxA
"""

#0. # unzip folders 
!unzip /content/colab_codes.zip
!unzip /content/colab_data.zip

#1. Install packages  
!python -m pip install hdf5storage --user

#2. Import packages
import time, os
 
# choose where you want your project files to be saved
project_folder = "/content/colab_codes/"

def create_and_set_working_directory(project_folder):
    # check if your project folder exists. if not, it will be created.
    if os.path.isdir(project_folder) == False:
        os.mkdir(project_folder)
        print(project_folder + ' did not exist but was created.')

    # change the OS to use your project folder as the working directory
    os.chdir(project_folder)

    # create a test file to make sure it shows up in the right place
    #!touch 'new_file_in_working_directory.txt'
    print('\nYour working directory was changed to ' + project_folder + \
            "\n\nAn empty text file was created there. You can also run !pwd to confirm the current working directory." )

create_and_set_working_directory(project_folder) 

#------------------------------------------------------------------------------#
# Run : 1. train | 2. extract remaining data  | 3. zip colab_results/
#------------------------------------------------------------------------------#
!pwd
start = time.time()

# Figure 3 Main paper 
for i in range(2): 
    if i == 0:
        # Train 
        !bash batch_train_fig_3.sh
    else:
        # test over plot data
        !bash batch_test_fig_3.sh     
     
end = time.time()
print('Task completed in: ' ,str((end - start)/60) ,'minutes' )
#------------------------------------------------------------------------------#
# Save data as .zip
!zip -r /content/colab_results.zip /content/colab_results