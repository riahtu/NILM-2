# Non-intrusive load monitoring project 
This is a non-intrusive load monitoring (NILM) project for GEIRINA. PNN classification algorithm is used. 

## Required software installation (Windows)
Install Anaconda https://docs.anaconda.com/anaconda/install/windows/

Install NILMTK https://github.com/nilmtk/nilmtk/blob/master/docs/manual/user_guide/install_user.md

Clone this repository

Open Anaconda Prompt, switch to nilmtk-env environment

``
activate nilmtk-env
``

Install pickle to read in trained PNN

``
pip install pickle
``

Install pyqt5

``conda install -c dsdale24 pyqt5``

Download iawe dataset https://www.iiitd.edu.in/~nipunb/iawe/iawe.h5

Create a directory named 'data' in 'NILM' directory and copy 'iawe.h5' into 'NILM/data'

## Run the program
Open Anaconda Prompt, switch to nilmtk-env environment

``activate nilmtk-env``

Change directory to your NILM directory (cloned file)

``cd YOUR_OWN_DIRECTORY``

Run the python script

``python pyqt_test.py``