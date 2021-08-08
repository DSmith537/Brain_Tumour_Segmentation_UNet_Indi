# Brain_Tumour_Segmentation_UNet_Indi

The system is executed using a Docker image, which is able to containerise the Python and external libraries used by the architecture. A high level description of this process is described below: 

•	Set up a Docker container with TensorFlow
    docker pull nvcr.io/nvidia/tensorflow:21.05-tf2-py3
    
•	Create a custom Docker file and requirements file:
    mkdir ~/tf2-custom
    touch ~/tf2-custom/Dockerfile
    touch ~/tf2-custom/requirements.txt
    cd ~/tf2-custom
    
•	Append the requirements.txt file with the following:
    pytest
    nilearn
    matplotlib
    sklearn
    keras
    seaborn
    SimpleITK

•	Build the custom image.
    docker build --build-arg local_uid=$(id -u) --build-arg local_user=$USER -t tf2-custom .

•	To execute the Individual Model run the following pipeline:
    1.	DataLoad.py
    2.	ClassWeights.py
    3.	IndiUNetModel.py
    4.	IndiUNetEval.py
