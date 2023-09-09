Hair Segmentation
===========
The purpose of this project is to segment people's hair


Pretrained model: https://drive.google.com/file/d/15HnmELuktfkZtoNzweTd2Z1Llyzh8iRh/view?usp=sharing




Requirements
------------
- python3.7
- install requirement.txt

Installation
---------------


.. code-block:: bash

    # install python3.7 and create environment
    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

Test
-------------------------------------

.. code-block:: bash

    python test.py -m pretrained_model_address -v image_folder_for_test
### Options
>- -m The address of the pre-trained model
>- -v Addresses of directories containing images

Train
---------------------------------------

.. code-block:: bash
    
    python main.py -t train_directory -v validation_directory -w directory_saved_weight -e number_of_epoch

### Options
>- -t Train Directory
>- -v Validation Directory
>- -i Test Directory
>- -w The storage location of the weights has been taught
>- -e Number of Epoch
>- -b Batch size
>- -l Learning Rate
>- -c Number of Classes