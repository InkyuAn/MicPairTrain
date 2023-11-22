"MicPairTraining" is a robust method for sound source localization, utilizing deep learning techniques.
Existing DL-based methods are typically tailored for specific types of microphone arrays.
This design choice can lead to issues when different types of microphone arrays are used.
This method addresses these issues by employing a two-step training process: microphone pair training and array geometry-aware training.
During the microphone pair training, various datasets collected by different types of microphone arrays can be utilized to train the DL model.
This method has been published in IEEE RA-L 2023, "Microphone Pair Training for Robust Sound Source Localization with Diverse Array Configurations".
For more details, please refer to the RA-L paper.

0. Initialization
   - Initialize a conda envirnment using "mic_pair_train.yaml"
     $ conda env create --file mic_pair_train.yaml
   - Download the datasets.
     : SSLR (https://www.idiap.ch/en/dataset/sslr)
     : DCASE2021 (https://zenodo.org/records/4844825)
     : TUT-CA (https://zenodo.org/records/1237752 and https://zenodo.org/records/1237754)
   - Modify "get_param.py" to suit your file paths, e.g., "dataset_dir_sslr", "dataset_dir_dcase", "dataset_dir_tut_ca", and "project_dir".
   - Option: If you want to put more datasets, you need to modify the source code manually.
   
2. Performing "Microphone Pair Training"
   - Execute "Mic_Pair_Train/train_TDOA_rTDOA.py"
     $ python Mic_Pair_Train/train_TDOA_rTDOA.py
   - The training outputs will be returned on the "folder_ATA_TDOA_result" directory, configured in "get_param.py".
   - If you want to train the baseline, "DeepGCC", execute "Mic_Pair_Train/train_TDOA_DeepGCC.py".
     
3. Performing "Array Geometry-Aware Training"
   - Execute "Mic_Pair_Train/train_DOA_rTDOA.py"
     $ python Mic_Pair_Train/train_DOA_rTDOA.py


Contact info: https://inkyuan.github.io/
