#!/bin/bash


### SSLR dataset
python Dataset_Management/sslr/gen_downsampled_audio.py
python Dataset_Management/sslr/packing_batch_sslr_data.py

### DCASE 2021 dataset
python Dataset_Management/dcase2021/gen_downsampled_audio.py
python Dataset_Management/dcase2021/packing_batch_dcase2021_data.py

### TUT 8-ch circular microphone array dataset
python Dataset_Management/tut_circular/gen_downsampled_audio.py
python Dataset_Management/tut_circular/packing_batch_tut_ca_data.py