# Brittleness of Language Reward Shaping

This is the code for preprint paper *A Reminder of its Brittleness: Language Reward Shaping May Hinder Learning for Instruction Following Agents*



## Running the code 

*Note that the pretrained models are not uploaded due to storage limitation of Github platform*

- The code is tested on Ubuntu 20.04 
- Anaconda Python environment is prefered 
- Run the code `pip install -r requirements.txt` to install the required packages
- wandb cli is required and should be activated during testing. 



1. To test the simulated LRS model, modify the config file `config/rnd_rl_train_test_false_positive.yaml`. An example of the command is as follows:

```bash
export PYTHONPATH=`pwd`

python rnd_rl_env_false_positive_simulation/train.py -m rl_data_files.rl_task=0,1 rl_params.whether_shorter_chunks=True,False rl_params.more_restrict_flag=True,False rl_params.follow_temporal_order=True,False CONSTANT.RANDOM_SEED=1,2,3,4,5,6,7,8,9,10
```



2. To test the non-simulated LRS model, one has to download the pretrained model and save them in to `saved_model` folder. After that, modify the config file `config/lang_rew_module.yaml`. an example of the command is as follows:

```bash
export PYTHONPATH=`pwd`

python models/lang_rew_shaping_binary_classification_ver/train_lang_rew_module.py -m lang_rew_shaping_params.use_action_prediction=True lang_rew_shaping_params.use_relative_offset=True data_files.pretrain_visual_encoder_use_object_detec=True lang_rew_shaping_params.normal_negative_ratio=0.7
```





 ## Code structure

```bash
.
├── backbones					# backbone modules for DL models
├── config					# config files for experiments
├── custom_utils				# utils 
├── data					# store training data, not uploaded
├── dataloader					# handling dataset using Nvidia dali library
├── models						# codes for non-sinmulated LRS models
│   ├── lang_rew_shaping_binary_classification_ver	# binary classification output layer ver
│   ├── lang_rew_shaping_event_detection_ver		# cosine similarity output layer ver
│   ├── video_action_predictor				# non-sim LRS observation encoder
│   └── video_autoencoder				# non-sim LRS observation encoder backbone
├── readme.md
├── requirements.txt		
├── rl_trained_models						# rl policy models trained using Non-sim LRS
├── rl_trained_models_test_false_positive			# rl policy models trained using Sim LRS
├── rnd_rl_env							# Env for Non-sim LRS models
├── rnd_rl_env_false_positive_simulation			# Env for Sim LRS models


```



## Data

- Raw training data can be downloaded from [Atari Grand Challenge dataset](http://atarigrandchallenge.com/data). 
- Sentence embedding encoders that we used in this project are from Huggingface platform



## Pretrained Models

- Pretrained models are required for testing non-simulated LRS model. Please download it from https://drive.google.com/drive/folders/1eQaej8V5Hwmoeew2-Bbso8OiwFHbsisj?usp=sharing
- Access permission is required. Please post an issue if you cannot download the pretrained model
