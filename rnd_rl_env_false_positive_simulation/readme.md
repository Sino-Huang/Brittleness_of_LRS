## code structure

`evaluate.py`

- it will use the model from `model_path = os.path.join(get_original_cwd(), cfg.rl_params.save_dir, f"task_id-{cfg.rl_data_files.rl_task}-"+cfg.CONSTANT.ENV_NAME + '.model')` to play the game and then evaluate with a GUI

`human_evaluate.py`

- it will use `lang_rew_module.py` to walkthrough data and saved state for loading.
- `walkthrough_data_arr` in the code is from `[f'{walkthrough_data_arr_folderpath}/constraints_info_walkthrough_sentence_{i}.pkl' for i in range(1, 6)]`

## how to test the simulated LRS model 

- see the paper for more information