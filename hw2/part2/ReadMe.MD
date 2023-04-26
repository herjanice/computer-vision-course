# How to run code:

## For model training, run the following code:
Pre-requisite: the original `part2` folder have no `p2_data` datasets. Hence, please first copy the `p2_data` datasets into the `part2` folder.

Once pre-requisite requirements are met, run the following code:
```
python main.py
```

The dataset path for training is by default `./p2_data/train`

The model will be saved in `save_dir/ResNet/best_model.pt`
<br><br>

## To evaluate the model, run the following code:
```
python eval.py
```

The model path is by default `./save_dir/ResNet/best_model.pt`

