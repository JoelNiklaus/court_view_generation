# Court View Generation

## Get Started

### Create conda environment and install pytorch
```
module load Workspace Anaconda3/2021.11-foss-2021a CUDA/11.8.0
eval "$(conda shell.bash hook)"

conda create -n court_gen python=3.9
conda activate court_gen
```

### Install dependencies
You can either use the requirements.txt:
```
pip install -r requirements.txt
```
or install the packages manually
```
pip install datasets transformers nltk rouge wandb evaluate sentencepiece numpy appdirs Click psutil jinja2 networkx sympy pandas attrs
pip install "protobuf<4.21.0"
```

### Install torch 
```
pip install https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp39-cp39-linux_x86_64.whl
```
If you don't use CUDA/11.8 adjust the torch version accordingly.

### Run
You can use the run_gen.sh to finetune or evaluate a model
```
python -m scripts.run_exp3 --finetune=True --model=google/mt5-small --train_size=1000 --eval_size=100 --test_size=200 --seq_length=2048 --grad_acc_steps=1 --epochs=5 --gm=24
```
The batch size gets selected based on the model, sequence length and gpu memory (gm) through the dict in util.py. 
Many values are not tested so you might have to adjust them.

### Troubleshooting
If you get an error like this:
```
/software.el7/software/Anaconda3/2021.11-foss-2021a/bin/python: No module named scripts.run_exp3
```
Try to install python 3.9 in your conda environment manually:
```
conda install python=3.9
```
