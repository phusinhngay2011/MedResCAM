# MedResCAM

## Installation

```bash
# Instal miniconda3
# ...

conda create --name medrescam python=3.8 --yes

pip install -r requirements.txt

```

## Experiment

### Train
```bash
python train.py --learning_rate 0.0001 --batch_size 64 --epoch 100
```

### Test
```bash
python test.py
```


## Results

![Visualization](./results/results.png)
