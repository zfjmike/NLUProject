# Course Project for DS-GA 1012 Natural Language Understanding

This is a repository for an NLU course project by Fangjun Zhang, Nimi Wang and Ruoyu Zhu. We aim at incorporating dependency information into natural language inference and fact verification task. Our model has comparable result to the state-of-the-art methods. We use [FEVER](fever.ai) dataset for our training data and FEVER score as our evaluation metric. You can find our report at [here](https://github.com/zfjmike/NLUProject/blob/master/Report.pdf)

#### Prerequisite

We implement our model based on several packages, including [Jack the Reader](https://github.com/uclmr/jack), [FEVER Baseline](https://github.com/sheffieldnlp/fever-naacl-2018) and additionally [UCL MR](https://github.com/uclmr/fever/tree/takuma-dev) data downloading script.

Please also install [StanfordNLP](https://stanfordnlp.github.io/stanfordnlp/) for dependency parsing preprocessing.


#### Training

Run the following script would start the whole pipeline for training and validation.

```shell
python pipeline.py --config configs/pytorch_depsa_esim_n5.json --overwrite --model [model_name]
```

Furthermore, if you are using Slrum system on HPC, you can run the `sbatch_*.sh` script by

```bash
sbatch sbatch_*.sh
```

#### Visualization

We also provide a simple script for visualizing the self-attention weight, dependency mask and dependency-enhanced self-attention weight.

```bash
python jack/visualize.py
```



