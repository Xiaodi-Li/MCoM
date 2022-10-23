# MCoMPytorch

This repository contains the code for [(springer link)](https://link.springer.com/chapter/10.1007/978-3-031-10684-2_4) 

```
@inproceedings{li2022mcom,
  title={MCoM: A Semi-Supervised Method for Imbalanced Tabular Security Data},
  author={Li, Xiaodi and Khan, Latifur and Zamani, Mahmoud and Wickramasuriya, Shamila and Hamlen, Kevin W and Thuraisingham, Bhavani},
  booktitle={IFIP Annual Conference on Data and Applications Security and Privacy},
  pages={48--67},
  year={2022},
  organization={Springer}
}
```


## Dependencies

The project was run on a conda virtual environment on Ubuntu 18.04.5 LTS.

Checkout the `requirements.txt` file, if you have conda pre-installed `cd` into the directory where you have downloaded the source code and run the following

```
conda create -n mcom python==3.7
conda activate mcom

pip install -r requirements.txt
```

## Running Experiments

To run experiments they are launched from the `train.py` file.  For example, to run MCoM on MNIST use the following command

`python train.py -c ./configs/mnist/mcom.json --pretrain`

The trainer, data loader, model, optimizer, settings are all specified in the `./configs/mnist/mcom.json` file. 
The `--pretrain` options specifies whether to run the pretraining phase (i.e. training the encoder).


## TODO

- [ ] Add instructions to run code
- [ ] File structure
- [ ] Config file instructions
- [ ] Comment/clean code
- [ ] Jupyter lab demo
