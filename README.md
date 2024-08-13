# SGCL-Master
This project involves the code and supplementary materials of paper "SGCL: Semantic-aware Graph Contrastive Learning with Lipschitz Graph Augmentation".

## Dependencies
* pytorch == 1.12.0
* numpy == 1.24.2
* scikit-learn == 1.2.1
* tqdm == 4.64.1
* scipy == 1.11.3
* seaborn == 0.13.0
* networkx == 3.0


## Run
Running SGCL is followed as:

    sh run_main.sh

If you want to run other datasets, please add them in run_main.sh.

## Cite

If you would like to use our code, please cite:
```
@inproceedings{DBLP:conf/icde/CuiCYDF024,
  author       = {Jinhao Cui and
                  Heyan Chai and
                  Xu Yang and
                  Ye Ding and
                  Binxing Fang and
                  Qing Liao},
  title        = {{SGCL:} Semantic-aware Graph Contrastive Learning with Lipschitz Graph
                  Augmentation},
  booktitle    = {40th {IEEE} International Conference on Data Engineering, {ICDE} 2024,
                  Utrecht, The Netherlands, May 13-16, 2024},
  pages        = {3028--3041},
  publisher    = {{IEEE}},
  year         = {2024},
  url          = {https://doi.org/10.1109/ICDE60146.2024.00235},
  doi          = {10.1109/ICDE60146.2024.00235},
}
```
