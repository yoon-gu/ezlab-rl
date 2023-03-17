# ezlab-rl

## Development Evirnoment

```bash
$ conda activate
(base) $ git clone https://github.com/yoon-gu/ezlab-rl
(base) $ cd ezlab-rl
(base) $ conda env create -f conda_ez.yaml
(base) $ conda activate ez
(ez) $ cd ..
(ez) $ git clone https://github.com/DLR-RM/stable-baselines3
(ez) $ cd stable-baselines3
(ez) $ pip install '.[extra_no_roms]'
(ez) $ cd ../ezlab-rl
(ez) $ python sir_ppo.py
```