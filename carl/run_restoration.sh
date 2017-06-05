#!/bin/bash
#BSUB -n 24
#BSUB -W 12:00
cd $HOME/git/CILProject/carl
module load new gcc/4.8.2 python/3.6.0
python road_renovation_multicore.py --n_processes 24 --data_path ~/git/CILProject/idil/processed --save_dir ~/git/CILProject/data/restoration --n_images 100
