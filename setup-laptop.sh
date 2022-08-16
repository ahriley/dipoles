#!/usr/bin/zsh

export PYTHONPATH=$HOME/others-code:$HOME/projects/dipoles/code:$PYTHONPATH

source ~/opt/miniconda3/etc/profile.d/conda.sh
conda activate binder

cd $HOME/projects/dipoles
code .
