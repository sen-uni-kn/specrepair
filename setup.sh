#!/bin/bash
# Sets up this project, including creating a python virtualenv
# and installing system dependencies
# will require sudo privileges for execution
# the script will prompt for sudo privileges when required

with_cuda=0
if [ "$(which nvidia-smi)" != "" ]; then
  with_cuda=1
fi

set -e  # exit when any command fails

echo "Installing git"
sudo apt install git git-lfs -y

echo "Updating submodules"
git submodule update --init eran

echo "Pull large files (git lfs). This may take a while."
git-lfs install
git lfs pull

echo "Installing System Level Requirements (python 3.8, m4, build-essential, cmake, autoconf, libtool, texlive-latex-base, libgmp, libprotobuf, protobuf-compiler)"
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get install python3.8 python3.8-venv python3.8-dev -y
sudo apt-get install bc m4 build-essential cmake autoconf libtool texlive-latex-base libgmp3-dev libprotobuf-dev protobuf-compiler -y
if [ "$with_cuda" == 1 ]; then
  sudo apt-get install nvidia-cuda-toolkit clang -y
fi
python3.8 -m ensurepip
python3.8 -m pip install virtualenv

echo "Creating virtual environment in $(pwd)"
python3.8 -m virtualenv env-nn-repair --python python3.8
echo "Setting environment variables in created virtualenv"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "This script assumes that a Gurobi license file is present in $HOME/gurobi.lic"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
{
  echo "export GUROBI_HOME=\"$(pwd)/eran/gurobi912/linux64\"";
  echo 'export PATH="$PATH:/usr/lib:$GUROBI_HOME/bin"';
  echo 'export CPATH="$CPATH:$GUROBI_HOME/include"';
  echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:$GUROBI_HOME/lib"';
  echo "export GRB_LICENSE_FILE=\"$HOME/gurobi.lic\""
  echo "export PYTHONPATH=\"\$PYTHONPATH:$(pwd)/eran/tf_verify:$(pwd)/eran/ELINA/python_interface:$(pwd)/eran/deepg/code\""
  echo "export PYTHONPATH=\"\$PYTHONPATH:$(pwd)\""
} >> env-nn-repair/bin/activate
source ./env-nn-repair/bin/activate

echo "Installing python dependencies"
if [ "$with_cuda" == 1 ]; then
  pip install -r requirements-cuda11.txt
else
  pip install -r requirements.txt
fi
pip install -e deep-opt/

echo "Installing ERAN"
cd eran/ || exit
if [ "$with_cuda" == 1 ]; then
  sudo bash -E ./install.sh --use-cuda
else
  sudo bash -E ./install.sh
fi
# cd gurobi*/linux64/ || exit
# export NOT_ROOT_USER="$USER"
# sudo chown -R "$NOT_ROOT_USER" .
# python setup.py install
cd ../../..
