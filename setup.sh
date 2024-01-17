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

echo "Installing System Level Requirements"
sudo apt install git git-lfs wget p7zip-full -y

if ! git status &>/dev/null; then
  echo "Project root directory is not a git directory. Setting up git...";
  git init
  git add .gitignore .gitattributes .gitmodules
  # set a user to make sure the command does not fail when there is not global git user (repo was just initialised)
  git -c user.name='SpecRepair' -c user.email='specrepair@sen.uni.kn' commit -m "Initial commit"
  if [ "$(ls -A eran/)" ]; then
      echo "The eran/ directory is not empty. Please remove manually to add ERAN as a submodule."
  else
      rm -r eran/
  fi
  git submodule add https://github.com/cherrywoods/eran eran
fi

echo "Updating submodules"
git submodule update --init eran

echo "Downloading additional files. This may take a while."
git-lfs install
git lfs pull
wget https://zenodo.org/records/7938547/files/resources.7z
echo "6372512f94ae910573af38b67c1a887e resources.7z" \
  md5sum -c || (
    echo "Downloading resources from Zenodo failed (checksum did not match). Aborting.";
    exit;
  )
7z x resources.7z
rm resources.7z

echo "Installing Further System Level Requirements (python 3.8, m4, build-essential, cmake, autoconf, libtool, texlive-latex-base, libgmp, libprotobuf, protobuf-compiler)"
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
