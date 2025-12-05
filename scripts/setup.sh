#!/usr/bin/env bash

set -e
set -u
set -o pipefail

project_root="$(pwd)"

# ESPNet+Versa installation (w/ quantization)
echo Installing ESPNet...

git clone https://github.com/elnaske/espnet.git
cd espnet
git checkout codec-quantization
cd tools
./setup_venv.sh $(command -v python3)
make

source venv/bin/activate

echo Installing VERSA...
bash installers/install_versa.sh

# Install local ESPNet
cd ${project_root}/espnet
pip install .

# Other dependencies
pip install fvcore ptflops

deactivate

echo Successfully installed ESPNet and Versa

# Pull LibriTTS test-clean
echo Prepared test set...
cd ${project_root}
bash scripts/data_prep.sh

echo Setup complete