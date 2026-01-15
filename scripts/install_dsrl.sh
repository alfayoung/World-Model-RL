# Install dsrl package
git clone --recurse-submodules git@github.com:ajwagen/dsrl.git third_party/dsrl
cd third_party/dsrl
conda create -n dsrl python=3.9 -y
conda activate dsrl
cd dppo
pip install -e .
pip install -e .[robomimic]
pip install -e .[gym]
cd ..
cd stable-baselines3
pip install -e .
cd ..

# Install Custom ManiSkill version
cd ../../
pip install -e simulation/ManiSkill/
python simulation/ManiSkill/download_and_extract_xsim.py "https://cornell.box.com/shared/static/qjfltim1ca96co8zdkhswes9cdnpbiap.zip"
