pip install -r requirements.txt
cd dust3r
pip install -r requirements.txt
cd ..

apt update
apt install -y build-essential cmake git libpython3-dev
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch