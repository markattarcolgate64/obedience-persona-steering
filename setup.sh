set -e

echo "setting up obedience persona vectors repo"

if [ -d revenv ]; then
    echo "revenv exists"
else 
    echo "creating revenv"
    python3 -m venv revenv
fi

echo "activating revenv"
source revenv/bin/activate

pip install --upgrade pip

echo "installing python packages"
pip install -r requirements.txt

echo "checking GPU availability"
python3 -c "import torch; print(f'cuda available: {torch.cuda.is_available()}\ngpu count: {torch.cuda.device_count()}')"
