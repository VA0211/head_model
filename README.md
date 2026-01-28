# Prediction of tissue-of-origin of early-stage cancers using serum miRNomes

This repository contains code for the paper *Prediction of tissue-of-origin of early-stage cancers using serum miRNomes*.

### Directory organization

- `README.md`: This file.
- `LICENSE`: License file.
- `head/`: Code for HEAD model.
- `preprocess/`: Code for data preprocess and DANN model.

For the usage of code in the subdirectories, see `README.md` therein.
---
# Set up project
**1. Set up env**
1. Download Python 3.10.11 from [website](https://www.python.org/downloads/release/python-31011/)

2. Powershell
```
# syntax: & "<path_to_python310>" -m venv <venv_name>
& "C:\Users\ldva0\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv_310
```
3. Activate env 
```
# Powershell syntax: .\<venv_name>\Scripts\Activate.ps1
.\venv_310\Scripts\Activate.ps1

# Bash syntax: source <venv_name>Scripts/activate
source venv_310/Scripts/activate
```
4. Install pkgs for training (Powershell)
```
pip install "numpy<2.0"
pip install cupy-cuda12x
# Moving the chainer (no longer supported) to using torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
```
cd head
pip install -r requirements.txt
```
or 
```
pip install tables scikit-learn scipy xgboost
```
5. Install pkgs for data preprocessing (Powershell) then check the `README.md` file in `preprocess` folder for details
```
cd preprocess
pip install -e .
```
