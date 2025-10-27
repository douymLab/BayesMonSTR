# install MoSTR-SC
echo "installing MoSTR-SC..."
cd MoSTR-SC/
pip install python/
echo "MoSTR-SC installed"

# install MoSTR-Bulk
echo "installing MoSTR-Bulk..."
cd ../
conda env create -f MoSTR-Bulk/environment.yml
echo "MoSTR-Bulk installed"
