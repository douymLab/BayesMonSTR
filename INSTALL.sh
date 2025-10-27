# install MoSTR-Bayes

echo "installing MoSTR-Bayes..."
cd MoSTR-Bayes/
pip install python/
echo "MoSTR-Bayes installed"

# install MoSTR-Bulk
echo "installing MoSTR-Bulk..."
cd ../
conda env create -f MoSTR-Bulk/environment.yml
echo "MoSTR-Bulk installed"
