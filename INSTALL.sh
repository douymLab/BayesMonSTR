# install MonSTR-Bayes

echo "installing MonSTR-Bayes..."
cd MonSTR-Bayes/
pip install python/
echo "MonSTR-Bayes installed"

# install MoSTR-Bulk
echo "installing MoSTR-Bulk..."
cd ../
conda env create -f MoSTR-Bulk/environment.yml
echo "MoSTR-Bulk installed"
