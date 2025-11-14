# install BayesMonSTR
echo "installing BayesMonSTR..."
cd BayesMonSTR/
pip install python/
echo "BayesMonSTR installed"

# install BayesMonSTR-Bulk
echo "installing BayesMonSTR-Bulk..."
cd ../
conda env create -f BayesMonSTR-Bulk/environment.yml
echo "BayesMonSTR-Bulk installed"
