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

# install BayesMonSTR-ATAC
echo "installing BayesMonSTR-ATAC..."
conda env create -f BayesMonSTR-ATAC/environment.yml
conda activate BayesMonSTR-ATAC
pip install -e BayesMonSTR-ATAC
echo "BayesMonSTR-ATAC installed"