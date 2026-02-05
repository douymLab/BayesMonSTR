# install BayesMonSTR
echo "installing BayesMonSTR..."
cd BayesMonSTR/
pip install python/
echo "BayesMonSTR installed"

# install BayesMonSTR-BulkMonSTR
echo "installing BayesMonSTR-BulkMonSTR..."
cd ../
conda env create -f BayesMonSTR-BulkMonSTR/environment.yml
echo "BayesMonSTR-BulkMonSTR installed"

# install BayesMonSTR-ATAC
echo "installing BayesMonSTR-ATAC..."
conda env create -f BayesMonSTR-ATAC/environment.yml
conda activate bayesmonstr-atac
pip install -e BayesMonSTR-ATAC
echo "BayesMonSTR-ATAC installed"
