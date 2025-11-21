# Installation Guide for INSIGHT

## Prerequisites

- **Python**: Version 3.8 or higher
- **R**: Version 4.4.0 or higher

## 1. Install Python Dependencies

You can install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

2. Install R Dependencies
INSIGHT uses R for generating visualizations. You need to install the following R packages.

Option A: Using Conda (Recommended)
If you are using Conda, you can install R and the required packages from the conda-forge channel:

```bash
conda install -c conda-forge r-base r-ggplot2 r-dplyr r-tidyr r-purrr r-readr r-gtable r-ragg
```

Option B: Using R Console
Alternatively, you can install them directly within R:

```R
install.packages(c("ggplot2", "dplyr", "tidyr", "purrr", "readr", "gtable", "ragg"))
```

3. Install INSIGHT
Once the dependencies are ready, you can install the INSIGHT package.

Standard Installation
To install the package into your current Python environment:

```bash
pip install .
```

Development Installation
If you plan to modify the code, install it in editable mode:

```bash
pip install -e .
```

4. Verification
After installation, verify that the command-line tools are available:

```r
bayesmonstr2insights --help
cf2insights --help
```
