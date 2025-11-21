# in-sight
Genome data visualization toolkit

## Installation

### Install from Source

```bash
# Method 1: Direct install via pip
pip install git+https://github.com/douymLab/INSIGHT.git
```

```bash
# Method 2: Manual installation (recommended for development)
git clone https://github.com/douymLab/INSIGHT.git
cd INSIGHT
pip install -e .
```

### R Dependencies

#### R Software
Ensure R (≥4.4.0) is installed

#### R Packages
Install required packages using one of these methods:

* **via conda (recommended):**
```bash
conda install r-base r-ggplot2=3.5.0 r-dplyr=1.1.0 r-tidyr r-purrr=1.0.0 r-readr=2.1.0 r-gtable=0.3.0 r-ragg=1.3.0 -c conda-forge
```

* **Automatic installation through rpy2:**
```bash
python -m rpy2.situation -m in_sight.r_script.plot_base.R
```

* **Manual installation:**
```bash
R -e "install.packages(c('ggplot2>=3.5.0','dplyr>=1.1.0','tidyr','purrr>=1.0.0','readr>=2.1.0','gtable>=0.3.0','ragg>=1.3.0'))"
```

## Basic Usage
```python
from in_sight import base_visualization

# Example usage
test = base_visualization(
    bam_path,
    'chr13:20222055-20222055',
    reference_fasta,
    ref_base="G",
    filter_tags={'CB': 'ACAATCGATCTTTATA-1'},
    r_script_path = "/storage/douyanmeiLab/xiayonghe/Projects/insight_SNP/in_sight/in_sight/r_script/plot_base.R",
    run_visualization = True
)
```

## Features
- Genome data visualization
- Multiple output formats support
- Integrated R plotting capabilities
- Batch processing support

## Documentation （TBD）
For detailed usage instructions, see our [documentation site](https://in-sight.readthedocs.io).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.

## License
[MIT](https://choosealicense.com/licenses/mit/)
