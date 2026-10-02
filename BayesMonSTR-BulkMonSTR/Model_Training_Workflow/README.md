# Model Training Workflow

BayesMonSTR-BulkMonSTR uses random forest (RF) classifiers to distinguish mosaic mutations from
germline heterozygous variants and artifacts among candidate STR mutations. The default models in
`../model/` were trained on Illumina data, and models trained on Element data are also provided
(`../model/Element_RF_indel.pkl`, `../model/Element_RF_mismatch.pkl`).

If your data were generated under different conditions (sequencing platform, instrument, library
preparation, read length or laboratory), you can train classifiers on your own labeled data with the
same features and training recipe, and use them in place of the default models. Two models are
trained, one for indels and one for mismatches.

```
Model_Training_Workflow/
├── README.md
├── 01_build_model_features.py     # compute the model features of labeled loci
├── 02_train_model.py              # train and evaluate an RF model
└── example/
    ├── manifest_example.tsv
    └── labels_example.tsv
```

## Requirements
- BayesMonSTR-BulkMonSTR (the same version for training and prediction)
- Python 3 with pandas, numpy, scipy and scikit-learn
- Sequencing data for which the true status of candidate loci can be determined (see Step 2)

## Overview
```
Step 0  stutter_model_estimation.py   estimate locus-specific stutter error rates from your samples
Step 1  BulkMonSTR workflow           run up to the feature output (extract_features.py)
Step 2  labels                        label candidate loci (Mosaic mutations / Germline Het / Artifacts)
Step 3  01_build_model_features.py    compute the model features of the labeled loci
Step 4  02_train_model.py             train and evaluate the indel and mismatch models
Step 5  BulkMonSTR_prediction.py      use the new models; evaluate them on an independent sample
```

## Step 0. Estimate the stutter model
BulkMonSTR uses locus-specific stutter error rates both in genotyping and as classifier features.
Estimate the stutter model with `src/stutter_model_estimation.py` from samples sequenced under the same
conditions as your data (see "Estimation of locus-based stutter model" in `../README.md`), and use it
in all subsequent runs, for training and for prediction.

## Step 1. Run BulkMonSTR up to feature extraction
Run the BulkMonSTR workflow (`../README.md`) on the training samples up to feature extraction. The
feature output of `src/extract_features.py` (a headerless CSV file, the input of
`src/BulkMonSTR_prediction.py`) is used in Step 3.

## Step 2. Label candidate loci
Create one label file per sample (tab-delimited; see `example/labels_example.tsv`):

| column | content |
|---|---|
| `STR_ID` | STR locus id, as in the feature output (e.g. `Human_STR_134067`) |
| `Label` | `Mosaic mutations`, `Germline Het` or `Artifacts` |

All three classes are required. Labels must be based on evidence that is independent of the sample
being labeled. Useful designs include:

- **Pedigrees**: a mutant allele present in a parent (e.g. VAF > 0.2) indicates a germline variant;
  a mutant allele that is not enriched in the offspring relative to the parents indicates an artifact.
- **Orthogonal sequencing of the same sample** (another platform or library): a mutant allele that
  is absent from adequately covered orthogonal data (e.g. depth > 30) indicates an artifact; a mutant
  allele supported in the orthogonal data supports a mosaic mutation.
- **In silico mixtures of samples with known genotypes**, such as tumor and matched normal samples, or
  a single-cell-derived colony and its source cells: mutations validated in the orthogonal samples are
  mosaic mutations; variants with high VAFs (e.g. > 0.2) in both orthogonal samples are germline
  variants; mutant alleles at very low VAFs in both orthogonal samples are artifacts.

Recommendations:
- Leave loci with insufficient evidence unlabeled rather than forcing a label.
- Cover the range of VAFs, depths, motifs and repeat lengths that you intend to analyze.
- Label as many loci per class as possible (ideally thousands); with few mosaic mutations, the model
  will learn mainly to separate germline variants from artifacts.

## Step 3. Build the model features
List the training samples in a manifest (tab-delimited; see `example/manifest_example.tsv`):

| column | content |
|---|---|
| `sample_name` | sample identifier |
| `feature_file` | feature output of the sample (Step 1) |
| `label_file` | label file of the sample (Step 2) |

```sh
python3 ~/BayesMonSTR-BulkMonSTR/Model_Training_Workflow/01_build_model_features.py \
    --src ~/BayesMonSTR-BulkMonSTR/src/BulkMonSTR_prediction.py \
    --manifest manifest.tsv \
    --out-dir features/
```

The script executes the feature-engineering code of `BulkMonSTR_prediction.py` itself, so the
features are identical to those computed at prediction time, and they are written in the order
expected by the prediction step.

Outputs:
- `features/model_features_indel.csv.gz`: `sample_name`, `STR_ID`, `Label` + indel model features
- `features/model_features_mismatch.csv.gz`: the same for mismatches
- `features/build_counts.tsv`: number of labeled loci and of loci with features per sample

## Step 4. Train the models
```sh
python3 ~/BayesMonSTR-BulkMonSTR/Model_Training_Workflow/02_train_model.py \
    --features features/model_features_indel.csv.gz --out models/MyData_RF_indel
python3 ~/BayesMonSTR-BulkMonSTR/Model_Training_Workflow/02_train_model.py \
    --features features/model_features_mismatch.csv.gz --out models/MyData_RF_mismatch
```

Training recipe (as for the provided models):
- mean imputation of missing values, followed by a random forest (`random_state = 666`);
- grid search over 50, 100 and 200 trees and a maximum depth of none, 10 or 20;
- 10-fold cross-validation, with accuracy as the selection score;
- an 80/20 train/test split and cross-validation folds grouped by `STR_ID`, so that the same locus
  (e.g. from several samples or depths) is never in both training and test data.

Options: `--test-size` (default 0.2), `--folds` (default 10), `--group-col` (default `STR_ID`),
`--threads` (default 8), and `--refit-all` to refit the selected model on all loci after evaluation.

Outputs (for `--out models/MyData_RF_indel`):

| file | content |
|---|---|
| `MyData_RF_indel.pkl` | trained model |
| `MyData_RF_indel_features.txt` | model features, in order |
| `MyData_RF_indel_grid_search.tsv` | cross-validation results |
| `MyData_RF_indel_test_metrics.tsv` | precision, recall and F1 score per class on the held-out loci |
| `MyData_RF_indel_test_confusion.tsv` | confusion matrix on the held-out loci |

Check the held-out metrics, in particular the precision and recall of `Mosaic mutations`.

## Step 5. Use and evaluate the models
Pass the new models to the prediction step in place of the default models:
```sh
python3 ~/BayesMonSTR-BulkMonSTR/src/BulkMonSTR_prediction.py \
-i your_feature_output.csv \
-o your_prediction_output.txt \
-l both \
-mm models/MyData_RF_mismatch.pkl \
-mid models/MyData_RF_indel.pkl \
-r your_reference_genome_FASTA_file \
-m both
```
(other parameters as described in `../README.md`).

The held-out loci come from the same samples as the training loci. Before using the models widely,
evaluate them on an independent sample that was not used for training, with orthogonal validation,
and compare them with the default models.

## Notes
- Use the same BayesMonSTR-BulkMonSTR version and the same stutter model for training and prediction;
  the model features depend on both.
- The saved models are scikit-learn objects; load them with the scikit-learn version used for training.
- Do not add or remove model features: the prediction step expects the features listed in
  `*_features.txt`, in that order.
