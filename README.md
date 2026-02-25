
# Congressional Oratory Research Archive (CORA)

**Paper Title:**
*Congressional Oratory Research Archive, A Comprehensive Data Set and Platform for Exploring and Analyzing the U.S. Congressional Record, 1873 to 2025*

**Authors:**
Shahid Rabbani
New York University Abu Dhabi, Division of Social Science
Abu Dhabi, United Arab Emirates
Email: [sr3987@nyu.edu](mailto:sr3987@nyu.edu)

*Corresponding Author:*
Aaron R. Kaufman
New York University Abu Dhabi
Email: [aaronkaufman@nyu.edu](mailto:aaronkaufman@nyu.edu)



## Overview

This repository contains the data processing scripts, topic modeling pipeline, and validation materials for the paper:

**Congressional Oratory Research Archive, A Comprehensive Data Set and Platform for Exploring and Analyzing the U.S. Congressional Record, 1873 to 2025**

The Congressional Oratory Research Archive, CORA, is a large scale, structured corpus of speeches drawn from the United States Congressional Record covering the period 1873 to 2025. The project integrates historical speech data, legislator metadata, and computational text analysis to produce a unified research platform for the study of legislative rhetoric, political communication, and representation.

The repository is organised to reflect the three primary stages of data construction described in the paper, followed by technical validation procedures and topic labeling workflows.



## Data Construction Pipeline

The data construction process is implemented in three sequential stages. Each stage is documented in the paper and corresponds to a dedicated directory in the repository.

### Stage 1

Stage 1 compiles and structures foundational speech and legislator data. It integrates biographical identifiers and prepares raw speech records for downstream processing.

**Directory:** `Stage 1/`

* `Stage1.py`
* `BioID/bioguide_profiles.csv`
* `BioID/link.txt`



### Stage 2

Stage 2 expands and harmonises speech level data, linking textual content with legislator identifiers and metadata.

**Directory:** `Stage 2/`

* `Stage2.py`
* `BioID/bioguide_profiles.csv`
* `BioID/link.txt`



### Stage 3

Stage 3 finalises dataset construction, performs additional cleaning and consolidation, and prepares the full CORA dataset for topic modeling and validation.

**Directory:** `Stage 3/`

* `Stage3a.py`
* `Stage3b.py`
* `BioID/bioguide_profiles.csv`



## Topic Labeling and Model Training

The repository includes scripts for training and applying the topic labeling model used to classify speeches.

**Directory:** `Topic_Labeling/`

* `1_Model_Training.py`
  Trains the topic classification model using labeled training data.

* `2_inference.py`
  Applies the trained model to the full speech corpus.

* `3_Updated_speech_data.py`
  Updates the master dataset with predicted topic labels.

* `training_data.csv`
  Labeled training dataset used for model development.



## Technical Validation

The repository includes multiple validation procedures designed to evaluate the reliability and substantive validity of the CORA dataset and associated topic classifications.

**Directory:** `Technical Validation/`

### Human Validation

Human validation was conducted under two conditions:

1. **Blind labeling condition**
   A human annotator was provided speeches without model generated topic labels and asked to independently assign topic categories.
   Results are stored in:

   * `Human/Comparison.csv`

2. **Verification condition**
   A human annotator was provided speeches along with model generated topic labels and asked to verify the correctness of those labels.
   Results are stored in:

   * `Human/Model_labeled_speeches.xlsx`



### Validation 1

Validation 1 replicates patterns identified in:

*Gender, language, and representation in the United States Senate*
Leah Windsor, Sara McLaughlin Mitchell, Tracy Osborn, Bryce Dietrich, Andrew J. Hampton

Affiliations:
The University of Memphis
University of Iowa
Christian Brothers University

This validation assesses whether CORA reproduces established findings on gendered speech patterns in the United States Senate.

**Directory:** `Validation_1/`

Key files include:

* `1-Generate_Scores.py`
* `3-Generate_Plots.py`
* `Full_Data_Validation/`

  * `2-Data.py`
  * `2-Scores_Sign_correction_FULL.ipynb`
  * `3-Validate-Fig1_FULL.ipynb`
  * `cora_1873_2025.csv`
  * `cora_1989_2006.csv`
  * `wilson_1989_2006.csv`

These scripts reproduce comparative analyses using both CORA and external benchmark datasets.



### Validation 2

Validation 2 draws on:

*Computational analysis of US congressional speeches reveals a shift from evidence to intuition*
Segun T. Aroyehun, Almog Simchon, Fabio Carrella, Jana Lasser, Stephan Lewandowsky, David Garcia

This validation evaluates whether CORA captures shifts in rhetorical style between evidence based and intuition based language over time.

**Directory:** `Validation_2/`

* `1_Compute_EMI_Score.py`
* `2_Compare_party_EMI.py`
* `evidence.txt`
* `intuition.txt`



## Directory Structure

```
CORA-Code
├── Stage 1
├── Stage 2
├── Stage 3
├── Topic_Labeling
├── Technical Validation
└── readme.md
```

Each top level directory corresponds to a distinct component of the research workflow: data construction, model development, and empirical validation.



## Replication Instructions

To reproduce the dataset and validation analyses:

1. Execute Stage 1 scripts.
2. Execute Stage 2 scripts.
3. Execute Stage 3 scripts.
4. Train the topic model using `Topic_Labeling/1_Model_Training.py`.
5. Run inference using `2_inference.py`.
6. Execute validation scripts in `Technical Validation/Validation_1` and `Validation_2`.

Detailed parameter settings and computational specifications are described in the paper.



## Data Scope

CORA covers speeches from the United States Congressional Record from 1873 through 2025. The dataset integrates:

* Speech level text data
* Legislator biographical identifiers
* Topic classifications generated through supervised machine learning
* Validation outputs



## Citation

If you use this dataset or code, please cite:

Rabbani, Shahid, and Kaufman, Aaron R.
*Congressional Oratory Research Archive, A Comprehensive Data Set and Platform for Exploring and Analyzing the U.S. Congressional Record, 1873 to 2025.*

Full citation details will be updated upon publication.



## Contact

For questions regarding data construction, validation, or replication:

Shahid Rabbani
New York University Abu Dhabi
[sr3987@nyu.edu](mailto:sr3987@nyu.edu)

Aaron R. Kaufman
New York University Abu Dhabi
[aaronkaufman@nyu.edu](mailto:aaronkaufman@nyu.edu)
