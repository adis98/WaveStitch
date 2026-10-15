# WaveStitch: Flexible and Fast Conditional Time Series Generation With Diffusion Models

**WaveStitch** is a deep generative framework for **conditional time series synthesis**. It enables the generation of realistic time series data conditioned on auxiliary features (e.g., labels, metadata) and signal anchors (e.g., partial observations). This codebase provides tools for experimentation and ablation studies.

## File Structure
- **TSImputers/**  
  Shared neural network backbones used in diffusion and TimeGAN models (integrates components from SSSD and TimeGAN codebases)

- **experiments/**  
  Results for experiments featured in the paper (e.g., parallelism vs autoregressive, encoding ablations, SOTA comparisons)

- **utils/**  
  Helper functions for the diffusion model backbone

- **synthesis_*.py**  
  Scripts for generating synthetic time series data (includes preconditioning models, TimeGAN, TimeWeaver, TSDiff, WaveStitch with repaint-based conditioning)

- **training_*.py**  
  Scripts for training different model types (e.g., TimeGAN, TimeWeaver, WaveStitch, etc.)

- **ablation_*.py / ablation_*.sh**  
  Pairs of scripts for ablation studies. `.sh` files generate synthetic data, and corresponding `.py` files analyze results and store them in CSVs. (To avoid regenerating data, run analysis scripts first, then modify shell scripts to only generate missing data)

- **\*_analysis.py / \*_plotter.py**  
  Scripts for analyzing synthetic data (autocorrelation, cross-correlation, etc.) and visualizing experiment results (separate from ablation studies)

- **\*.sh**  
  Shell scripts for orchestrating runs, including training all backbones

- **WaveStitch_Appendix.pdf**  
  Contains algorithm details and illustrations for repaint-based WaveStitch implementation

- **generated/**  
  Directory for storing generated data during experiments  
  - **[dataset name]/**  
    - **c/** (Coarse-grained tasks - Root-level conditions)  
    - **m/** (Medium-grained tasks - Intermediate-level conditions)  
    - **f/** (Fine-grained tasks - Bottom-level conditions)

- **saved_models/**  
  Directory where trained models are saved after running training scripts

- **data_utils.py**  
  Preprocessing functions for each dataset

- **metasynth.py**  
  Generates tasks from datasets for different experiment configurations

## Reproducing the results

The experiments in this repository were conducted using **Python 3.12, with PyTorch 2.2.2 + CUDA 12.1**. Our setup uses a **Ubuntu 20.04 LTS** with a **RTX 3090 NVIDIA GPU**. We recommend using the same Python version to reproduce the reported results.

### 1. Environment Setup

Clone the repository and create a Python environment:

```bash
git clone https://github.com/adis98/WaveStitch.git
cd WaveStitch

python[VERSION] -m venv venv
source venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
> **Note:** Exact reproducibility may depend on the installed PyTorch/CUDA versions, GPU architecture, random seeds, and the availability of the original datasets.

### 2. Dataset Preparation

Before running the experiments, download and prepare the required datasets.
Here are the available links:
1. Panama Energy: https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama
2. Beijing Air Quality: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
3. Australia Tourism: https://www.kaggle.com/datasets/luisblanche/quarterly-tourism-in-australia
4. Rossman Sales: https://www.kaggle.com/competitions/rossmann-store-sales


The expected data directory structure is:

```text
WaveStitch/
├── PanamaEnergy/
│   └── continuous dataset.csv
├── RossmanSales/
│   └── train.csv
		..
├── QuarterlyTourismAustralia/
│   └── tourism.csv
├── MetroInterstateTrafficVolume/
│   └── Metro_Interstate_Traffic_Volume.csv/
└── BeijingAirQuality/
	└── beijing+multi+site+air+quality+data/
    ├── PRSA_Data_Aotizhongxin_...
    ├── ...
    └── PRSA_Data_Wanshouxigong_...
```

Dataset-specific preprocessing is implemented in `data_utils.py`.

The task configurations used by the experiments are generated using:

```bash
python metasynth.py
```

Before training the model, please install the Cauchy multiplication extensions available here: https://github.com/state-spaces/s4

Place the directory named "extensions" within the project root: 
```text
WaveStitch/
├── extensions/
```

### 3. Running the Experiments

The repository contains scripts for training the models, generating synthetic data, and analysing the results.

#### Training

Training scripts are named `training_*.py`. For example:

```bash
python training_wavestitch.py
```

Other training scripts can be used to reproduce the baseline models:

```bash
python training_timegan.py
python training_timeweaver.py
...
```

Trained models are stored in:

```text
saved_models/
```

#### Synthetic Data Generation

Synthetic data can be generated using the corresponding `synthesis_*.py` scripts. For example:

```bash
python synthesis_wavestitch_pipeline_strided_preconditioning.py
```

Generated data are stored in:

```text
generated/
└── [dataset name]/
    ├── c/
    ├── m/
    └── f/
```

where `c`, `m`, and `f` correspond to coarse-, medium-, and fine-grained conditioning tasks, respectively.

### 4. Reproducing the Paper Experiments

The experiments reported in the paper can be reproduced using the provided shell scripts.

For example:

```bash
bash ablation_[EXPERIMENT].sh
```

The shell scripts generate the required synthetic data, while the corresponding Python scripts analyse the results and store the resulting metrics in CSV files.

For example:

```bash
bash ablation_[EXPERIMENT].sh
python ablation_[EXPERIMENT].py
```

**Important:** The `.sh` scripts may regenerate synthetic data that already exist in `generated/`. To avoid unnecessarily repeating expensive generation, run the corresponding analysis script first. If the required generated data are already available, modify the shell script to skip those generation commands and run only the missing experiments.

### 5. Analysis and Plotting

Analysis scripts are provided separately from the ablation scripts.

Examples include:

```bash
python [analysis_script].py
python [plotter_script].py
```

These scripts compute metrics such as autocorrelation and cross-correlation and generate the plots used to evaluate the models.

The analysis scripts generally operate on the generated data stored in `generated/`.

### 6. Recommended Reproduction Order

For a clean reproduction of an experiment, we recommend following this order:

1. **Set up the Python environment** using the specified Python and PyTorch versions.
2. **Prepare the datasets** using the instructions above.
3. **Generate the experiment configurations** using `metasynth.py`, if required.
4. **Train the required models** using the appropriate `training_*.py` scripts.
5. **Generate synthetic data** using the corresponding `synthesis_*.py` scripts.
6. **Run the analysis scripts** to compute the reported metrics.
7. **Run the plotting scripts** to reproduce the figures.
8. **Run the ablation scripts** only when the corresponding generated data are not already available.

### 8. Experiment Configuration

The main experiment settings, including model configurations, conditioning tasks, and evaluation settings, are specified in the corresponding Python and shell scripts.

The repository is organised so that each experiment can be traced through the following pipeline:

```text
Dataset
   ↓
metasynth.py
   ↓
training_*.py
   ↓
saved_models/
   ↓
synthesis_*.py
   ↓
generated/
   ↓
*_analysis.py
   ↓
CSV results
   ↓
*_plotter.py
   ↓
Figures
```


