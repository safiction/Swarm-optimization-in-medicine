# Diabetes Detection using Particle Swarm Optimization (PSO) for Feature Selection

## Project Description
This project aims to improve diabetes detection by applying PSO for feature selection. The goal is to enhance prediction performance while reducing the number of features in the dataset.

## Objectives
- Build baseline machine learning models using all features;
- Apply PSO to select optimal feature subsets;
- Compare model performance before and after feature selection;
- Analyze the stability of selected features across multiple runs.

---

## Dataset
We use the **2023 BRFSS Diabetes Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/spandanjit2005/brfss-diabetes-dataset/data

### Dataset Characteristics:
- ~416,000 instances
- 27 features
- Target variable: `DIABETES_STATUS` (binary classification)
  - 0: No diabetes  
  - 1: Diabetes / Prediabetes / Pregnancy diabetes  

## Methods
### Baseline Models
- Logistic Regression  
- Random Forest  
- CatBoost  

### Feature Selection
- Particle Swarm Optimization (PSO)
- Binary encoding of feature subsets
- Fitness function:
  - Maximizes model performance (F1-score)
  - Penalizes large feature subsets

### Evaluation Metrics
- Accuracy  
- F1-score 
- Precision  
- Recall  

### Additional Analysis
- Feature reduction impact
- Stability of selected features
- Performance comparison before and after PSO
- Data preprocessing and cleaning;
- Baseline ML models (Logistic Regression, Random Forest, Catboost);
- Particle Swarm Optimization (PSO) for feature selection;
- Model evaluation and comparison;
- Stability analysis.
---

## Tech Stack
- Python 3.11
- Jupyter Notebook (Anaconda)
- scikit-learn
- CatBoost
- NumPy, Pandas
- Matplotlib, Seaborn

## How to Run using Docker (recommended)
1. Clone the repository
```bash
git clone https://github.com/safiction/Swarm-optimization-in-medicine.git
cd Swarm-optimization-in-medicine
```

2. Build image:
```bash
docker build -t pso-medical-project .
```

3. Run container:
```bash
docker run -it pso-medical-project
```

4. Explore the PSO algorithm in the `notebooks/PSO_analysis.ipynb` file (run cells inside, change parameters)

## Option 2: Local Setup
1. Clone the repository
```bash
git clone https://github.com/safiction/Swarm-optimization-in-medicine.git
cd Swarm-optimization-in-medicine

pip install -r requirements.txt
```

2. Run baseline models:
```bash
python notebooks/baseline_models.py
```

3. Run PSO experiments:
```bash
python notebooks/PSO_analysis.ipynb
```

## Project Structure
```text
Swarm-optimization-in-medicine/
│
├── data/
│   ├── raw/                # Original dataset (2023 BRFSS)
│   └── processed/          # Cleaned & split data
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── baseline_models.py
│   └── PSO_analysis.ipynb
│
├── src/
│   ├── models.py           # ML models
│   ├── evaluation.py       # Metrics
│   ├── preprocessing.py    # Data processing
│   ├── pso.py              # PSO implementation
│   └── pso_algorithm.py    # Core PSO logic
│
├── results/
│   ├── figures/            # Visualizations
│   │   ├── baseline_vs_pso_accuracy.png
│   │   ├── baseline_vs_pso_f1.png
│   │   └── pso_feature_stability_bar.png
│   │
│   └── metrics/            # Experimental results
│       ├── baseline_results.csv
│       ├── pso_experiments.csv
│       ├── baseline_vs_pso_comparison.csv
│       └── pso_stability.csv
│
├── scripts/
│   └── run_preprocessing.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Team Members and Roles
- __Elvina__ – EDA, data analysis, stability analysis;
- __Safina__ – preprocessing, PSO implementation, repository management;
- __Ekaterina__ – research, baseline models, documentation.

## Project Timeline
- Feb 28 – Mar 7: Data preparation & EDA
- Mar 8 – Mar 20: Baseline models
- Mar 21 – Apr 5: PSO implementation
- Apr 6 – Apr 14: Stability analysis & final report

## Expected Outcome
- Improved model performance;
- Reduced feature set;
- Analysis of feature stability.

## Results
- PSO successfully reduced the number of features
- Comparable performance was achieved with fewer inputs
- Feature selection showed stability across multiple runs
- CatBoost consistently achieved strong performance

## Key Insights
- Feature reduction does not significantly degrade performance
- PSO effectively balances performance and model simplicity
- Ensemble models benefit most from optimized feature subsets

## License
This project is for academic purposes only.