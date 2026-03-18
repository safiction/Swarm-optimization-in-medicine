# Diabetes Detection using Particle Swarm Optimization (PSO) for Feature Selection

## Project Description
This project aims to improve diabetes detection by applying PSO for feature selection. The goal is to enhance prediction performance while reducing the number of features in the dataset.

## Objectives
- Build baseline machine learning models using all features;
- Apply PSO to select optimal feature subsets;
- Compare optimized models with baseline performance;
- Analyze the stability of selected features.

## Dataset
We use the BRFSS Diabetes Dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/spandanjit2005/brfss-diabetes-dataset/data

### Dataset Characteristics:
- ~416000 instances
- 27 features
- Target variable: `DIABETES_STATUS` (multi-class)
  - 0: No diabetes  
  - 1: Prediabetes  
  - 2: Diabetes during pregnancy  
  - 3: Diabetes  

## Methods
- Data preprocessing and cleaning;
- Baseline ML models (Logistic Regression, Random Forest);
- Particle Swarm Optimization (PSO) for feature selection;
- Model evaluation and comparison;
- Stability analysis.

## Tech Stack
- Python
- Jupyter Notebook (Anaconda)
- scikit-learn
- NumPy, Pandas
- Matplotlib / Seaborn

## How to Run

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
jupyter notebook
```

## Project Structure
project/
│── data/
│── notebooks/
│── src/
│── results/
│── docs/

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

## License
This project is for academic purposes.