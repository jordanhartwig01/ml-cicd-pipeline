# Machine Learning CI/CD Pipeline Tutorial

[![ML CI/CD Pipeline](https://github.com/YOUR_USERNAME/ml-cicd-pipeline/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/ml-cicd-pipeline/actions/workflows/ml-pipeline.yml)

## 📚 Overview

This project demonstrates a complete Machine Learning Continuous Integration and Deployment (CI/CD) pipeline using GitHub Actions. The pipeline automatically runs tests, trains a model, and saves it as an artifact whenever code is pushed to the repository.

## 🎯 Assignment Requirements Met

✅ **Git Version Control**: All code is stored in GitHub with version control  
✅ **Automated Pipeline**: GitHub Actions executes the complete process from data processing to model deployment  
✅ **Unit Tests**: Comprehensive unit tests for data processing functions  
✅ **Model Deployment**: Trained model is saved as a GitHub artifact  

## 🏗️ Project Structure

```
ml-cicd-pipeline/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml      # GitHub Actions workflow definition
├── src/
│   ├── data_processing.py       # Data loading and preprocessing
│   └── train_model.py           # Model training and evaluation
├── tests/
│   └── test_data_processing.py  # Unit tests for data processing
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 How It Works

### Pipeline Stages

1. **Test Stage**
   - Runs all unit tests using pytest
   - Validates data processing functions
   - Generates test report
   - Uploads test results as artifact

2. **Train Stage** (only runs if tests pass)
   - Loads synthetic data (3-class classification problem)
   - Preprocesses data (validation, splitting, scaling)
   - Trains Random Forest classifier
   - Evaluates model performance
   - Saves trained model as artifact

### What the Code Does

**Data Processing (`data_processing.py`)**:
- Generates synthetic data similar to the Iris dataset
- Validates data for quality (no NaN, correct shapes, etc.)
- Splits data into training and test sets
- Applies standard scaling to features

**Model Training (`train_model.py`)**:
- Trains a Random Forest classifier
- Evaluates model on test data
- Requires minimum 70% accuracy
- Saves model and scaler using joblib

**Unit Tests (`test_data_processing.py`)**:
- Tests data loading functionality
- Tests data validation rules
- Tests preprocessing pipeline
- Tests data statistics computation
- 20+ test cases ensuring code quality

## 📦 Setup Instructions

### Step 1: Fork or Clone This Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-cicd-pipeline.git
cd ml-cicd-pipeline
```

### Step 2: Set Up Local Environment (Optional)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run Locally (Optional)

```bash
# Run unit tests
pytest tests/ -v

# Train model
python train_model.py
```

### Step 4: Push to GitHub

```bash
# Make sure you're on the main branch
git checkout -b main

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: ML CI/CD pipeline"

# Push to GitHub
git push -u origin main
```

## 🔄 GitHub Actions Pipeline

Once you push to GitHub, the pipeline automatically runs:

1. **View Pipeline Status**:
   - Go to your repository on GitHub
   - Click on "Actions" tab
   - You'll see the pipeline running

2. **Access Artifacts**:
   - Click on a completed workflow run
   - Scroll down to "Artifacts" section
   - Download `trained-model` artifact (contains model.pkl)
   - Download `test-report` artifact (contains test results)

### Workflow File Explained

The `.github/workflows/ml-pipeline.yml` file defines the pipeline:

```yaml
# Triggers: runs on push to main or pull requests
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

# Two jobs: test and train
jobs:
  test:
    # Runs unit tests first
    
  train:
    needs: test  # Only runs if tests pass
    # Trains and saves model
```

## 📊 Understanding the Results

### Test Results

The pipeline runs 20+ unit tests checking:
- Data loading correctness
- Data validation rules
- Preprocessing functionality
- Edge cases and error handling

### Model Performance

The trained model:
- Uses Random Forest algorithm
- Trains on 120 samples (80%)
- Tests on 30 samples (20%)
- Typically achieves >90% accuracy
- Must achieve >70% to pass

### Artifacts

Two artifacts are produced:
1. **trained-model**: Contains the serialized model (model.pkl)
2. **test-report**: Contains detailed test execution results

## 🔧 Customization Options

### Modify the Data

Edit `data_processing.py` to load your own data:

```python
def load_data():
    # Replace with your data loading logic
    # For example, load from CSV:
    # df = pd.read_csv('your_data.csv')
    # X = df.drop('target', axis=1).values
    # y = df['target'].values
    # return X, y
```

### Change the Model

Edit `train_model.py` to use a different algorithm:

```python
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
```

### Add More Tests

Add new test files to the `tests/` directory. They will automatically be discovered and run.

## 🐛 Troubleshooting

### Pipeline Fails on First Run

**Problem**: GitHub Actions may need permissions to run.

**Solution**: 
1. Go to repository Settings → Actions → General
2. Under "Workflow permissions", select "Read and write permissions"
3. Click Save

### Tests Fail Locally but Pass on GitHub

**Problem**: Different Python versions or missing dependencies.

**Solution**:
```bash
# Ensure you're using Python 3.9
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Model Artifact Not Created

**Problem**: Training job didn't complete successfully.

**Solution**:
- Check the Actions tab for error messages
- Ensure tests pass first (train job depends on test job)
- Verify `model.pkl` is created after running locally

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [pytest Documentation](https://docs.pytest.org/)

## 🎓 Learning Outcomes

By completing this project, you will understand:

1. **Version Control**: How to use Git and GitHub for ML projects
2. **CI/CD Pipelines**: How to automate testing and deployment
3. **Unit Testing**: How to write and run tests for ML code
4. **ML Workflows**: Complete pipeline from data to deployed model
5. **Artifact Management**: How to store and version ML models

## 📝 Assignment Checklist

Use this checklist to verify you've met all requirements:

- [ ] Code is stored in GitHub repository
- [ ] GitHub Actions workflow file exists (`.github/workflows/ml-pipeline.yml`)
- [ ] Pipeline runs automatically on push
- [ ] Unit tests are included (`tests/test_data_processing.py`)
- [ ] Tests run as part of pipeline
- [ ] Model training is automated
- [ ] Trained model is saved as artifact
- [ ] Pipeline has run successfully at least once
- [ ] Can download model artifact from GitHub

## 📧 Questions?

If you encounter issues:
1. Check the GitHub Actions logs for error messages
2. Verify all files are in correct locations
3. Ensure Python version matches (3.9)
4. Check that dependencies install correctly

## 📄 License

This project is for educational purposes.

---

**Note**: Remember to replace `YOUR_USERNAME` in the URLs with your actual GitHub username!
