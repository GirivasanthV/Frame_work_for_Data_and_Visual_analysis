# Loan Prediction Analysis Dashboard

A comprehensive, data-driven framework for assessing the likelihood of loan approval based on applicant demographic, financial, and credit-related attributes. This project leverages machine learning algorithms and provides an interactive web dashboard for loan prediction and analysis.

## Features

### 🤖 Machine Learning Models
- **Decision Tree Classifier**: Interpretable tree-based model
- **Random Forest Classifier**: Ensemble method for improved accuracy
- **Logistic Regression**: Statistical approach for binary classification

### 📊 Interactive Dashboard
- **Real-time Predictions**: Input applicant data and get instant loan approval predictions
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Data Insights**: Comprehensive analysis of loan approval patterns
- **Feature Importance**: Understanding which factors matter most

### 🎯 Key Analytics
- Loan approval distribution analysis
- Income vs loan amount correlations
- Credit history impact assessment
- Property area and education factor analysis
- Model performance comparisons

## Project Structure

```
loan-prediction-analysis/
├── train_model.py          # Model training and evaluation script
├── backend.py              # Flask web application
├── train.csv              # Training dataset
├── loan_model.joblib      # Trained model (generated after training)
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Main dashboard template
├── static/
│   ├── styles.css         # Dashboard styling
│   └── loan_analysis_dashboard.png  # Generated visualizations
└── README.md              # This file
```

## Installation & Setup

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd loan-prediction-analysis

# Or download and extract the files
```

### 2. Install Dependencies
```bash
# Install required Python packages
pip install -r requirements.txt
```

### 3. Train the Model
```bash
# Run the training script to generate the model
python train_model.py
```

This will:
- Load and preprocess the training data
- Train multiple ML models (Decision Tree, Random Forest, Logistic Regression)
- Generate comprehensive visualizations
- Save the best model as `loan_model.joblib`

### 4. Start the Web Application
```bash
# Launch the Flask dashboard
python backend.py
```

The dashboard will be available at: `http://localhost:5000`

## Using the Dashboard

### 📈 Dashboard Tab
- View key statistics (total applications, approval rate, average amounts)
- Interactive charts showing loan approval patterns
- Distribution analysis by various factors

### 🔮 Loan Prediction Tab
- Fill out the loan application form
- Get instant predictions with confidence scores
- See approval probability percentages

### 📊 Analytics Tab
- Advanced scatter plots and correlation analysis
- Feature importance rankings
- Model performance metrics

### 💡 Insights Tab
- Key findings from the data analysis
- Recommendations for loan applicants
- Factor-wise impact analysis

## Model Features

The system uses the following features for prediction:

### Demographic Features
- **Gender**: Male/Female
- **Married**: Yes/No
- **Dependents**: 0, 1, 2, 3+
- **Education**: Graduate/Not Graduate
- **Self_Employed**: Yes/No

### Financial Features
- **ApplicantIncome**: Primary applicant's income
- **CoapplicantIncome**: Co-applicant's income (if any)
- **LoanAmount**: Requested loan amount (in thousands)
- **Loan_Amount_Term**: Loan term in months

### Credit & Property Features
- **Credit_History**: 1 (has credit history) / 0 (no credit history)
- **Property_Area**: Urban/Semiurban/Rural

### Engineered Features
- **TotalIncome**: Combined income of applicant and co-applicant
- **Income_LoanAmount_Ratio**: Income to loan amount ratio
- **LoanAmount_per_Term**: Loan amount per term ratio
- **Log transformations**: For handling skewed distributions

## Model Performance

The system trains and compares three models:
- **Random Forest**: Typically achieves highest accuracy (~80-85%)
- **Decision Tree**: Good interpretability (~75-80%)
- **Logistic Regression**: Fast and reliable baseline (~78-82%)

Performance metrics include:
- Cross-validation accuracy
- Test set accuracy
- AUC-ROC scores
- Confusion matrices

## Key Insights

Based on the analysis, the most important factors for loan approval are:

1. **Credit History**: Most significant predictor
2. **Total Income**: Higher income increases approval chances
3. **Loan Amount**: Reasonable amounts relative to income
4. **Property Area**: Semiurban properties have highest approval rates
5. **Education**: Graduate status positively impacts approval
6. **Marital Status**: Married applicants have higher approval rates

## Technical Details

### Data Preprocessing
- Missing value imputation (mode for categorical, median for numerical)
- Feature engineering (income ratios, log transformations)
- Label encoding for categorical variables
- Feature scaling using StandardScaler

### Model Training
- Stratified train-test split (80-20)
- 5-fold cross-validation
- Hyperparameter tuning capabilities
- Model comparison and selection

### Web Framework
- Flask backend with RESTful API endpoints
- Responsive HTML/CSS frontend
- Interactive Plotly visualizations
- Real-time prediction capabilities

## API Endpoints

- `GET /`: Main dashboard page
- `POST /predict`: Loan prediction API
- `GET /api/insights`: Data insights JSON
- `GET /api/plots`: Plot data JSON

## Customization

### Adding New Features
1. Update the feature list in `train_model.py`
2. Modify the preprocessing pipeline
3. Update the web form in `templates/index.html`
4. Adjust the prediction function in `backend.py`

### Model Improvements
- Add more sophisticated models (XGBoost, Neural Networks)
- Implement hyperparameter tuning
- Add ensemble methods
- Include more feature engineering

### Dashboard Enhancements
- Add more interactive filters
- Include time-series analysis
- Add export functionality
- Implement user authentication

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Make sure to run `python train_model.py` first
   - Check if `loan_model.joblib` exists

2. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.7+ is installed

3. **Port already in use**
   - Change the port in `backend.py`: `app.run(port=5001)`
   - Or kill the process using the port

4. **Visualization not loading**
   - Check browser console for JavaScript errors
   - Ensure Plotly CDN is accessible

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Contact

For questions, suggestions, or issues, please create an issue in the repository or contact the development team.

---

**Note**: This is a demonstration project for educational purposes. For production use, additional validation, security measures, and testing would be required.