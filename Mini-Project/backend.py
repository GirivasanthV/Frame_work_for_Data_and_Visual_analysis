#!/usr/bin/env python3
"""
Loan Prediction Analysis - Flask Backend Application
Interactive web dashboard for loan approval prediction
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objs as go
import plotly.utils
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os

app = Flask(__name__)

class LoanPredictionApp:
    def __init__(self):
        self.model_package = None
        self.df = None
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """Load trained model and original data"""
        try:
            # Load model package
            if os.path.exists('loan_model.joblib'):
                self.model_package = joblib.load('loan_model.joblib')
                print("Model loaded successfully!")
            else:
                print("Model not found. Please run train_model.py first.")
            
            # Load original data for analysis
            if os.path.exists('train.csv'):
                self.df = pd.read_csv('train.csv')
                print("Data loaded successfully!")
            else:
                print("Training data not found.")
                
        except Exception as e:
            print(f"Error loading model or data: {e}")
    
    def predict_loan(self, applicant_data):
        """Predict loan approval for given applicant data"""
        if not self.model_package:
            return {"error": "Model not loaded"}
        
        try:
            # Extract components
            model = self.model_package['model']
            scaler = self.model_package['scaler']
            label_encoders = self.model_package['label_encoders']
            feature_columns = self.model_package['feature_columns']
            
            # Create DataFrame from input (data is already in INR from frontend)
            input_df = pd.DataFrame([applicant_data])
            
            # Convert INR values back to USD for model prediction (model was trained on USD data)
            usd_to_inr_rate = 83
            input_df['ApplicantIncome'] = input_df['ApplicantIncome'] / usd_to_inr_rate
            input_df['CoapplicantIncome'] = input_df['CoapplicantIncome'] / usd_to_inr_rate
            input_df['LoanAmount'] = input_df['LoanAmount'] / usd_to_inr_rate
            
            # Feature engineering (same as training)
            input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
            input_df['Income_LoanAmount_Ratio'] = input_df['TotalIncome'] / (input_df['LoanAmount'] + 1)
            input_df['LoanAmount_per_Term'] = input_df['LoanAmount'] / (input_df['Loan_Amount_Term'] + 1)
            input_df['Log_ApplicantIncome'] = np.log1p(input_df['ApplicantIncome'])
            input_df['Log_TotalIncome'] = np.log1p(input_df['TotalIncome'])
            input_df['Log_LoanAmount'] = np.log1p(input_df['LoanAmount'])
            
            # Encode categorical variables
            categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
            for feature in categorical_features:
                if feature in input_df.columns and feature in label_encoders:
                    # Handle unseen categories
                    try:
                        input_df[feature] = label_encoders[feature].transform(input_df[feature].astype(str))
                    except ValueError:
                        # Use most common class for unseen categories
                        input_df[feature] = 0
            
            # Select and order features
            X_input = input_df[feature_columns]
            
            # Scale features
            X_scaled = scaler.transform(X_input)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            prediction_proba = model.predict_proba(X_scaled)[0]
            
            # Convert back to original labels
            loan_status = label_encoders['Loan_Status'].inverse_transform([prediction])[0]
            
            return {
                "prediction": loan_status,
                "probability": {
                    "approved": float(prediction_proba[1]),
                    "rejected": float(prediction_proba[0])
                },
                "confidence": float(max(prediction_proba))
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def get_data_insights(self):
        """Generate data insights for dashboard"""
        if self.df is None:
            return {}
        
        insights = {}
        
        # Basic statistics (convert to INR for display)
        insights['total_applications'] = len(self.df)
        insights['approval_rate'] = (self.df['Loan_Status'] == 'Y').mean() * 100
        insights['avg_loan_amount'] = self.df['LoanAmount'].mean()  # Will be converted in template
        insights['avg_applicant_income'] = self.df['ApplicantIncome'].mean()  # Will be converted in template
        
        # Categorical distributions
        insights['gender_dist'] = self.df['Gender'].value_counts().to_dict()
        insights['education_dist'] = self.df['Education'].value_counts().to_dict()
        insights['property_area_dist'] = self.df['Property_Area'].value_counts().to_dict()
        insights['married_dist'] = self.df['Married'].value_counts().to_dict()
        
        return insights
    
    def create_interactive_plots(self):
        """Create interactive Plotly visualizations"""
        if self.df is None:
            return {}
        
        plots = {}
        
        # 1. Loan Status Distribution
        loan_counts = self.df['Loan_Status'].value_counts()
        fig1 = go.Figure(data=[go.Pie(
            labels=['Approved', 'Rejected'],
            values=[loan_counts.get('Y', 0), loan_counts.get('N', 0)],
            hole=0.4,
            marker_colors=['#2E8B57', '#DC143C']
        )])
        fig1.update_layout(title="Loan Approval Distribution", height=400)
        plots['loan_status_pie'] = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 2. Income vs Loan Amount Scatter (convert to INR for display)
        df_inr = self.df.copy()
        df_inr['ApplicantIncome_INR'] = df_inr['ApplicantIncome'] * 83
        df_inr['LoanAmount_INR'] = df_inr['LoanAmount'] * 83
        
        fig2 = px.scatter(
            df_inr, 
            x='ApplicantIncome_INR', 
            y='LoanAmount_INR',
            color='Loan_Status',
            title='Applicant Income vs Loan Amount (₹)',
            color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'},
            labels={'Loan_Status': 'Status', 'ApplicantIncome_INR': 'Applicant Income (₹)', 'LoanAmount_INR': 'Loan Amount (₹ thousands)'}
        )
        fig2.update_layout(height=500)
        plots['income_loan_scatter'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 3. Credit History Impact
        credit_cross = pd.crosstab(self.df['Credit_History'], self.df['Loan_Status'], normalize='index') * 100
        fig3 = go.Figure(data=[
            go.Bar(name='Rejected', x=['No Credit History', 'Has Credit History'], 
                   y=[credit_cross.loc[0, 'N'] if 0 in credit_cross.index else 0,
                      credit_cross.loc[1, 'N'] if 1 in credit_cross.index else 0],
                   marker_color='#DC143C'),
            go.Bar(name='Approved', x=['No Credit History', 'Has Credit History'], 
                   y=[credit_cross.loc[0, 'Y'] if 0 in credit_cross.index else 0,
                      credit_cross.loc[1, 'Y'] if 1 in credit_cross.index else 0],
                   marker_color='#2E8B57')
        ])
        fig3.update_layout(barmode='stack', title='Credit History Impact on Loan Approval', 
                          yaxis_title='Percentage', height=400)
        plots['credit_history_bar'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 4. Property Area Analysis
        property_cross = pd.crosstab(self.df['Property_Area'], self.df['Loan_Status'], normalize='index') * 100
        fig4 = go.Figure(data=[
            go.Bar(name='Rejected', x=property_cross.index, y=property_cross['N'], marker_color='#DC143C'),
            go.Bar(name='Approved', x=property_cross.index, y=property_cross['Y'], marker_color='#2E8B57')
        ])
        fig4.update_layout(barmode='stack', title='Property Area vs Loan Approval Rate', 
                          yaxis_title='Percentage', height=400)
        plots['property_area_bar'] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 5. Loan Amount Distribution (convert to INR)
        fig5 = go.Figure()
        
        # Get loan amounts and convert to INR, ensuring we have valid data
        approved_loans = self.df[self.df['Loan_Status'] == 'Y']['LoanAmount'].dropna()
        rejected_loans = self.df[self.df['Loan_Status'] == 'N']['LoanAmount'].dropna()
        
        if len(approved_loans) > 0 and len(rejected_loans) > 0:
            # Convert to INR
            approved_loans_inr = approved_loans * 83
            rejected_loans_inr = rejected_loans * 83
            
            # Create histograms with proper parameters
            fig5.add_trace(go.Histogram(
                x=approved_loans_inr, 
                name='Approved', 
                opacity=0.7, 
                marker_color='#2E8B57',
                nbinsx=25
            ))
            fig5.add_trace(go.Histogram(
                x=rejected_loans_inr, 
                name='Rejected', 
                opacity=0.7, 
                marker_color='#DC143C',
                nbinsx=25
            ))
            
            fig5.update_layout(
                title='Loan Amount Distribution by Status (₹)', 
                xaxis_title='Loan Amount (₹ thousands)', 
                yaxis_title='Frequency',
                barmode='overlay', 
                height=400,
                showlegend=True,
                bargap=0.1
            )
        else:
            # Fallback if no data
            fig5.add_annotation(
                text="No data available for loan amount distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig5.update_layout(title='Loan Amount Distribution by Status (₹)', height=400)
        
        plots['loan_amount_hist'] = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 6. Feature Importance (if available)
        if self.model_package and 'feature_importance' in self.model_package:
            feature_imp = self.model_package['feature_importance'].head(10)
            fig6 = go.Figure(go.Bar(
                x=feature_imp['importance'],
                y=feature_imp['feature'],
                orientation='h',
                marker_color='#4682B4'
            ))
            fig6.update_layout(title='Top 10 Feature Importance', 
                              xaxis_title='Importance', height=500)
            plots['feature_importance'] = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
        
        return plots

# Initialize the app
loan_app = LoanPredictionApp()

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        insights = loan_app.get_data_insights()
        plots = loan_app.create_interactive_plots()
        print(f"Insights keys: {list(insights.keys()) if insights else 'None'}")
        print(f"Plots keys: {list(plots.keys()) if plots else 'None'}")
        return render_template('index.html', insights=insights, plots=plots)
    except Exception as e:
        print(f"Error in index route: {e}")
        return f"Error: {e}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for loan prediction"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                          'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                          'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Make prediction
        result = loan_app.predict_loan(data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/insights')
def api_insights():
    """API endpoint for data insights"""
    insights = loan_app.get_data_insights()
    return jsonify(insights)

@app.route('/api/plots')
def api_plots():
    """API endpoint for plot data"""
    plots = loan_app.create_interactive_plots()
    return jsonify(plots)

@app.route('/debug')
def debug_plots():
    """Debug page for testing plots"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Debug Plots</title>
        <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        <style>
            .plot-container { width: 100%; height: 400px; border: 1px solid #ccc; margin: 10px 0; }
            .debug-info { background: #f0f0f0; padding: 10px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Debug: Loan Prediction Plots</h1>
        <div class="debug-info">
            <h3>Debug Information</h3>
            <p>Plotly loaded: <span id="plotly-status">Checking...</span></p>
            <p>API Status: <span id="api-status">Checking...</span></p>
            <p>Plots found: <span id="plots-count">0</span></p>
        </div>
        
        <div id="loan-status-pie" class="plot-container"></div>
        <div id="credit-history-bar" class="plot-container"></div>
        
        <script>
            // Check Plotly
            document.getElementById('plotly-status').textContent = typeof Plotly !== 'undefined' ? 'Yes' : 'No';
            
            if (typeof Plotly !== 'undefined') {
                console.log('Plotly version:', Plotly.version);
                
                // Fetch and render plots
                fetch('/api/plots')
                    .then(response => {
                        document.getElementById('api-status').textContent = response.status;
                        return response.json();
                    })
                    .then(plots => {
                        const plotNames = Object.keys(plots);
                        document.getElementById('plots-count').textContent = plotNames.length;
                        console.log('Available plots:', plotNames);
                        
                        // Render first two plots for testing
                        plotNames.slice(0, 2).forEach((plotName, index) => {
                            const elementId = plotName.replace(/_/g, '-');
                            const element = document.getElementById(elementId);
                            
                            if (element) {
                                try {
                                    const plotData = JSON.parse(plots[plotName]);
                                    console.log(`Rendering ${plotName}:`, plotData);
                                    
                                    Plotly.newPlot(elementId, plotData.data, plotData.layout || {}, {
                                        responsive: true
                                    }).then(() => {
                                        console.log(`✓ ${plotName} rendered successfully`);
                                        element.style.border = '2px solid green';
                                    }).catch(error => {
                                        console.error(`✗ ${plotName} render error:`, error);
                                        element.style.border = '2px solid red';
                                        element.innerHTML = `<p>Error: ${error.message}</p>`;
                                    });
                                    
                                } catch (error) {
                                    console.error(`Parse error for ${plotName}:`, error);
                                    element.style.border = '2px solid orange';
                                    element.innerHTML = `<p>Parse Error: ${error.message}</p>`;
                                }
                            }
                        });
                    })
                    .catch(error => {
                        console.error('Fetch error:', error);
                        document.getElementById('api-status').textContent = 'Error: ' + error.message;
                    });
            } else {
                document.body.innerHTML += '<p style="color: red;">Plotly failed to load!</p>';
            }
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("Starting Loan Prediction Analysis Dashboard...")
    print("Make sure to run train_model.py first to generate the model!")
    app.run(debug=True, host='0.0.0.0', port=5002)