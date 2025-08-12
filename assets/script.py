# Create comprehensive sample data for the NBFC analytics platform
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import uuid

# Set random seed for reproducibility
np.random.seed(42)

# 1. Create sample borrower data
n_borrowers = 1000

# Generate synthetic borrower data
borrower_data = {
    'borrower_id': [f'BRW{str(i).zfill(6)}' for i in range(1, n_borrowers + 1)],
    'age': np.random.normal(35, 10, n_borrowers).astype(int),
    'income': np.random.lognormal(10.5, 0.8, n_borrowers).astype(int),
    'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer'], n_borrowers, p=[0.4, 0.3, 0.2, 0.1]),
    'city_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n_borrowers, p=[0.3, 0.4, 0.3]),
    'education': np.random.choice(['Graduate', 'Post-Graduate', 'Undergraduate', 'Diploma'], n_borrowers, p=[0.4, 0.25, 0.25, 0.1]),
    'existing_loans': np.random.poisson(1.2, n_borrowers),
    'credit_history_months': np.random.exponential(24, n_borrowers).astype(int),
    'bank_account_age_months': np.random.exponential(36, n_borrowers).astype(int)
}

# Ensure realistic constraints
borrower_data['age'] = np.clip(borrower_data['age'], 18, 70)
borrower_data['income'] = np.clip(borrower_data['income'], 15000, 2000000)
borrower_data['credit_history_months'] = np.clip(borrower_data['credit_history_months'], 0, 120)
borrower_data['bank_account_age_months'] = np.clip(borrower_data['bank_account_age_months'], 3, 240)

borrower_df = pd.DataFrame(borrower_data)

print("Sample Borrower Data:")
print(borrower_df.head(10))
print(f"\nShape: {borrower_df.shape}")

# 2. Create alternative data features
alt_data = {
    'borrower_id': borrower_df['borrower_id'],
    'upi_transactions_monthly': np.random.poisson(45, n_borrowers),
    'utility_bill_payment_consistency': np.random.beta(8, 2, n_borrowers),  # Higher means more consistent
    'gst_returns_filed': np.random.binomial(12, 0.7, n_borrowers),  # Out of 12 months
    'mobile_recharge_frequency': np.random.poisson(6, n_borrowers),  # Per month
    'ecommerce_transactions_monthly': np.random.poisson(8, n_borrowers),
    'digital_wallet_balance_avg': np.random.lognormal(6, 1.5, n_borrowers).astype(int),
    'social_media_financial_mentions': np.random.poisson(2, n_borrowers),  # Risk indicator
    'app_usage_financial_minutes_daily': np.random.exponential(30, n_borrowers).astype(int),
    'location_stability_score': np.random.beta(7, 3, n_borrowers)  # Higher means more stable
}

alt_data_df = pd.DataFrame(alt_data)

print("\nSample Alternative Data:")
print(alt_data_df.head(10))

# 3. Create loan application data with target variable
def calculate_default_probability(row, alt_row):
    """Calculate default probability based on features"""
    risk_score = 0
    
    # Age factor (middle-aged lower risk)
    if 25 <= row['age'] <= 45:
        risk_score -= 0.1
    else:
        risk_score += 0.05
    
    # Income factor
    if row['income'] > 50000:
        risk_score -= 0.15
    elif row['income'] < 25000:
        risk_score += 0.2
    
    # Employment stability
    if row['employment_type'] == 'Salaried':
        risk_score -= 0.1
    elif row['employment_type'] == 'Self-Employed':
        risk_score += 0.05
    
    # Alternative data factors
    if alt_row['utility_bill_payment_consistency'] > 0.8:
        risk_score -= 0.1
    if alt_row['upi_transactions_monthly'] > 30:
        risk_score -= 0.05
    if alt_row['location_stability_score'] > 0.7:
        risk_score -= 0.08
    
    # Credit history
    if row['credit_history_months'] > 24:
        risk_score -= 0.1
    
    # Random factor
    risk_score += np.random.normal(0, 0.1)
    
    return max(0.02, min(0.95, 0.15 + risk_score))  # Bounded between 2% and 95%

# Calculate default probabilities and outcomes
default_probs = []
for i in range(n_borrowers):
    prob = calculate_default_probability(borrower_df.iloc[i], alt_data_df.iloc[i])
    default_probs.append(prob)

loan_data = {
    'loan_id': [f'LN{str(i).zfill(8)}' for i in range(1, n_borrowers + 1)],
    'borrower_id': borrower_df['borrower_id'],
    'loan_amount': np.random.lognormal(10, 0.8, n_borrowers).astype(int),
    'loan_tenure_months': np.random.choice([12, 18, 24, 36, 48], n_borrowers, p=[0.2, 0.2, 0.3, 0.2, 0.1]),
    'interest_rate': np.random.normal(16, 3, n_borrowers),
    'loan_purpose': np.random.choice(['Personal', 'Business', 'Education', 'Medical', 'Home Improvement'], 
                                   n_borrowers, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
    'application_date': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(n_borrowers)],
    'default_probability': default_probs,
    'is_default': [np.random.random() < prob for prob in default_probs],
    'credit_score_traditional': np.random.normal(650, 100, n_borrowers).astype(int),
}

# Ensure realistic constraints
loan_data['loan_amount'] = np.clip(loan_data['loan_amount'], 10000, 1000000)
loan_data['interest_rate'] = np.clip(loan_data['interest_rate'], 8, 25)
loan_data['credit_score_traditional'] = np.clip(loan_data['credit_score_traditional'], 300, 850)

loan_df = pd.DataFrame(loan_data)

print("\nSample Loan Data:")
print(loan_df.head(10))
print(f"\nDefault Rate: {loan_df['is_default'].mean():.2%}")

# 4. Save sample data to CSV files
borrower_df.to_csv('sample_borrower_data.csv', index=False)
alt_data_df.to_csv('sample_alternative_data.csv', index=False)
loan_df.to_csv('sample_loan_data.csv', index=False)

print("\nSample data files created:")
print("- sample_borrower_data.csv")
print("- sample_alternative_data.csv") 
print("- sample_loan_data.csv")

# 5. Create data generation guidelines and prompts
data_generation_guidelines = """
# Sample Data Generation Guidelines for CreditIQ Analytics

## Overview
This document provides guidelines for generating synthetic data to validate the CreditIQ Analytics platform for NBFC credit risk assessment.

## Data Sources and Types

### 1. Traditional Financial Data
- **Borrower Demographics**: Age, income, employment type, education, city tier
- **Credit History**: Existing loans, credit history length, traditional credit scores
- **Loan Details**: Amount, tenure, purpose, interest rates

### 2. Alternative Data Sources
- **UPI Transaction Data**: Monthly transaction frequency and patterns
- **Utility Payment Data**: Consistency in bill payments (electricity, mobile, internet)
- **GST Returns**: Filing frequency and compliance for business borrowers
- **Digital Wallet Usage**: Average balance and transaction frequency
- **E-commerce Behavior**: Online purchase patterns and frequency
- **Location Data**: Stability score based on location consistency
- **Mobile App Usage**: Time spent on financial apps

## ChatGPT Prompts for Data Generation

### Prompt 1: Borrower Profile Generation
"Generate 100 realistic borrower profiles for an Indian NBFC with the following fields:
- Age (18-65), Income (15K-20L INR), Employment type (Salaried/Self-employed/Business), 
- City tier (1/2/3), Education level, Existing loan count, Credit history months.
Make the data representative of India's demographic distribution."

### Prompt 2: Alternative Data Generation  
"Create alternative credit data for Indian borrowers including:
- Monthly UPI transactions (0-100), Utility bill payment consistency (0-1 score),
- GST returns filed (0-12 per year), Mobile recharge frequency, E-commerce transactions,
- Digital wallet average balance, Location stability score (0-1).
Ensure realistic correlations between variables."

### Prompt 3: Loan Default Scenarios
"Generate loan default scenarios considering:
- Higher default risk for: Lower income, irregular alt-data patterns, new credit users
- Lower default risk for: Consistent bill payments, stable location, regular UPI usage
- Include seasonal and economic factors affecting default rates"

## Validation Metrics
- Default rate should be between 3-15% (typical for Indian NBFCs)
- Strong correlation between alternative data consistency and lower default risk
- Income and employment stability should be primary risk factors

## Data Quality Checks
1. No missing values in critical fields
2. Realistic value ranges for all variables
3. Logical correlations between related variables
4. Balanced representation across demographic segments
"""

with open('data_generation_guidelines.txt', 'w') as f:
    f.write(data_generation_guidelines)

print("\n✅ Sample data generation completed!")
print("📄 Guidelines saved to: data_generation_guidelines.txt")