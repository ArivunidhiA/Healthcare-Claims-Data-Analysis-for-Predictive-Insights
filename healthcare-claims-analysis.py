import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HealthcareClaims:
    def __init__(self, data_path):
        """Initialize the HealthcareClaims analyzer."""
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load and preprocess the CMS claims data."""
        # Load the data
        self.df = pd.read_csv(self.data_path)
        
        # Basic cleaning
        self.df = self.df.dropna()
        return self
    
    def create_features(self):
        """Create relevant features for analysis."""
        # Calculate total claims per beneficiary
        claims_per_beneficiary = self.df.groupby('DESYNPUF_ID').agg({
            'CLM_PMT_AMT': ['sum', 'mean', 'count'],
            'NCH_PRMRY_PYR_CLM_PD_AMT': ['sum', 'mean'],
            'NCH_BENE_IP_DDCTBL_AMT': ['sum', 'mean'],
            'NCH_BENE_BLOOD_DDCTBL_AMT': ['sum']
        }).reset_index()
        
        # Flatten column names
        claims_per_beneficiary.columns = ['DESYNPUF_ID', 'total_claims', 'avg_claim_amount', 
                                        'claim_count', 'total_primary_payer', 'avg_primary_payer',
                                        'total_deductible', 'avg_deductible', 'total_blood_deductible']
        
        # Define high-cost patients (top 10%)
        threshold = claims_per_beneficiary['total_claims'].quantile(0.9)
        claims_per_beneficiary['is_high_cost'] = (
            claims_per_beneficiary['total_claims'] > threshold).astype(int)
        
        self.features_df = claims_per_beneficiary
        return self
    
    def train_model(self):
        """Train a Random Forest model to predict high-cost patients."""
        # Prepare features and target
        X = self.features_df.drop(['DESYNPUF_ID', 'is_high_cost'], axis=1)
        y = self.features_df['is_high_cost']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        return self
    
    def analyze_patterns(self):
        """Analyze and visualize patterns in the claims data."""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Distribution of claim amounts
        plt.subplot(1, 3, 1)
        sns.histplot(data=self.df, x='CLM_PMT_AMT', bins=50)
        plt.title('Distribution of Claim Amounts')
        plt.xlabel('Claim Amount ($)')
        plt.xticks(rotation=45)
        
        # Plot 2: Average claim amount by diagnosis
        plt.subplot(1, 3, 2)
        diagnosis_costs = self.df.groupby('ICD9_DGNS_CD_1')['CLM_PMT_AMT'].mean().sort_values(
            ascending=False)[:10]
        sns.barplot(x=diagnosis_costs.index, y=diagnosis_costs.values)
        plt.title('Top 10 Diagnoses by Average Claim Amount')
        plt.xlabel('ICD9 Diagnosis Code')
        plt.ylabel('Average Claim Amount ($)')
        plt.xticks(rotation=45)
        
        # Plot 3: Feature importance
        plt.subplot(1, 3, 3)
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.features_df.drop(['DESYNPUF_ID', 'is_high_cost'], axis=1).columns
        ).sort_values(ascending=True)
        sns.barplot(x=importances.values, y=importances.index)
        plt.title('Feature Importance in Predicting High-Cost Patients')
        
        plt.tight_layout()
        plt.savefig('claims_analysis.png')
        plt.close()
        
        return self

def main():
    """Main function to run the analysis."""
    # Initialize and run analysis
    analyzer = HealthcareClaims('DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv')
    analyzer.load_data().create_features().train_model().analyze_patterns()
    
    print("\nAnalysis complete! Check claims_analysis.png for visualizations.")

if __name__ == "__main__":
    main()
