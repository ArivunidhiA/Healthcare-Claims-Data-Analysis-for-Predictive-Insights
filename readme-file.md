# Healthcare Claims Analysis Project

## Overview
This project analyzes healthcare claims data to identify patterns in patient treatment costs and predict high-cost patients. It uses the CMS DE-SynPUF (Medicare Claims Synthetic Public Use Files) dataset to demonstrate practical applications of data analysis in healthcare.

## Features
- Load and preprocess healthcare claims data
- Create meaningful features from raw claims data
- Train a Random Forest model to predict high-cost patients
- Generate visualizations of key patterns and insights
- Automated reporting and analysis workflow

## Requirements
- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/healthcare-claims-analysis.git
cd healthcare-claims-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the CMS DE-SynPUF dataset:
- Visit [CMS website](https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF)
- Download the Beneficiary Summary File
- Place the CSV file in the project directory

## Usage
Run the analysis:
```bash
python main.py
```

The script will:
1. Load and preprocess the claims data
2. Create features for analysis
3. Train a predictive model
4. Generate visualizations saved as 'claims_analysis.png'

## Output
- Model performance metrics in the console
- Visualizations saved as 'claims_analysis.png':
  - Distribution of claim amounts
  - Top 10 diagnoses by average claim amount
  - Feature importance in predicting high-cost patients

## Project Structure
```
healthcare-claims-analysis/
├── main.py                 # Main analysis script
├── requirements.txt        # Required Python packages
├── README.md              # Project documentation
└── claims_analysis.png    # Generated visualizations
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- CMS for providing the DE-SynPUF dataset
- Healthcare data analysis community for insights and best practices
