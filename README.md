#  Loan Guard: Loan Default Prediction & Deployment
A deployed ML system that predicts loan default risk in real time comparing Logistic Regression, XGBoost, and Neural Network with SHAP explainability built for banking regulatory standards.
---

## 🧩 The Business Problem

Traditional credit scoring is interpretable but linear. Real world default risk is not.

---

## 🏆 Solution & Results

Three models were built and compared from interpretable baseline to deep learning  to find the best balance of performance and regulatory explainability.

| Model | Accuracy | ROC-AUC | Default Recall | Verdict |
|-------|----------|---------|----------------|---------|
| **Logistic Regression** | **86.84%** | **0.745** | **0.50** | 
| XGBoost | 100% | 1.0 | 1.0 |
| MLP Neural Network | 99.94% | 0.9995 | 1.0 |

> 📝 **Note on near-perfect scores:** XGBoost and MLP achieved 100% and 99.94% respectively. After investigating, no data leakage was found. Logistic Regression's more modest score (86.84%) is consistent with a linear model struggling on non linear synthetic patterns.

---

## 🔑 Top Risk Drivers (SHAP + Feature Importance)

SHAP explainability and XGBoost feature importance consistently agreed on the key default signals:

| Feature | Business Signal |
|---------|----------------|
| `Interest_rate_spread` | #1 driver high spread signals lender assessed risk |
| `rate_of_interest` | Higher rates correlate with riskier borrower profiles |
| `Upfront_charges` | Large fees indicate non standard loan structures |
| `age_45_54` | Mid-career borrowers show distinct default patterns |
| `loan_limit` | Borrowers at the edge of their limit are higher risk |
| `property_value` | Collateral quality impacts default recovery |
| `dtir1` | Debt to income ratio core affordability signal |
| `income` | Lower income increases default probability |
| `LTV` | High loan-to-value less equity cushion |
| `Credit_Score` | Traditional score still matters, but not the only signal |

---

## ⚙️ Technical Approach

**Data Challenges Solved:**
- Missing values: mode-filled (categorical), mean-filled (numerical) pre-split
- Outliers capped using IQR Winsorization discovered aggressive capping caused skewed features to collapse to zero, requiring careful calibration
- Imbalanced dataset handled with SMOTE (minority class oversampling)
- Feature scaling with StandardScaler for Logistic Regression and MLP

**Encoding Strategy (post split  no leakage):**
- One-Hot: Gender, loan_type, loan_purpose, occupancy_type, age, Region
- Binary: loan_limit, approv_in_adv, Credit_Worthiness, Neg_ammortization, Security_Type + 8 others

**Explainability:**
- SHAP LinearExplainer for Logistic Regression
- XGBoost native feature importance
- Both point to the same top risk factors  consistent and defensible

---

## 🚀 Deployment Render App

The Xgboost model is deployed as an interactive render application:

- Loan officers input applicant details and receive a real time default probability
- Risk score with feature contribution breakdown
- Designed for non technical users in credit teams
- https://loan-default-risk-3yo4.onrender.com/docs (Deployed)

```bash
streamlit run app.py
```

---

## 📈 Business Impact

Assuming a mid size lender, 10,000 applications/month, avg. loan £15,000, 24% default rate:

| Scenario | Annual Default Losses |
|----------|-----------------------|
| No model — approve all | £43,200,000 |
| Manual review (70% catch rate) | £12,960,000 |
| **Logistic Regression (50% default recall)** | **~£21,600,000** |
| **Future goal improved recall to 75%** | **~£10,800,000** |

*Improving recall on the default class is the primary optimisation target for v2.*

---

---

Author @Oluwatosin Abimbola

**Dataset:** [Kaggle — Loan Default Dataset](https://www.kaggle.com/datasets/oluwanifemiabimbola/loandata)

---

*Built for the question every credit team faces daily: is this applicant going to pay us back?*
