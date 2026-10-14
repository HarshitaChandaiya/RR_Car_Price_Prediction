# Used Car Price Prediction with Ridge & Lasso Regression

Predicting used car prices from AutoScout listings (German online car marketplace) using linear regression, with Ridge and Lasso regularisation to control overfitting and identify which features actually drive price.

## Business problem

A used-car reseller needs to price inventory accurately. A good pricing model helps to:

- **Price competitively** while protecting margins
- **Understand price drivers** (mileage, age, power, fuel type, etc.) to guide inventory and marketing decisions
- **Avoid overfitting**, so the model generalises to new stock rather than memorising past listings

## Dataset

- Source: AutoScout24 listings, publicly available on Kaggle
- 15,915 listings, 23 columns (engine specs, weight, power, previous owners, condition, and bundled comfort/safety/entertainment feature lists)
- Target: `price`

## Approach

1. **Data cleaning**: no NaNs, but 1,673 exact duplicate rows (10.5%) were removed, leaving 14,242 listings
2. **Class imbalance**: categorical levels under 1% frequency grouped into `Other`
3. **Target transform**: `price` is right-skewed (skew ≈ 1.27), so the models predict `log(1 + price)` (skew ≈ -0.04 after transform)
4. **Outliers**: IQR-based capping (winsorising) on `km`, `hp_kW`, `Displacement_cc`, `Weight_kg`, `cons_comb`
5. **Feature engineering**:
   - Parsed the four bundled spec columns (`Comfort_Convenience`, `Entertainment_Media`, `Extras`, `Safety_Security`) into binary flags, keeping items present in 5% to 95% of listings (71 flags)
   - Engineered `power_to_weight`, then dropped it after VIF analysis showed it was near-redundant with `hp_kW` and `Weight_kg`
   - One-hot encoding, 70/30 train/test split, `StandardScaler` fit on training data only (no leakage)
6. **Modelling**: baseline Linear Regression, Ridge and Lasso, each tuned with 5-fold cross-validated MAE using a coarse-then-fine alpha search
7. **Diagnostics**: residuals vs fitted, residual normality (Q-Q plot), and VIF for multicollinearity

## Results

| Model | Test R² | Test RMSE (log price) | Test MAE (log price) |
|---|---|---|---|
| Linear Regression | 0.9276 | 0.1076 | 0.0791 |
| Ridge (alpha = 26.75) | 0.9275 | 0.1076 | 0.0791 |
| Lasso (alpha = 0.00026) | 0.9274 | 0.1077 | 0.0792 |

- All three models perform almost identically. The baseline wasn't overfitting much (train and test R² were already equal), so regularisation had little to correct.
- A test MAE of about 0.079 on the log scale corresponds to roughly **±8% typical price error**.
- Lasso zeroed out 6 of 105 features (e.g. Radio, Hands-free equipment, Electronic stability control), a useful signal for which optional spec columns add little pricing value.

### Strongest price drivers

- `hp_kW` (more power, higher price)
- `make_model` (the largest effects overall; e.g. Opel Corsa and Renault Clio price well below the Audi A1 reference level)
- `km` and `age` (both negative: depreciation)
- `Gearing_Type_Manual` (negative: manual cars price lower than automatic)

### Recommendation

Use **Ridge** for day-to-day pricing (keeps all features, stable coefficients), and run **Lasso** periodically as a feature-audit tool to see which optional spec columns can be dropped from data collection.

## Repository contents

| File | Description |
|---|---|
| `RR_Car_Price_Prediction_Harshita_Chandaiya.ipynb` | Full analysis notebook with outputs |
| `RR_Car_Price_Prediction_Harshita_Chandaiya_Report.pdf` | Written report with visualisations, results and assumptions |
| `Car_Price_data.csv` | Raw dataset |

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
jupyter notebook RR_Car_Price_Prediction_Harshita_Chandaiya.ipynb
```

Keep `Car_Price_data.csv` in the same folder as the notebook, then run all cells top to bottom.

## Tech stack

Python, pandas, NumPy, scikit-learn, statsmodels, matplotlib, seaborn

## Author

Harshita Chandaiya
