# Car Accident Severity Analysis

**Language:** R  
**Libraries:** `tidyverse`, `GGally`, `caret`, `rpart`, `rpart.plot`, `ipred`, `readr`, `class`, `bestglm`  
**Dataset:** U.S. Car Accidents (Kaggle) — 500K sampled accidents (2016–2023) across 49 states  
**Group Project:** Completed as part of a collaborative team effort.

---

## Objective

Predict accident severity (1–4) using demographic, environmental, and weather-related factors.

---

## Data Description

- Covers **U.S. car accidents (2016–2023)**
- Data collected from **traffic sensors, cameras, and APIs**
- **Target:** Severity level (1–4)  
- **Predictors:** Temperature, humidity, distance, wind speed, population, density, etc.  
- Sampled **500K** records from Kaggle across **49 states**
- Data joined with U.S. city demographics for richer context  

**Data cleaning steps:**
1. Removed missing values (`NA`)
2. Joined accident data with `uscities.csv` on `City` and `State`
3. Selected relevant features and standardized variables

---

## Exploratory Data Analysis (EDA)

- Visualized severity distribution  
- Plotted variable relationships with `GGally::ggpairs`  
- Examined accident frequency by month and hour  
- Mapped accidents by city using `leaflet`  

*Visuals: (Insert generated plots here)*  
- Histogram of Severity  
- Accidents per Month  
- Accidents by Hour  
- City Distributions (Miami, Los Angeles, Orlando, Dallas, Houston)  
- Miami Severity Map  

---

## Milestone 1: k-Nearest Neighbors (kNN)

**Goal:** Classify accident severity using environmental and demographic predictors.

**Steps:**
1. Stratified and sampled data by severity class  
2. Split into 70/30 train-test sets  
3. Trained kNN model with varying `k` values  
4. Evaluated performance using confusion matrices  

**Best Model:**  
- **k = 23** gave the highest accuracy  
- *Accuracy:* ~72%  

*Visual: Accuracy vs. Neighborhood Size plot*

---

## Part 2: Classification Tree & Bagging

**Goal:** Identify the most influential predictors of accident severity.

**Methods:**
- Built a **classification tree** using `rpart`
- Pruned using optimal CP value
- Applied **bagging** with both `ipred` and `caret` for model stability

**Key Findings:**
- **Distance, Humidity, and Population** were major split variables  
- **Low distance + low humidity → more severe crashes (Severity 3)**  
- **High population density → linked to Severity 4**  
- **Low population → linked to moderate crashes (Severity 2)**  
- **Population** emerged as the strongest predictor, followed by **density** and **humidity**  

*Visuals:*  
- Classification tree diagram  
- Variable importance plot (from `vip`)

---

## Part 3: Logistic Regression Models

**Goal:** Predict binary severity (Mild vs. Severe) using logistic regression.

**Approach:**
- Created binary variable:  
  - `Severity < 3 → Mild (0)`  
  - `Severity ≥ 3 → Severe (1)`  
- Trained multiple logistic regression models with environmental and demographic variables  
- Used stepwise selection and `bestglm` for best subset selection  

**Results:**
- Logistic regression models achieved **moderate performance**  
- Weather features (temperature, humidity, wind) had weaker effects  
- **Demographic context (population, density)** remained dominant predictors  

*Visuals:*  
- Correlation matrix (`GGally`)  
- Logistic regression fit plots
 

---

## Key Findings

### Severity Insights
- Urban and demographic context (population, density) drives accident severity more than weather conditions  
- Low distance + low humidity → more severe crashes (Severity 3)  
- High population → more severe crashes (Severity 4)  
- Low population → moderate crashes (Severity 2)  

### Model Performance
- Bagging ensemble outperforms single decision tree  
- Logistic regression less effective due to complex, non-linear relationships  
