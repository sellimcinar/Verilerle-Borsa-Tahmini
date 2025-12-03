# 📈 BIST AI Analyst (Borsa Kahini)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Finance](https://img.shields.io/badge/Domain-Finance-green)

**BIST AI Analyst** is an advanced financial dashboard designed to analyze Borsa Istanbul (BIST) stocks using Artificial Intelligence and Statistical Modeling. This project bridges the gap between Data Science and Economics, providing real-time insights, forecasting, and risk assessment.

🔗 **Live Demo:** [https://verilerle-borsa-tahmini-2025.streamlit.app/]

## 🚀 Key Features

### 1. 🤖 AI-Powered Forecasting (Prophet)
* Utilizes Facebook's **Prophet** model to forecast stock prices for the next **30 days**.
* Analyzes trends, seasonality, and holiday effects automatically.

### 2. 🎲 Risk Simulation (Monte Carlo)
* Implements **Geometric Brownian Motion (GBM)** to simulate **1,000 future price scenarios**.
* Calculates **Value at Risk (VaR)** and expected best/worst-case outcomes to quantify investment risk.

### 3. 🔍 Historical Pattern Matching
* Uses a **Similarity Search Algorithm** (Euclidean Distance) to scan historical data.
* Identifies past market movements that resemble the current chart and projects potential future trends based on history.

### 4. 🕸️ Fundamental Health Radar
* Visualizes key financial ratios (P/E, P/B, ROE, Profit Margin) on an interactive **Radar Chart**.
* Provides a quick "health check" snapshot of the company's fundamentals.

### 5. 📰 Sentiment & Market Data
* Fetches real-time data using `yfinance`.
* Includes a "Heads-Up Display" (HUD) for instant executive summary.

## 🛠️ Tech Stack
* **Core:** Python
* **Frontend:** Streamlit
* **Data & ML:** Pandas, NumPy, Facebook Prophet, Scikit-learn
* **Visualization:** Plotly, Matplotlib
* **Financial Data:** yfinance API

## 📉 Methodology
* **Forecasting:** Additive Regression Model (Prophet).
* **Simulation:** Stochastic Calculus (GBM: `dS = μSdt + σSdW`).
* **Analysis:** Time-series correlation and normalization techniques.

---
*Disclaimer: This project is for educational and informational purposes only. It does not constitute investment advice.*
