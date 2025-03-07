# Weinstein Ticker Analyzer - Streamlit Version

This is a Streamlit application that applies Stan Weinstein's Stage Analysis method to analyze stock tickers. The app provides detailed technical analysis, chart visualization, support/resistance levels, and volume profile analysis.

## Features

- **Stage Analysis**: Automatically identifies the current Weinstein stage (Base, Uptrend, Top Formation, Downtrend)
- **Interactive Charts**: Candlestick charts with moving averages, Bollinger Bands, and volume analysis
- **Support & Resistance**: Identifies key price levels based on historical price and volume data
- **Volume Profile**: Visualizes volume distribution across price levels
- **Market Context**: Compares the ticker to overall market and sector conditions
- **Detailed Analysis**: Generates written analysis based on current technical conditions
- **Trade Recommendations**: Provides actionable recommendations based on Weinstein's methodology

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Clone or download the code

Save the Streamlit analyzer code as `weinstein_analyzer.py`.

### Step 2: Install required dependencies

```bash
pip install streamlit yfinance pandas numpy plotly
```

*Optional*: Create a virtual environment first to isolate your dependencies:

```bash
python -m venv weinstein_env
source weinstein_env/bin/activate  # On Windows: weinstein_env\Scripts\activate
pip install streamlit yfinance pandas numpy plotly
```

## Usage

### Starting the application

Run the following command in your terminal:

```bash
streamlit run weinstein_analyzer.py
```

The app will open in your default web browser. If it doesn't open automatically, navigate to `http://localhost:8501`.

### Analyzing a ticker

1. Enter a ticker symbol in the sidebar (e.g., "AAPL", "MSFT", "^GSPC" for S&P 500)
2. Select your desired time period and interval
3. Click the "Analyze" button
4. Navigate through the tabs to see different aspects of the analysis

## Understanding Weinstein Stage Analysis

Stan Weinstein's stage analysis method divides market cycles into four key stages:

1. **Stage 1 - Base Formation**: Consolidation after a downtrend, characterized by price moving sideways and accumulation by smart money. The 30-week moving average flattens.

2. **Stage 2 - Uptrend**: The most profitable stage, marked by higher highs and higher lows. Price breaks out above the base with increased volume, and the 30-week moving average slopes upward.

3. **Stage 3 - Top Formation**: Distribution phase where smart money begins to sell. The 30-week moving average begins to flatten or turn down, and price action becomes more volatile.

4. **Stage 4 - Downtrend**: Characterized by lower highs and lower lows. Price moves below the 30-week moving average, which is now sloping downward.

The analyzer applies these principles to identify the current stage and generate appropriate recommendations.

## Troubleshooting

- If you encounter a "No module named..." error, ensure you've installed all required packages.
- For ticker data issues, verify that the symbol is correct and has sufficient historical data.
- Check the log file (`weinstein_analyzer_YYYYMMDD.log`) for detailed error information.

## Credits

This application is based on Stan Weinstein's methodology from his book "Secrets for Profiting in Bull and Bear Markets."
