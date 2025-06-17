
# Stock Sentiment Analysis Backend

This backend provides real-time stock data, technical indicators, and news sentiment analysis for the Stock Sentiment Analyzer application.

## Features

- Real-time stock data from Yahoo Finance
- Technical indicators calculation (RSI, SMA, EMA, MACD, Bollinger Bands)
- News sentiment analysis using TextBlob
- News retrieval from News API

## Setup and Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```
   export NEWS_API_KEY=your_news_api_key
   ```

3. Run the application:
   ```
   python app.py
   ```

## API Endpoints

### GET /api/stock
Get stock data, technical indicators, and news sentiment analysis.

Query Parameters:
- `symbol`: Stock symbol (required)
- `marketType`: Market type ('Indian' or 'Foreign', default: 'Foreign')

## Deployment

This application is ready for deployment on platforms like Heroku.

```
git push heroku main
```

## Technologies Used

- Flask
- yfinance
- TextBlob
- News API
- Pandas
