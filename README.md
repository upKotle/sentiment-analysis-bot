# Russian Sentiment Analysis Telegram Bot

Telegram bot for multiclass sentiment analysis of Russian text  
(negative / neutral / positive).

## ML Pipeline
- Text cleaning (regex-based)
- TF-IDF (unigrams + bigrams)
- Logistic Regression (One-vs-Rest)
- Class balancing

## Metrics
Macro F1 ≈ 0.6  
Dataset: RuSentiment (Twitter)

## Features
- Probability-based predictions
- Confidence estimation
- Telegram interface

## How to run
```bash
python -m src.train
python -m bot.bot

### Environment variables

Create `.env` file in project root:

BOT_TOKEN=your_token_here
