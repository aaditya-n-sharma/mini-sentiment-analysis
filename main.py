import pytz
from datetime import datetime
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

#intialise finbert
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)

#clean finbert output
def sentiment(headline):
    output = finbert(headline)
    label = output[0]["label"]
    score = output[0]["score"]
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0

#get news aggregate webpage 
ticker = "AMZN"
root = "https://finviz.com/quote.ashx?t="
url = root + ticker

req = Request(url=url, headers={"user-agent": "Mozilla/5.0"}) 
response = urlopen(req) 
html = BeautifulSoup(response, "html.parser")
news_table = html.find(id="news-table")

#parse headlines and timestamps from webpage
rows = []
for row in news_table.find_all("tr"):
    headline = row.a.text
    compound_time = row.td.text.strip().split(" ")
        
    if len(compound_time) == 1:
        time_str = compound_time[0]
    else:
        date_str = compound_time[0]
        time_str = compound_time[1]

    if date_str == "Today":
        date_str = datetime.now(pytz.timezone("US/Eastern")).strftime("%b-%d-%y")

    try:
        date_str = datetime.strptime(date_str, "%b-%d-%y").strftime("%d %m %Y")
    except Exception:
        pass

    time_str = datetime.strptime(time_str, "%I:%M%p").strftime("%H:%M")
    rows.append([date_str, time_str, headline])

#database of all parsed data
df = pd.DataFrame(rows, columns=["Date", "Time", "Headline"])
df["Sentiment"] = df["Headline"].apply(sentiment)

#sort chronologically
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d %m %Y %H:%M")
df.sort_values("Datetime", inplace=True)
df.set_index("Datetime", inplace=True)

start_date = df.index.min().date().strftime("%Y-%m-%d")
end_date = df.index.max().date().strftime("%Y-%m-%d")

stock_df = yf.download(ticker, start=start_date, end=end_date, interval="15m")
stock_df.index = stock_df.index.tz_convert("UTC")

#plot historical stock data in blue
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(stock_df.index, stock_df["Close"], color="blue", label="Stock Price")
ax1.set_ylabel("Stock Price", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

#plot sentiment in green/red for positive/negative
ax2 = ax1.twinx()
colors = df["Sentiment"].apply(lambda x: "green" if x > 0 else ("red" if x < 0 else "gray"))
ax2.bar(df.index, df["Sentiment"], color=colors, alpha=0.6, width=0.02, label="Sentiment")
ax2.set_ylabel("Sentiment Score", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.tight_layout()
plt.show()


