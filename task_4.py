import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("twitter_training.csv", header=None)

# Assign column names
df.columns = ["Tweet_ID", "Brand", "Sentiment", "Tweet"]

# Basic Information
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# 1. Sentiment Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(x="Sentiment", data=df, order=df["Sentiment"].value_counts().index)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# -----------------------------
# 2. Top 10 Most Discussed Brands
# -----------------------------
top_brands = df["Brand"].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_brands.index, y=top_brands.values)
plt.title("Top 10 Most Discussed Brands")
plt.xlabel("Brand")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_brands.png")
plt.show()

# -----------------------------
# 3. Brand-wise Sentiment Analysis
# -----------------------------
top5_brands = df["Brand"].value_counts().head(5).index

brand_sentiment = df[df["Brand"].isin(top5_brands)]

plt.figure(figsize=(10, 6))
sns.countplot(
    data=brand_sentiment,
    x="Brand",
    hue="Sentiment"
)
plt.title("Sentiment by Top 5 Brands")
plt.xlabel("Brand")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("brand_sentiment.png")
plt.show()

print("\nTask 4 Completed Successfully!")