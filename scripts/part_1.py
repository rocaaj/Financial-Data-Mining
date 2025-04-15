import pandas as pd
import random 
import time 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter 

# Load the data from CSV files
fact_stocks = pd.read_csv("output/factStocks.csv", dtype={"keyCompany": str, "keyTime": str})
dim_company = pd.read_csv("output/dimCompany.csv", dtype={"keyCompany": str})
dim_time = pd.read_csv("output/dimTime.csv", dtype={"keyTime": str})

# Debug: Check column names
print("📊 factStocks columns:", fact_stocks.columns.tolist())
print("📊 dimCompany columns:", dim_company.columns.tolist())
print("📊 dimTime columns:", dim_time.columns.tolist())

# Merge using correct keys
merged = fact_stocks.merge(dim_company, on="keyCompany")
merged = merged.merge(dim_time, on="keyTime")

# ✅ FIX 1: Apply sector filter properly (you forgot to assign it back)
merged = merged[merged["sectorCompany"] == "BM&FBOVESPA BASIC MATERIALS INDEX (IMAT)"]

# ✅ FIX 2: Use real datetime column for filtering
merged = merged[(merged["datetime"] >= "2010-01-01") & (merged["datetime"] <= "2020-12-31")]

# Use actual datetime as grouping key
date_col = "datetime"

# Group by time and get lists of stocks per time period
transactions = merged.groupby(date_col)["stockCodeCompany"].apply(list).tolist()

# ✅ FIX 3: Apply filtering BEFORE encoding
# Count stock frequencies across all transactions
flat_stock_list = [stock for tx in transactions for stock in tx]
stock_counts = Counter(flat_stock_list)

# Keep only stocks that occur > 100 times
top_stocks = set([stock for stock, count in stock_counts.items() if count > 100])
filtered_transactions = [[stock for stock in tx if stock in top_stocks] for tx in transactions]

# ✅ FIX 4: Only encode filtered transactions
if len(filtered_transactions) > 50000:
    filtered_transactions = random.sample(filtered_transactions, 50000)

# Encode transactions
te = TransactionEncoder()
te_array = te.fit_transform(filtered_transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# Run Apriori algorithm
timer = time.time()
print("Running Apriori...")
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
total_time = time.time() - timer 
print(f"Apriori completed in {total_time:.2f} seconds.")

# Show results
print("\n📈 Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
