import pandas as pd

# load each table
fact_stocks = pd.read_csv("factStocks.csv")
dim_time = pd.read_csv("dimTime.csv")
dim_company = pd.read_csv("dimCompany.csv")

# join tables
stocks_time = fact_stocks.merge(dim_time, on="keyTime", how="left")
stocks_full = stocks_time.merge(dim_company, on="keyCompany", how="left")

# keep features directly
features = stocks_full[[
    "datetime",
    "stockCodeCompany",
    "openValueStock",
    "closeValueStock",
    "highValueStock",
    "lowValueStock",
    "quantityStock",
    "monthTime",
    "yearTime",
    "sectorCompany"  # optional categorical feature
]]

# Save final version to CSV
features.to_csv("ml_stocks.csv", index=False)

print("Saved to ml_stocks.csv ✅")
