# Count stock frequencies
stock_counts = Counter([s for tx in transactions for s in tx])
top_stocks = set([stock for stock, count in stock_counts.items() if count > 50])

# Filter transactions to only include top stocks
transactions = [[s for s in tx if s in top_stocks] for tx in transactions]
