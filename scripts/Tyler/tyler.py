import pandas as pd 
from sklearn.linear_model import LinearRegression
import numpy as np
import sys
import pipeline as pl
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt






def run_pipeline(input_file, output_file):
    stocks = pd.read_csv(input_file)
    stocks = stocks.iloc[1:]
    stocks = stocks.dropna()

    #cluster stocks that have similiar times together and see their respective closing value
    #KMeans clustering

    stocks['Time'] = pd.to_datetime(stocks['Time'])
     # Convert 'Time' to numeric for clustering
    stocks['Time'] = pd.to_datetime(stocks['Time'])
    stocks['TimeNumeric'] = stocks['Time'].map(pd.Timestamp.toordinal)

    # Prepare data for clustering
    features = stocks[['TimeNumeric', 'Close']].values

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)  
    stocks['Cluster'] = kmeans.fit_predict(features)


  
    # Save the clustered data to the output file
    stocks.to_csv(output_file, index=False)
    print(f"Clustered data saved to {output_file}")

def main():
    if(len(sys.argv) != 3):
        print("Usage: python tyler.py <input_file> <output_file>")
        return
    if(sys.argv[1] == sys.argv[2]):
        print("Input and output directories cannot be the same.")
        return
    input_file = sys.argv[1]
    output_file = sys.argv[2]

  


if __name__ == "__main__":
    main()
    

    
