import pandas as pd
import matplotlib.pyplot as plt
import os

# Data
data =pd.read_csv("../data/structured/Exchange Rate_Bloomberg_TCURR_ 06 Feb 22(Sheet1).csv")

# Create DataFrame
df = pd.DataFrame(data)

# Convert 'Valid from' to datetime format
df["Valid from"] = pd.to_datetime(df["Valid from"], format="%d.%m.%Y")

# Sort by date (if not already sorted)
df = df.sort_values(by="Valid from")

# Descriptive Statistics
print("Descriptive Statistics:")
print(df["Exch. Rate"].describe())

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(df["Valid from"], df["Exch. Rate"], marker="o", linestyle="-", color="b", label="Exchange Rate")
plt.title("Exchange Rate (AED to AOA) Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Exchange Rate", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()