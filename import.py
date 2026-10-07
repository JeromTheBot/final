import truststore
truststore.inject_into_ssl()
import pandas as pd

df = pd.read_csv("hf://datasets/fenar/5g_netops/5G_netops_data_100K.csv.xz")
print(df.head())
print(len(df.columns))
print("Loaded Successfully....")