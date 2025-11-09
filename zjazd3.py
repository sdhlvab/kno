import pandas as pd

# wczytanie danych z pliku csv
wine = pd.read_csv('wine.data', header=None)



print(wine)