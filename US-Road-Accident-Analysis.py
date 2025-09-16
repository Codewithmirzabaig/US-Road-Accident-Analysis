import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 1. Load dataset
# ----------------------
df = pd.read_csv(r"C:\Users\Mirza Shareef Baig\OneDrive\Documents\US_Accidents_March23.csv", nrows=10000)

# ----------------------
# 2. Convert Start_Time to datetime
# ----------------------
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

# Extract Year, Month, Day, Hour
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['Day'] = df['Start_Time'].dt.day
df['Hour'] = df['Start_Time'].dt.hour

# ----------------------
# 3. Accidents per Year
# ----------------------
accidents_per_year = df.groupby('Year').size()
plt.figure(figsize=(8,5))
accidents_per_year.plot(kind='bar', color='orange')
plt.title('Accidents per Year')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=0)
plt.show()

# ----------------------
# 4. Accidents per Month
# ----------------------
accidents_per_month = df.groupby('Month').size()
plt.figure(figsize=(8,5))
accidents_per_month.plot(kind='bar', color='green')
plt.title('Accidents per Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=0)
plt.show()

# ----------------------
# 5. Peak Accident Hours (with NumPy)
# ----------------------
hours = df['Hour'].to_numpy()
peak_hour = np.argmax(np.bincount(hours))
mean_hour = np.mean(hours)
median_hour = np.median(hours)
std_hour = np.std(hours)

print(f"Peak accident hour: {peak_hour}")
print(f"Mean accident hour: {mean_hour:.2f}")
print(f"Median accident hour: {median_hour}")
print(f"Std deviation of accident hours: {std_hour:.2f}")

# Plot accidents per hour
accidents_per_hour = df.groupby('Hour').size()
plt.figure(figsize=(10,5))
accidents_per_hour.plot(kind='bar', color='red')
plt.title('Accidents per Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=0)
plt.show()

# ----------------------
# 6. Top 10 Accident-Prone Cities
# ----------------------
cities = df['City'].to_numpy()
top_cities = df['City'].value_counts().head(10)
plt.figure(figsize=(10,5))
top_cities.plot(kind='bar', color='blue')
plt.title('Top 10 Accident-Prone Cities')
plt.xlabel('City')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

# ----------------------
# 7. Accident Severity Analysis (with NumPy)
# ----------------------
severity = df['Severity'].to_numpy()
severity_counts = np.bincount(severity)
severity_levels = np.arange(len(severity_counts))

plt.figure(figsize=(8,5))
plt.bar(severity_levels, severity_counts, color='purple')
plt.title('Accident Severity Distribution')
plt.xlabel('Severity Level')
plt.ylabel('Number of Accidents')
plt.xticks(severity_levels)
plt.show()


