import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load dataset
df = pd.read_csv("dataSet.csv")
df["Category"] = df["Category"].str.strip()

# Compute Z-score within each category
df["zscore_calories"] = df.groupby("Category")["Calories per 100g"].transform(zscore)

# Mark anomalies: Z-score > 1.1
df["Anomaly"] = df["zscore_calories"].abs() > 1.1

# --- Plot: Mean Calories per Category + One Anomaly per Category ---
mean_values = df.groupby("Category")["Calories per 100g"].mean().sort_values()
sorted_categories = mean_values.index.tolist()
df["Category_sorted"] = pd.Categorical(df["Category"], categories=sorted_categories, ordered=True)

plt.figure(figsize=(14, 6))
colors = sns.color_palette("RdYlGn_r", len(mean_values))
sns.barplot(x=mean_values.index, y=mean_values.values, palette=colors)

# Get one anomaly per category
example_anomalies = df[df["Anomaly"]].groupby("Category").first().reset_index()

for i, cat in enumerate(sorted_categories):
    example = example_anomalies[example_anomalies["Category"] == cat]
    if not example.empty:
        row = example.iloc[0]
        plt.scatter(i, row["Calories per 100g"], color="red", s=100, zorder=5)
        plt.text(i, row["Calories per 100g"] + 5, row["Food Name"], ha="center", fontsize=9, color="darkred")

plt.title("Average Calories per 100g by Category with One Anomaly Example Highlighted")
plt.ylabel("Calories per 100g")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# --- Console Output: Explain why each food is an anomaly ---
nutrient_columns = [
    "Fat (g) per 100g",
    "Protein (g) per 100g",
    "Carbs (g) per 100g",
    "Water (g) per 100g",
    "Fiber (g) per 100g",
]

print("\n--- Anomaly Analysis: Nutrient Deviation per Category ---\n")

for _, row in example_anomalies.iterrows():
    cat = row["Category"]
    food = row["Food Name"]
    food_cals = row["Calories per 100g"]

    cat_df = df[df["Category"] == cat]
    cat_mean = cat_df[nutrient_columns].mean()
    food_values = row[nutrient_columns]

    comparison = pd.DataFrame({
        "Nutrient": nutrient_columns,
        "Food": food_values.values,
        "Category Avg": cat_mean.values
    })
    comparison["Difference"] = comparison["Food"] - comparison["Category Avg"]
    comparison["% Diff"] = (comparison["Difference"] / comparison["Category Avg"]) * 100

    print(f"Category: {cat}")
    print(f"Anomalous Food: {food}")
    print(f"Calories: {food_cals:.1f}")
    print("Nutrient differences:")

    for _, r in comparison.iterrows():
        status = ""
        if abs(r["% Diff"]) >= 20:
            status = " <== significant"
        print(f"  {r['Nutrient']}: {r['Food']:.2f} (avg: {r['Category Avg']:.2f}, Δ = {r['% Diff']:.1f}%) {status}")

    print("-" * 60)
