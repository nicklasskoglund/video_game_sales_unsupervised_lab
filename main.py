"""
🎮 Video Game Sales — Unsupervised ML Lab
==========================================
Runs the full experiment pipeline:
  1. Load & clean data
  2. EDA
  3. Feature engineering
  4. K-Means clustering (regional patterns)
  5. Hierarchical clustering + PCA (game archetypes)
  6. DBSCAN (critic vs user outliers + blockbusters)

All plots are saved to outputs/figures/
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "data/vgsales.csv"
OUTPUT_DIR   = "outputs/figures"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    print("\n📂 Loading data...")
    df = pd.read_csv(path)
    print(f"   Raw shape: {df.shape}")

    # Drop rows missing critical columns
    df.dropna(subset=["Year_of_Release", "Publisher"], inplace=True)

    # Fix User_Score: "tbd" → NaN, then cast to float
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
    # Scale User_Score to 0–100 to match Critic_Score
    df["User_Score_scaled"] = df["User_Score"] * 10

    # Year as int
    df["Year_of_Release"] = df["Year_of_Release"].astype(int)

    # Remove obvious data errors
    df = df[df["Year_of_Release"] >= 1980]
    df = df[df["Global_Sales"] > 0]

    print(f"   Clean shape: {df.shape}")
    print(f"   Critic_Score missing: {df['Critic_Score'].isna().sum()} rows")
    print(f"   User_Score missing:   {df['User_Score'].isna().sum()} rows")
    return df


# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    print("\n📊 Running EDA...")

    # --- Sales by genre ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    genre_sales = df.groupby("Genre")["Global_Sales"].sum().sort_values()
    genre_sales.plot(kind="barh", ax=axes[0], color=sns.color_palette("muted", len(genre_sales)))
    axes[0].set_title("Global Sales by Genre (millions)")
    axes[0].set_xlabel("Sales (M)")

    # --- Sales over time ---
    year_sales = df.groupby("Year_of_Release")["Global_Sales"].sum()
    axes[1].plot(year_sales.index, year_sales.values, marker="o", markersize=3, linewidth=1.5)
    axes[1].set_title("Global Sales Over Time")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Sales (M)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_sales_overview.png")
    plt.close()

    # --- Regional correlation heatmap ---
    fig, ax = plt.subplots(figsize=(7, 5))
    region_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    corr = df[region_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Regional Sales Correlation")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_regional_correlation.png")
    plt.close()

    # --- Critic vs User score scatter ---
    df_scores = df.dropna(subset=["Critic_Score", "User_Score_scaled"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_scores["Critic_Score"], df_scores["User_Score_scaled"],
               alpha=0.3, s=10, color="steelblue")
    ax.set_xlabel("Critic Score (0–100)")
    ax.set_ylabel("User Score scaled (0–100)")
    ax.set_title("Critic Score vs User Score")
    # Diagonal = perfect agreement
    ax.plot([0, 100], [0, 100], "r--", linewidth=1, label="Perfect agreement")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_critic_vs_user.png")
    plt.close()

    print("   ✅ EDA plots saved.")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 Engineering features...")

    # Regional sales ratios (share of global)
    for region in ["NA", "EU", "JP", "Other"]:
        df[f"{region}_ratio"] = df[f"{region}_Sales"] / (df["Global_Sales"] + 1e-9)

    # Platform generation grouping
    gen_map = {
        "NES": "Retro", "SNES": "Retro", "GB": "Retro", "GEN": "Retro",
        "2600": "Retro", "GG": "Retro", "SCD": "Retro", "NG": "Retro",
        "N64": "5th-gen", "PS": "5th-gen", "SAT": "5th-gen",
        "PS2": "6th-gen", "GC": "6th-gen", "XB": "6th-gen", "GBA": "6th-gen",
        "PS3": "7th-gen", "X360": "7th-gen", "Wii": "7th-gen", "DS": "7th-gen", "PSP": "7th-gen",
        "PS4": "8th-gen", "XOne": "8th-gen", "WiiU": "8th-gen", "3DS": "8th-gen",
        "PC": "PC", "PSV": "Handheld",
    }
    df["Platform_Gen"] = df["Platform"].map(gen_map).fillna("Other")

    # Encode Genre numerically for clustering
    df["Genre_encoded"] = df["Genre"].astype("category").cat.codes

    print(f"   Platform generations: {df['Platform_Gen'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# HELPER: Elbow + Silhouette
# ─────────────────────────────────────────────
def find_optimal_k(X_scaled: np.ndarray, k_range: range, title: str, filename: str):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "bo-")
    axes[0].set_title(f"Elbow Method — {title}")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, "gs-")
    axes[1].set_title(f"Silhouette Score — {title}")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"   Best K by silhouette: {best_k}")
    return best_k


# ─────────────────────────────────────────────
# EXPERIMENT 1 — Regional Sales Patterns (K-Means)
# ─────────────────────────────────────────────
def experiment_1_regional_kmeans(df: pd.DataFrame):
    print("\n🧪 Experiment 1 — Regional Sales Patterns (K-Means)")

    features = ["NA_ratio", "EU_ratio", "JP_ratio", "Other_ratio"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = find_optimal_k(X_scaled, range(2, 9), "Regional Patterns", "exp1_elbow.png")

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df.loc[X.index, "Cluster_Regional"] = km.fit_predict(X_scaled)
    df["Cluster_Regional"] = df["Cluster_Regional"].astype("Int64")

    # Visualize: mean ratios per cluster
    cluster_summary = df.groupby("Cluster_Regional")[features].mean()
    cluster_summary.plot(kind="bar", figsize=(10, 5), colormap="Set2")
    plt.title(f"Experiment 1 — Regional Sales Profile per Cluster (K={best_k})")
    plt.ylabel("Mean ratio of Global Sales")
    plt.xlabel("Cluster")
    plt.xticks(rotation=0)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp1_cluster_profiles.png")
    plt.close()

    # Top genres per cluster
    print("\n   Top genres per regional cluster:")
    for c in sorted(df["Cluster_Regional"].dropna().unique()):
        top = df[df["Cluster_Regional"] == c]["Genre"].value_counts().head(3)
        print(f"   Cluster {int(c)}: {top.to_dict()}")

    return df


# ─────────────────────────────────────────────
# EXPERIMENT 2 — Critic vs User Outliers (DBSCAN)
# ─────────────────────────────────────────────
def experiment_2_dbscan_scores(df: pd.DataFrame):
    print("\n🧪 Experiment 2 — Critic vs User Score Outliers (DBSCAN)")

    df_scores = df.dropna(subset=["Critic_Score", "User_Score_scaled"]).copy()
    X = df_scores[["Critic_Score", "User_Score_scaled"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=0.4, min_samples=10)
    df_scores["DBSCAN_label"] = db.fit_predict(X_scaled)

    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise    = (db.labels_ == -1).sum()
    print(f"   Clusters found: {n_clusters}, Noise/outliers: {n_noise}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {-1: "red"}
    for label in sorted(df_scores["DBSCAN_label"].unique()):
        if label == -1:
            mask = df_scores["DBSCAN_label"] == -1
            ax.scatter(df_scores.loc[mask, "Critic_Score"],
                       df_scores.loc[mask, "User_Score_scaled"],
                       s=12, alpha=0.6, color="red", label="Outlier", zorder=3)
        else:
            mask = df_scores["DBSCAN_label"] == label
            ax.scatter(df_scores.loc[mask, "Critic_Score"],
                       df_scores.loc[mask, "User_Score_scaled"],
                       s=8, alpha=0.3, label=f"Cluster {label}")

    ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="Perfect agreement")
    ax.set_xlabel("Critic Score (0–100)")
    ax.set_ylabel("User Score scaled (0–100)")
    ax.set_title("Experiment 2 — DBSCAN Outliers: Critics vs Users")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp2_dbscan_scores.png")
    plt.close()

    # Show most controversial outlier games
    outliers = df_scores[df_scores["DBSCAN_label"] == -1].copy()
    outliers["score_gap"] = abs(outliers["Critic_Score"] - outliers["User_Score_scaled"])
    print("\n   Most controversial games (outliers with largest score gap):")
    print(outliers.sort_values("score_gap", ascending=False)[
        ["Name", "Genre", "Critic_Score", "User_Score_scaled", "score_gap"]
    ].head(10).to_string(index=False))

    return df


# ─────────────────────────────────────────────
# EXPERIMENT 3 — Game Archetypes (Hierarchical + PCA)
# ─────────────────────────────────────────────
def experiment_3_hierarchical_pca(df: pd.DataFrame):
    print("\n🧪 Experiment 3 — Game Archetypes (Hierarchical Clustering + PCA)")

    features = ["NA_ratio", "EU_ratio", "JP_ratio", "Other_ratio",
                "Global_Sales", "Genre_encoded"]
    df_sub = df[features + ["Name", "Genre", "Year_of_Release"]].dropna()

    # Sample for dendrogram readability
    sample = df_sub.sample(min(300, len(df_sub)), random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample[features])

    # Dendrogram
    Z = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, truncate_mode="level", p=5, leaf_rotation=90, leaf_font_size=8)
    ax.set_title("Experiment 3 — Dendrogram: Game Archetypes (Ward linkage, sample n=300)")
    ax.set_xlabel("Games (truncated)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp3_dendrogram.png")
    plt.close()

    # PCA 2D scatter — full dataset
    df_full = df[features + ["Genre", "Year_of_Release"]].dropna()
    X_full  = scaler.fit_transform(df_full[features])
    pca     = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_full)
    print(f"   PCA explained variance: {pca.explained_variance_ratio_.round(3)}")

    df_pca = pd.DataFrame(components, columns=["PC1", "PC2"], index=df_full.index)
    df_pca["Genre"] = df_full["Genre"].values
    df_pca["Year_of_Release"] = df_full["Year_of_Release"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color by Genre
    genres = df_pca["Genre"].unique()
    cmap   = cm.get_cmap("tab20", len(genres))
    for i, genre in enumerate(genres):
        mask = df_pca["Genre"] == genre
        axes[0].scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                        s=5, alpha=0.4, color=cmap(i), label=genre)
    axes[0].set_title("PCA — Colored by Genre")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[0].legend(fontsize=6, markerscale=2, loc="upper right", ncol=2)

    # Color by Era
    bins  = [1979, 1994, 2002, 2010, 2030]
    labels= ["Retro (≤94)", "5th–6th gen (95–02)", "7th gen (03–10)", "8th gen (11+)"]
    df_pca["Era"] = pd.cut(df_pca["Year_of_Release"], bins=bins, labels=labels)
    era_colors = {"Retro (≤94)": "#e07b54", "5th–6th gen (95–02)": "#5b8db8",
                  "7th gen (03–10)": "#6abf8a", "8th gen (11+)": "#c97bbf"}
    for era, color in era_colors.items():
        mask = df_pca["Era"] == era
        axes[1].scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                        s=5, alpha=0.4, color=color, label=era)
    axes[1].set_title("PCA — Colored by Era")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[1].legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp3_pca.png")
    plt.close()
    print("   ✅ Dendrogram + PCA saved.")

    return df


# ─────────────────────────────────────────────
# EXPERIMENT 4 — Blockbusters vs Long Tail (DBSCAN)
# ─────────────────────────────────────────────
def experiment_4_blockbusters(df: pd.DataFrame):
    print("\n🧪 Experiment 4 — Blockbusters vs Long Tail (DBSCAN)")

    features = ["Global_Sales", "NA_ratio", "EU_ratio", "JP_ratio"]
    df_sub = df[features + ["Name", "Genre", "Platform"]].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sub[features])

    db = DBSCAN(eps=0.3, min_samples=15)
    df_sub = df_sub.copy()
    df_sub["DBSCAN_label"] = db.fit_predict(X_scaled)

    n_noise = (df_sub["DBSCAN_label"] == -1).sum()
    print(f"   Noise (potential blockbusters/anomalies): {n_noise}")

    # Scatter: Global Sales vs NA ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    normal  = df_sub[df_sub["DBSCAN_label"] != -1]
    outlier = df_sub[df_sub["DBSCAN_label"] == -1]

    ax.scatter(normal["Global_Sales"], normal["NA_ratio"],
               s=8, alpha=0.3, color="steelblue", label="Regular games")
    ax.scatter(outlier["Global_Sales"], outlier["NA_ratio"],
               s=20, alpha=0.8, color="crimson", label="Anomaly / Blockbuster", zorder=3)

    # Label biggest outliers
    top_outliers = outlier.sort_values("Global_Sales", ascending=False).head(8)
    for _, row in top_outliers.iterrows():
        ax.annotate(row["Name"], (row["Global_Sales"], row["NA_ratio"]),
                    fontsize=6, alpha=0.8, xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel("Global Sales (M)")
    ax.set_ylabel("NA Sales Ratio")
    ax.set_title("Experiment 4 — DBSCAN: Blockbusters as Anomalies")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp4_blockbusters.png")
    plt.close()

    print("\n   Top anomalies by global sales:")
    print(outlier.sort_values("Global_Sales", ascending=False)[
        ["Name", "Genre", "Platform", "Global_Sales"]
    ].head(10).to_string(index=False))

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🎮 Video Game Sales — Unsupervised ML Lab")
    print("=" * 55)

    df = load_and_clean(DATA_PATH)
    run_eda(df)
    df = engineer_features(df)

    df = experiment_1_regional_kmeans(df)
    df = experiment_2_dbscan_scores(df)
    df = experiment_3_hierarchical_pca(df)
    df = experiment_4_blockbusters(df)

    print(f"\n✅ All done! Figures saved to: {OUTPUT_DIR}/")
    print("=" * 55)