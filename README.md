# 🎮 Video Game Sales — Unsupervised ML Lab

> An exploratory machine learning project using unsupervised techniques to uncover hidden patterns in video game sales data across regions, genres, platforms and critic/user ratings.

---

## 📖 Project Description

This is a group lab project in Unsupervised Machine Learning. The goal is **not** to find a "right" answer, but to experiment with clustering and dimensionality reduction techniques and interpret what we find.

We use the [Video Game Sales with Ratings dataset from Kaggle](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings) which contains sales data for over 16,000 games across platforms, genres, regions and review scores.

---

## 🧪 Experiments

| # | Experiment | Technique | Question |
|---|---|---|---|
| 1 | Regional Sales Patterns | K-Means | What makes a game dominate in Japan vs USA vs EU? |
| 2 | Critic vs User Score Outliers | DBSCAN | Who disagrees more — critics or players? |
| 3 | Game Archetypes Through Time | Hierarchical Clustering + PCA | Can we find eras without looking at the year? |
| 4 | Blockbusters vs Long Tail | DBSCAN | Are the best-selling games just anomalies? |

---

## 📁 Project Structure

```
video_game_sales_unsupervised_lab/
│
├── main.py                   # Runs the full pipeline
├── README.md
├── requirements.txt
│
├── data/
│   └── vgsales.csv           # Download from Kaggle (see below)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Data cleaning & Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering.ipynb   # K-Means + Hierarchical + PCA
│   └── 04_extras.ipynb       # DBSCAN + extra experiments
│
└── outputs/
    └── figures/              # Saved plots (auto-generated)
```

---

## 📦 Dataset

**Source:** [Kaggle — Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)

Download the CSV and rename it to `vgsales.csv`, then place it in the `data/` folder.

**Key columns:**
| Column | Description |
|---|---|
| `Name` | Game title |
| `Platform` | Console/platform (PS2, Wii, X360 ...) |
| `Year_of_Release` | Release year |
| `Genre` | Game genre |
| `Publisher` | Publisher name |
| `NA_Sales` | North America sales (millions) |
| `EU_Sales` | Europe sales (millions) |
| `JP_Sales` | Japan sales (millions) |
| `Other_Sales` | Rest of world sales (millions) |
| `Global_Sales` | Total global sales (millions) |
| `Critic_Score` | Metacritic score (0–100) |
| `Critic_Count` | Number of critic reviews |
| `User_Score` | User score (0–10, stored as string!) |
| `User_Count` | Number of user reviews |
| `Developer` | Game developer |
| `Rating` | ESRB rating |

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/nicklasskoglund/video_game_sales_unsupervised_lab.git
cd video_game_sales_unsupervised_lab

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle, rename to vgsales.csv and place in data/
# data/vgsales.csv

# Run the full pipeline
python main.py

# Or open notebooks individually
jupyter notebook notebooks/01_eda.ipynb
```

---

## ⚠️ Known Data Issues

| Issue | Column | Solution |
|---|---|---|
| ~51% missing values | `Critic_Score` | Analyze only subset with scores |
| ~40% missing values | `User_Score` | Analyze only subset with scores |
| Stored as string with `"tbd"` | `User_Score` | Convert to float, coerce "tbd" to NaN |
| 31 unique platforms | `Platform` | Group into platform generations |
| Data sparse after 2015 | `Year_of_Release` | Dataset collected Dec 2016 — recent years incomplete |

---

## 👥 Group Members

- Nicklas Skoglund
- Constantine Diamantis

---

## 🏫 Course Info

**Course:** Unsupervised Machine Learning  
**School:** Jensen Education  
**Date:** 27 April 2026