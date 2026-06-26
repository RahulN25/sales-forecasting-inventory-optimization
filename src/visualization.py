import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def setup_style():
    """Sets standard styling for visual output."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)


def save_or_show(save_path=None):
    """Saves figure if path is given, otherwise shows it."""
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sales_trend(df, save_path=None):
    """Plots total weekly sales across all stores over time."""
    setup_style()

    timeline_sales = df.groupby("Date")["Weekly_Sales"].sum().reset_index()

    plt.figure(figsize=(15, 6))
    plt.plot(
        timeline_sales["Date"],
        timeline_sales["Weekly_Sales"],
        linewidth=2,
        marker="o",
        markersize=4
    )

    plt.title("Total Weekly Sales Across All Stores", fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Total Weekly Sales")

    # Automatically mark holiday weeks
    holiday_dates = df[df["IsHoliday"] == True]["Date"].drop_duplicates()

    for holiday_date in holiday_dates:
        plt.axvline(
            holiday_date,
            linestyle="--",
            alpha=0.4
        )

    save_or_show(save_path)


def plot_sales_by_type(df, save_path=None):
    """Plots weekly sales distribution and store size by store type."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(
        ax=axes[0],
        data=df,
        x="Type",
        y="Weekly_Sales",
        showfliers=False
    )

    axes[0].set_title("Weekly Sales Distribution by Store Type", fontweight="bold")
    axes[0].set_xlabel("Store Type")
    axes[0].set_ylabel("Weekly Sales")

    store_info = df[["Store", "Type", "Size"]].drop_duplicates()

    sns.boxplot(
        ax=axes[1],
        data=store_info,
        x="Type",
        y="Size"
    )

    axes[1].set_title("Store Size Distribution by Store Type", fontweight="bold")
    axes[1].set_xlabel("Store Type")
    axes[1].set_ylabel("Store Size")

    save_or_show(save_path)


def plot_holiday_impact(df, save_path=None):
    """Compares total weekly sales for holiday and non-holiday weeks."""
    setup_style()

    weekly_sales = (
        df.groupby(["Date", "IsHoliday"])["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=weekly_sales,
        x="IsHoliday",
        y="Weekly_Sales",
        errorbar=("ci", 95)
    )

    plt.title("Total Weekly Sales: Holiday vs Non-Holiday Weeks", fontsize=14, fontweight="bold")
    plt.xlabel("Holiday Week")
    plt.ylabel("Total Weekly Sales")

    save_or_show(save_path)


def plot_correlation_matrix(df, save_path=None):
    """Plots a correlation heatmap for sales and numerical features."""
    setup_style()

    corr_df = df.copy()

    if "IsHoliday" in corr_df.columns:
        corr_df["IsHoliday"] = corr_df["IsHoliday"].astype(int)

    cols_to_corr = [
        "Weekly_Sales",
        "Size",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "IsHoliday",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    cols_to_corr = [col for col in cols_to_corr if col in corr_df.columns]

    corr_matrix = corr_df[cols_to_corr].corr()

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )

    plt.title("Correlation Matrix of Sales and Features", fontsize=16, fontweight="bold")

    save_or_show(save_path)


def plot_monthly_sales(df, save_path=None):
    """Plots monthly sales trend."""
    setup_style()

    monthly_sales = (
        df.set_index("Date")
        .resample("M")["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_sales["Date"], monthly_sales["Weekly_Sales"], marker="o", linewidth=2)

    plt.title("Monthly Sales Trend", fontsize=16, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")

    save_or_show(save_path)


def plot_top_stores(df, save_path=None):
    """Plots top 10 stores by total sales."""
    setup_style()

    top_stores = (
        df.groupby("Store")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(data=top_stores, x="Store", y="Weekly_Sales", order=top_stores["Store"])

    plt.title("Top 10 Stores by Total Sales", fontsize=16, fontweight="bold")
    plt.xlabel("Store")
    plt.ylabel("Total Sales")

    save_or_show(save_path)


def plot_top_departments(df, save_path=None):
    """Plots top 10 departments by total sales."""
    setup_style()

    top_depts = (
        df.groupby("Dept")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(data=top_depts, x="Dept", y="Weekly_Sales", order=top_depts["Dept"])

    plt.title("Top 10 Departments by Total Sales", fontsize=16, fontweight="bold")
    plt.xlabel("Department")
    plt.ylabel("Total Sales")

    save_or_show(save_path)


def plot_missing_values(df, save_path=None):
    """Plots missing values by column."""
    setup_style()

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    plt.figure(figsize=(12, 6))

    if missing.empty:
        plt.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        sns.barplot(x=missing.values, y=missing.index)
        plt.title("Missing Values by Column", fontsize=16, fontweight="bold")
        plt.xlabel("Number of Missing Values")
        plt.ylabel("Column")

    save_or_show(save_path)