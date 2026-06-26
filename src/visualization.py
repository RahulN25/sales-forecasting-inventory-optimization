import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def setup_style():
    """Sets standard styling for visual output."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)

def plot_sales_trend(df, save_path=None):
    """Plots total weekly sales across all stores over time."""
    setup_style()
    timeline_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    plt.figure(figsize=(15, 6))
    plt.plot(timeline_sales['Date'], timeline_sales['Weekly_Sales'], color='royalblue', linewidth=2, marker='o', markersize=4)
    plt.title('Total Weekly Sales Across All Stores (2010 - 2012)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    
    # Add vertical lines for Thanksgiving and Christmas
    holidays = {
        '2010-11-26': ('red', 'Thanksgiving 2010'),
        '2010-12-24': ('darkred', 'Christmas 2010'),
        '2011-11-25': ('red', 'Thanksgiving 2011'),
        '2011-12-23': ('darkred', 'Christmas 2011')
    }
    
    for date_str, (color, label) in holidays.items():
        plt.axvline(pd.to_datetime(date_str), color=color, linestyle='--', alpha=0.7, label=label)
        
    plt.legend(frameon=True)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_sales_by_type(df, save_path=None):
    """Plots sales distribution and store size by type."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.boxplot(ax=axes[0], data=df, x='Type', y='Weekly_Sales', showfliers=False, palette='muted')
    axes[0].set_title('Weekly Sales Distribution by Store Type', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Store Type')
    axes[0].set_ylabel('Weekly Sales ($)')
    
    sns.boxplot(ax=axes[1], data=df.drop_duplicates(subset=['Store']), x='Type', y='Size', palette='muted')
    axes[1].set_title('Store Size Distribution by Store Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Store Type')
    axes[1].set_ylabel('Store Size (sq ft)')
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_holiday_impact(df, save_path=None):
    """Plots average sales comparing holiday and non-holiday weeks."""
    setup_style()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='IsHoliday', y='Weekly_Sales', ci=95, palette='Set2')
    plt.title('Average Weekly Sales: Holiday vs. Non-Holiday Weeks', fontsize=14, fontweight='bold')
    plt.xlabel('Is Holiday Week')
    plt.ylabel('Average Weekly Sales ($)')
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_correlation_matrix(df, save_path=None):
    """Plots a correlation heatmap for sales and numerical features."""
    setup_style()
    cols_to_corr = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 
                    'CPI', 'Unemployment', 'IsHoliday', 
                    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    
    # Filter out columns that don't exist in df
    cols_to_corr = [col for col in cols_to_corr if col in df.columns]
    
    corr_matrix = df[cols_to_corr].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Sales and Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
