import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data():
    """Create visualizations for the dataset"""
    df = pd.read_csv('housing_data.csv')
    
    # Set style
    sns.set_style('whitegrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Price Distribution
    axes[0, 0].hist(df['Price'], bins=50, edgecolor='black')
    axes[0, 0].set_title('House Price Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Price (in $100,000s)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Correlation Heatmap
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                ax=axes[0, 1], cbar_kws={'label': 'Correlation'})
    axes[0, 1].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    # 3. Price vs MedInc (Median Income)
    axes[1, 0].scatter(df['MedInc'], df['Price'], alpha=0.5)
    axes[1, 0].set_title('Price vs Median Income', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Median Income')
    axes[1, 0].set_ylabel('Price')
    
    # 4. House Age Distribution
    axes[1, 1].hist(df['HouseAge'], bins=30, edgecolor='black', color='orange')
    axes[1, 1].set_title('House Age Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('House Age (years)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=300)
    plt.show()
    
    print("✅ Visualizations saved as 'data_visualization.png'")

if __name__ == "__main__":
    visualize_data()
