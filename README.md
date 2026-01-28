# Customer Behavior Analysis and Segmentation in Banking Using PySpark

A Master's-level research project implementing scalable customer segmentation for retail banking using Apache Spark's distributed machine learning library (PySpark MLlib).

## Project Overview

This project demonstrates a production-ready customer segmentation pipeline that:

- Implements K-Means and Bisecting K-Means clustering using PySpark MLlib
- Performs rigorous cluster evaluation using Silhouette scores across multiple configurations (k=2 to k=10)
- Generates business-interpretable customer segments with comprehensive profiling
- Analyzes post-segmentation churn patterns for validation
- Emphasizes scalable, distributed computing architecture applicable to large-scale banking data

## Key Differentiation

This study differs from traditional customer segmentation approaches by emphasizing:

- **Scalable PySpark-based pipelines** suitable for production deployment
- **Rigorous clustering evaluation methodology** rather than standalone results
- **Production-ready architecture** with distributed computing design patterns
- **Comprehensive cluster profiling** for business interpretability

## Dataset

**File**: `Bank Customer Churn Prediction.csv`

**Records**: 10,001 customer records

**Features**: 12 attributes including demographics (age, gender, country), behavioral metrics (balance, products, tenure, credit score), and engagement indicators (active member, credit card)

## Project Structure

```
mrvpyspark/
├── Bank Customer Churn Prediction.csv    # Dataset
├── customer_segmentation_pyspark.py      # Main PySpark implementation
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── outputs/                               # Generated outputs (created on execution)
    ├── silhouette_scores.png              # Evaluation visualization
    ├── cluster_projection.png             # Cluster projection visualization
    └── churn_by_cluster.png               # Churn distribution visualization
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Java 8 or higher (required for PySpark)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/mrvpyspark
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify PySpark installation**:
   ```bash
   python -c "import pyspark; print(pyspark.__version__)"
   ```

## Execution Guide

### Running the Segmentation Pipeline

Execute the main PySpark script:

```bash
python customer_segmentation_pyspark.py
```

### Expected Execution Flow

The pipeline executes the following steps:

1. **Data Loading and Schema Inspection**: Loads 10,001 customer records and displays schema
2. **Data Quality Validation**: Checks for missing values and data integrity
3. **Feature Selection and Preprocessing**: Encodes categorical variables (country, gender)
4. **Feature Engineering Pipeline**: Creates standardized feature vectors using VectorAssembler and StandardScaler
5. **Clustering Evaluation**: Evaluates K-Means for k=2 to k=10 using Silhouette scores
6. **Final Model Training**: Trains K-Means with optimal k value
7. **Algorithm Comparison**: Compares K-Means with Bisecting K-Means
8. **Cluster Profiling**: Generates comprehensive segment profiles with descriptive statistics
9. **Post-Segmentation Churn Analysis**: Analyzes churn distribution across segments
10. **Results Saving**: Exports all results to `outputs/` directory

### Execution Time

Expected runtime: **2-5 minutes** on a standard laptop (varies based on system specifications)

## Output Description

### CSV Files

- **`kmeans_evaluation.csv`**: Silhouette scores for each k value (2-10)
- **`cluster_profiles.csv`**: Mean values for all features within each cluster
- **`churn_by_cluster.csv`**: Churn rates and customer counts per cluster
- **`clustered_customers.csv`**: Original dataset with cluster assignments

### Visualizations

- **`silhouette_scores.png`**: Line plot showing Silhouette scores vs. k values
- **`churn_by_cluster.png`**: Bar chart of churn rates across segments

### Console Output

The script provides detailed console output including:
- Data loading statistics
- Schema information
- Clustering evaluation results
- Cluster profiles with business interpretations
- Churn analysis findings



The comprehensive academic report (`academic_report.md`) includes:

- **Abstract**: Research summary and key findings
- **Introduction**: Problem framing and PySpark rationale
- **Related Work**: Literature review and differentiation
- **Methodology**: Detailed technical approach
- **Results**: Cluster evaluation, profiles, and churn analysis
- **Discussion**: Managerial implications and strategies
- **Limitations**: Dataset and methodological constraints
- **Conclusion**: Summary and future directions
- **References**: Academic citations

## Key Findings

The analysis identifies **4 distinct customer segments**:

1. **Standard Balance Customers** (30.8%): Moderate balances, youngest demographic
2. **Zero-Balance Account Holders** (28.9%): Dormant accounts, highest churn risk (26.7%)
3. **High-Balance Savers** (25.2%): Affluent customers, significant deposits
4. **Multi-Product Engagers** (15.1%): Highest product holdings, lowest churn (12.8%)

## Technical Highlights

### PySpark ML Pipeline

```python
StringIndexer → VectorAssembler → StandardScaler → Clustering Algorithm
```

### Clustering Algorithms

- **K-Means**: Partitional clustering with iterative refinement
- **Bisecting K-Means**: Divisive hierarchical clustering

### Evaluation Metric

- **Silhouette Score**: Measures cluster cohesion and separation (range: -1 to 1)

### Distributed Computing Features

- Horizontal scalability through Spark cluster architecture
- Fault tolerance via Resilient Distributed Datasets (RDDs)
- Parallel distance computation and aggregation
- Integration with enterprise Hadoop ecosystems

## Troubleshooting

### Common Issues

**Issue**: `Java not found` error
- **Solution**: Install Java 8+ and set `JAVA_HOME` environment variable

**Issue**: `pyspark module not found`
- **Solution**: Ensure PySpark is installed: `pip install pyspark`

**Issue**: Memory errors during execution
- **Solution**: Increase Spark driver memory in the script:
  ```python
  .config("spark.driver.memory", "4g")
  ```

**Issue**: Visualizations not saving
- **Solution**: Ensure matplotlib backend is configured correctly. The script uses non-interactive backend.

## Author

Master's Level Research Project  
January 2026

## License

This project is created for academic purposes.

## Acknowledgments

- Apache Spark MLlib documentation and community
- Banking dataset contributors
- Academic literature on customer segmentation and distributed computing
