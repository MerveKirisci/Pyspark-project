"""
Customer Behavior Analysis and Segmentation in Banking Using PySpark

This module implements a scalable customer segmentation pipeline for retail banking
using PySpark MLlib. The implementation demonstrates production-ready distributed
computing techniques applicable to large-scale banking datasets.

Author: Merve KİRİŞCİ
Date: January 2026
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev, min as spark_min, max as spark_max
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class BankCustomerSegmentation:
    """
    A scalable customer segmentation pipeline using PySpark MLlib.
    
    This class encapsulates the entire segmentation workflow including data loading,
    preprocessing, feature engineering, clustering, and evaluation.
    """
    
    def __init__(self, data_path, output_dir="outputs"):
        """
        Initialize the segmentation pipeline.
        
        Args:
            data_path (str): Path to the customer data CSV file
            output_dir (str): Directory for saving outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Spark session with optimized configuration
        self.spark = SparkSession.builder \
            .appName("BankCustomerSegmentation") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        # Set log level to reduce verbosity
        self.spark.sparkContext.setLogLevel("WARN")
        
        self.df = None
        self.processed_df = None
        self.feature_cols = None
        self.scaled_df = None
        self.best_k = None
        self.best_model = None
        
    def load_data(self):
        """
        Load customer data from CSV into Spark DataFrame.
        
        Returns:
            pyspark.sql.DataFrame: Loaded customer data
        """
        print("=" * 80)
        print("STEP 1: DATA LOADING AND SCHEMA INSPECTION")
        print("=" * 80)
        
        # Load data with header and infer schema
        self.df = self.spark.read.csv(
            self.data_path,
            header=True,
            inferSchema=True
        )
        
        print(f"\n✓ Successfully loaded {self.df.count():,} customer records")
        print(f"✓ Number of features: {len(self.df.columns)}")
        
        print("\nDataset Schema:")
        self.df.printSchema()
        
        print("\nFirst 5 records:")
        self.df.show(5, truncate=False)
        
        print("\nBasic Statistics:")
        self.df.describe().show()
        
        return self.df
    
    def validate_data(self):
        """
        Perform data quality validation including missing value analysis.
        """
        print("\n" + "=" * 80)
        print("STEP 2: DATA QUALITY VALIDATION")
        print("=" * 80)
        
        # Check for missing values
        print("\nMissing Value Analysis:")
        missing_counts = self.df.select([
            count(col(c)).alias(c) for c in self.df.columns
        ])
        
        total_rows = self.df.count()
        print(f"Total rows: {total_rows:,}")
        
        for column in self.df.columns:
            non_null = missing_counts.select(column).collect()[0][0]
            missing = total_rows - non_null
            if missing > 0:
                print(f"  {column}: {missing} missing ({missing/total_rows*100:.2f}%)")
        
        print("\n✓ Data quality validation complete")
    
    def preprocess_data(self):
        """
        Preprocess data by selecting relevant features and encoding categorical variables.
        
        Note: The 'churn' variable is intentionally excluded from clustering inputs
        to maintain unsupervised learning integrity. It will be reintroduced only
        for post-hoc segment analysis.
        """
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE SELECTION AND PREPROCESSING")
        print("=" * 80)
        
        # Select features for clustering (exclude customer_id and churn)
        # Behavioral features: credit_score, age, tenure, balance, products_number, estimated_salary
        # Engagement features: credit_card, active_member
        # Demographic features: country, gender
        
        print("\nFeature Selection Strategy:")
        print("  Behavioral: credit_score, age, tenure, balance, products_number, estimated_salary")
        print("  Engagement: credit_card, active_member")
        print("  Demographic: country, gender")
        print("  Excluded: customer_id (identifier), churn (target for post-analysis)")
        
        # Encode categorical variables
        print("\nEncoding categorical variables...")
        
        # StringIndexer for country
        country_indexer = StringIndexer(
            inputCol="country",
            outputCol="country_index",
            handleInvalid="keep"
        )
        
        # StringIndexer for gender
        gender_indexer = StringIndexer(
            inputCol="gender",
            outputCol="gender_index",
            handleInvalid="keep"
        )
        
        # Apply indexers
        country_model = country_indexer.fit(self.df)
        self.processed_df = country_model.transform(self.df)
        
        gender_model = gender_indexer.fit(self.processed_df)
        self.processed_df = gender_model.transform(self.processed_df)
        
        print("✓ Categorical encoding complete")
        
        # Define feature columns for clustering
        self.feature_cols = [
            'credit_score', 'age', 'tenure', 'balance', 'products_number',
            'credit_card', 'active_member', 'estimated_salary',
            'country_index', 'gender_index'
        ]
        
        print(f"\n✓ Selected {len(self.feature_cols)} features for clustering")
        print(f"  Features: {', '.join(self.feature_cols)}")
        
        return self.processed_df
    
    def create_feature_pipeline(self):
        """
        Create a feature engineering pipeline with VectorAssembler and StandardScaler.
        
        StandardScaler is critical for distance-based clustering algorithms as it
        ensures all features contribute equally to distance calculations, preventing
        features with larger scales from dominating the clustering process.
        """
        print("\n" + "=" * 80)
        print("STEP 4: FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
        
        print("\nCreating feature engineering pipeline:")
        print("  1. VectorAssembler: Combine features into a single vector")
        print("  2. StandardScaler: Normalize features (mean=0, std=1)")
        
        print("\nRationale for StandardScaler:")
        print("  - Distance-based clustering (K-Means) is sensitive to feature scales")
        print("  - Features like 'balance' (0-200K) would dominate 'tenure' (0-10)")
        print("  - Standardization ensures equal contribution from all features")
        
        # VectorAssembler to combine features
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features_unscaled"
        )
        
        # StandardScaler for normalization
        scaler = StandardScaler(
            inputCol="features_unscaled",
            outputCol="features",
            withMean=True,
            withStd=True
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        
        # Fit and transform
        print("\nFitting feature pipeline...")
        pipeline_model = pipeline.fit(self.processed_df)
        self.scaled_df = pipeline_model.transform(self.processed_df)
        
        print("✓ Feature engineering pipeline complete")
        print(f"✓ Scaled feature vectors created for {self.scaled_df.count():,} records")
        
        return self.scaled_df
    
    def evaluate_clustering_range(self, min_k=2, max_k=10):
        """
        Evaluate K-Means clustering for a range of k values using Silhouette score.
        
        Args:
            min_k (int): Minimum number of clusters
            max_k (int): Maximum number of clusters
            
        Returns:
            dict: Dictionary mapping k values to silhouette scores
        """
        print("\n" + "=" * 80)
        print("STEP 5: CLUSTERING EVALUATION (K-MEANS)")
        print("=" * 80)
        
        print(f"\nEvaluating K-Means for k = {min_k} to {max_k}")
        print("Evaluation Metric: Silhouette Score (higher is better)")
        print("  - Range: [-1, 1]")
        print("  - > 0.5: Strong cluster structure")
        print("  - 0.25-0.5: Moderate cluster structure")
        print("  - < 0.25: Weak cluster structure")
        
        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        
        silhouette_scores = {}
        
        print("\nK-Means Evaluation Results:")
        print("-" * 60)
        print(f"{'k':<5} {'Silhouette Score':<20} {'Cluster Sizes':<30}")
        print("-" * 60)
        
        for k in range(min_k, max_k + 1):
            # Train K-Means
            kmeans = KMeans(
                featuresCol="features",
                predictionCol="prediction",
                k=k,
                seed=42,
                maxIter=100
            )
            
            model = kmeans.fit(self.scaled_df)
            predictions = model.transform(self.scaled_df)
            
            # Evaluate
            score = evaluator.evaluate(predictions)
            silhouette_scores[k] = score
            
            # Get cluster sizes
            cluster_sizes = predictions.groupBy("prediction").count().orderBy("prediction").collect()
            sizes_str = ", ".join([f"C{row['prediction']}:{row['count']}" for row in cluster_sizes])
            
            print(f"{k:<5} {score:<20.4f} {sizes_str:<30}")
        
        print("-" * 60)
        
        # Find best k
        self.best_k = max(silhouette_scores, key=silhouette_scores.get)
        print(f"\n✓ Best k: {self.best_k} (Silhouette Score: {silhouette_scores[self.best_k]:.4f})")
        
        # Save evaluation results
        eval_df = pd.DataFrame(list(silhouette_scores.items()), columns=['k', 'silhouette_score'])
        eval_df.to_csv(f"{self.output_dir}/kmeans_evaluation.csv", index=False)
        print(f"✓ Evaluation results saved to {self.output_dir}/kmeans_evaluation.csv")
        
        # Plot silhouette scores
        self._plot_silhouette_scores(silhouette_scores)
        
        return silhouette_scores
    
    def _plot_silhouette_scores(self, scores):
        """
        Plot silhouette scores for different k values.
        
        Args:
            scores (dict): Dictionary mapping k values to silhouette scores
        """
        plt.figure(figsize=(10, 6))
        k_values = sorted(scores.keys())
        silhouette_values = [scores[k] for k in k_values]
        
        plt.plot(k_values, silhouette_values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('K-Means Clustering Evaluation: Silhouette Score vs. k', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        
        # Highlight best k
        best_k = max(scores, key=scores.get)
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/silhouette_scores.png", dpi=300, bbox_inches='tight')
        print(f"✓ Silhouette score plot saved to {self.output_dir}/silhouette_scores.png")
        plt.close()
    
    def train_final_model(self, k=None):
        """
        Train the final K-Means model with the optimal k value.
        
        Args:
            k (int): Number of clusters (uses best_k if not specified)
        """
        print("\n" + "=" * 80)
        print("STEP 6: TRAINING FINAL CLUSTERING MODEL")
        print("=" * 80)
        
        if k is None:
            k = self.best_k
        
        print(f"\nTraining K-Means with k={k}...")
        
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=k,
            seed=42,
            maxIter=100
        )
        
        self.best_model = kmeans.fit(self.scaled_df)
        self.scaled_df = self.best_model.transform(self.scaled_df)
        
        print(f"✓ Final model trained successfully")
        print(f"✓ Cluster assignments added to dataset")
        
        return self.best_model
    
    def compare_bisecting_kmeans(self):
        """
        Compare K-Means with Bisecting K-Means algorithm.
        
        Bisecting K-Means uses a divisive hierarchical approach, which can be
        more efficient for large datasets and may produce different cluster structures.
        """
        print("\n" + "=" * 80)
        print("STEP 7: COMPARING WITH BISECTING K-MEANS")
        print("=" * 80)
        
        print("\nBisecting K-Means Characteristics:")
        print("  - Divisive hierarchical clustering approach")
        print("  - Starts with all data in one cluster, recursively splits")
        print("  - Often more efficient for large datasets")
        print("  - May produce different cluster structures than standard K-Means")
        
        print(f"\nTraining Bisecting K-Means with k={self.best_k}...")
        
        bisecting_kmeans = BisectingKMeans(
            featuresCol="features",
            predictionCol="bisecting_prediction",
            k=self.best_k,
            seed=42,
            maxIter=100
        )
        
        bisecting_model = bisecting_kmeans.fit(self.scaled_df)
        bisecting_predictions = bisecting_model.transform(self.scaled_df)
        
        # Evaluate both models
        evaluator = ClusteringEvaluator(
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        
        kmeans_score = evaluator.evaluate(
            self.scaled_df.withColumnRenamed("cluster", "prediction")
        )
        
        bisecting_score = evaluator.evaluate(
            bisecting_predictions.withColumnRenamed("bisecting_prediction", "prediction")
        )
        
        print("\nComparison Results:")
        print("-" * 60)
        print(f"{'Algorithm':<25} {'Silhouette Score':<20}")
        print("-" * 60)
        print(f"{'K-Means':<25} {kmeans_score:<20.4f}")
        print(f"{'Bisecting K-Means':<25} {bisecting_score:<20.4f}")
        print("-" * 60)
        
        if kmeans_score > bisecting_score:
            print(f"\n✓ K-Means performs better (Δ = {kmeans_score - bisecting_score:.4f})")
        else:
            print(f"\n✓ Bisecting K-Means performs better (Δ = {bisecting_score - kmeans_score:.4f})")
        
        return {
            'kmeans': kmeans_score,
            'bisecting_kmeans': bisecting_score
        }
    
    def profile_clusters(self):
        """
        Generate comprehensive cluster profiles with descriptive statistics.
        """
        print("\n" + "=" * 80)
        print("STEP 8: CLUSTER PROFILING AND INTERPRETATION")
        print("=" * 80)
        
        print("\nGenerating cluster profiles...")
        
        # Aggregate statistics by cluster
        cluster_profiles = self.scaled_df.groupBy("cluster").agg(
            count("*").alias("size"),
            mean("credit_score").alias("avg_credit_score"),
            mean("age").alias("avg_age"),
            mean("tenure").alias("avg_tenure"),
            mean("balance").alias("avg_balance"),
            mean("products_number").alias("avg_products"),
            mean("estimated_salary").alias("avg_salary"),
            mean("credit_card").alias("pct_credit_card"),
            mean("active_member").alias("pct_active"),
            mean("country_index").alias("country_mode"),
            mean("gender_index").alias("gender_mode")
        ).orderBy("cluster")
        
        print("\nCluster Profiles:")
        cluster_profiles.show(truncate=False)
        
        # Convert to Pandas for easier analysis
        profiles_pd = cluster_profiles.toPandas()
        
        # Save to CSV
        profiles_pd.to_csv(f"{self.output_dir}/cluster_profiles.csv", index=False)
        print(f"✓ Cluster profiles saved to {self.output_dir}/cluster_profiles.csv")
        
        # Assign business-oriented segment names
        print("\n" + "=" * 80)
        print("BUSINESS-ORIENTED SEGMENT INTERPRETATION")
        print("=" * 80)
        
        segment_names = self._assign_segment_names(profiles_pd)
        
        for cluster_id, name in segment_names.items():
            profile = profiles_pd[profiles_pd['cluster'] == cluster_id].iloc[0]
            print(f"\n{'='*60}")
            print(f"Cluster {cluster_id}: {name}")
            print(f"{'='*60}")
            print(f"  Size: {int(profile['size']):,} customers ({profile['size']/profiles_pd['size'].sum()*100:.1f}%)")
            print(f"  Avg Credit Score: {profile['avg_credit_score']:.0f}")
            print(f"  Avg Age: {profile['avg_age']:.1f} years")
            print(f"  Avg Tenure: {profile['avg_tenure']:.1f} years")
            print(f"  Avg Balance: ${profile['avg_balance']:,.2f}")
            print(f"  Avg Products: {profile['avg_products']:.2f}")
            print(f"  Avg Salary: ${profile['avg_salary']:,.2f}")
            print(f"  Credit Card Holders: {profile['pct_credit_card']*100:.1f}%")
            print(f"  Active Members: {profile['pct_active']*100:.1f}%")
        
        return profiles_pd, segment_names
    
    def _assign_segment_names(self, profiles_pd):
        """
        Assign meaningful business-oriented names to clusters based on characteristics.
        
        Args:
            profiles_pd (pd.DataFrame): Cluster profiles
            
        Returns:
            dict: Mapping of cluster IDs to segment names
        """
        segment_names = {}
        
        for idx, row in profiles_pd.iterrows():
            cluster_id = int(row['cluster'])
            
            # Heuristic-based naming
            if row['avg_balance'] > profiles_pd['avg_balance'].median() and \
               row['avg_salary'] > profiles_pd['avg_salary'].median():
                segment_names[cluster_id] = "High-Value Customers"
            elif row['avg_age'] < 35 and row['avg_tenure'] < 3:
                segment_names[cluster_id] = "Young Newcomers"
            elif row['pct_active'] < 0.5:
                segment_names[cluster_id] = "Inactive/At-Risk"
            elif row['avg_products'] > profiles_pd['avg_products'].median():
                segment_names[cluster_id] = "Multi-Product Engagers"
            elif row['avg_balance'] < profiles_pd['avg_balance'].median():
                segment_names[cluster_id] = "Low-Balance Savers"
            else:
                segment_names[cluster_id] = f"Standard Segment {cluster_id}"
        
        return segment_names
    
    def analyze_churn_by_segment(self):
        """
        Post-segmentation analysis: Analyze churn distribution across clusters.
        
        Note: This is a post-hoc analysis. The churn variable was NOT used in
        clustering to maintain unsupervised learning integrity.
        """
        print("\n" + "=" * 80)
        print("STEP 9: POST-SEGMENTATION CHURN ANALYSIS")
        print("=" * 80)
        
        print("\nIMPORTANT: Churn was NOT used in clustering (unsupervised learning)")
        print("This analysis reintroduces churn to understand segment characteristics")
        
        # Analyze churn by cluster
        churn_analysis = self.scaled_df.groupBy("cluster").agg(
            count("*").alias("total_customers"),
            mean("churn").alias("churn_rate")
        ).orderBy("cluster")
        
        print("\nChurn Rate by Cluster:")
        churn_analysis.show()
        
        # Convert to Pandas
        churn_pd = churn_analysis.toPandas()
        churn_pd.to_csv(f"{self.output_dir}/churn_by_cluster.csv", index=False)
        print(f"✓ Churn analysis saved to {self.output_dir}/churn_by_cluster.csv")
        
        # Visualize churn rates
        self._plot_churn_by_cluster(churn_pd)
        
        return churn_pd
    
    def _plot_churn_by_cluster(self, churn_pd):
        """
        Visualize churn rates across clusters.
        
        Args:
            churn_pd (pd.DataFrame): Churn analysis results
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(churn_pd['cluster'], churn_pd['churn_rate'] * 100, 
                      color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Churn Rate (%)', fontsize=12)
        ax.set_title('Churn Rate by Customer Segment', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/churn_by_cluster.png", dpi=300, bbox_inches='tight')
        print(f"✓ Churn visualization saved to {self.output_dir}/churn_by_cluster.png")
        plt.close()
    
    def save_results(self):
        """
        Save final clustered dataset.
        """
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Select relevant columns for output
        output_df = self.scaled_df.select(
            'customer_id', 'cluster', 'credit_score', 'country', 'gender',
            'age', 'tenure', 'balance', 'products_number', 'credit_card',
            'active_member', 'estimated_salary', 'churn'
        )
        
        # Convert to Pandas and save
        output_pd = output_df.toPandas()
        output_pd.to_csv(f"{self.output_dir}/clustered_customers.csv", index=False)
        print(f"✓ Clustered dataset saved to {self.output_dir}/clustered_customers.csv")
        print(f"  Total records: {len(output_pd):,}")
        
    def stop(self):
        """
        Stop the Spark session.
        """
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print("\n✓ All analyses complete")
        print(f"✓ Results saved to: {self.output_dir}/")
        self.spark.stop()


def main():
    """
    Main execution function for the customer segmentation pipeline.
    """
    print("\n" + "=" * 80)
    print("CUSTOMER BEHAVIOR ANALYSIS AND SEGMENTATION IN BANKING")
    print("Using PySpark for Scalable Distributed Computing")
    print("=" * 80)
    
    # Initialize pipeline
    segmentation = BankCustomerSegmentation(
        data_path="Bank Customer Churn Prediction.csv",
        output_dir="outputs"
    )
    
    # Execute pipeline
    try:
        # Data loading and validation
        segmentation.load_data()
        segmentation.validate_data()
        
        # Preprocessing and feature engineering
        segmentation.preprocess_data()
        segmentation.create_feature_pipeline()
        
        # Clustering evaluation and model training
        segmentation.evaluate_clustering_range(min_k=2, max_k=10)
        segmentation.train_final_model()
        
        # Algorithm comparison
        segmentation.compare_bisecting_kmeans()
        
        # Cluster profiling and interpretation
        segmentation.profile_clusters()
        
        # Post-segmentation analysis
        segmentation.analyze_churn_by_segment()
        
        # Save results
        segmentation.save_results()
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise
    finally:
        # Clean up
        segmentation.stop()


if __name__ == "__main__":
    main()
