from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window
import pyspark.sql.functions as F

import seaborn as sns
import matplotlib.pyplot as plt

import logging
import time

logging.basicConfig(level=logging.INFO)
logging.info("Starting script")

start_time = time.time()

# Launch and configure Spark
spark = (SparkSession
    .builder
    .master("local[*]")
    .appName("relevanC test")
    .config("spark.driver.memory", "10g")
    .getOrCreate()
)

# Load the dataframe from the file
raw_df = spark.read.option("header", "true").option("sep", "|").csv("randomized-transactions-202009.psv")

# Adjust date and float types (default is string)
df = (raw_df.withColumn("date_encaissement", F.to_date(F.col("date_encaissement")))
            .withColumn("prix", F.col("prix").cast("float"))
            #.limit(10000)  # for tests only
            .cache()
)

df.show(10)
df.printSchema()

#---------Top 50 stores-----------#
logging.info("Calculating top 50 stores by CA")

top_50_stores_df: DataFrame = (df.select("code_magasin", "prix")
                                 .groupBy("code_magasin").agg(F.sum("prix").alias("ca"))
                                 .orderBy("ca", ascending=False)
                                 .limit(50)
)

# Write dataframe to CSV file
top_50_stores_df.coalesce(1).write.save("top-50-stores", header="true", format="csv", mode="overwrite", sep="|")

# Visual of the KPI
sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=((12, 15)))
sns.set_color_codes("pastel")
sns_plot = sns.barplot(x="ca", y="code_magasin", data=top_50_stores_df.toPandas(), color="b")
ax.set(ylabel="", xlabel="Chiffre d'affaire", title="Classement des 50 premiers magasins, par chiffre d'affaire")
sns_plot.get_figure().savefig("top-50-stores/top-50-stores.png")
plt.close()

#---------Top 100 products-----------#
logging.info("Calculating the top 100 products for each store")

# For each store, calculate the top 100
grouped_df = (df.groupBy("code_magasin", "identifiant_produit")
                .agg({'identifiant_produit': 'count', 'prix': 'sum'})
                .withColumnRenamed("sum(prix)", "ca")
                .withColumnRenamed("count(identifiant_produit)", "nb_ventes_produit")
                .orderBy("identifiant_produit", "nb_ventes_produit", ascending=False)
)

# Create window to rank the products of each store
window = Window.partitionBy([F.col('code_magasin'), F.col('identifiant_produit')]
                            ).orderBy(grouped_df['nb_ventes_produit'].desc())

ordered_df = (grouped_df.select('*', F.rank().over(window).alias('rank')) 
                        .filter(F.col('rank') <= 100)
                        .select("code_magasin", "identifiant_produit", "ca")
)

logging.info("Writing top 100 CSVs on disk")
ordered_df.repartition(1).write.save("top-100-products-by-store",
                                  header="true",
                                  format="csv",
                                  mode="overwrite",
                                  sep="|",
                                  partitionBy="code_magasin")

logging.info("Writing done, exiting Pyspark")

spark.stop()
"""
# Visual of the KPI
f, ax = plt.subplots(figsize=((12, 15)))
sns.set_color_codes("pastel")
sns_plot = sns.barplot(x="ca", y="code_magasin", data=top_100_products_df.toPandas(), color="b")
ax.set(ylabel="", xlabel="Chiffre d'affaire",
        title="Classement des 50 produits les plus populaires, par chiffre d'affaire")
sns_plot.get_figure().savefig(f"relevanC/top-products-by-store/top-100-products-store-{magasin[0]}/"
                                f"top-100-products-store-{magasin[0]}.png")
plt.close()
"""
logging.info(f"--- Time taken to run the script: {time.time() - start_time} ---")

