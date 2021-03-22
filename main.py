from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F

import seaborn as sns
import matplotlib.pyplot as plt

import time


start_time = time.time()

# Launch and configure Spark
spark = (SparkSession
    .builder
    .appName("relevanC test")
    #.config("spark.driver.memory", "17g")
    .getOrCreate()
)

# Load the dataframe from the file
raw_df = spark.read.option("header", "true").option("sep", "|").csv("randomized-transactions-202009.psv")

# Adjust date and float types (default is string)
df = (raw_df.withColumn("date_encaissement", F.to_date(F.col("date_encaissement")))
            .withColumn("prix", F.col("prix").cast("float"))
            .limit(10000)  # for tests only
            .cache()
)

df.show(10)
df.printSchema()

#---------Top 50 stores-----------#
top_50_stores_df: DataFrame = (df.select("code_magasin", "prix")
                                 .groupBy("code_magasin").agg(sum("prix").alias("ca"))
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
# Get list of unique store ids
code_magasin_list = df.select("code_magasin").distinct().collect()


# For each store, calculate the top 100
for magasin in code_magasin_list:
    top_100_products_df = (df.filter(F.col("code_magasin") == magasin[0])
                             .select("identifiant_produit", "prix")
                             .groupBy("identifiant_produit")
                             .agg({'identifiant_produit': 'count', 'prix': 'sum'})
                             .withColumnRenamed("sum(prix)", "ca")
                             .withColumn("code_magasin", F.lit(magasin[0]))
                             .orderBy("count(identifiant_produit)", ascending=False)
                             .select("code_magasin", "identifiant_produit", "ca")
                             .limit(100)
    )
    # Write dataframe to CSV file
    top_100_products_df.write.save(f"relevanC/top-products-by-store/top-100-products-store-{magasin[0]}", header="true", format="csv", mode="overwrite", sep="|")

    # Visual of the KPI
    f, ax = plt.subplots(figsize=((12, 15)))
    sns.set_color_codes("pastel")
    sns_plot = sns.barplot(x="ca", y="code_magasin", data=top_100_products_df.toPandas(), color="b")
    ax.set(ylabel="", xlabel="Chiffre d'affaire", title="Classement des 50 produits les plus populaires, par chiffre d'affaire")
    sns_plot.get_figure().savefig(f"relevanC/top-products-by-store/top-100-products-store-{magasin[0]}/top-100-products-store-{magasin[0]}.png")
    plt.close()

print(f"--- Time taken to run the script: {time.time() - start_time} ---")
