from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import os

spark = SparkSession\
    .builder\
    .appName("Airlines Part2 Data Engineering ")\
    .config("spark.executor.memory","8g")\
    .config("spark.executor.cores","4")\
    .config("spark.driver.memory","6g")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://prod-cdptrialuser19-trycdp-com")\
    .getOrCreate()
    
flights_path="s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/airlines_csv/*"

from IPython.core.display import HTML
HTML('<a href="http://spark-{}.{}">Spark UI</a>'.format(os.getenv("CDSW_ENGINE_ID"),os.getenv("CDSW_DOMAIN")))


# ## Read Data from file

schema = StructType([StructField("FL_DATE", TimestampType(), True),
    StructField("OP_CARRIER", StringType(), True),
    StructField("OP_CARRIER_FL_NUM", StringType(), True),
    StructField("ORIGIN", StringType(), True),
    StructField("DEST", StringType(), True),
    StructField("CRS_DEP_TIME", StringType(), True),
    StructField("DEP_TIME", StringType(), True),
    StructField("DEP_DELAY", DoubleType(), True),
    StructField("TAXI_OUT", DoubleType(), True),
    StructField("WHEELS_OFF", StringType(), True),
    StructField("WHEELS_ON", StringType(), True),
    StructField("TAXI_IN", DoubleType(), True),
    StructField("CRS_ARR_TIME", StringType(), True),
    StructField("ARR_TIME", StringType(), True),
    StructField("ARR_DELAY", DoubleType(), True),
    StructField("CANCELLED", DoubleType(), True),
    StructField("CANCELLATION_CODE", StringType(), True),
    StructField("DIVERTED", DoubleType(), True),
    StructField("CRS_ELAPSED_TIME", DoubleType(), True),
    StructField("ACTUAL_ELAPSED_TIME", DoubleType(), True),
    StructField("AIR_TIME", DoubleType(), True),
    StructField("DISTANCE", DoubleType(), True),
    StructField("CARRIER_DELAY", DoubleType(), True),
    StructField("WEATHER_DELAY", DoubleType(), True),
    StructField("NAS_DELAY", DoubleType(), True),
    StructField("SECURITY_DELAY", DoubleType(), True),
    StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True)])


flight_raw_df = spark.read.csv(
    path=flights_path,
    header=True,
    schema=schema,
    sep=',',
    nullValue='NA'
)

#from pyspark.sql.types import StringType	
#from pyspark.sql.functions import udf,weekofyear

flight_raw_df = flight_raw_df.withColumn('WEEK',weekofyear('FL_DATE').cast('double'))

smaller_data_set = flight_raw_df.select(	
  "WEEK",	
  "FL_DATE",
  "OP_CARRIER",
  "OP_CARRIER_FL_NUM",
  "ORIGIN",
  "DEST",
  "CRS_DEP_TIME",
  "CRS_ARR_TIME",
  "CANCELLED",
  "CRS_ELAPSED_TIME",
  "DISTANCE"
)

# #### Commented out as it has already been run
#smaller_data_set.write.parquet(
#  path="s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/airlines/airline_parquet",
#  mode='overwrite',
#  compression="snappy")

spark.sql("select * from default.smaller_flight_table limit 10").show()

# ## This is added info for Hive optimisation
#
# #### Simple data enrichment
# #### Extract year and month and day and enrich the dataframe 

flight_year_month = flight_raw_df \
  .withColumn("YEAR", year("FL_DATE")) \
  .withColumn("MONTH", month("FL_DATE")) \
  .withColumn("DAYOFMONTH", dayofmonth("FL_DATE"))
  
flight_year_month.cache()
flight_year_month.createOrReplaceTempView('flights_raw')

# #### Save table in with no optimisation

# #### Commented out as it has already been run
#flight_year_month.write.saveAsTable(
#  'default.flight_not_partitioned', 
#   format='orc', 
#   mode='overwrite', 
#   path='s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/airlines/flight_not_partitioned')


# ## optimize - order + partition
# When saving data to Hive or Impala, it is important to optimize the data for reading
# This usually entails : 
# 1. Some type of ordering (in this case by Year, Month, Day)
# 2. Some type of partionning of the table 
#    This has a *major impact* when filtering / slicing 

flight_year_month = flight_year_month.sortWithinPartitions(["YEAR","MONTH","DAYOFMONTH"])
print('Dataset has {} lines'.format(flight_year_month.count()))

# #### Save data in Hive 

# Required to insert into a partitionned table
spark.sql("SET hive.exec.dynamic.partition = true")
spark.sql("SET hive.exec.dynamic.partition.mode = nonstrict")

# #### Commented out as it has already been run
#flight_year_month.write.saveAsTable(
#  'default.flight_partitioned', 
#   format='orc', 
#   mode='overwrite', 
#   path='s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/airlines/flight_partitioned',
#   partitionBy=('YEAR', 'MONTH')) 
  
spark.sql("SHOW PARTITIONS default.flight_partitioned").show(100)
spark.sql("ANALYZE TABLE default.flight_partitioned COMPUTE STATISTICS")

#
# --------- Read test comparison --------------
# ## Read Data - difference in read with partitionned table
# Predicate filtering
Year = 2015
Month = 10

# ### Complete dataset size

dataset_count = spark.sql("select count(*) from default.flight_partitioned").first()
print('Dataset has {} lines'.format(dataset_count['count(1)']))

# ###  Read non partitionned table
import time
start_time1 = time.time()
statement = 'select * from default.flight_not_partitioned where YEAR = {} and MONTH = {}'.format(Year, Month)
print(statement)
flight_df1 = spark.sql(statement)

print('dataset has {} lines'.format(flight_df1.count()))
print("--- %s seconds ---" % (time.time() - start_time1))


# ###  Read partitionned table
import time
start_time2 = time.time()
statement = 'select * from default.flight_partitioned where YEAR = {} and MONTH = {}'.format(Year, Month)
print(statement)
flight_df2 = spark.sql(statement)

print('dataset has {} lines'.format(flight_df2.count()))
print("--- %s seconds ---" % (time.time() - start_time2))
