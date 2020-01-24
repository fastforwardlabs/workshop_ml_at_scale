from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os


spark = SparkSession\
    .builder\
    .appName("Airline")\
    .config("spark.executor.memory","8g")\
    .config("spark.executor.cores","4")\
    .config("spark.driver.memory","6g")\
    .config("spark.executor.instances","3")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://prod-cdptrialuser19-trycdp-com")\
    .getOrCreate()


from IPython.core.display import HTML
HTML('<a href="http://spark-{}.{}">Spark UI</a>'.format(os.getenv("CDSW_ENGINE_ID"),os.getenv("CDSW_DOMAIN")))

schema = StructType(
  [
    StructField("FL_DATE", TimestampType(), True),
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
    StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True)
  ]
)

flight_df = spark.read.csv(
  path="s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/airlines_csv/*",
  header=True,
  schema=schema
)

from pyspark.sql.types import StringType
from pyspark.sql.functions import udf,weekofyear

# This has been added to help with partitioning.
flight_df = flight_df\
  .withColumn('WEEK',weekofyear('FL_DATE').cast('double'))

smaller_data_set = flight_df.select(
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

smaller_data_set.show()


from pyspark.sql  import SQLContext
sqlContext = SQLContext(spark)

spark.sql("show databases").show()
spark.sql("show tables in default").show()

sqlContext.registerDataFrameAsTable(flight_df, "temp_flight_df")
spark.sql("select count(*) from temp_flight_df").show()


# This is commented out as it has already been run
#smaller_data_set.write.parquet(
#  path="s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/airlines/airline_parquet",
#  mode='overwrite', 
#  compression="snappy")


# This will write the table to Hive to be used for other SQL services.
#smaller_data_set.write.saveAsTable(
#  'default.smaller_flight_table', 
#  format='parquet', 
#  mode='overwrite', 
#  path='s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/warehouse/tablespace/external/hive/smaller_flight_table')

spark.sql("select count(*) from flight_test_table").show()

spark.sql("select * from default.smaller_flight_table limit 10").show()

