from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, OneHotEncoderEstimator
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf,substring,weekofyear,concat,col,when,length,lit
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession\
    .builder\
    .appName("Airline ML")\
    .config("spark.executor.memory","16g")\
    .config("spark.executor.cores","4")\
    .config("spark.driver.memory","6g")\
    .config("spark.executor.instances","10")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field")\
.getOrCreate()

#.config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\

flight_df_original =spark.read.parquet(
  "s3a://ml-field/demo/flight-analysis/data/airline_parquet_partitioned/",
)

# You can uncomment this and comment the previous line to get the able from Hive rather.
#!cp /home/cdsw/hive-site.xml /etc/hadoop/conf/
#flight_df_original = spark.sql("select * from default.flight_test_table")

flight_df = flight_df_original.na.drop()
flight_df.persist()


flight_df = flight_df\
   .withColumn(
       'CRS_DEP_HOUR',
       when(
           length(col("CRS_DEP_TIME")) == 4,col("CRS_DEP_TIME")
       )\
       .otherwise(concat(lit("0"),col("CRS_DEP_TIME")))
   )

flight_df = flight_df.withColumn('CRS_DEP_HOUR',col('CRS_DEP_HOUR').cast('double'))
flight_df = flight_df.withColumn('WEEK',weekofyear('FL_DATE').cast('double'))


numeric_cols = ["CRS_ELAPSED_TIME","DISTANCE","WEEK","CRS_DEP_HOUR"]

op_carrier_indexer = StringIndexer(inputCol ='OP_CARRIER', outputCol = 'OP_CARRIER_INDEXED',handleInvalid="keep")

origin_indexer = StringIndexer(inputCol ='ORIGIN', outputCol = 'ORIGIN_INDEXED',handleInvalid="keep")

dest_indexer = StringIndexer(inputCol ='DEST', outputCol = 'DEST_INDEXED',handleInvalid="keep")


indexer_encoder = OneHotEncoderEstimator(
    inputCols = ['OP_CARRIER_INDEXED','ORIGIN_INDEXED','DEST_INDEXED'],
    outputCols= ['OP_CARRIER_ENCODED','ORIGIN_ENCODED','DEST_ENCODED']
)

input_cols=[
    'OP_CARRIER_ENCODED',
    'ORIGIN_ENCODED',
    'DEST_ENCODED'] + numeric_cols

assembler = VectorAssembler(
    inputCols = input_cols,
    outputCol = 'features')


lr = LogisticRegression(featuresCol = 'features', labelCol = 'CANCELLED', maxIter=15, elasticNetParam = 0.0,regParam = 0.01)

pipeline = Pipeline(
  stages=[        
    op_carrier_indexer,
    origin_indexer,
    dest_indexer,
    indexer_encoder,
    assembler,
    lr
  ]
)

(train, test) = flight_df.randomSplit([0.7, 0.3])

lrModel = pipeline.fit(train)

predictionslr = lrModel.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="CANCELLED",metricName="areaUnderROC")
evaluator.evaluate(predictionslr)

## Already Done
#lrModel.write().overwrite().save("s3a://ml-field/demo/flight-analysis/data/models/large_model")
