# # Part 3 and 4, Building and training the Model

# This notebook goes through the process of selecting and building the best model to use with this problem. There are several ML algorithm options available in the SparkML library. As this is a binary classification problem, we will use Logistic Regression, Random Forest and Gradient Boosted Tree classifiers.

# The aim here is to train all three kinds of models and then test to see which provides the best results. The next step is to optimise i.e. the chosen models hyper parameters to then have the best option for solving this problem. 

#---

# Load libraries
library(ggplot2)
library(sparklyr)
library(dplyr)


#---

# ## Start the Spark Session
# Note the changes to `spark.executor.memory` and `spark.executor.instances`. SparkML processes are more memory intensive and therefore require a different configuration to run efficiently.

#spark_home_set("/etc/spark/")

config <- spark_config()
config$spark.executor.memory <- "8g"
config$spark.executor.cores <- "2"
config$spark.driver.memory <- "6g"
config$spark.executor.instances <- "3"
config$spark.yarn.access.hadoopFileSystems <- "s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/"
spark <- spark_connect(master = "yarn-client", config=config)

#---

#Adding a link to the Spark UI for demo purposes

library(cdsw)
html(paste("<a href='http://spark-",Sys.getenv("CDSW_ENGINE_ID"),".",Sys.getenv("CDSW_DOMAIN"),"' target='_blank'>Spark UI<a>",sep=""))

#---

# ## Import the Data
# The data that is imported was the smaller data set created in Part 2. This is now stored in S3 in parquet format and will therefore load much quicker. To get the larger ML jobs to complete in a reasonable amount of time for a workshop audience, there total data set size is limited to 1,000,000 rows.

#flight_df_original = spark.sql("select * from smaller_flight_table")

flight_df <- sdf_sql(spark, "select * from smaller_flight_table")

#flight_df <- spark_read_parquet(
#  spark,
#  name = "flight_df",
#  path = "s3a://ml-field/demo/flight-analysis/data/airline_parquet_partitioned/",
#)


sdf_schema(flight_df) %>% as.data.frame %>% t

flight_df <- flight_df %>% na.omit() %>% sdf_sample(fraction = 1/600, replacement = FALSE, seed = 111)

#---

# ## Basic Feature Engineering
# Before applying any training to an ML algorithm, you need to check if the format of columns in your dataset will work with the chosen algorithm. The first change that we need to make is extracting the hour of the day from the time. This is based on an assumption of seasonality during the day. The second change is to extract week of the year using `weekofyear`. The ML aglorithms needs floating point numbers for optimal internal matrix maths, therefore these two new columns are cast using `cast('double')`.

# > HANDY TIP
# > 
# > Each new release of sparkly brings with it more useful functions that makes it easier to do things that you had to use a UDF to do. Also given that you can run spark functions directly means not have to do create custom UDFs.
# > Even with Apache Arrow used for memory storage, using an R UDF in a sparklyr tranform is going to be slower than running pure Spark functions. After spending some time reading through the pyspark function list, I found that I could do what my udf did using a the `ifelse` verb.

# This was the previous UDF created before I discoverd that I can run spark functions directly
#```convert_time_to_hour <- function(x) {
#  if (nchar(x) == 4) {
#    return(x)
#    } 
#  else {
#      return(paste("0",x,sep=""))
#    }
#}
#
#flight_df <- flight_df %>% spark_apply(convert_time_to_hour,columns=c("CRS_DEP_HOUR"))```


flight_df_mutated <- flight_df %>%
  mutate(CRS_DEP_HOUR = ifelse(length(CRS_DEP_TIME) == 4,CRS_DEP_TIME,paste("0",CRS_DEP_TIME,sep=""))) %>% 
  mutate(CRS_DEP_HOUR = as.numeric(substring(CRS_DEP_HOUR,0,2))) %>%
  mutate(WEEK = as.numeric(weekofyear(FL_DATE))) %>%
  select(CANCELLED,DISTANCE,ORIGIN,DEST,WEEK,CRS_DEP_HOUR,OP_CARRIER,CRS_ELAPSED_TIME)

sdf_schema(flight_df) %>% as.data.frame %>% t

#---
# ## Additional Feature Indexing and Encoding

# The next feature engineering requirement is to convert the categorical variables into a numeric form. The `OP_CARRIER`,`ORIGIN` and `DEST` columns are the ones that will be used. All three are `string` type categorical variables. There are 2 methods of dealing with these, one is using a [String Indexer](https://spark.rstudio.com/reference/ft_string_indexer/) and the other is using a [One Hot Encoder](https://spark.rstudio.com/reference/ft_one_hot_encoder/). Logsitic Regression requires one hot encoding, [tree based algorithims do not](https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/). The code below creates a [Spark Pipeline](https://spark.rstudio.com/guides/pipelines/) that assembles the process to get the columns needed, do the indexing and encoding and create an packaged vector that the various models will be trained on.

# sparklyr has not yet implemented the features for running a param grid builder, therefore this sections will simply use the default paramters from the logistic model run in the python section, train the model and write the output to S3.

flight_partitions <- flight_df_mutated %>%
  sdf_random_split(training = 0.7, testing = 0.3, seed = 1111)

flights_pipeline <- ml_pipeline(spark) %>%
  ft_string_indexer(
    input_col = "OP_CARRIER",
    output_col = "OP_CARRIER_INDEXED",
    handle_invalid = "skip"
  ) %>%
  ft_one_hot_encoder(
    input_col = "OP_CARRIER_INDEXED",
    output_col = "OP_CARRIER_ENCODED",
  ) %>%
  ft_string_indexer(
    input_col = "ORIGIN",
    output_col = "ORIGIN_INDEXED",
    handle_invalid = "skip"
  ) %>%
  ft_one_hot_encoder(
    input_col = "ORIGIN_INDEXED",
    output_col = "ORIGIN_ENCODED",
  ) %>%
  ft_string_indexer(
    input_col = "DEST",
    output_col = "DEST_INDEXED",
    handle_invalid = "skip"
  ) %>%
  ft_one_hot_encoder(
    input_col = "DEST_INDEXED",
    output_col = "DEST_ENCODED",
  ) %>%
  ft_r_formula(
    CANCELLED ~ 
    CRS_ELAPSED_TIME +
    OP_CARRIER_ENCODED +
    CRS_DEP_HOUR + 
    DISTANCE + 
    ORIGIN_ENCODED + 
    DEST_ENCODED + 
    WEEK
  ) %>% 
  ml_logistic_regression()

fitted_pipeline <- ml_fit(
  flights_pipeline,
  flight_partitions$training
)

predictions <- ml_predict(fitted_pipeline,flight_partitions$testing)

ml_binary_classification_evaluator(predictions)

## This has already been run
# ml_save(
#   fitted_pipeline,
#   "s3a://ml-field/demo/flight-analysis/data/models/fitted_pipeline_r",
#   overwrite = TRUE
# )

