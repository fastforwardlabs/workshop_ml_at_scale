library(sparklyr)
library(dplyr)
library(jsonlite)
#library(arrow)

## Connect to Spark. Check spark_defaults.conf for the correct 
##spark_home_set("/etc/spark/")

config <- spark_config()
config$spark.hadoop.fs.s3a.aws.credentials.provider  <- "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"
config$spark.driver.memory <- "6g"
config$spark.sql.catalogImplementation <- "in-memory"
config$spark.yarn.access.hadoopFileSystems <- "s3a://ml-field"

# config$sparklyr.defaultPackages <- c("org.apache.hadoop:hadoop-aws:2.7.3")
# config$spark.hadoop.fs.s3a.access.key <- "<YOUR AWS ACCESS KEY>"
# config$spark.hadoop.fs.s3a.secret.key <- "<YOUR AWS SECRET KEY>""

spark <- spark_connect(master = "local", config=config)

reloaded_model <- ml_load(spark, "s3a://ml-field/demo/flight-analysis/data/models/fitted_pipeline_r")


cols <- c(
  "OP_CARRIER",
  "ORIGIN",
  "DEST",
  "CRS_DEP_TIME",
  "CRS_ELAPSED_TIME",
  "WEEK",
  "DISTANCE"
)

flights_pipeline <- ml_pipeline(spark) %>%
  ft_string_indexer(
    input_col = "OP_CARRIER",
    output_col = "OP_CARRIER_INDEXED"
  ) %>%
  ft_one_hot_encoder(
    input_col = "OP_CARRIER_INDEXED",
    output_col = "OP_CARRIER_ENCODED"
  ) %>%
  ft_string_indexer(
    input_col = "ORIGIN",
    output_col = "ORIGIN_INDEXED"
  ) %>%
  ft_one_hot_encoder(
    input_col = "ORIGIN_INDEXED",
    output_col = "ORIGIN_ENCODED"
  ) %>%
  ft_string_indexer(
    input_col = "DEST",
    output_col = "DEST_INDEXED"
  ) %>%
  ft_one_hot_encoder(
    input_col = "DEST_INDEXED",
    output_col = "DEST_ENCODED"
  )

#args = {"feature":"AA,ICT,DFW,1135,85,11,328"}
#json_data <- fromJSON('{"feature":"AA,ICT,DFW,935,85,11,328"}')


predict_lr <- function(json_data) {
  feature_df <- as.data.frame(c(
  strsplit(json_data$feature,",")[[1]][1],
  strsplit(json_data$feature,",")[[1]][2],
  strsplit(json_data$feature,",")[[1]][3],
  strsplit(json_data$feature,",")[[1]][4],
  strsplit(json_data$feature,",")[[1]][5],
  strsplit(json_data$feature,",")[[1]][6],
  strsplit(json_data$feature,",")[[1]][7]
) %>% t)

colnames(feature_df) <- cols

feature_sdf <- copy_to(spark, feature_df, overwrite=TRUE)

feature_sdf <- feature_sdf %>% 
  mutate(CRS_ELAPSED_TIME = as.numeric(CRS_ELAPSED_TIME)) %>%
  mutate(WEEK = as.numeric(WEEK)) %>%
  mutate(DISTANCE = as.numeric(DISTANCE)) %>%
  mutate(CRS_DEP_TIME = ifelse(length(CRS_DEP_TIME) == 4,CRS_DEP_TIME,paste("0",CRS_DEP_TIME,sep=""))) %>% 
  mutate(CRS_DEP_HOUR = as.numeric(substring(CRS_DEP_TIME,0,2)))

  list(out_val = (ml_predict(reloaded_model,feature_sdf) %>% select(prediction) %>% collect)[[1]])
}
