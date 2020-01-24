## Part 1 - Data Exploration
#For this project, the requirement is to use the flights dataset to predict if a particular flight in the future will be cancelled. This first notebook is used to explore the data.
#
#The original dataset comes from [Kaggle](https://www.kaggle.com/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018) and has a fairly good data dictionary:
#
#+ **FL_DATE** Date of the flight, yy/mm/dd
#+ **OP_CARRIER** Airline Identifier
#+ **OP_CARRIER_FL_NUM** Flight Number
#+ **ORIGIN** Starting Airport Code
#+ **DEST** Destination Airport Code
#+ **CRS_DEP_TIME** Planned Departure Time
#+ **DEP_TIME** Actual Departure Time
#+ **DEP_DELAY** Total Delay on Departure in minutes
#+ **TAXI_OUT** The time duration elapsed between departure from the origin airport gate and wheels off
#+ **WHEELS_OFF** The time point that the aircraft's wheels leave the ground
#+ **WHEELS_ON** The time point that the aircraft's wheels touch on the ground
#+ **TAXI_IN** The time duration elapsed between wheels-on and gate arrival at the destination airport
#+ **CRS_ARR_TIME** Planned arrival time
#+ **ARR_TIME** Actual Arrival Time
#+ **ARR_DELAY** Total Delay on Arrival in minutes
#+ **CANCELLED** Flight Cancelled (1 = cancelled)
#+ **CANCELLATION_CODE** Reason for Cancellation of flight:
#    A - Airline/Carrier;
#    B - Weather;
#    C - National Air System;
#    D - Security
#+ **DIVERTED** Aircraft landed on airport that out of schedule
#+ **CRS_ELAPSED_TIME** Planned time amount needed for the flight trip
#+ **ACTUAL_ELAPSED_TIME** The time duration between wheels_off and wheels_on time
#+ **AIR_TIME** Time spent in the air
#+ **DISTANCE** Distance between two airports
#+ **CARRIER_DELAY** Delay caused by the airline in minutes
#+ **WEATHER_DELAY** Delay caused by weather
#+ **NAS_DELAY** Delay caused by air system
#+ **SECURITY_DELAY** Delay caused by security
#+ **LATE_AIRCRAFT_DELAY** Delay cause by late arriving aircraft
#+ **Unnamed: 27** Useless column

#---

#The various imports
library(sparklyr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(psych)
library(reshape2)
library(leaflet)

#---

### Create the Spark session
#Connect to spark using the standard Spark Session connector. I've put the connection parameters into each file directly they change dependingo the type of job that is running. You should adjust the following for your specific Spark environment.
#
#+ `spark.executor.memory`
#+ `spark.executor.cores`
#+ `spark.driver.memory` 
#+ `spark.executor.instances` 
#
#Spark will use the default `master` setting when connecting to the resource manager. With Cloudera CML, this will be Spark on Kubernetes. If you are working on you local machine for testing, add `.master("local[*]")\` before `.getOrCreate()`     
#
#The `spark.hadoop.fs.s3a.aws.credentials.provider org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider` will let you access the S3 bucket anonymously, but it doesn't always work. First make sure you have the hadoop-aws class path. It should be present on most recent versions of Spark. 
#
#+ `spark.jars.packages org.apache.hadoop:hadoop-aws:2.7.3`
#
#You might need to also set:
#
#+ `spark.hadoop.fs.s3a.access.key <YOUR AWS_ACCESS_KEY>`
#+ `spark.hadoop.fs.s3a.secret.key <YOUR AWS SECRET_KEY>`  
#
#The config below is specific to the Cloudera CML setup:
#
#`spark.yarn.access.hadoopFileSystems s3a://ml-field`

#spark_home_set("/etc/spark/")
config <- spark_config()
config$spark.executor.memory <- "8g"
config$spark.executor.cores <- "2"
config$spark.driver.memory <- "6g"
config$spark.executor.instances <- "5"
config$spark.yarn.access.hadoopFileSystems <- "s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/"
#config$spark.sql.catalogImplementation <- "in-memory"
sc <- spark_connect(master = "yarn-client", config=config)

#---

## Load the Spark UI
#This creates a link the Spark UI. Its specific to CML and needed because of an issue 
#with TLS that is being fixed. If you are running locally, the Sparklyr connection panel 
#shoudl provide you with the Spark UI link.

library(cdsw)
html(paste("<a href='http://spark-",Sys.getenv("CDSW_ENGINE_ID"),".",Sys.getenv("CDSW_DOMAIN"),"' target='_blank'>Spark UI<a>",sep=""))

#---

### Import the data
#This file was downloaded from 
#[Kaggle](https://www.kaggle.com/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018) as a CSV and 
#uploaded to S3. Since we know the schema already, we can make its correct by defining the schema for the 
# import rather than relying on inferSchema. Its also faster! 
#
#_Note: If you are working in local mode, you should limit the number of rows that are returned._
#
#> _HANDY TIP_
#> 
#> Use `.persist()` on a data frame that you are work a lot with to prevent Spark from fetching the data 
#everytime you run query. It will store that dataframe in memory and all operations on that dataframe will 
#run on the in-memory version.
#

s3_link_all <-
  "s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/airlines_csv/*"
#  "s3a://ml-field/demo/flight-analysis/data/airlines_csv/2010.csv"

cols = list(
  FL_DATE = "date",
  OP_CARRIER = "character",
  OP_CARRIER_FL_NUM = "character",
  ORIGIN = "character",
  DEST = "character",
  CRS_DEP_TIME = "character",
  DEP_TIME = "character",
  DEP_DELAY = "double",
  TAXI_OUT = "double",
  WHEELS_OFF = "character",
  WHEELS_ON = "character",
  TAXI_IN = "double",
  CRS_ARR_TIME = "character",
  ARR_TIME = "character",
  ARR_DELAY = "double",
  CANCELLED = "double",
  CANCELLATION_CODE = "character",
  DIVERTED = "double",
  CRS_ELAPSED_TIME = "double",
  ACTUAL_ELAPSED_TIME = "double",
  AIR_TIME = "double",
  DISTANCE = "double",
  CARRIER_DELAY = "double",
  WEATHER_DELAY = "double",
  NAS_DELAY = "double",
  SECURITY_DELAY = "double",
  LATE_AIRCRAFT_DELAY = "double",
  'Unnamed: 27' = "logical"
)

spark_read_csv(
  sc,
  name = "flight_data",
  path = s3_link_all,
  infer_schema = FALSE,
  columns = cols,
  header = TRUE
)

airlines <- tbl(sc, "flight_data")

airlines %>% count()

airlines %>% sample_n(10) %>% as.data.frame

#---

### Cancelled Flights by Carrier
#The first bit of data exploration is to check the flight cancellations by carrier. This is best done by 
# showing which carrier has the highet percentage of cancelled flights rather than the total number of 
# cancelled flights. 
#
#Concepts introduced in this section:
#+ `dplyr` verbs. `mutate`,`summarise` etc.
#+ `filter()`,`group_by` etc.
#+ `withColumn` and `withColumnRenamed`
#+ `toPandas()`

cancelled_flights_by_carrier <-
  airlines %>% 
  group_by(OP_CARRIER) %>% 
  filter(CANCELLED == 1) %>%
  summarise(count_delays = n()) %>%
  arrange(desc(count_delays)) 

flights_by_carrier <-
  airlines %>% 
  group_by(OP_CARRIER) %>% 
  summarise(count = n()) %>%
  arrange(desc(count))

flights_by_carrier %>% 
  left_join(cancelled_flights_by_carrier, by = "OP_CARRIER") %>% 
  mutate(delay_percent = (count_delays/count)*100) %>%
  arrange(desc(delay_percent))

#---

### Cancelled flights by Year
#This is not necessarily useful as a predictive metric, but it is still interesting. 
#This is the first plot done with a Spark DataFrame using ggplot. Stuff all just works in R

#> HANDY TIP
#>
#> This is important, you can run `spark.sql` functions directly inside an `mutate`.
#> The complete list is available [here](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions)

flight_counts_by_year <-
  airlines %>% 
  mutate(year = year(FL_DATE)) %>%
  group_by(year) %>% 
  summarise(count = n())

cancel_counts_by_year <-
  airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(year = year(FL_DATE)) %>%
  group_by(year) %>% 
  summarise(count_delays = n())

flight_cancel_percent_year <-
  flight_counts_by_year %>% 
  left_join(cancel_counts_by_year, by = "year") %>% 
  mutate(delay_percent = (count_delays/count)*100) %>%
  arrange(desc(delay_percent))

g <- 
  ggplot(flight_cancel_percent_year, aes(year, delay_percent)) + 
  theme_tufte(base_size=14, ticks=F) + 
  geom_col(width=0.75, fill = "grey") +
  theme(axis.title=element_blank()) +
  scale_x_continuous(breaks=seq(2008,2018,1)) +
  ylab("%") +
  scale_y_continuous() + 
  ggtitle("Percentage Cancelled Flights by Year") + 
  geom_hline(yintercept=seq(1, 2, 1), col="white", lwd=0.5)
plot(g)

#---

## Cancelled flights per Week of Year
#This is a more intersting statistic and likely to have more predictive power. 
#Week of Year will be a seasonal and generally the flight patterns will have busier 
#vs less busy times of the year. This will also show the effect that seasonal 
#weather conditions can have on flight cancellations.

flight_counts_by_week <-
  airlines %>% 
  mutate(week = weekofyear(FL_DATE)) %>%
  group_by(week) %>% 
  summarise(count = n())

cancel_counts_by_week <-
  airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(week = weekofyear(FL_DATE)) %>%
  group_by(week) %>% 
  summarise(count_delays = n())

flight_cancel_percent_week <- 
  flight_counts_by_week %>% 
  left_join(cancel_counts_by_week, by = "week") %>% 
  mutate(delay_percent = (count_delays/count)*100) 

g <- 
  ggplot(flight_cancel_percent_week, aes(week, delay_percent)) + 
  theme_tufte(base_size=14, ticks=F) + 
  geom_col(width=0.75, fill = "grey") +
  theme(axis.title=element_blank()) +
  scale_x_continuous(breaks=seq(1,54,2)) +
  ylab("%") +
  scale_y_continuous() + 
  ggtitle("Percentage Cancelled Flights by Week of Year") + 
  geom_hline(yintercept=seq(1, 4, 1), col="white", lwd=0.5)
plot(g)

#---

## Calculating Cancelled Routes
#To work out if route is likely to cancalled, the easiest way is to create a 
#string that combines the origin and destination. However that works only in on 
#direction, so calculate for both directions, the code below uses `hash` to create 
#an interger that is the sum of the a of the origin and a hash of the destination. 
#This creates a commutative process for any route.

all_routes <- airlines %>% 
  mutate(combo_hash= hash(ORIGIN) + hash(DEST),combo = paste(ORIGIN,DEST,sep="")) %>% 
  select(combo_hash, combo,ORIGIN, DEST) %>%
  group_by(combo_hash) %>%
  summarize(count_all = n(), first_val = first_value(combo)) %>%
  arrange(desc(count_all)) %>%
  collect

cancelled_routes_all <- airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(combo_hash= hash(ORIGIN) + hash(DEST),combo = paste(ORIGIN,DEST,sep="")) %>% 
  select(combo_hash, combo,ORIGIN, DEST) %>%
  group_by(combo_hash) %>%
  summarize(count = n(), first_val = first_value(combo)) %>%
  arrange(desc(count)) %>%
  collect

cancelled_routes_percentage <-
  cancelled_routes_all %>% 
  inner_join(all_routes,by="combo_hash") %>%
  mutate(
    route = paste(
          substr(first_val.x,0,3), "<>",dest = substr(first_val.x,4,6),sep = ""
        ), 
    cancelled_percent = count/count_all*100) %>% 
  select(route,count_all,count_all,cancelled_percent) %>%
  arrange(desc(cancelled_percent)) 
  
cancelled_routes_percentage %>% as.data.frame

#---

### Side Note
# Interestingly most popular routes have similar numbers of cancelled flights in 
# either direction.

cancelled_by_route_non_combo <- airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(combo = paste(ORIGIN,DEST,sep="")) %>% 
  group_by(combo) %>%
  summarize(count_all = n()) %>%
  select(combo,count_all) %>%
  arrange(desc(count_all))

cancelled_by_route_non_combo %>% head(10) %>% as.data.frame

#---

## Plotting Cancelled Routes on Map
#
#A good practice for good data exploration is using visualisations. The next cell 
#fetches additional data about airports to join with the cancelled data flights and 
#then plots this on a `leaflet` map.

spark_read_csv(
  sc,
  name = "airports",
  path = "s3a://prod-cdptrialuser19-trycdp-com/cdp-lake/data/airports_orig.csv",
  infer_schema = TRUE,
  header = TRUE
)

airports  <- tbl(sc, "airports")

airports <- airports %>% collect


cancelled_routes_combo <- airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(combo_hash= hash(ORIGIN) + hash(DEST),combo = paste(ORIGIN,DEST,sep="")) %>% 
  select(combo_hash, combo,ORIGIN, DEST) %>%
  group_by(combo_hash) %>%
  summarize(count = n(), first_val = first_value(combo)) %>%
  arrange(desc(count)) %>%
  collect

cancelled_routes_combo <- cancelled_routes_combo %>% 
  mutate(orig = substr(first_val,0,3), dest = substr(first_val,4,6)) %>%
  select(count,orig,dest)%>%
  inner_join(airports, by=c("orig"="iata"))%>%
  mutate(orig_lat = lat, orig_long = long) %>%
  select(count,orig,dest,orig_lat,orig_long) %>% 
  inner_join(airports, by=c("dest"="iata"))%>%
  mutate(dest_lat = lat, dest_long = long) %>%
  select(count,orig,dest,orig_lat,orig_long,dest_lat,dest_long) %>% 
  filter(between(orig_lat, 20, 50),count > 500)


map3 = leaflet(cancelled_routes_combo) %>% 
  addProviderTiles(providers$CartoDB.Positron)

for(i in 1:nrow(cancelled_routes_combo)){
    map3 <- 
      addPolylines(
        map3, 
        lat = as.numeric(cancelled_routes_combo[i,c(4,6)]), 
        lng = as.numeric(cancelled_routes_combo[i,c(5,7)]),
        weight = 10*(as.numeric(cancelled_routes_combo[i,1]/9881)+0.1), 
        opacity = 0.8*(as.numeric(cancelled_routes_combo[i,1]/9881)+0.05),
        color = "#888"
      )
}
map3

#---

## Find Unused Columns
# Given that our aim is to calculate a prediction for the CANCELLED variable, many of the
# other columns are no longer relevant. You don't have an actual wheels down time for a
# cancelled flight. The code below lists the colums that have lots of NA values on 
# cancelled flights.

#> HANDY TIP
#>
#> The `dplyr` verbs include `summarise_all` and `select_if` also work on sparklyr

unused_columns <- airlines %>%
  filter(CANCELLED == 1) %>%
  summarise_all(~sum(as.integer(is.na(.)))) %>%
  select_if(~sum(.) > 0) 

unused_columns %>% as.data.frame

unused_columns %>% colnames




