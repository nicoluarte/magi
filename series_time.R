setwd("/home/nicoluarte/Downloads")
library(rio)
library(ggplot2)
library(forecast)
library(tseries)
library(tidyverse)

## read data
data(co2)
df <- co2

## look at the data
plot.ts(df)

## we need to eliminate the trend!
df_detrend <- diff(df)
plot.ts(df_detrend)
## plot acf to see if there's seasonality
acf(df_detrend)
## clearly there's
df_seasonal <- diff(df_detrend, lag = 12) ## yearly
plot.ts(df_seasonal)
acf(df_seasonal)
pacf(df_seasonal)
## seasonality is dropped!
## test it
adf.test(df_seasonal)
Box.test(x = df_seasonal)
## nailed it
## MA = 5 SMA = 1
## AR = 4 SAR = 2
mdl <- arima(df, order = c(1,2,1),)
acf(mdl$residuals)
mdl.auto <- auto.arima(df, seasonal = 12)
acf(mdl.auto$residuals)
pacf(mdl.auto$residuals)
fcast <- forecast(mdl.auto, h = 5*20)
plot(fcast)
