## load libraries
pacman::p_load(tidyr, purrr, broom, dplyr,
               modelr, tidyverse, ggplot2,
               popbio)

## load csv
setwd("/home/nicoluarte/Downloads/")
data <- read.csv("data_fondecyt.csv")
head(data)

## subset to only relevant data
nonRelevant <- names(data) %in% c("Valence", "Pupil", "Socioeconomic", "Neighbourhood")
dataRelevant <- data[!nonRelevant]
head(dataRelevant)
dataRelevant <- as_tibble(dataRelevant)
dataRelevant <- dataRelevant %>%
    mutate(Socioeconomic_num = as.factor(Socioeconomic_num),
           Neighbourhood_num = as.factor(Neighbourhood_num))

## nest the data
dataNest <- dataRelevant %>%
    group_by(Socioeconomic_num, Subjects) %>%
    nest()

## plot 1: valence ~ ., by = Socioeconomic
dataRelevant %>%
    gather(-Valence_num, -Socioeconomic_num, key = "var", value = "value") %>%
    ggplot(aes(x = value, y = Valence_num, color = Socioeconomic_num)) +
    geom_jitter() +
    facet_wrap(~ var, scales = "free") +
    theme_bw()

## plot 2: valence ~ ., by = Neighbourhood
dataRelevant %>%
    gather(-Valence_num, -Neighbourhood_num, key = "var", value = "value") %>%
    ggplot(aes(x = value, y = Valence_num, color = Neighbourhood_num)) +
    geom_jitter() +
    facet_wrap(~ var, scales = "free") +
    theme_bw()

## plot 3: Noise
dataRelevant %>%
    gather(-Noise, key = "var", value = "value") %>%
    ggplot(aes(x = value, y = Noise)) +
    geom_jitter() +
    facet_wrap(~ var, scales = "free") +
    theme_bw()

## model fitting function
socioeconomicModel <- function(df) {
    glm(Valence_num ~ Noise, data = df)
}

## apply model by socioeconomic
dataNest <- dataNest %>%
    mutate(glm_mdl = map(data, socioeconomicModel))
dataNest

## add residuals
dataNest <- dataNest %>%
    mutate(
        resids = map2(data, glm_mdl, add_residuals)
    )
## unlist them
resids <- unnest(dataNest, resids)
## plot them by socioeconomic
resids %>%
    ggplot(aes(Noise, resid, group = Socioeconomic_num)) +
    geom_line(alpha = 1 / 3) +
    geom_smooth(se = FALSE) +
    facet_wrap(~Socioeconomic_num)

## nest the data
by_se <- dataRelevant %>%
    group_by(Socioeconomic_num, Neighbourhood_num) %>%
    nest()

by_se <- by_se %>%
    mutate(mdl_noise = map(data, socioeconomicModel))

by_se <- by_se %>%
    mutate(resids = map2(data, mdl_noise, add_residuals))

## unlist them
resids <- unnest(by_se, resids)
## plot them by socioeconomic
resids %>%
    ggplot(aes(Noise, resid, group = Socioeconomic_num)) +
    geom_line(alpha = 1 / 3) +
    geom_smooth(se = FALSE) +
    facet_wrap(~Socioeconomic_num)

glance <- by_se %>%
    mutate(glance = map(mdl_noise, broom::glance)) %>%
    unnest(glance, .drop = TRUE)
glance %>% arrange(AIC)

glance %>%
    ggplot(aes(Socioeconomic_num, AIC)) +
    geom_jitter(width = 0.5)

unnest(by_se, data) %>%
    ggplot(aes(x = Noise, y = Valence_num, colour = Socioeconomic_num)) +
    geom_point() +
    facet_wrap(~ Socioeconomic_num)
