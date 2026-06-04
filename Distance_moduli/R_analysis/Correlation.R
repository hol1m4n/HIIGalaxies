

setwd("~/HIIGalaxies/Distance_moduli")

library(readr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(tidyr)

library(corrplot)

sm_r <- read_csv("Results_tables/sm_r.csv") # Cefeidas sin correccion por metalicidad solo errores aleatorios


tabla_errores <- sm_r %>%
  filter(N_dat > 1) %>%
  select(sigma_w, sigma_br, sigma_C, sigma_L, sigma_Lcorr)

tabla_errores_num <- tabla_errores %>%
  mutate(across(everything(), as.numeric))

matriz_corr <- cor(tabla_errores_num, method = "spearman", use = "complete.obs")
print(matriz_corr)  

corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)



matriz_corr <- cor(tabla_errores_num, method = "pearson", use = "complete.obs")
print(matriz_corr)  

corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)


cm_r <- read_csv("Results_tables/cm_r.csv") # Cefeidas con correccion por metalicidad solo errores aleatorios


tabla_errores <- cm_r %>%
  filter(N_dat > 1) %>%
  select(sigma_w, sigma_br, sigma_C, sigma_L, sigma_Lcorr)

tabla_errores_num <- tabla_errores %>%
  mutate(across(everything(), as.numeric))

matriz_corr <- cor(tabla_errores_num, method = "spearman", use = "complete.obs")
print(matriz_corr)  

corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)



matriz_corr <- cor(tabla_errores_num, method = "pearson", use = "complete.obs")
print(matriz_corr)  

corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)



t_r <- read_csv("Results_tables/t_r.csv") # Cefeidas sin correccion por metalicidad solo errores aleatorios


tabla_errores <- t_r %>%
  filter(N_dat > 1) %>%
  select(sigma_w, sigma_br, sigma_C, sigma_L, sigma_Lcorr)

tabla_errores_num <- tabla_errores %>%
  mutate(across(everything(), as.numeric))

matriz_corr <- cor(tabla_errores_num, method = "spearman", use = "complete.obs")
print(matriz_corr)  

corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)



matriz_corr <- cor(tabla_errores_num, method = "pearson", use = "complete.obs")
print(matriz_corr)  

corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)
























cm_r <- read_csv("Results_tables/cm_r.csv") # Cefeidas con correccion por metalicidad solo errores aleatorios


tabla_errores <- cm_r %>%
  filter(N_dat > 1) %>%
  select(sigma_w, sigma_br, sigma_C, sigma_L, sigma_Lcorr)

tabla_errores_num <- tabla_errores %>%
  mutate(across(everything(), as.numeric))

matriz_corr <- cor(tabla_errores_num, method = "spearman", use = "complete.obs")
print(matriz_corr)  

corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)




t_r <- read_csv("Results_tables/t_r.csv") # TRGB solo errores aleatorios


tabla_errores <- t_r %>%
  filter(N_dat > 1) %>%
  select(sigma_w, sigma_br, sigma_C, sigma_L, sigma_Lcorr)

tabla_errores_num <- tabla_errores %>%
  mutate(across(everything(), as.numeric))

matriz_corr <- cor(tabla_errores_num, method = "spearman", use = "complete.obs")
print(matriz_corr)  

corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)















