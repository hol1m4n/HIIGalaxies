setwd("~/HIIGalaxies/Distance_moduli")

library(readr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(tidyr)

sm_r <- read_csv("Results_tables/sm_r.csv")
t_r <- read_csv("Results_tables/cm_r.csv")

datos_pareados <- inner_join(sm_r, t_r, by = "Galaxia")



diferencias <- datos_pareados$mu_w.x - datos_pareados$mu_w.y
shapiro.test(diferencias) # Si p > 0.05, es normal

# Ejecutamos el T-Test Pareado
resultado_ttest <- t.test(datos_pareados$mu_w.x, 
                          datos_pareados$mu_w.y, 
                          paired = TRUE, 
                          alternative = "two.sided")
print(resultado_ttest)

resultado_wilcoxon <- wilcox.test(datos_pareados$mu_w.x, 
                                  datos_pareados$mu_w.y, 
                                  paired = TRUE, 
                                  alternative = "two.sided")
print(resultado_wilcoxon)


# Convertimos el vector de diferencias en un data frame para ggplot
df_dif <- data.frame(diferencias = diferencias)

ggplot(df_dif, aes(x = diferencias)) +
  # Histograma base
  geom_histogram(aes(y = ..density..), bins = 10, fill = "royalblue", alpha = 0.5, color = "white") +
  # Curva de densidad suavizada de tus datos
  geom_density(color = "blue", size = 1.2) +
  # Curva Gaussiana teórica ideal para comparar
  stat_function(fun = dnorm, 
                args = list(mean = mean(df_dif$diferencias), sd = sd(df_dif$diferencias)), 
                color = "red", linetype = "dashed", size = 1) +
  labs(title = "Distribución de Diferencias (Cefeidas - TRGB)",
       subtitle = "La línea roja discontinua representa una distribución normal teórica",
       x = "Delta mu (mag)",
       y = "Densidad") +
  theme_minimal()



ggplot(datos_pareados, aes(x = mu_w.y, y = mu_w.x)) +
  # Línea de identidad (Y = X). Si fueran idénticos, todos los puntos caerían aquí.
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50", size = 1) +
  # Puntos astronómicos (Galaxias)
  geom_point(color = "darkorange", size = 3, alpha = 0.8) +
  # Etiquetas con los nombres de las galaxias para identificar outliers de inmediato
  geom_text(aes(label = Galaxia), hjust = -0.2, vjust = -0.2, size = 3, check_overlap = TRUE) +
  labs(title = "Módulo de Distancia: Cefeidas vs TRGB por Galaxia",
       x = "Módulo de Distancia TRGB (mu_w.y)",
       y = "Módulo de Distancia Cefeidas (mu_w.x)") +
  theme_minimal()


# Necesitamos transformar los datos pareados a formato largo solo para este gráfico
datos_long_grafico <- datos_pareados %>%
  select(Galaxia, Cefeidas = mu_w.x, TRGB = mu_w.y) %>%
  pivot_longer(cols = c(Cefeidas, TRGB), names_to = "Metodo", values_to = "Mu")

ggplot(datos_long_grafico, aes(x = Metodo, y = Mu, group = Galaxia)) +
  # Líneas que conectan la misma galaxia entre ambos métodos
  geom_line(color = "purple", alpha = 0.4, size = 0.8) +
  # Puntos para los valores de cada método
  geom_point(aes(color = Metodo), size = 2.5) +
  scale_color_manual(values = c("Cefeidas" = "deeppink4", "TRGB" = "darkgreen")) +
  labs(title = "Efecto del Calibrador en el Módulo de Distancia",
       subtitle = "Cada línea une las estimaciones de una misma galaxia",
       x = "Método Calibrador",
       y = "Módulo de Distancia (mag)") +
  theme_light() +
  theme(legend.position = "none") # Quitamos la leyenda porque el eje X ya lo dice







library(corrplot)

sm_r <- read_csv("Results_tables/cm_r.csv")


# Filtramos las galaxias que tienen estimaciones de error completas (N > 1)
tabla_errores <- sm_r %>%
  filter(N_dat > 1) %>%
  select(sigma_w, sigma_br, sigma_C, sigma_L, sigma_Lcorr)

tabla_errores_num <- tabla_errores %>%
  mutate(across(everything(), as.numeric))

# R transformará cualquier celda con texto extraño en un valor "NA" (vacío válido)
matriz_corr <- cor(tabla_errores_num, method = "spearman", use = "complete.obs")
print(matriz_corr)  


# Graficamos un mapa de calor elíptico espectacular para tu entrega
corrplot(matriz_corr, method = "ellipse", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


str(tabla_errores)























