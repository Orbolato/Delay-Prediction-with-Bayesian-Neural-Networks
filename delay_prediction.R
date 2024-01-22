# Clearing workspace
rm(list = ls())

# Packages
library(lubridate)
library(brnn)
library(BayesFluxR)
library(Julia)
library(JuliaCall)
library(devtools)

# Functions
min_date <- function(x, y){
  if(x <= y){
    return(x)
  }else{
    return(y)
  }
}

max_date <- function(x,y){
  if(x >= y){
    return(x)
  }else{
    return(y)
  }
}

# Loading flight data from ANAC's VRA
folder <- "C:/Users/lucas/Desktop/Doutorado/Dados/Paper Delay Prediction/Dados/Atrasos - VRA"
file_list <- list.files(folder, full.names = TRUE)
setwd(folder)
vra_data <- data.frame()
for(file_path in file_list){
  if(substr(file_path,nchar(file_path)-2,nchar(file_path)) == 'csv'){
    aux <- read.csv(file_path, header = TRUE, sep = ';')
    vra_data <- rbind(vra_data, aux)
  }
}
for(i in 1:ncol(vra_data)){
  vra_data <- vra_data[!is.na(vra_data[i]),]
}

# Loading ATFM measure data
folder <- "C:/Users/lucas/Desktop/Doutorado/Dados/Paper Delay Prediction/Dados/Merged TMI + METAR/TMI"
file <- paste0(folder, "/tmi_sp.Rda")
load(file)
file <- paste0(folder, "/tmi_sbgr.Rda")
load(file)
file <- paste0(folder, "/tmi_2019.Rda")
load(file)
for(i in 1:ncol(tmi_sbgr)){
  tmi_sbgr <- tmi_sbgr[!is.na(tmi_sbgr[i]),]
}

# Loading METAR data
folder <- "C:/Users/lucas/Desktop/Doutorado/Dados/Paper Delay Prediction/Dados/Merged TMI + METAR/METAR"
file <- paste0(folder, "/metar_sbgr.Rda")
load(file)
metar_sbgr <- metar
file <- paste0(folder, "/metar_sbkp.Rda")
load(file)
metar_sbkp <- metar
file <- paste0(folder, "/metar_sbsp.Rda")
load(file)
metar_sbsp <- metar
metar <- rbind(metar, metar_sbkp)
metar <- rbind(metar, metar_sbgr)
metar <- metar[order(metar$valid),]
for(i in 1:ncol(metar_sbgr)){
  metar_sbgr <- metar_sbgr[!is.na(metar_sbgr[i]),]
}

# Filtering SBGR data from VRA data
vra_data <- vra_data[vra_data$situacao == 'REALIZADO',]
vra_data <- vra_data[vra_data$cd_tipo_linha != 'I',]
vra_data <- vra_data[order(vra_data$dt_partida_prevista),]
vra_data_sp <- vra_data[vra_data$sg_icao_destino == 'SBGR' | vra_data$sg_icao_destino == 'SBKP' | vra_data$sg_icao_destino == 'SBSP' | vra_data$sg_icao_destino == 'SBXP',]
vra_data_sbgr <- vra_data[vra_data$sg_icao_destino == 'SBGR',]

# Removing some useless columns from METAR data
metar_sbgr <- subset(metar_sbgr, select = -c(`mslp`, `gust`, `skyc1`, `skyc2`, `skyc3`, `skyc4`, `skyl1`, `skyl2`, `skyl3`, `skyl4`, `wxcodes`, `ice_accretion_1hr`, `ice_accretion_3hr`, `ice_accretion_6hr`, `peak_wind_gust`, `peak_wind_drct`, `peak_wind_time`, `metar`))
attributes(metar_sbgr$valid)$tzone <- 'America/Sao_Paulo'
for(i in 3:ncol(metar_sbgr)){
  metar_sbgr[,i] <- as.numeric(metar_sbgr[,i])
}
for(i in 1:ncol(metar_sbgr)){
  metar_sbgr <- metar_sbgr[!is.na(metar_sbgr[i]),]
}
# metar_sbgr <- cbind(rep('', nrow(metar_sbgr)), metar_sbgr)
# colnames(metar_sbgr)[1] <- 'date'
# metar_sbgr$date <- date(metar_sbgr$valid)
# metar_sbgr <- metar_sbgr[metar_sbgr$valid >= as.POSIXlt('2019-01-01') & metar_sbgr$valid <= as.POSIXlt('2019-12-31'),]

# Pre-processing some information from TMI data
# 'T' stands for terminal, indicating an ATFM measure applied to the TMA limits
# 'E' stands for en-route, indicating an ATFM measure applied between FIRs
# Rates are in nautical-miles (NM)
tmi_sbgr$tmi <- as.character(tmi_sbgr$tmi)
for(i in 1:nrow(tmi_sbgr)){
  if(grepl("por aeródromo", tmi_sbgr$tmi[i], ignore.case = TRUE)){
    tmi_sbgr$tmi[i] <- 'T'
  }else{
    tmi_sbgr$tmi[i] <- 'E'
  }
  tmi_sbgr$rate[i] <- as.integer(strsplit(tmi_sbgr$rate[i], ' ')[[1]][1])
}
colnames(tmi_sbgr)[colnames(tmi_sbgr) == "rate"] <- "rate_nm"
attributes(tmi_sbgr$t_begin)$tzone <- 'America/Sao_Paulo'
# tmi_sbgr <- cbind(rep('', nrow(tmi_sbgr)), tmi_sbgr)
# colnames(tmi_sbgr)[1] <- 'd_end'
# tmi_sbgr$d_end <- date(tmi_sbgr$t_end)
# tmi_sbgr <- cbind(rep('', nrow(tmi_sbgr)), tmi_sbgr)
# colnames(tmi_sbgr)[1] <- 'd_begin'
# tmi_sbgr$d_begin <- date(tmi_sbgr$t_begin)

# Calculating en-route delay
vra_data_sbgr$dt_chegada_prevista <- as.POSIXlt(strptime(vra_data_sbgr$dt_chegada_prevista, '%d/%m/%Y %H:%M', tz = 'America/Sao_Paulo')) # 'Etc/GMT+3' to remove the summer time
vra_data_sbgr$dt_chegada_real <- as.POSIXlt(strptime(vra_data_sbgr$dt_chegada_real, '%d/%m/%Y %H:%M', tz = 'America/Sao_Paulo'))
vra_data_sbgr$dt_partida_prevista <- as.POSIXlt(strptime(vra_data_sbgr$dt_partida_prevista, '%d/%m/%Y %H:%M', tz = 'America/Sao_Paulo'))
vra_data_sbgr$dt_partida_real <- as.POSIXlt(strptime(vra_data_sbgr$dt_partida_real, '%d/%m/%Y %H:%M', tz = 'America/Sao_Paulo'))
vra_data_sbgr <- cbind(vra_data_sbgr, rep(0, nrow(vra_data_sbgr)))
colnames(vra_data_sbgr)[ncol(vra_data_sbgr)] <- 'en_route_delay'
vra_data_sbgr$en_route_delay <- difftime(vra_data_sbgr$dt_chegada_real, vra_data_sbgr$dt_partida_real, units = 'mins') - difftime(vra_data_sbgr$dt_chegada_prevista, vra_data_sbgr$dt_partida_prevista, units = 'mins')
par(mfrow = c(1,1))
x <- as.numeric(vra_data_sbgr$en_route_delay)
hist(x, main = 'En-route delay histogram (SBGR)', xlab = 'En-route delay (min)')
x <- x[!x %in% boxplot.stats(x, coef = 3)$out]
#par(mfrow = c(1,2))
#plot(density(x), main = 'En-route delay distribution (SBGR)', xlab = 'En-route delay (min)')
hist(x, main = 'En-route delay histogram (SBGR)', xlab = 'En-route delay (min)')

# Generating final data
final_data_sbgr <- subset(vra_data_sbgr, select = c(`sg_empresa_icao`, `sg_icao_origem`, `en_route_delay`))

final_data_sbgr <- cbind(rep(0, nrow(final_data_sbgr)), final_data_sbgr)
final_data_sbgr <- cbind(rep(0, nrow(final_data_sbgr)), final_data_sbgr)
colnames(final_data_sbgr)[1:2] <- c('tmi_duration', 'rate_nm')

for(i in 1:(ncol(metar_sbgr) - 2)){ # Including METAR information
  final_data_sbgr <- cbind(rep(0, nrow(final_data_sbgr)), final_data_sbgr)
}
colnames(final_data_sbgr)[1:(ncol(metar_sbgr) - 2)] <- colnames(metar_sbgr)[3:ncol(metar_sbgr)]

for(i in 1:nrow(final_data_sbgr)){
  print(i)
  # Loading TMI information
  aux <- tmi_sbgr[tmi_sbgr$t_begin <= vra_data_sbgr$dt_chegada_real[i] & tmi_sbgr$t_end >= vra_data_sbgr$dt_partida_real[i] & tmi_sbgr$orig == final_data_sbgr$sg_icao_origem[i],]
  if(nrow(aux) > 0){
    for(j in 1:nrow(aux)){
      duration <- difftime(min_date(aux$t_end[j], vra_data_sbgr$dt_chegada_real[i]), max_date(aux$t_begin[j], vra_data_sbgr$dt_partida_real[i]), units = 'mins')
      final_data_sbgr$tmi_duration[i] <- final_data_sbgr$tmi_duration[i] + duration
      final_data_sbgr$rate_nm[i] <- final_data_sbgr$rate_nm[i] + duration*aux$rate_nm[j]
    }
    final_data_sbgr$rate_nm[i] <- final_data_sbgr$rate_nm[i]/final_data_sbgr$tmi_duration[i]
  }
  # Loading METAR information
  aux <- metar_sbgr[vra_data_sbgr$dt_partida_real[i] <= metar_sbgr$valid & metar_sbgr$valid <= vra_data_sbgr$dt_chegada_real[i], 3:ncol(metar_sbgr)]
  if(nrow(aux) > 0){
    final_data_sbgr[i,1:(ncol(metar_sbgr) - 2)] <- colMeans(aux)
  }else{
    aux <- metar_sbgr[vra_data_sbgr$dt_partida_real[i] - lubridate::days(1) <= metar_sbgr$valid & metar_sbgr$valid <= vra_data_sbgr$dt_chegada_real[i] + lubridate::days(1), 3:ncol(metar_sbgr)]
    if(nrow(aux) > 0){
      final_data_sbgr[i,1:(ncol(metar_sbgr) - 2)] <- colMeans(aux)
    }
  }
}

# Exporting CSV
# write.csv(final_data_sbgr, "C:/Users/lucas/Desktop/Doutorado/Dados/Paper Delay Prediction/final_data_sbgr.csv", row.names=FALSE)

# Creating formula
# formula <- colnames(final_data_sbgr)[ncol(final_data_sbgr)]
# formula <- paste0(formula, ' ~ ')
# formula <- paste0(formula, colnames(final_data_sbgr)[1])
# for(i in 2:(ncol(final_data_sbgr)-1)){
#   formula <- paste0(formula, ' + ')
#   formula <- paste0(formula, colnames(final_data_sbgr)[i])
# }
# formula <- as.formula(formula)
# print(formula)

# Loading BayesFluxR
# julia_setup(JULIA_HOME = "C:\\Users\\lucas\\AppData\\Local\\Programs\\Julia-1.9.3\\bin", verbose = TRUE, install = TRUE, force = FALSE, useRCall = TRUE)
# BayesFluxR_setup()
# BayesFluxR_setup(JULIA_HOME = "C:\\Users\\lucas\\AppData\\Local\\Programs\\Julia-1.9.3\\bin", pkg_check =  TRUE)

# Network Architecture
# net <- Chain(Dense(ncol(final_data_sbgr) - 1, 2*ncol(final_data_sbgr) - 1, 'identity'), Dense(2*ncol(final_data_sbgr) - 1, 1, 'identity'))

# Likelihood
# like <- likelihood.feedforward_normal(net, Gamma(2.0, 0.5))

# Prior distribution
# prior <- prior.gaussian(net, 0.5)

# Initialization
# init <- initialise.allsame(Normal(0, 0.5), like, prior)

# Creating datasets
# x <- as.matrix(subset(final_data_sbgr, select = -c(`sg_empresa_icao`,`sg_icao_origem`,`en_route_delay`)))
# y <- as.numeric(final_data_sbgr$en_route_delay)
# 
# for(i in 1:ncol(x)){
#   x[i,] <- as.numeric(x[i,])
# }
# 
# x <- t(x)

# x <- subset(final_data_sbgr, select = -c(`sg_empresa_icao`,`sg_icao_origem`,`en_route_delay`))
# x <- t(x)
# y <- final_data_sbgr$en_route_delay

# Creating the model
# bnn <- BNN(x, y, like, prior, init)

# Sample from the BNN using Monte Carlo Markov Chains
# sampler <- sampler.SGLD()
# ch <- mcmc(bnn, 10, 1000, sampler)
