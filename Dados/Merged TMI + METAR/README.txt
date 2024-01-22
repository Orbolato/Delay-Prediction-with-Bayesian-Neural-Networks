

# A base de dados final é resultado da fusão da base de dados de restrição miles-in-trail com a base de dados de meteorologia.


# Como a maioria das restrições foram aplicadas para SBXP (área terminal São Paulo), vamos desenvolver o modelo de previsão para esta área terminal.



# Descrição das variáveis na base de dados final:


date <- data

orig <- origem do fluxo para o qual é aplicada a restrição miles-in-trail 

dest <- destino do fluxo para o qual é aplicada a restrição miles-in-trail (SBXP - área terminal São Paulo)

tmi <- descrição da restrição miles-in-trail

tmi_rate <- separação utilizada (20 NM ou 30 NM)  

wind_dir - direção do vento (wind_dir_sbgr = direção do vento no aeroporto de Guarulhos; wind_dir_sbsp = direção do vento no aeroporto de Congonhas; wind_dir_sbkp = direção do vento no aeroporto de Viracopos)

wind_speed - velocidade do vento (kt)

visibility - visibilidade (NM)

ceiling - teto (ft)

ifr - variável binária; = 1 se condição Instrument Flight Rules (IFR) presente, isto é, teto < 1000 ft ou visibilidade < 3 NM

lifr - variável binária; = 1 se condição Low Instrument Flight Rules (IFR) presente, isto é, teto < 500 ft ou visibilidade < 1 NM

ts - variável binária; = 1 se houve registro de atividade convectiva naquela hora

wind_gust - velocidade do vento de rajada (NM)

hour - hora do dia

weekdays - dia da semana


 


# Exemplo de modelo de classificação a ser treinado:

Variável de output: tmi_rate (modelo de classificação com 3 classes: 'None', '20 NM', '30 NM')

Variáveis de input: variáveis meteorológicas, origem do fluxo, hora do dia (proxy para volume de demanda)

Testar diferentes técnicas de aprendizado de máquina

Exemplo:

model <-randomForest(tmi_rate ~ orig + hour + wind_dir_sbgr + wind_speed_sbgr + wind_gust_sbgr + visibility_sbgr + ceiling_sbgr + lifr_sbgr + ts_sbgr + wind_dir_sbsp + wind_speed_sbsp + wind_gust_sbsp + visibility_sbsp + ceiling_sbsp + lifr_sbsp + ts_sbsp + wind_dir_sbkp + wind_speed_sbkp + wind_gust_sbkp + visibility_sbkp + ceiling_sbkp + lifr_sbkp + ts_sbkp, data = training, ntree = 100, classwt=c(0.3,0.3,0.3), importance = TRUE)



# Importante: as classes estão desbalanceadas (há muito mais 'None'). Isso deve ser tratado durante o treinamento do modelo. 

Ver aula 6 de TRA-48.






