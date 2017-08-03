
data(sales, package="DMwR2")
sales <- filter(sales,!(is.na(Quant) & is.na(Val)))
sales <- filter(sales,!(Prod %in% c("p2442", "p2443"))) %>% droplevels()

sales[noQuant,'Quant'] <- ceiling(sales[noQuant,'Val'] /noQuantMedPrices)
sales[noVal,'Val'] <- sales[noVal,'Quant'] * noValMedPrices
#sales$Uprice <- sales$Val/sales$Quant

###############################################

nrow(sales) == nrow(na.omit(sales))

sales <- droplevels(sales %>% filter(Insp %in% c("ok", "fraud")))
table(sales$Insp)
sales <- sales %>% select(-ID)
sales <- sales %>% mutate(Insp = as.numeric(Insp)-1)
sales

swip <- with(sales,
            data.frame(model.matrix(~Prod-1,sales),
                       Quant,Val,Insp))

swop <- select(swip, c(Quant, Val, Insp))

s <- swop

fraud_indices <- which(s$Insp == 1)
length(fraud_indices)

non_fraud_indices <- setdiff(1:nrow(s), fraud_indices)

X_train
y_train
