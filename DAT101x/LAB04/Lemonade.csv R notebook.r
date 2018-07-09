
library("AzureML")
ws <- workspace()
dat <- download.datasets(ws, "Lemonade.csv")

head(dat)

# Print statistics for Temperature and Sales
summary(dat[c('Temperature', 'Sales')])
print('Standard Deviations:')
apply(dat[c('Temperature', 'Sales')],2,sd)

# Print correlation for temperature vs Sales
print('Correlation:')
cor(dat[c('Temperature', 'Sales')])

# Plot Temperature vs Sales
plot(dat$Temperature, dat$Sales, xlab="Temperature", ylab="Sales") 
