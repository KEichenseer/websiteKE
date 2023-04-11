
set.seed(1) # change the seed for a different sequence of random numbers
n <- 30 # number of total data points
x <- runif(n,-2,2) # generate the true x
y <- 2 + 0.75 * x # generate the true y
# define the standard deviations of the normal uncertainties with which x and y where observed
sigma_x <- runif(n,0.3,0.5)
sigma_y <- runif(n,0.05,0.1)
# generate observations from x and y, given these uncertainties
x_obs <- rnorm(n,x,sigma_x)
y_obs <- rnorm(n,y,sigma_y)

# function to show uncertainty shadings
ciPoly <- function(x,en,ep,color=rgb(0,0,0,0.2)) {
  polygon( c(x[1], x, x[length(x)], x[length(x)], rev(x), x[1]),
           c((ep)[1],ep, (ep)[length(ep)], (en)[length(en)], rev(en), (en)[1]),
           border = NA, col = color)}

par(mar = c(4,4,1,1), las = 1, mgp = c(2.25,0.75,0), cex = 1.35)

plot(x_obs,y_obs,pch=21,col=NA,bg=rgb(0,0,0,0.5),
     xlab = "x", ylab = "y")

lm_true <- lm(y~x)
points(range(x_obs),lm_true$coefficients[1]+lm_true$coefficients[2]*range(x_obs),type="l",lwd=2,col = rgb(0.5,0.5,0.5,1))

lm1 <- lm(y_obs~x_obs)
x_seq <- seq(min(x_obs),max(x_obs),length.out = 100)
x_pred <- predict.lm(lm1,newdata = data.frame(x_obs=x_seq), interval = "confidence")
ciPoly(x_seq,x_pred[,2],x_pred[,3])
points(x_seq,x_pred[,1],lwd=2,lty=2,type = "l", col = rgb(0,0,0,1))
legend("topleft", legend = c("observations","true relationship", "linear regression"), pt.bg  = c(rgb(0,0,0,0.5), NA, NA), pch = c(21,NA, NA), lwd = c(NA,2,2),  pt.cex = c(1,NA,NA), bty = "n", pt.lwd = c(0,NA,NA), col = c(NA,rgb(.5,.5,.5,1),rgb(0,0,0,1)), lty = c(NA,1,2), cex = .65)

errors_in_variables_regression_jags <- function() {
  ## Likelihood
  for (i in 1:n){  
    y_est[i] ~ dnorm(mu[i], tau) # JAGS uses precision `tau` instead of sigma
    mu[i] <- alpha + beta * x_est[i]
    x_est[i] ~ dnorm(x_obs[i], 1/(sigma_x[i]*sigma_x[i])) # precision = 1/sigma^2
    y_obs[i] ~ dnorm(y_est[i], 1/(sigma_y[i]*sigma_y[i])) # precision = 1/sigma^2
  }
  ## Priors
  tau ~ dgamma(1, 1)  # gamma prior for precision
  sigma <- 1/sqrt(tau) # calculate residual standard deviation
  alpha ~ dnorm(0, 1/(10^2)) # normal prior with standard deviation = 10
  beta ~ dnorm(0, 1/(10^2)) # normal prior with standard deviation = 10
}

library(R2jags)

regression_data <- list("x_obs", "y_obs", "sigma_x", "sigma_y","n")

lm_jags  <- jags(data = regression_data,
                 parameters.to.save = c("alpha",
                                        "beta",
                                        "sigma",
                                        "y_est",
                                        "x_est"
                 ),
                 n.iter = 5000,
                 n.thin = 1,
                 n.chains =  3, # Other values set at default (for simplicity)
                 model.file = errors_in_variables_regression_jags)


par(mar = c(4,4,1,1), las = 1, mgp = c(2.25,0.75,0), cex = 1.35)

plot(x_obs,y_obs,pch=21,col=NA,bg=rgb(0,0,0,0.5),
     xlab = "x", ylab = "y")
lm_true <- lm(y~x)
points(range(x_obs),lm_true$coefficients[1]+lm_true$coefficients[2]*range(x_obs),type="l",lwd=2,col = rgb(0.5,0.5,0.5,1))

lm1 <- lm(y_obs~x_obs)
x_seq <- seq(min(x_obs),max(x_obs),length.out = 100)
x_pred <- predict.lm(lm1,newdata = data.frame(x_obs=x_seq), interval = "confidence")
ciPoly(x_seq,x_pred[,2],x_pred[,3])
legend("topleft", legend = c("observations","true relationship", "lm() regression"
), pt.bg  = c(rgb(0,0,0,0.5), NA, NA), pch = c(21,NA, NA), lwd = c(NA,2,2),  pt.cex = c(1,NA,NA), pt.lwd = c(0,NA,NA), col = c(NA,rgb(0.5,0.5,0.5,1),"black"), lty = c(NA,1,2,2), cex = .65)

legend("bottomright", legend = c("JAGS estimates",
                                 "JAGS regression"), pt.bg  = c(rgb(0.9,0.33,0,0.5), NA), pch = c(24,NA), lwd = c(NA,2),  pt.cex = c(1,NA,NA,NA), pt.lwd = c(0,NA), col = c(NA,rgb(0.9,0.33,0,1)), lty = c(NA,2), cex = .65)

x_seq <- seq(min(x_obs),max(x_obs),length.out = 100)
regmat <- matrix(NA,nrow = 3000, ncol = length(x_seq))

for(i in 1:3000) {
  regmat[i,] <- lm_jags$BUGSoutput$sims.list$alpha[i] + lm_jags$BUGSoutput$sims.list$beta[i]*x_seq
}

reg_025 <- apply(regmat, 2, function(x) quantile(x, probs = 0.025))
reg_975 <- apply(regmat, 2, function(x) quantile(x, probs = 0.975))

ciPoly(x_seq, reg_975,reg_025, col = rgb(0.9,.33,0,0.25))
points(range(x_obs), lm_jags$BUGSoutput$mean$alpha + range(x_obs)*lm_jags$BUGSoutput$mean$beta, type = "l", col = rgb(0.9,.33,0,1),lwd=2, lty=2)

points(x_seq,x_pred[,1],lwd=2,lty=2,type = "l")
points(lm_jags$BUGSoutput$mean$x_est,lm_jags$BUGSoutput$mean$y_est,pch=24, col = NA, bg = rgb(0.9,0.33,0,0.5))
sapply(1:n,function(a) points(points(c(lm_jags$BUGSoutput$mean$x_est[a],x_obs[a]),c(lm_jags$BUGSoutput$mean$y_est[a],y_obs[a]), type = "l", lty = 3, col = rgb(0.9,0.33,0,0.5))
))


X <- cbind(rep(1,length(x)),x_obs)

# pre-compute

n_params = 2
n_obs = n

beta = c(1,1) # starting value
sigma2 = 1 #starting value

X_est = X
y_est = y_obs
y_pred = (X_est %*% beta)[,1]

n_iterations = 20000

beta_out = matrix(data=NA, nrow=n_iterations, ncol=n_params)
sigma2_out = matrix(data = NA, nrow = n_iterations, ncol=1)

x_est_out = matrix(data=NA, nrow=n_iterations, ncol=n)
y_est_out = matrix(data=NA, nrow=n_iterations, ncol=n)

for (i in 1:n_iterations){
  
  
  y_est =
    rnorm(n_obs,
          sigma2/(y_sd^2+sigma2)*y_obs + y_sd^2/(y_sd^2+sigma2)*y_pred,
          sqrt(1/(1/sigma2 + 1/y_sd^2))) # I think that one at least should be correct now... If x_sd is
  #  set to very small values, then the results mostly agree with those of the jags implementation
  
  
  if(beta[2] == 0) x_pred = rep(beta[1],length(x_obs)) else x_pred = (y_pred-beta[1])/beta[2]
  
  
  X_est[,2] =
    #   rnorm(n_obs,
    #                   ((beta[2] * x_obs / x_sd^2) + ((y_est - beta[1] - y_pred) * beta[2]/ sigma2)) / ((beta[2]^2 / x_sd^2) + (1 / sigma2)),
    #                   1 / ((beta[1]^2 / x_sd^2) + (1 / sigma2))
    # )
    rnorm(n_obs, 
          sigma2/(x_sd^2+sigma2)*x_obs + x_sd^2/(x_sd^2+sigma2)*x_pred,
          sqrt(1/(1/sigma2 + 1/x_sd^2)))
  
  rnorm(n_obs,
        (beta[2]*(y_est - beta[1]) / sigma2 + x_obs / x_sd^2) / (1 / x_sd^2 + beta[2]^2 / sigma2), #(x_obs / x_sd^2 + beta[2]*y_est / sigma2) / (1 / x_sd^2 + beta[2]^2 / sigma2),
        sqrt(1 / (1 / x_sd^2 + beta[2]^2 / sigma2))
  )
  
  # rnorm(n_obs,
  #       (beta[2]*y_est + beta[1]*x_sd^2/x_obs) / (beta[2]^2*x_sd^2/x_obs + 1),
  #       (sigma2*x_sd^2) / (beta[2]^2*x_sd^2/x_obs + 1)
  # )
  #
  #
  XtX = t(X_est) %*% X_est
  beta_hat = solve(XtX, t(X_est) %*% y_est)
  XtXi = solve(XtX)
  
  beta = mvnfast::rmvn(n=1, beta_hat, sigma2 * XtXi)[1,] # Beta is N(XtXiXty, sigma^2XtXi)
  
  y_pred = (X_est %*% beta)[,1]
  
  resid = (y_pred-y_est)
  
  sigma2 = 1/rgamma(1, 1+n_obs/2, 1+sum(resid^2) * .5 ) # sigma^2 is IG(n/2, ....)
  
  # save the results.
  beta_out[i,] = beta
  sigma2_out[i,] = sigma2
  x_est_out[i,] = X_est[,2]
  y_est_out[i,] = y_est
}

abline(a=median(beta_out[1000:20000,1]), b=median(beta_out[1000:20000,2]), col = "green",lwd=2, lty = 2)

#abline(a=mean(beta_samples[1000:10000,1]), b=mean(beta_samples[1000:10000,2]), col = "red")
regrange <- seq(min(x_obs), max(x_obs), 0.1)
regmat <- matrix(NA,nrow = 10000, ncol = length(regrange))

for(i in 10001:20000) {
  regmat[i-10000,] <- beta_out[i,1] + beta_out[i,2]*regrange
}

reg_025 <- apply(regmat, 2, function(x) quantile(x, probs = 0.025))
reg_975 <- apply(regmat, 2, function(x) quantile(x, probs = 0.975))
ciPoly(regrange, reg_025,reg_975, col = rgb(0,1,0,0.2))


points(apply(x_est_out,2,mean),apply(y_est_out,2,mean),pch = 21, col = NA, bg = rgb(0,0.9,0,0.33))


### OH booy. Maybe just use MH for getting x_est. Not sure how to get the conditional posterior for this in a nice form.