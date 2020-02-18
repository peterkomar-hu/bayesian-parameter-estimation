#!/usr/bin/python
import numpy
import pystan

# Define model
model_code = """
data {
    int<lower=1> N;      // number of waiting times
    real<lower=0> t[N];  // waiting times [in days]
}
parameters {
    real<lower=0> tau;   // lifetime [in days]
}
model {
    target += log(1/tau);  // log-prior
    for (n in 1:N){
        target += log(1/tau) - t[n] / tau;  // log-likelihood
    }
}
"""
model = pystan.StanModel(model_code=model_code)

# Get data and run sampling
data = {'N': 4, 
        't': [31, 51, 40, 64]}
fit = model.sampling(data=data,
                     algorithm='NUTS',
                     chains=4,
                     iter=2000,
                     warmup=1000,
                     thin=1,
                     seed=123456)

# Post-process the results
tau_samples = fit.extract()['tau']
print('mean(tau): ' + str(numpy.mean(tau_samples)))
print('std(tau):  ' + str(numpy.std(tau_samples)))
fit.plot().savefig('./tau.pdf', bbox_inches='tight')
