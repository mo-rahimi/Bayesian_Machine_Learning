# Bayesian Machine Learning:
- `(Gaussian Process regression, Bayesian optimization)`
  
This repository is created  for different ML topics, such as Graph ML, Bayesian ML, and etc. for those who are interested to learn these topics. 

## Project Overview:

We can start with a function and sample that at FOUR equally spaced points between its lower and upper bounds (the two bounds should be included in the four sampled points).

A. Then we use this sampled data to train a `Gaussian Process regression model`, by using `RBF kernel`.

B. Then we try to optimize the function using Bayesian optimization:

C. We will try a different acquisition function, called **Upper Confidence Bound (UCB)**, to use within Bayesian optimization and to minimize the function. 
- UCB is defined as:
           `𝑈𝐶𝐵(𝑥) = 𝑦)(𝑥) − 𝜅𝑠(𝑥)`
          



<img src="GIF/animation.gif" width="800" height="400"/>
