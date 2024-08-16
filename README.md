# Machine-Learning-2019
This repository is created  for different ML topics, such as Graph ML, Bayesian ML, and ... for any one interested to learn these topics. 
Crrently you can find topic related to Bayesian ML and classification nd clustreing in this repo.
you can find related information and the list of content in each corresponding folder.

## List of projects:
1. Bayesian Machine Learning:(Gaussian Process regression, Bayesian optimization)




We start with a function and sample that at FOUR equally spaced points between its lower and upper bounds (the two bounds should be included in the four sampled points).
A. Then we use this sampled data to train a `Gaussian Process regression model`, by using `RBF kernel`.
Produce a figure which shows: (i) the four points sampled, (ii) the true function, (iii) the Gaussian Process regression prediction trained on the four sampled points, and (iv) the standard deviation s(x) in the prediction. 

B. Then we try to optimize the function using Bayesian optimization:
• Use the four equally spaced points from part (a) to create the initial model.
• Sample a further EIGHT points using the Expected Improvement acquisition
function.
• At each iteration, print the value of x selected by the Expected Improvement
acquisition function for evaluation, and the value of the function evaluated at
x.
• After all iterations are complete, print the best (i.e. lowest) point found (both x
and the function value at x).
[6 marks]
(c) You are now going to implement an alternative acquisition function, called Upper Confidence Bound (UCB), to use within Bayesian optimization and to minimize your chosen function. UCB is defined as:
𝑈𝐶𝐵(𝑥) = 𝑦)(𝑥) − 𝜅𝑠(𝑥)
          
 COM761 Machine Learning Ulster University 2022/2023 Prof Zheng & Dr. Hawe where 𝑦)(𝑥) is the Gaussian Process prediction at x, s(x) is the standard deviation in the prediction at x, and κ is a non-negative parameter whose value can be set.
Use UCB as the acquisition function within Bayesian Optimization to optimize your chosen function from part (a). Do this three times:
• In experiment one you should use 𝜅 = 0.5
• In experiment two you should use 𝜅 = 3
• In experiment three you should use 𝜅 = !, where n is the iteration number "
(not including the first four points; i.e. n=1 for the first point selected using
UCB; n=2 for the second point selected using UCB; and so on).
• In each experiment, you should use the same four initial points as in part (a).
• In each experiment, you should use UCB to select a total of EIGHT further
points for evaluation.
• At each iteration of each experiment, print the value of x selected by the UCB
acquisition function, and the value of the function evaluated at x. After all iterations are complete, print the best (i.e. lowest) point found (both x and the function value at x).
Comment on which value of 𝜅 worked best for the optimization of your function, and whether it outperformed Expected Improvement.

