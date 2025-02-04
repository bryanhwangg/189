# Soft-Margin Support Vector Machines
Solves 2 problems:
Hard-margin SVMs fail if data is not linearly separable <br>
Sensitive to outliers <br>
Small differences in data can cause big changes in decision boundary <br>


Idea: Allow some points to violate the margin, with slack variables. Modifies constraint for point i:  <br> 
$y_i(X_i \cdot w + \alpha) \geq 1 - \xi$ <br>
$\xi_i \geq 0$

Slack variable $\xi$ gives some slack so that certain training points can violate the margin. A $\xi$ value of 0 would give the same results as a hard-margin SVM <br>

A point has a positive $\xi$ if the point is inside the "slab" or is on the completely wrong side of the decision boundary. All other points have a $\xi$ of 0. <br>

To prevent abuse of slack, modify objective function to include a loss term that penalizes the use of slack. <br>
Optimization problem: <br>
Find $w, \alpha, \xi_i$ that minimize $||w||^2 + C\sum^n_{i=1}\xi_i$ subject to $y_i(x_i \cdot + \alpha) \geq 1 - \xi_i$
<br>
For all $i \in [1, n]$ <br>
$\xi_i \geq 0$ <br>

As C increases, soft margin support vector machine gets closer to a hard margin support vector machine, so gets hard for computer to find decision boundary with non-linearly separable data. <br>

Falls into a class of optimization called quadratic program in d+n+1 dimensions and 2n constraints. <br>

$C > 0$ is a scalar regularization hyperparameter that trades off <br>

For a small C, you want to maximize margin $\frac{1}{||w||}$ and for big C you want to keep most slack variables 0 or small. <br>

For small C, you have danger of underfitting (misclasssifies a lot of training data). For big C, there is danger of overfitting. <br>

For small C, model is less sensitive to outliers. For big C, model is very sensitive to outliers. <br>

For small C, boundaries are more "flat". For big C, boundaries are more sinous <br>


