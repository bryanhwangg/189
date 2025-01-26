# ESL Chapter 1: Introduction
Supervised Learning gets its name because of the presence of the outcome variable to guide the learning process. Variable that we are predicting will be in our training data. <br>

Unsupervised Learning only observes the features and has no measurements of the outcome, and rather looks at patterns of how the data is organized or clustered. <br>

Classification Problems: problems where we want to group something into a category. <br>

Regression Problems: problems where we have a quantifiable result that we want to predict. <br>


# ESL Chapter 4.5-4.5.1
## Separating Hyperplanes
When using hyperplanes to separate and classify data into categories, there are infinitely many hyperplanes in a given $\mathbb{R}^d$. Typically, using a least squares linear decision boundary will make errors when trying to create a boundary between classes. <br>

## Rosenblatt's Perceptron Learning Algorithm
The perceptron learning algorithm tries to find a separating hyperplane by minimizing the distance of misclassified points to the decision boundary. <br>

If a response $y_i = 1$ is incorrectly predicted, then $x^T_i\beta+\beta_0<0$ and the opposite for an incorrect prediction for $y_i = - 1$ . <br>

The goal is to minimize $D(\beta, \beta_0) = -\sum_{i \in M}y_i(x_i^T\beta + \beta_0)$






# Overview of Supervised Learning: MISC
## Types of Variables 
Types of variables we want to predict: qualitative and quantitative <br>

Qualitative Variables: also referred to as categorical or discrete variables as well as factors. Problems where we are predicting qualitative variables are referred to as __classification__ problems. <br>

Quantitative Variables: typically numbers and is distinct from qualitative variables since there is a sense of measurement that we can use. Problems where we are predicting a quantitative variable are referred to as __regression__ problems. <br>

Ordered Categorical Variables: ordering between values, but no metric notation that is applicable. Ex. Small, Medium, Large. <br>

Typically qualitative variables will be numericized, for example a classification function can represent its output between two categories as 1 or 0. For classification problems that have greater than two categories, dummy variables are typically used. <br>

We can loosely state all learning tasks as: given the value of an input vector $X$, make a good prediction of the output $Y$, which is denoted by $\hat{Y}$. If $Y$ $\in$ $\mathbb{R}$, then $\hat{Y}$ $\in$ $\mathbb{R}$ as well. This logic is the same for categorical inputs and outputs as well. <br>

For a two class classification problem where $G$ is the possible categories of the input vector, one common approach is to denote the binary coded target as $Y$, and then treat the binary coded $G$ as the quantitative variable $Y$. By predicting $G$ as a $\hat{Y}$, the prediction $\hat{Y}$ will typically fall between the range [0, 1]. By setting a threshold for $\hat{Y}$ to be at least 0.5 to be encoded as a 1 in the binary representation of $G$, you can successfully predict a qualitative variable by predicting a quantitative variable. <br>

This approach generalizes to $K$-level qualitative outputs as well by setting different thresholds for each $k_i$ category in $K$ outputs. <br>


## Two Simple Approaches to Prediction: Least Squares and Nearest Neighbors
The linear model of least squares makes huge assumptions about structure and yields stable but possible innacurate predictions. The $k$-nearest-neighbor prediction rule on the other hand makes very mild structural assumptions, and provides accurate but unstable predictions. <br>

### Linear Models and Least Squares
The linear model for an input vector $X^T$ = ($X_1$, $X_2$, $X_3$, ..., $X_p$) we can predict the output $Y$ using the following model: 
$\hat{Y}$ = $\hat{\beta_0}$ + $\sum^p_{j=1}$ $X_j$ $\hat{\beta_j}$ <br>

The term $\hat{\beta_0}$ is the intercept, or bias. A more condensed representation of the linear model is $\hat{Y}$ = $X^T \hat{\beta}$, after you include $\hat{\beta_0}$ in the $\hat{\beta}$ vector. <br>

Since this model is a regression model of a single quantitative value, $\hat{Y}$ is a scalar, but $\hat{Y}$ can also be a $K$-vector as well, where $\hat{\beta}$ would have to be a $p$ $\times$ $K$ matrix of coefficients. <br>

In the ($p$ + 1) dimensional input-output space, (X, $\hat{Y}$) represents a hyperplane. If $\hat{\beta_0}$ is included in in the $\hat{X}$ input vector vector, then the hyperspace includes the origin and is a subspace. Otherwise, the hyperspace is a an affine set cutting the Y-axis at the point (0, $\hat{\beta}$). <br>

As a function over the $p$ dimensional input space, $f(X) = X^T \beta$ is linear. The gradient $f'(X) = \beta$ is a vector in the input space that points in the steepest uphill direction. <br>

Least squares method to fit a linear model to a set of training data picks the coefficients in $\beta$ to minimize the residual sum of squares <br>

$RSS(\beta) = \sum^N_{i=1} (y_i - x_i^T \beta)^2$ <br>

Since the equation for the $RSS$ is quadratic, there will always be a minimina, but the minimina may not be unique. The equation can be differentiated to get the normal equation: <br>

$\hat{\beta} = (X^TX)^{-1}X^Ty$ <br>

The aforementioned method to convert a qualitative variable prediction to a quantitative prediction utilizes a decision boundary, which was defined as 0.5 for the classification for an input vector to be assigned the encoded category of 1. <br>

If you have data from a bivariate normal distribution, a linear decision boundary will be the best way to classify between the two categories. However, if you have a mixture of Gaussian distributions (10 different Gaussian distributions where the means of the Gaussians are drawn from a Gaussian as well for example), then a linear decision boundary is less likely to be optimal. <br>

## Nearest Neighbor Models
$k$-nearest neighbor formula is as follows:
$\hat{Y}(x) = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i$ <br>
Where $N_k(x)$ is the neighborhood of points to $x$ that are defined as the $k$ closest points to $x_i$ in the training sample. Closeness is typically determined by Euclidian distance in case of 2D data like the example shown in the textbook. <br>

## From Least Squares to Nearest Neighbors
Linear decision boundaries from least squares are typically very smooth and are relatively stable to fit, but maeks the assumption that a linear decision boundary is the best choice for the data. In other words, the least squares linear decision boundary has low variance and high bias. <br>


