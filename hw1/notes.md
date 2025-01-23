# ESL Chapter 1: Introduction
Supervised Learning gets its name because of the presence of the outcome variable to guide the learning process. Variable that we are predicting will be in our training data. <br>

Unsupervised Learning only observes the features and has no measurements of the outcome, and rather looks at patterns of how the data is organized or clustered. <br>

Classification Problems: problems where we want to group something into a category. <br>

Regression Problems: problems where we have a quantifiable result that we want to predict. <br>

# Overview of Supervised Learning
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
