# Theory of Hard-Margin Support Vector Machines

Decision Rule / Classifier is a function $r$: $\mathbb{R}^d$ $\rightarrow$ $\pm$ 1 <br/>
$r$ will map a vector (or test point) to +1 if it is predicted to be in a class and -1 otherwise <br>
<br>
$
r(x) =
\begin{cases} 
+1 & \text{if } w \cdot x + \alpha \geq 0 \\
-1 & \text{otherwise}
\end{cases}
$
where w $\in$ $\mathbb{R}^d$ and $\alpha$ $\in$ $\mathbb{R}$ are the parameters of the support vector machine <br>

To determine the best decision boundary for our SVM, let there be $n$ training data points and $n$ corresponding data points. <br>

Suppose there is a training point $X_i$ $\in$ $\mathbb{R^d}$ where there are $d$ columns in $X_i$ since $X_i$ must have the same dimensionality of our classifier function $r$.
Let there also be a corresponding label to this data $y_i$ $\in$ $[-1, + 1]$. <br>



