library(ggplot2)
library(tidyverse)
# Creating a Matrix -------------------------------------------------------
A <- matrix(c(1, 2, 1, 1), nrow = 2, ncol = 2)
A
b <- c(4, 5)
b
solve(A, b)

# Example42 ---------------------------------------------------------------
# x1 + 2x2 + 3x3 = 6
# -2x1 + 3x2 - 2x3 = -1
# -x + 2x2 + x3 = 2
m <- matrix(c(1, -2, -1, 2, 3, 2, 3, -2, 1, 6, -1, 2), nrow = 3, ncol = 4)
print(m)
A <- m[ ,-4]

b <- m[ ,4]
solve(A, b)


m <- matrix(c(1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1), nrow = 4, ncol = 4)
m
b <- c(475, 489, 542, 422)
solve(m, b)

# matrix_arithmetic -------------------------------------------------------
# library(magick)
v1 <- c(2, -1, 3)
v2 <- c(-1, 0, 4)
v1 + v2

# sum of two matrices -----------------------------------------------------

A <- matrix(c(3, -1, 0, -3, -5, 4), nrow = 2, ncol = 3)
B <- matrix(c(-5, 1, 5, -2, 2, 0), nrow = 2, ncol = 3)
A + B

# scalar multiplication ---------------------------------------------------
-3 * A

# Dot product -------------------------------------------------------------

v1 %*% v2

B <- matrix(c(-5, 2, -2, 5, 1, 0), nrow = 3, ncol = 2)

A %*% B

# Transpose ---------------------------------------------------------------

A <- matrix(c(4, 0, -1, 1, -5, -2), nrow = 2, ncol = 3)
A_t <- matrix(c(4, -1, -5, 0, 1, -2), nrow = 3, ncol = 2)
# or 
t(A)

# System Linear Equations as a Matrix -------------------------------------

# 0 + 1x2 + 3x3 - 1x4 = 1
# -1x1 + 1x2 - 4x3 + 0 = 1
# 1x1 + 0 + 0 + 2x3 + 4x4 = 5
# 0 +1x2 + 0 - 4x4 = -2

# first column a1, the second column be a2, the third be a3, the fourth be a 4, and the last be b

# a1 = [0, -1, 1, 0], a2 = [1, 1, 0, 1], a3 = [3, -4, 2, 0], a4 = [-1, 0, 4, -4]

# x1 * [ ,1] = x1 * [ 0] x2 * [ 1] x3 * [ 3] x4 * [-1] = A * x where x = c(x1, x2, x3, x4)
#                   [-1]      [ 1]      [-4]      [ 0]
#                   [ 1]      [ 0]      [ 2]      [ 4]
#                   [ 0]      [ 1]      [ 0]      [-4]
#
# A * x = b

# Exercise 2.3 ------------------------------------------------------------

A <- matrix(c(2, -5, -4, -1, 4, -3), nrow = 2, ncol = 3, byrow = TRUE)
B <- matrix(c(5, 0, 1, -2, -5, -3), nrow = 2, ncol = 3, byrow = TRUE)
C <- matrix(c(-2, 0, -3, -1, 3, 5, 2, 4, -5), nrow = 3, ncol = 3, byrow = TRUE)
D <- matrix(c(0, 1, 1, 1, 3, 3, 1, 5, 3), nrow = 3, ncol = 3, byrow = TRUE)
E <- matrix(c(0, -2, 4, 2, 3, 5, -3, -4, 3, -4, -1, -1), 3, 4, byrow = TRUE)
F <- matrix(c(1, -4, -5, -1, -1, 0, -2, 1, 0, 1, 1, 4), 3, 4, byrow = TRUE) 

# 1. (-2) * A = matrix(c(-4, 10, 8, 2, -8, 6), nrow = 2, ncol = 3, byrow = TRUE)
matrix(c(-4, 10, 8, 2, -8, 6), nrow = 2, ncol = 3, byrow = TRUE)
-2 * A

# 2 A + B = matrix(c(2 + 5, 0 + -5, -4 + 1, -2 + -1, -5 + 4, -3 + -3), 2, 3, byrow = TRUE)  
matrix(c(2 + 5, 0 + -5, -4 + 1, -2 + -1, -5 + 4, -3 + -3), 2, 3, byrow = TRUE)
A + B

# 3 t(C) or C^T = matrix(c(-2, 0, -3, -1, 3, 5, 2, 4, -5), nrow = 3, ncol = 3)
matrix(c(-2, 0, -3, -1, 3, 5, 2, 4, -5), nrow = 3, ncol = 3)
t(C)

# 11 C * D - E * F^T
# C * D = 3 x 3
matrix(c(-2 * 0 + 0 * 1 + -3 * 1, -2 * 1 + 0 * 3 + -3 * 5, -2 * 1 + 0 * 3 + -3 * 3,
         -1 * 0 + 3 * 1 + 5 * 1, -1 * 1 + 3 * 3 + 5 * 5, -1 * 1 + 3 * 3 + 5 * 3,
         2 * 0 + 4 * 1 + -5 * 1, 2 * 1 + 4 * 3 + -5 * 5, 2 * 1 + 4 * 3 + -5 * 3), nrow = 3, ncol = 3, byrow = TRUE)
C %*% D

matrix(c(1, -4, -5, -1, -1, 0, -2, 1, 0, 1, 1, 4), ncol = 3, nrow = 4)
t(F)
# final answer
C%*% D - E %*% t(F)

# 2.2.8 Practical Applications

# standard deviation
sigma <- matrix(c( 4, 2, 2, 3), ncol = 2, nrow = 2)
# mean
mu <- c(1, 2)
n <- 1000
set.seed(123)
library(mvtnorm)
x <- rmvnorm(n = n, mean = mu, sigma = sigma)

df <- data.frame(x)  
View(x)  
ggplot(data = df, mapping = aes(x = X1, y = X2)) +
  geom_point(alpha = .5) +
  geom_density_2d()
  
# to get to the origin of the graph
# mu <- c(-1, -2) since the center is at (1, 2)
y <- x - mu
E <- eigen(sigma)
E$vectors
inv <- solve(E$vectors)
y <- y %*% t(inv)

df2 <- data.frame(y)
View(y)

ggplot(df2, mapping = aes(x = X1, y = X2)) +
  geom_point(alpha = .5) +
  geom_density_2d()


# Commutative law for addition --------------------------------------------

A <- matrix(1:9, 3, byrow = TRUE)
B <- matrix(10:18, 3, byrow = TRUE)
C <- matrix(18:26, 3, byrow = TRUE)
A + B
B + A

# Associative -------------------------------------------------------------

A + B == B + A

A +(B + C) == (A + B) + C

2 * (A + B)
2 * A + 2 * B

3 * (A - B)
3 * A - 3 * B

(2 + 3) * C == 2 * C + 3 * C
(3 - 1) * C == 3 * C - 1 * C

# Inverse -----------------------------------------------------------------

# If A is a square matrix m x m then the inverse of a matrix A is an m x m matrix A ^ -1 such that 
# A * A ^ -1 = I
# If A has an inverse then A is invertible

A <- matrix(c(-1, 1, 4, -3), nrow = 2, ncol = 2)
solve(A) %*% A

A <- solve(A)
solve(A) # the inverse of an inverse is the original matrix
library(MASS)
A <- matrix(c(1, -2, -1, 2, 3, 2, 3, -2, 1), ncol = 3)
fractions(solve(A))

# A * x = b

# x1 + 2x2 + 3x3 = 6 
# -2x1 + 3x2 - 2x3 = -1
# -1x1 + 2x2 + 1x3 = 2

A <- matrix(c(1, -2, -1, 2, 3, 2, 3, -2, 1), ncol = 3)
b = matrix(c(6, -1, 2), ncol = 1)

solve(A) %*% b
# same as solve(A, b)


# Theorem 2.16 ------------------------------------------------------------

# (A^T)^-1 = (A^-1)^T

solve(t(A)) %>% MASS::fractions()
solve(A) %>% t() %>% MASS::fractions()

# Theorem 2.17 ------------------------------------------------------------

# If A is a square and symmetric matrix, then A^T = A

A <- matrix(c(3, 1, 4, 1, 5, 0, 4, 0, -2), ncol = 3)

t(A)

# Theorem 2.20 ------------------------------------------------------------

# M = [A|In] where A is an n x matrix and In is the identity matrix of size n

# the reduced row echelon from of M is R
# R = [I|A^-1]

A <- matrix(c(-1, 4, 1, -3), nrow = 2, byrow = TRUE)

I <- matrix(c(1, 0, 0, 1), nrow = 2)

cbind(A, I)
R <- cbind(I, solve(A))
R



# Example 79

A <- matrix(c(1, 2, 2, -4, 3, 5, 2, -2, 2, -2, -3, -1, 1, -3, -3, 4), nrow = 4)
A
I <- matrix(c(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), ncol = 4)
# diag(4)
cbind(A, I)
R <- cbind(I, solve(A) %>% MASS::fractions())
R

# determinants ------------------------------------------------------------

A <- matrix(c(-2, -1, -5, 4, -1, 0, -5, 1, -3), ncol = 3)

det(A)

A<- matrix(c(0, 2, 5, -3, -1, 4, 0,0, -1), ncol = 3)

det(A)

# exercise 3.3
A <- matrix(c(-3, -1, 1, 0, 2, 1, 3, 2, -1), ncol = 3)
det(A)


# Lab exercise 101 --------------------------------------------------------

A <- matrix(sample(-10:10, 25, replace = TRUE), ncol = 5, nrow = 5)
print(A)
# 1
det(-A)

# 2
det(2 * A)

# 3

det(5 * A)

# 4

det(-10 * A)

det(A)

# Lab 102

det(A^2)

det((2 * A)**2)

det(solve(A))

install.packages("plotly")


# Verifying some stuff ----------------------------------------------------

A <- matrix(c(0, 1, 3, -1, -1, 1, -4, 0, 1, 0, 2, 4, 0, 1, 0, -4), byrow = TRUE, ncol = 4)
A
E <- matrix(c(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1), ncol = 4)

E %*% A

det(E %*% A)
# det(A %*% B) = det(A) %*% det(B)
det(E)
det(A)

det(t(A))
det(A)

det(solve(A))
1/30
