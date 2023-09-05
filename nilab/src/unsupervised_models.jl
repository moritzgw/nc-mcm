using Statistics
using LinearAlgebra

export pca

"""
	V, lambda = pca(x)

	PCA returns eigenvectors in columns of V sorted in descending order of eigenvalues lambda.

"""
function pca(x)

	X = cov(x, dims = 1)
	V = eigvecs(X)
	lambda = diag(V'*X*V)
	i = sortperm(lambda, rev = true)
	lambda = lambda[i]
	V = V[:,i]

	return V, lambda
	
end