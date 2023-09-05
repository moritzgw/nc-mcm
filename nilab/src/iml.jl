using Random
using Statistics

export shared_pred_inf

"""
	
	shared_pred_inf()

	Estimates shared contributions of features for predicting a target:

	x 		-- 	input data of format [samples, features]
	y 		--	labels of format [samples]
	model 	-- 	model class (not an actual model, i.e., need to pass class without '()')
	G 		--	number of features in the groups to compare
				(G = 2 (default): pairwise contributions, G > 2: random group sampling)
	K 		--	number of context features to randomly sample per group
	L 		-- 	number of iterations for random sampling

	Output

	P 		--	Matrix of estimated shared predictive information of dimension [features, features]

"""
function shared_pred_inf(x, y, model_class; G = 2, K = 0, L = 100)

	# extract parameters
	N, M = size(x)

	# A = zeros(M, M)

	if G == 2 # compute pairwise contributions

		P = zeros(M, M)

		for m1 = 1:M

			for m2 = 1:M

				SA = m1
				SB = m2

				v = zeros(L)
				a = zeros(L)

				for l = 1:L

					# create three models
					modelA = model_class()
					modelB = model_class()
					modelAB = model_class()

					# randomly sample context
					S = randperm(M)[1:K]

					xS = x[:, S]
					xA = x[:, SA]
					xB = x[:, SB]

					# create permuted versions
					ip = randperm(N)
					xAp = xA[ip, :]
					xBp = xB[ip, :]

					# train and compare models
					train!(modelA, hcat(xA, xBp, xS), y)
					train!(modelB, hcat(xAp, xB, xS), y)
					train!(modelAB, hcat(xA, xB, xS), y)

					loss_empty = var(y)

					v[l] = ((loss_empty - modelA.loss_train) + (loss_empty - modelB.loss_train) - (loss_empty - modelAB.loss_train)) / (loss_empty - modelA.loss_train)
					# v[l] = modelAB.loss_train / modelB.loss_train
					a[l] = modelAB.loss_train

				end

				P[SA, SB] += mean(v)
				# A[SA, SB] += mean(a)

			end

		end

	else  # run random sampling

		V = zeros(M, M) # shared info
		# E = zeros(M, M) # loss
		C = zeros(M, M) # counter

		for l = 1:L

			# create three models
			modelA = model_class()
			modelB = model_class()
			modelAB = model_class()

			# randomly sample features
			S = randperm(M)
			
			SA = S[1:G]
			SB = S[(G+1):2*G]
			SC = S[(2*G+1):(2*G+1)]

			xA = x[:, SA]
			xB = x[:, SB]
			xSC = x[:, SC]

			# create permuted versions
			ip = randperm(N)
			xAp = xA[ip, :]
			xBp = xB[ip, :]

			# train and compare models
			# train!(modelA, hcat(xA, xBp, xSC), y)
			train!(modelB, hcat(xAp, xB, xSC), y)
			train!(modelAB, hcat(xA, xB, xSC), y)

			v = modelAB.loss_train / modelB.loss_train

			V[SA, SB] .+= v
			C[SA, SB] .+= 1

		end

		P = V ./ C
		P[findall(isnan.(P))] .= 0

	end

	return P

end


function sample_test_dag(N = 1000)

	# x_1 -> x_2 -> y <- x_3 <- x_4 -> x_5; x_1 <- x_6 -> y; y <- h_1 -> x_7; x_3 <- h_2 -> x_8

	eps = randn(N, 8)
	h1 = randn(N, 1)
	h2 = randn(N, 1)

	# eps[:,3] = eps[:, 3]

	A = [	1 0 0 0 0 1 0 0 ; 
			1 1 0 0 0 0 0 0 ; 
			0 0 1 1 0 0 0 0 ;
			0 0 0 1 0 0 0 0 ;
			0 0 0 1 1 0 0 0 ;
			0 0 0 0 0 1 0 0 ;
			0 0 0 0 0 0 1 0 ;
			0 0 0 0 0 0 0 1 ]

	x = (A * eps')'

	x[:, 7] = x[:, 7] + h1
	x[:, 8] = x[:, 8] + h2
	x[:, 3] = x[:, 3] + h2

	y = x[:, 2] + x[:, 3] + x[:, 6] + h1 + randn(N)

	return x, y

end