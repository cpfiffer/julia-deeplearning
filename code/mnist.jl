### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 3726383c-2a91-11ee-2be6-8f64410b6b5b
# Setting up imports
begin 
	import Pkg; Pkg.activate("..")
	using Revise
	using Flux
	using FluxTraining
	using MLDatasets
	using MLUtils
	using cuDNN
	using CUDA
	using Colors
	using Statistics
	using OneHotArrays
	using Random
	using Printf
	using Plots
	using ProgressLogging
	using PlutoUI
	plotlyjs()
end

# ╔═╡ ced62b30-0860-4b46-8c52-db460cf8c29e
md"""
The first part is loading all the packages we're going to use! In this case, I know I'm going to use Flux.jl (Julia's deep learning library) and MLDatasets.jl, which lets you acces a bunch of standard datasets in machine learning.
"""

# ╔═╡ 989319a2-6ca8-4030-846c-b16145419e81
md"""
Next up, let's pull in the MNIST dataset. MNIST is essentially the "hello world" of computer vision datasets. It's a bunch of handwritten digits (0-9) in grayscale! You can read more about how `MLDatasets` structures this stuff in [their docs](https://juliaml.github.io/MLDatasets.jl/stable/datasets/vision/#MLDatasets.MNIST)
"""

# ╔═╡ 3b986a76-849f-462c-a86e-b0be03d8f56d
dataset = MLDatasets.MNIST(:train)

# ╔═╡ 4dda4aba-053e-4afe-8c9a-d3a45236b16f
md"""
Let's take a look at one of the observations. Conveniently, calling `dataset[1]` will return the feature and target for the first observation, so we can "unpack" the feature and target into new variables, `x1` and `y1`.
"""

# ╔═╡ 40608d61-775b-41f2-a5f5-1352fb545d22
x1, y1 = dataset[1];

# ╔═╡ 25a1adc5-5a9e-4663-8878-62706c1d562c
md"""
The `Colors` package provides some tools for us to be able to make various images. In our case, we can call `Grays.(x1)` to make a greyscale image of our first letter:
"""

# ╔═╡ 5f2722de-80ad-463d-aa75-acf1368d3db5
Gray.(x1')

# ╔═╡ b3a694e2-7a15-4021-85cd-a4bc538c4568
md"""
Uh. Well, I have no idea what that is. Let's look at `y1` to see what this digit actually is.
"""

# ╔═╡ 3964ccf5-a408-4b96-ba26-09306b275ea8
# Figure out what the character is coded as
y1

# ╔═╡ 2bad0e09-c03e-490e-994f-b31a6f969e8e
md"""
A five? That's a five. Whatever man. Work on your handwriting because holy hell that's bad.

Anyway. Let's move on. Our next challenge is to get all of our data into a shape that we can work with. There's a few steps we need to take:

- We have $28 \times 28$ matrices that we need to flatten into vectors of length $28^2$. 
- We want to one-hot encode our outcomes, i.e. we can't have `4` be our output, we need to have the outcome be a vector of length 10 with a 1 where the result is. In the case of 4, we want to have `[0,0,0,0,1,0,0,0,0,0]`. Note that the 1 is in the fifth spot -- the first element of this vector indicates whether the digit is a 0.
- We want to create a validation set! We'll default to 20% of the sample as validation.
- We want to bundle everything up into a [`DataLoader`](https://fluxml.ai/Flux.jl/stable/data/mlutils/#MLUtils.DataLoader), which handles batching, shuffling, etc.
"""

# ╔═╡ 6b16aa0d-a7db-4e9c-a8e2-b36c30eff83e
function process(dt; pct=0.2, batchsize=512)
	# Extract all the features and targets
	X_raw, y_raw = dt[:]

	# Reshape the X's from a 28 x 28 x N to a 28^2 x N matrix.
	# We also divide by 255 to scale the features to between 0 and 1.
	X = reshape(X_raw, (28^2, size(X_raw, 3))) ./ 255

	# One-hot encode the Ys
	y = onehotbatch(y_raw, collect(0:9))

	# Draw pct samples for validation
	valid_inds = shuffle(eachindex(y_raw))
	cutpoint = Int(floor(size(X, 2) * 0.2))
	X_valid, y_valid = X[:,valid_inds[1:cutpoint]], y[:,valid_inds[1:cutpoint]]
	X_train, y_train = X[:,valid_inds[(cutpoint+1):end]],
		y[:,valid_inds[(cutpoint+1):end]]

	@info "Data size" size(X_valid) size(y_valid) size(X_train) size(y_train)

	# Return dataloaders for validation and train
	return (
		train=DataLoader((X_train, y_train) |> gpu, shuffle=true, batchsize=batchsize),
		valid=DataLoader((X_valid, y_valid) |> gpu, shuffle=true, batchsize=batchsize),
	)
end

# ╔═╡ f5d9cfaa-27f7-40dd-9bed-bbbda0521155
md"""
Let's take the `MNIST(:train)` dataset and process it using the function we just made.
"""

# ╔═╡ 66a8c5b1-017b-463e-83ca-4224cf9bcd06
# Generate our training and validation samples
trainload, validload = process(MNIST(:train));

# ╔═╡ 99e776bf-a521-4c0b-9e0c-2cdb80e74090
md"""
How "thick" do you want the hidden layers to be? $(@bind thickness NumberField(1:30, default=10)) units.

How many layers do you want to use? $(@bind n_layers NumberField(1:10, default=5)) layers.

How many epochs do you want to train for? $(@bind epochs NumberField(1:10, default=10)) epochs.
"""

# ╔═╡ 56c33691-3f7d-4cf1-b0a4-8fdced123921
model = Chain(
	Dense(28^2 => thickness, celu),
	repeat([Dense(thickness => thickness, celu)], n_layers)...,
	Dense(thickness => 10, celu),
	softmax
) |> gpu

# ╔═╡ 7962c59a-5ded-47a3-a81e-371d8fc160ba
function accuracy(model::Flux.Chain, data::DataLoader)
	# Compute the percent that the model ascribes to the true outcome
	# mean(sum(map(x -> model(x[1]) .* x[2], data), dims=1))
	# for (feature, label) in data
	# 	yhat = sum(model(feature) .* label, dims=1)
	# end
	mean(map(x -> sum(model(x[1]) .* x[2], dims=1), data)) |> mean |> cpu
end

# ╔═╡ c6403f4a-936f-4359-8b69-19bf8cc41f21
state = Flux.setup(Flux.Adam(0.001), model)

# ╔═╡ 3cadc0c0-131a-49c8-ac10-29018ca76035
train_accuracy = zeros(Float32, epochs);

# ╔═╡ e9ba68f1-ee10-4663-901e-0586e47cafb5
valid_accuracy = zeros(Float32, epochs);

# ╔═╡ 93dbef0a-7c9b-4be8-b0c0-01e273bf8360
# Manual training
begin
	done = false
	@progress for epoch in 1:epochs
		# @info "Epoch $epoch"
		# Iterate through samples
		for (feature, label) in trainload
			# Compute the gradients
			grads = Flux.gradient(model) do m
				result = m(feature)
				Flux.Losses.crossentropy(result, label)
			end
			# Update the model
			Flux.update!(state, model, grads[1])
		end
	
		# Compute accuracy
		train_accuracy[epoch] = accuracy(model, trainload)
		valid_accuracy[epoch] = accuracy(model, validload)
	end
	done=true
end


# ╔═╡ f5136eb5-329b-4692-a6f1-91b0387dc499
let 
	done 
	p = plot(valid_accuracy, label="valid");
	plot!(p, train_accuracy, label="train")
end

# ╔═╡ 35151dec-88d3-4cdc-8d3b-3a1fd75ef0cf
function predict(img, model)
	predictions = model(vec(img) |> gpu) |> cpu
	prob, i = findmax(predictions)
	# display(predictions)
	return (prob, i-1)
end

# ╔═╡ 49e36999-2f21-47ca-98a1-c9e4fe587aef
predict(dataset.features[:,:,3], model)

# ╔═╡ 2f3888e2-1df4-4df5-8966-f64def460992
@bind sample_number Slider(1:30)

# ╔═╡ f9defc31-e394-4192-b586-cb241dd180a0
Gray.(MNIST(:test).features[:,:,sample_number]')

# ╔═╡ c192f6dc-1797-49c7-a439-f5246fd6ae51
let
	prob, thing = predict(MNIST(:test).features[:,:,sample_number], model)
	md"""
	The number is probably a $(thing), with probability $(prob)!
	"""
end

# ╔═╡ Cell order:
# ╟─ced62b30-0860-4b46-8c52-db460cf8c29e
# ╠═3726383c-2a91-11ee-2be6-8f64410b6b5b
# ╟─989319a2-6ca8-4030-846c-b16145419e81
# ╠═3b986a76-849f-462c-a86e-b0be03d8f56d
# ╟─4dda4aba-053e-4afe-8c9a-d3a45236b16f
# ╠═40608d61-775b-41f2-a5f5-1352fb545d22
# ╟─25a1adc5-5a9e-4663-8878-62706c1d562c
# ╠═5f2722de-80ad-463d-aa75-acf1368d3db5
# ╟─b3a694e2-7a15-4021-85cd-a4bc538c4568
# ╠═3964ccf5-a408-4b96-ba26-09306b275ea8
# ╟─2bad0e09-c03e-490e-994f-b31a6f969e8e
# ╠═6b16aa0d-a7db-4e9c-a8e2-b36c30eff83e
# ╟─f5d9cfaa-27f7-40dd-9bed-bbbda0521155
# ╠═66a8c5b1-017b-463e-83ca-4224cf9bcd06
# ╟─99e776bf-a521-4c0b-9e0c-2cdb80e74090
# ╠═56c33691-3f7d-4cf1-b0a4-8fdced123921
# ╠═7962c59a-5ded-47a3-a81e-371d8fc160ba
# ╠═c6403f4a-936f-4359-8b69-19bf8cc41f21
# ╠═3cadc0c0-131a-49c8-ac10-29018ca76035
# ╠═e9ba68f1-ee10-4663-901e-0586e47cafb5
# ╠═93dbef0a-7c9b-4be8-b0c0-01e273bf8360
# ╠═f5136eb5-329b-4692-a6f1-91b0387dc499
# ╠═35151dec-88d3-4cdc-8d3b-3a1fd75ef0cf
# ╠═49e36999-2f21-47ca-98a1-c9e4fe587aef
# ╠═2f3888e2-1df4-4df5-8966-f64def460992
# ╠═f9defc31-e394-4192-b586-cb241dd180a0
# ╟─c192f6dc-1797-49c7-a439-f5246fd6ae51
