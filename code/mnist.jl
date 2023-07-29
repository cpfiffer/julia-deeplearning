### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 04b7adaf-c369-4d22-80aa-24d0f0301c37
using Pkg; Pkg.activate("..")

# ╔═╡ 3726383c-2a91-11ee-2be6-8f64410b6b5b
# Setting up imports
begin 
	using Flux
	using FluxTraining
	using MLDatasets
	using MLUtils
	using cuDNN
	using CUDA
	using Colors
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

Anyway. Let's move on. 
"""

# ╔═╡ 66a8c5b1-017b-463e-83ca-4224cf9bcd06
X, y = dataset[:];

# ╔═╡ a2250120-c293-4932-9ca7-5bd30ba2ffee

vec_X = reshape(X[:,:,:], (28^2, 60_000))' |> Array

# ╔═╡ 0a0b518e-9256-4a39-8a11-6c274831fc6f
loader = DataLoader((vec_X, y), batchsize=32, shuffle=true);

# ╔═╡ 56c33691-3f7d-4cf1-b0a4-8fdced123921
model = Chain(
	Dense(28^2 => 10, relu),
	Dense(10 => 10, relu),
	Dense(10 => 10, softmax)
)

# ╔═╡ 407a726c-1922-4dfe-8fe5-56ea613cc981
learner = Learner(model, Flux.Losses.logitcrossentropy)

# ╔═╡ 3ed8922e-8f74-44b8-839d-462061af02b8
fit!(learner, 10, (10, 10))

# ╔═╡ Cell order:
# ╠═04b7adaf-c369-4d22-80aa-24d0f0301c37
# ╟─ced62b30-0860-4b46-8c52-db460cf8c29e
# ╠═3726383c-2a91-11ee-2be6-8f64410b6b5b
# ╟─989319a2-6ca8-4030-846c-b16145419e81
# ╠═3b986a76-849f-462c-a86e-b0be03d8f56d
# ╟─4dda4aba-053e-4afe-8c9a-d3a45236b16f
# ╠═40608d61-775b-41f2-a5f5-1352fb545d22
# ╟─25a1adc5-5a9e-4663-8878-62706c1d562c
# ╠═5f2722de-80ad-463d-aa75-acf1368d3db5
# ╠═b3a694e2-7a15-4021-85cd-a4bc538c4568
# ╠═3964ccf5-a408-4b96-ba26-09306b275ea8
# ╠═2bad0e09-c03e-490e-994f-b31a6f969e8e
# ╠═66a8c5b1-017b-463e-83ca-4224cf9bcd06
# ╠═a2250120-c293-4932-9ca7-5bd30ba2ffee
# ╠═0a0b518e-9256-4a39-8a11-6c274831fc6f
# ╠═56c33691-3f7d-4cf1-b0a4-8fdced123921
# ╠═407a726c-1922-4dfe-8fe5-56ea613cc981
# ╠═3ed8922e-8f74-44b8-839d-462061af02b8
