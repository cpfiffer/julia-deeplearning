# Programming Environment

There are a few ways of setting up your programming environment for this course, each with their own pros and cons. This page provides a brief overview.


## Local

For working locally, we recommend using [Visual Studio Code](https://code.visualstudio.com) with the [Julia extension](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia).


## Google Colab (Cloud)

You can use Google Colab to get free GPU compute (when available). To set up Julia on Colab, you can follow this [example notebook](https://colab.research.google.com/drive/1rp0OXWr1fPbnm8vWwRHdHP0lLlLbL8d9?usp=sharing).

Note the following two potential pitfalls:

* You need to re-load the notebook after running the Julia installation (otherwise the notebook still runs with a python kernel and won't run Julia code)
* If you want to switch the Runtime (e.g. to enable one with a GPU), you need to run the Julia install *after* switching the runtime, because the switch loads a fresh runtime without Julia installed, regardless of whether you've installed it already in your previous runtime.