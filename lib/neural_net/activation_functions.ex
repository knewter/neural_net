defmodule NeuralNet.ActivationFunctions do
  @moduledoc "A few activation functions and their derivatives."

  @doc "Tanh has a range of -1 to 1."
  def tanh(x), do: :math.tanh(x)
  @doc "Returns the derivative of tanh at x."
  def tanh_prime(x), do: 1/:math.pow(:math.cosh(x), 2)

  @doc "This sigmoid has a range of 0 to 1.  It uses 1 / (1 + e^-x)"
  def sigmoid(x), do: 1 / (1 + :math.exp(-x))
  @doc "Returns the derivative of this sigmoid function at x."
  def sigmoid_prime(x), do: sigmoid(x) * (1 - sigmoid(x))
end
