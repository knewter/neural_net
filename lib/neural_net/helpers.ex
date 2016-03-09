defmodule NeuralNet.Helpers do
  import NeuralNet.Constructor
  import NeuralNet.ActivationFunctions

  @moduledoc "These helper functions provide a sort of DSL for specifying network architectures. Look at lib/gru.ex for an example implementation of the Gated Recurrent Unit architecture."

  def set_size(size, vec_names) do
    update! :size_defs, fn size_defs ->
      Enum.reduce vec_names, size_defs, fn vec_name, size_defs ->
        Map.put(size_defs, vec_name, size)
      end
    end
  end

  def mult(inputs, output) do
    link(inputs, output)
    add_operation(output, {:mult, inputs})
  end

  def mult_const(inputs, const, output) do
    link(inputs, output)
    add_operation(output, {:mult_const, inputs, const})
  end

  def add(inputs, output) do
    link(inputs, output)
    add_operation(output, {:add, inputs})
  end

  def customNetLayer(activation_function, activation_function_prime, inputs, output) do
    link(inputs, output)
    add_net_layer(output, {{activation_function, activation_function_prime}, inputs})
  end

  def sigmoid(inputs, output) do
    customNetLayer(&sigmoid/1, &sigmoid_prime/1, inputs, output)
  end

  def tanh(inputs, output) do
    customNetLayer(&tanh/1, &tanh_prime/1, inputs, output)
  end

  def uid, do: :erlang.unique_integer([:monotonic])

  def input, do: :input
  def output, do: :output
  def previous(name), do: {:previous, name}
end
