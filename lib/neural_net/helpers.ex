defmodule NeuralNet.Helpers do
  import NeuralNet.Constructor
  import NeuralNet.ActivationFunctions

  @moduledoc "These helper functions provide a sort of DSL for specifying network architectures. Look at lib/gru.ex for an example implementation of the Gated Recurrent Unit architecture."

  def def_vec(ids, vec_names) do
    update! :vec_defs, fn vec_defs ->
      Enum.reduce vec_names, vec_defs, fn vec_name, vec_defs ->
        Map.put(vec_defs, vec_name, ids)
      end
    end
  end

  def def_vec_by_size(size, vec_names) do
    def_vec(Enum.map(1..size, &uid/0), vec_names)
  end

  def mult(inputs, output) do
    link(inputs, output)
    add_operation(output, {:mult, inputs})
  end

  def mult_const(inputs, const, output) do #only takes 1 input!
    link(inputs, output)
    add_operation(output, {{:mult_const, const}, inputs})
  end

  def add(inputs, output) do
    link(inputs, output)
    add_operation(output, {:add, inputs})
  end

  def add_const(inputs, const, output) do #only takes 1 input!
    link(inputs, output)
    add_operation(output, {{:add_const, const}, inputs})
  end

  def custom_net_layer(activation_function, activation_function_prime, inputs, output) do
    link(inputs, output)
    add_net_layer(output, {{:net_layer, activation_function, activation_function_prime}, inputs})
  end

  def sigmoid(inputs, output) do
    custom_net_layer(&sigmoid/1, &sigmoid_prime/1, inputs, output)
  end

  def tanh(inputs, output) do
    custom_net_layer(&tanh/1, &tanh_prime/1, inputs, output)
  end

  def uid, do: :erlang.unique_integer([:monotonic])

  def input, do: :input
  def output, do: :output
  def previous(name), do: {:previous, name}
end
