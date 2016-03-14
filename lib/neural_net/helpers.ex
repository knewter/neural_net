defmodule NeuralNet.Helpers do
  import NeuralNet.Constructor
  alias NeuralNet.ActivationFunctions

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

  def mult(inputs, output \\ uid) do
    link(inputs, output)
    add_operation(output, {:mult, inputs})
  end

  def mult_const(input, const, output \\ uid) do #only takes 1 input!
    link([input], output)
    add_operation(output, {{:mult_const, const}, [input]})
  end

  def add(inputs, output \\ uid) do
    link(inputs, output)
    add_operation(output, {:add, inputs})
  end

  def add_const(input, const, output \\ uid) do #only takes 1 input!
    link([input], output)
    add_operation(output, {{:add_const, const}, [input]})
  end

  def custom_net_layer(activation_function, activation_function_prime, inputs, output \\ uid) do
    link(inputs, output)
    add_net_layer(output, {{:net_layer, activation_function, activation_function_prime}, inputs})
  end

  def sigmoid(inputs, output \\ uid) do
    custom_net_layer(&ActivationFunctions.sigmoid/1, &ActivationFunctions.sigmoid_prime/1, inputs, output)
  end

  def tanh(inputs, output \\ uid) do
    custom_net_layer(&ActivationFunctions.tanh/1, &ActivationFunctions.tanh_prime/1, inputs, output)
  end

  def uid, do: :erlang.unique_integer([:monotonic])

  def input, do: :input
  def output, do: :output
  def previous(name), do: {:previous, name}
end
