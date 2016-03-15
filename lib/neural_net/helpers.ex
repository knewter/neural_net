defmodule NeuralNet.Helpers do
  import NeuralNet.Constructor
  alias NeuralNet.ActivationFunctions

  @moduledoc """
  These helper functions provide a sort of DSL for specifying network architectures. Look at lib/gru.ex for an example implementation of the Gated Recurrent Unit architecture.
  The functions `mult`, `mult_const`, `add`, `add_const`, `custom_net_layer`, `sigmoid`, and `tanh`, all take input(s), and create a resulting output vector. A random uid will be generated and returned, for use as reference. A custom uid, for keeping track of neural network vectors (other than :input and :output) beyond the construction phase of the neural network, can be provided as the final argument.
  """

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

  @doc "Multiplies together the corresponding components from each of the input vectors."
  def mult(inputs, output \\ uid) do
    link(inputs, output)
    add_operation(output, {:mult, inputs})
  end

  @doc "Multiplies each component of the single input vector by the provided constant."
  def mult_const(input, const, output \\ uid) do #only takes 1 input!
    link([input], output)
    add_operation(output, {{:mult_const, const}, [input]})
  end

  @doc "Adds together the corresponding components from each of the input vectors."
  def add(inputs, output \\ uid) do
    link(inputs, output)
    add_operation(output, {:add, inputs})
  end

  @doc "Adds the provided constant to each component of the single input vector."
  def add_const(input, const, output \\ uid) do #only takes 1 input!
    link([input], output)
    add_operation(output, {{:add_const, const}, [input]})
  end

  def pointwise_tanh(input, output \\ uid) do
    link([input], output)
    add_operation(output, {:pointwise_tanh, [input]})
  end

  @doc "Network layers are train-able connections that take into account 1 or more input vectors in creating the values for an output vector. For a sigmoid network layer, vector components will have values from 0 to 1. For a tanh network layer, vector components will have values from -1 to 1. `custom_net_layer` allows for custom network layers using a provided `activation_function` (instead of tanh or a sigmoid). A function for caluclating the derivative at any value x must also be supplied as `activation_function_prime`."
  def custom_net_layer(activation_function, activation_function_prime, inputs, output \\ uid) do
    link(inputs, output)
    add_net_layer(output, {{:net_layer, activation_function, activation_function_prime}, inputs})
  end

  @doc "A sigmoid network layer. The output vector's components will have values between 0 and 1."
  def sigmoid(inputs, output \\ uid) do
    custom_net_layer(&ActivationFunctions.sigmoid/1, &ActivationFunctions.sigmoid_prime/1, inputs, output)
  end

  @doc "A tanh network layer. The output vector's components will have values between -1 and 1."
  def tanh(inputs, output \\ uid) do
    custom_net_layer(&ActivationFunctions.tanh/1, &ActivationFunctions.tanh_prime/1, inputs, output)
  end

  @doc "Generates a uid (using `:erlang.unique_integer([:monotonic])`)."
  def uid, do: :erlang.unique_integer([:monotonic])

  @doc "Use this to reference the input vector."
  def input, do: :input
  @doc "Use this to reference the output vector."
  def output, do: :output
  @doc "Use this to reference a vector from a previous time frame."
  def previous(name), do: {:previous, name}
end
