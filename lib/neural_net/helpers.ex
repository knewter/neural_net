defmodule NeuralNet.Helpers do
  import NeuralNet.Constructor
  alias NeuralNet.ActivationFunctions

  @moduledoc """
  These helper functions provide a DSL for specifying network architectures. Look at lib/gru.ex for an example implementation of the Gated Recurrent Unit architecture.
  The functions `mult`, `mult_const`, `add`, `add_const`, `custom_net_layer`, `sigmoid`, `tanh`, `pointwise_tanh`, `tanh_given_weights`, and `custom_net_layer` all take input(s), and create a resulting output vector. A random uid will be generated and returned, for use as reference. A custom uid for important neural network vectors (like for the network's input and output), can be provided as the final argument for any of these functions.

  `tanh` and `sigmoid` are network layers. Network layers are train-able connections (that can later be developed through training algorithms like backpropogation) that take into account 1 or more input vectors in creating the values for an output vector. For a sigmoid network layer, output vector components will have values from 0 to 1. For a tanh network layer, output vector components will have values from -1 to 1.

  `input()` returns a reference to the neural network's input, and `output()` to its output.
  `uid()` returns a unique identifier (which is used for vector names if a name is not supplied).
  `previous(vector) returns a reference to the given vector from the previous time frame. This is how recurrent neural networks are built.`

  `mult`, `mult_const`, `add`, `add_const`, and `pointwise_tanh` are all considered pointwise operations.

  All vectors must have component definitions, that is, a list of their component_names. Input and output vectors on either side of a pointwise operation must all have the same component definitions. Because of this, vectors are automatically put into groups. These groups of vectors must all share the same definition, so the user need only define 1 vector per group, and the rest of the vectors will automatically obtain their definitions. If creating a new architecture, try running it once done, and an error message will print out the vector groupings that require definitions. Use `def_vec` and `def_vec_by_size` to provide vector definitions.
  Because of this feature, a lot of network architectures only need the network's ultimate input and output to be define, and the rest of the vectors will recieve their definitions automatically.
  """

  @doc "Specifies a vector component definition given the `vec_name`, and `ids`, which is a list of component names."
  def def_vec(vec_name, ids) do
    net = NeuralNet.Constructor.get_neural_net()
    vec_names = Enum.find Map.get(net.construction_data, :vec_groupings, []), fn group -> MapSet.member?(group, vec_name) end
    if vec_names == nil, do: raise "Vector #{inspect(vec_name)} was never used during construction."
    update! :vec_defs, fn vec_defs ->
      Enum.reduce vec_names, vec_defs, fn vec_name, vec_defs ->
        if Map.has_key?(vec_defs, vec_name) do
          conflicting_def = Map.get(vec_defs, vec_name)
          if conflicting_def != ids do
            raise "Vector #{inspect(vec_name)} has conflicting definition of #{inspect(ids)} and #{inspect(conflicting_def)}.\nThe list of vector groupings that must share the same definitions are as follows: #{inspect(net.vec_groupings)}"
          else
            raise "Vector #{inspect(vec_name)} was already defined through association, additional specification is not needed."
          end
        end
        Map.put(vec_defs, vec_name, ids)
      end
    end
  end

  @doc "Specifies a vector component definition given the `vec_name`, and a `size`. Based on the size, a list of component names will be generated of size `size` using UIDs."
  def def_vec_by_size(vec_name, size) do
    def_vec(vec_name, Enum.map(1..size, fn _ -> uid() end))
  end

  @doc "Multiplies together the corresponding components from each of the input vectors."
  def mult(inputs, output \\ uid) do
    link(inputs, output)
    add_operation(output, {:mult, inputs})
  end

  @doc "Multiplies each component of the single input vector by the provided constant. Note, this only takes 1 input vector."
  def mult_const(input, const, output \\ uid) do
    confirm_not_list("mult_const", input)
    link([input], output)
    add_operation(output, {{:mult_const, const}, [input]})
  end

  @doc "Adds together the corresponding components from each of the input vectors."
  def add(inputs, output \\ uid) do
    link(inputs, output)
    add_operation(output, {:add, inputs})
  end

  @doc "Adds the provided constant to each component of the single input vector. Note, this only takes 1 input vector."
  def add_const(input, const, output \\ uid) do
    confirm_not_list("add_const", input)
    link([input], output)
    add_operation(output, {{:add_const, const}, [input]})
  end

  @doc "Applies the function `tanh` to all components of the vector, compressing values to be between -1 and 1. Note, this only takes 1 input vector."
  def pointwise_tanh(input, output \\ uid) do
    confirm_not_list("pointwise_tanh", input)
    link([input], output)
    add_operation(output, {:pointwise_tanh, [input]})
  end

  @doc "This allows for custom network layers using a provided `activation_function` (instead of tanh or a sigmoid). A function for caluclating the derivative at any value x must also be supplied as `activation_function_prime`."
  def custom_net_layer(activation_function, activation_function_prime, inputs, layers \\ 1, output \\ uid) do
    if is_number(layers) && layers < 1, do: raise "Error, layers cannot be an integer less than 1, got #{inspect(layers)}"
    if layers == 1 or layers == [] do
      link(inputs, output)
      add_net_layer(output, {{:net_layer, activation_function, activation_function_prime}, inputs})
    else
      id = custom_net_layer(activation_function, activation_function_prime, inputs)
      if is_number(layers) do
        add_vec_grouping([id, output])
        custom_net_layer(activation_function, activation_function_prime, [id], layers - 1, output)
      else #is list
        def_vec_by_size(id, hd(layers))
        custom_net_layer(activation_function, activation_function_prime, [id], tl(layers), output)
      end
    end
  end

  @doc "A sigmoid network layer. The output vector's components will have values between 0 and 1."
  def sigmoid(inputs, num_layers \\ 1, output \\ uid) do
    custom_net_layer(&ActivationFunctions.sigmoid/1, &ActivationFunctions.sigmoid_prime/1, inputs, num_layers, output)
  end

  @doc "A tanh network layer. The output vector's components will have values between -1 and 1."
  def tanh(inputs, num_layers \\ 1, output \\ uid) do
    custom_net_layer(&ActivationFunctions.tanh/1, &ActivationFunctions.tanh_prime/1, inputs, num_layers, output)
  end

  @doc "This provides experimental functionality. This is sort of like a network layer, only values for weights are supplied as a vector, instead of developed through training algorithms."
  def tanh_given_weights(inputs, weight_vec, output \\ uid) do
    link([weight_vec | inputs], output)
    add_special(output, {{:tanh_given_weights, weight_vec, inputs}, [weight_vec | inputs]})
  end

  @doc "Generates a uid (using `:erlang.unique_integer([:monotonic])`)."
  def uid, do: :erlang.unique_integer([:monotonic])

  @doc "Use this to reference the input vector."
  def input, do: :input
  @doc "Use this to reference the output vector."
  def output, do: :output
  @doc "Use this to reference a vector from a previous time frame."
  def previous(name), do: {:previous, name}

  defp confirm_not_list(fun_name, input) do
    if is_list(input) do
      raise "#{fun_name} only takes 1 input, not a list of inputs. Was given #{inspect(input)}"
    end
  end
end
