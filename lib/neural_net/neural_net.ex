defmodule NeuralNet do
  @moduledoc """
  This module allows you to define and train neural networks with complex architectures. See lib/gru.ex for an example implementation of the Gated Recurrent Unit architecture. For making a new architecture, use the template at lib/.template.ex. Just copy the file and change the module name to get started.

  For a great post on LSTMs, GRUs, and other recurrent neural network architectures, see here: http://colah.github.io/posts/2015-08-Understanding-LSTMs/.

  This module of course allows for the creation of non recurrent neural networks as well. An architecture that makes no reference to vectors from previous time frames is not recurrent.

  Networks are evaluated and trained using "vectors across time". A vector-across-time is a list of time_frames, where each element is a vector, which is a map of {component_name, value} key-val-pairs.
  ### Example
      [%{x: 1.0, y: -1.0}, %{x: 0.5, -0.5}, %{x: 0.0, y: 0.0}]
  """

  defstruct vec_defs: %{}, construction_data: %{}, affects: %{}, roots: MapSet.new, operations: %{}, net_layers: %{}, weight_map: %{}

  alias NeuralNet.Backprop

  defmacro __using__(_) do
    quote location: :keep do
      import NeuralNet.Helpers

      def new(args) do
        import NeuralNet.Constructor
        define(args)
        set_special_vec_definitions()
        net = get_neural_net() #retrieves constructed network from the Process dictionary
        |> confirm_groupings_defined()
        |> gen_random_weight_map()
        put_neural_net(%NeuralNet{})
        net
      end
    end
  end

  @doc "Evaluates a network. The input should be a vector-across-time. The function returns a {output_values, time_frames} tuple. `time_frames` holds all the data for the networks state across its time frames of execution. If you want to continue evaluation with more inputs, run the eval function again passing in the previous data as `given_time_frames`."
  def eval(net, input, given_time_frames \\ [%{}]) do
    {output_data, acc} = Backprop.get_feedforward(net, input, false, given_time_frames)
    {output_data.values, acc}
  end

  @doc "`train` uses backpropogation to train any neural_net. `training_data` should be a list of {inputs, expected_output} tuples, where `inputs` is a vector-across-time. `expected_output` can be either a vector representing final expected output, or it can be a list of expected outputs for every time frame. `learn_val` should be a positive constant which controls the effect of each batch. Higher values can cause faster learning, but also may introduce trouble finding a minimum error. `batch_size` specifies the number of training pairs to be run in parallel. At each training iteration, `batch_size` number of training sessions run in parallel, and the results are averaged together. Small batch sizes of 1-3 tend to work best. The `training_complete?(info)` function should take 1 argument of miscellaneous info, and return true when training should be stopped. This function can also be used for debugging or monitoring, and can print out information at regular intervals. The time (in seconds) between each call to `training_complete?(info)` is specified by the argument `completion_checking_interval`."
  def train(net, training_data, learn_val \\ 2, batch_size \\ 1, training_complete? \\ fn info -> info.eval_time > 60 end, completion_checking_interval \\ 1) do
    train(net, training_data, learn_val, batch_size, training_complete?, completion_checking_interval, monotonic_time() - completion_checking_interval, monotonic_time(), 1)
  end
  defp train(net, training_data, learn_val, batch_size, training_complete?, completion_checking_interval, last_check, start_time, iterations) do
    batch = Enum.map(1..batch_size, fn _ ->
      Task.async(fn ->
        {inputs, exp_output} = Enum.random(training_data)
        {error_sum, acc} = Backprop.get_backprop(net, inputs, exp_output)
        train_val_map = Enum.reduce(net.weight_map, %{}, fn {output, weights}, train_val_map ->
          Map.put(train_val_map, output,
            Enum.reduce(weights, %{}, fn {id={{input, input_component}, output_component}, _}, train_vals ->
              len = length(acc) - 1
              Map.put(train_vals, id,
                Enum.reduce(1..len, 0, fn time, train_val ->
                  output_value = Map.fetch!(Backprop.fetch!(acc, time, input).values, input_component)
                  backprop_value = Map.fetch!(Backprop.fetch!(acc, time, output).backprops, output_component)
                  backprop_value = if (is_number(backprop_value)), do: backprop_value, else: Map.fetch!(backprop_value, input)
                  train_val + (-learn_val*output_value*backprop_value / len)
                end)
              )
            end)
          )
        end)
        {error_sum, train_val_map}
      end)
    end) |> Enum.map(fn task -> Task.await(task, 24*60*60*1000) end)

    {avg_error, weight_map} = Enum.reduce batch, {0, net.weight_map}, fn {error_sum, train_val_map}, {avg_error, weight_map} ->
      {
        avg_error + error_sum/batch_size,
        Map.merge(train_val_map, weight_map, fn _k, train_vals, weight_vals ->
          Dict.merge(train_vals, weight_vals, fn _k, train_val, weight_val ->
            weight_val + train_val/batch_size
          end)
        end)
      }
    end
    net = Map.put(net, :weight_map, weight_map)

    time = monotonic_time()
    info = %{
      eval_time: (time - start_time),
      error: avg_error,
      iterations: iterations,
      net: net
    }
    if (time - last_check >= completion_checking_interval) do
      if training_complete?.(info) do
        {net, info}
      else
        train(net, training_data, learn_val, batch_size, training_complete?, completion_checking_interval, time, start_time, iterations + 1)
      end
    else
      train(net, training_data, learn_val, batch_size, training_complete?, completion_checking_interval, last_check, start_time, iterations + 1)
    end
  end

  # def train(net, training_data, batch_size, evaluation_data, minimum_error, time_out, completion_checking_interval \\ 1) do
  #
  # end

  @doc "Retrieves the list of the named components that make up the given vector."
  def get_vec_def(net, vec), do: Map.fetch!(net.vec_defs, deconstruct(vec))

  @doc "Breaks down a vector name into its core name. When given `{:next, vec}` or `{:previous, vec}`, it will just return `vec.` :next and :previous are used in the fashion to keep track of references to vectors from future or past time frames."
  def deconstruct({:next, vec}), do: vec
  def deconstruct({:previous, vec}), do: vec
  def deconstruct(vec), do: vec

  defp monotonic_time do
    :erlang.monotonic_time(:milli_seconds) / 1000
  end

  @doc "Given a vector, returns the component name with the greatest value."
  def get_max_component(vec) do
    {comp, _val} = Enum.max_by(vec, fn {_comp, val} -> val end)
    comp
  end

  @doc "Given a list of component names for a vector, returns a complete vector where each component has value `value` (default is 0)."
  def get_blank_vector(components, value \\ 0) do
    Enum.reduce components, %{}, fn component, vec ->
      Map.put(vec, component, value)
    end
  end
end
