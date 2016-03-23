defmodule NeuralNet do
  @moduledoc """
  This module allows you to define and train neural networks with complex architectures. See lib/gru.ex for an example implementation of the Gated Recurrent Unit architecture.

  A vector-across-time is a list of input vectors across time. Index 1 is the beginning of time. Index 0 is the moment just before the start of time. This is where initial values for vectors that are used recurrently will be stored.
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

  @doc "Evaluates a network. The input should be a vector-across-time. The function returns a {output_values, acc} tuple. The acc holds all the data for the networks state across its time frames of execution. If you want to continue evaluation with more inputs, run the eval function again passing in the acc as the given_acc."
  def eval(net, input, given_acc \\ [%{}]) do
    {output_data, acc} = Backprop.get_feedforward(net, input, given_acc, false)
    {output_data.values, acc}
  end

  @doc "train uses backpropogation to train any neural_net. `training_data` should be a list of {inputs, expected_output} tuples, where `inputs` is a vector-across-time. `learn_val` should be a positive constant which controls the effect of each batch. Higher values can cause faster learning, but also may have trouble finding a minimum error. `batch_size` specifies the number of training pairs to be run in parallel. The results of the batch are averaged together. Small batch sizes of 1-3 tend to work best. The `training_complete?` should return true when training should be stopped. The time (in seconds) between each call to `training_complete?` is specified by the argument `completion_checking_interval`."
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
    end) |> Enum.map(&Task.await/1)

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
      iterations: iterations
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

  def deconstruct({:next, vec}), do: vec
  def deconstruct({:previous, vec}), do: vec
  def deconstruct(vec), do: vec

  defp monotonic_time do
    :erlang.monotonic_time(:milli_seconds) / 1000
  end
end
