defmodule NeuralNet do
  @moduledoc """
  This module allows you to define and train neural networks with complex architectures. See lib/gru.ex for an example implementation of the Gated Recurrent Unit architecture.

  A vector-across-time is a list of input vectors across time. Index 1 is the beginning of time. Index 0 is the moment just before the start of time. This is where initial values for vectors that are used recurrently will be stored.
  """

  defstruct vec_defs: %{}, deps: %{}, affects: %{}, operations: %{}, net_layers: %{}, weight_map: %{}

  defmacro __using__(_) do
    quote location: :keep do
      import NeuralNet.Helpers

      def new(args) do
        define(args)
        NeuralNet.Constructor.get_neural_net #retrieves constructed network from the Process dictionary
        |> NeuralNet.Constructor.gen_random_weight_map()
      end
    end
  end

  def get_vec_def(net, {:previous, vec}), do: get_vec_def(net, vec)
  def get_vec_def(net, vec), do: Map.fetch!(net.vec_defs, vec)

  def get_weight_ids(net, output) do
    {{:net_layer, _, _}, inputs} = Map.fetch!(net.net_layers, output)
    get_weight_ids(net, output, inputs)
  end
  def get_weight_ids(net, output, inputs) do
    Enum.flat_map(inputs, fn input ->
      Enum.flat_map(NeuralNet.get_vec_def(net, input), fn input_component ->
        Enum.map(NeuralNet.get_vec_def(net, output), fn output_component ->
          {{input, input_component}, output_component}
        end)
      end)
    end)
  end

  @doc "The input should be a vector-across-time."
  def eval(net, input) do
    get_feedforward(net, input, false)
  end

  def get_feedforward(net, input, given_acc \\ [%{}], calc_partial_derivs \\ true) do
    acc = %{time_frames: given_acc ++ Enum.map(input, fn input_frame ->
      feedforward = %{values: input_frame}
      %{input: (if calc_partial_derivs, do: Map.put(feedforward, :partial_derivs, %{}), else: feedforward)}
    end)}
    get_feedforward(net, calc_partial_derivs, acc, length(input), :output)
  end
  def get_feedforward(net, _calc_partial_derivs, acc, time = 0, vec) do
    feedforward = %{values:
      Enum.reduce(Map.fetch!(net.vec_defs, vec), %{}, fn id, map ->
        Map.put(map, id, 0)
      end)
    }
    {feedforward, update_acc(acc, time, vec, feedforward)}
  end
  def get_feedforward(net, calc_partial_derivs, acc, time, {:previous, vec}) do
    get_feedforward(net, calc_partial_derivs, acc, time - 1, vec)
  end
  def get_feedforward(net, calc_partial_derivs, acc, time, vec) do
    time_frames = Enum.at(acc.time_frames, time)
    if Map.has_key? time_frames, vec do
      {Map.fetch!(time_frames, vec), acc}
    else
      {vec_specs, inputs} = Map.fetch!(Map.merge(net.operations, net.net_layers), vec)
      {input_map, acc} = Enum.reduce inputs, {%{}, acc}, fn input, {input_map, acc} ->
        {feedforward, acc} = get_feedforward(net, calc_partial_derivs, acc, time, input)
        IO.puts "\nEvaluating #{inspect input}, got #{inspect feedforward}"
        {Map.put(input_map, input, feedforward.values), acc}
      end
      feedforward = %{values: %{}}
      feedforward = if calc_partial_derivs, do: Map.put(feedforward, :partial_derivs, %{}), else: feedforward
      feedforward = Enum.reduce NeuralNet.get_vec_def(net, vec), feedforward, fn output_component, feedforward ->
        {values_fun, partial_derivs_fun} = case vec_specs do
          :mult ->
            product = Enum.reduce(input_map, 1, fn {_input_name, input_vals}, product ->
              product * Dict.fetch!(input_vals, output_component)
            end)
            {
              fn -> product end,
              fn ->
                Enum.reduce(input_map, %{}, fn {input_name, input_vals}, partial_derivs_map ->
                  input_part_val = Map.fetch!(input_vals, output_component)
                   Map.put(partial_derivs_map, input_name,
                     (if input_part_val != 0, do: product / input_part_val, else: 0)
                   )
                end)
              end
            }
          {:mult_const, const} ->
            {
              fn ->
                [input] = inputs
                const * Map.fetch!(Map.fetch!(input_map, input), output_component)
              end,
              fn -> const end
            }
          :add ->
            {
              fn ->
                Enum.reduce(input_map, 0, fn {_input_name, input_vals}, sum ->
                  sum + Dict.fetch!(input_vals, output_component)
                end)
              end,
              fn -> 1 end
            }
          {:add_const, const} ->
            {
              fn ->
                [input] = inputs
                const + Map.fetch!(Map.fetch!(input_map, input), output_component)
              end,
              fn -> 1 end
            }
          {:net_layer, activation_function, activation_function_prime} ->
            sum = Enum.reduce input_map, 0, fn {input_name, input_vals}, sum ->
              Enum.reduce(input_vals, sum, fn {input_component, val}, sum ->
                sum + val * Map.fetch!(
                  Map.fetch!(net.weight_map, vec),
                  {{input_name, input_component}, output_component}
                )
              end)
            end
            { fn -> activation_function.(sum) end, fn -> activation_function_prime.(sum) end }
        end
        feedforward = Map.update!(feedforward, :values, fn values ->
          Map.put(values, output_component, values_fun.())
        end)
        if calc_partial_derivs do
          Map.update!(feedforward, :partial_derivs, fn partial_derivs ->
            Map.put(partial_derivs, output_component, partial_derivs_fun.())
          end)
        else
          feedforward
        end
      end
      {feedforward, update_acc(acc, time, vec, feedforward)}
    end
  end
  def update_acc(acc, time, vec, value) do
    Map.update! acc, :time_frames, fn time_frames ->
      List.update_at(time_frames, time, fn time_frame ->
        Map.put(time_frame, vec, value)
      end)
    end
  end
end
