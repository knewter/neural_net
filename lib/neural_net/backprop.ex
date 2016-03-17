defmodule NeuralNet.Backprop do
  @moduledoc "This module provides the code that generates the feedforward and backprop data used for training. The feedforward can also be used for normal network evaluation in which only the minimum computation required for evaluation is executed."

  alias NeuralNet.ActivationFunctions

  def get_feedforward(net, input, given_acc \\ [%{}], calc_partial_derivs \\ true) do
    acc = given_acc ++ Enum.map(input, fn input_frame ->
      feedforward = %{values: input_frame}
      %{input: (if calc_partial_derivs, do: Map.put(feedforward, :partial_derivs, %{}), else: feedforward)}
    end)
    acc = if calc_partial_derivs do
      #make sure outputs from every time frame are calculated
      Enum.reduce 1..length(input), acc, fn time, acc ->
        {_, acc} = get_feedforward(net, calc_partial_derivs, acc, time, :output)
        acc
      end
    else
      acc
    end
    get_feedforward(net, calc_partial_derivs, acc, length(input), :output)
  end
  def get_feedforward(net, calc_partial_derivs, acc, time = 0, vec) do
    feedforward = %{values:
      Enum.reduce(Map.fetch!(net.vec_defs, vec), %{}, fn id, map ->
        Map.put(map, id, 0)
      end)
    }
    feedforward = if calc_partial_derivs, do: Map.put(feedforward, :partial_derivs, feedforward.values), else: feedforward
    {feedforward, update_acc(acc, time, vec, feedforward)}
  end
  def get_feedforward(net, calc_partial_derivs, acc, time, {:previous, vec}) do
    get_feedforward(net, calc_partial_derivs, acc, time - 1, vec)
  end
  def get_feedforward(net, calc_partial_derivs, acc, time, vec) do
    time_frame = Enum.at(acc, time)
    if Map.has_key? time_frame, vec do
      {Map.fetch!(time_frame, vec), acc}
    else
      {vec_specs, inputs} = Map.fetch!(Map.merge(net.operations, net.net_layers), vec)
      {input_map, acc} = Enum.reduce inputs, {%{}, acc}, fn input, {input_map, acc} ->
        {feedforward, acc} = get_feedforward(net, calc_partial_derivs, acc, time, input)
        {Map.put(input_map, input, feedforward.values), acc}
      end
      feedforward = %{values: %{}}
      feedforward = if calc_partial_derivs, do: Map.put(feedforward, :partial_derivs, %{}), else: feedforward
      feedforward = Enum.reduce NeuralNet.get_vec_def(net, vec), feedforward, fn output_component, feedforward ->
        {values_fun, partial_derivs_fun} = case vec_specs do
          :mult ->
            product = Enum.reduce(input_map, 1, fn {_input_name, input_vals}, product ->
              product * Map.fetch!(input_vals, output_component)
            end)
            {
              fn -> product end,
              fn ->
                Enum.reduce(input_map, %{}, fn {input_name, input_vals}, partial_derivs_map ->
                  input_part_val = Map.fetch!(input_vals, output_component)
                  input_name = case input_name do
                    {:previous, input_name} -> input_name
                    input_name -> input_name
                  end
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
          :pointwise_tanh ->
            {
              fn ->
                [input] = inputs
                x = Map.fetch!(Map.fetch!(input_map, input), output_component)
                ActivationFunctions.tanh(x)
              end,
              fn ->
                [input] = inputs
                x = Map.fetch!(Map.fetch!(input_map, input), output_component)
                ActivationFunctions.tanh_prime(x)
              end
            }
          {:tanh_given_weights, weight_vec, actual_inputs} ->
            weight_vals = Map.fetch!(input_map, weight_vec)
            sum = Enum.reduce(actual_inputs, 0, fn input_name, sum ->
              Enum.reduce(Map.fetch!(input_map, input_name), sum, fn {input_component, component_val}, sum ->
                weight = Map.fetch!(weight_vals, {{input_name, input_component}, output_component})
                sum + component_val * weight
              end)
            end)
            {
              fn -> ActivationFunctions.tanh(sum) end,
              fn ->
                partial_derivs_map = Enum.reduce(actual_inputs, %{}, fn input_name, partial_derivs_map ->
                  Enum.reduce(Map.fetch!(input_map, input_name), partial_derivs_map, fn {input_component, _component_val}, partial_derivs_map ->
                    weight_val = Map.fetch!(weight_vals, {{input_name, input_component}, output_component})
                    Map.put(partial_derivs_map, {NeuralNet.deconstruct(input_name), input_component}, weight_val * ActivationFunctions.tanh_prime(sum))
                  end)
                end)
                Enum.reduce(Map.fetch!(input_map, weight_vec), partial_derivs_map, fn
                 {weight_component={{input_name, input_component}, ^output_component}, _weight_val}, partial_derivs_map ->
                  input_val = Map.fetch!(Map.fetch!(input_map, input_name), input_component)
                  Map.put(partial_derivs_map, {NeuralNet.deconstruct(weight_vec), weight_component}, input_val * ActivationFunctions.tanh_prime(sum))
                  {{{_, _}, _}, _}, partial_derivs_map ->
                    partial_derivs_map
                end)
              end
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

  def get_backprop(net, input, exp_output) do
    {output, acc} = get_feedforward(net, input)
    {backprop_error, error_sum} = Enum.reduce(output.values, {%{}, 0}, fn {component, value}, {acc, error_sum} ->
      difference = value - Map.fetch!(exp_output, component)
      {Map.put(acc, component, difference), error_sum + (0.5 * :math.pow(difference, 2))}
    end)
    acc = update_acc(acc, -1, :output, apply_backprop(output, backprop_error))
    acc = Enum.reduce(net.roots, acc, fn root, acc ->
      {_, acc} = get_backprop(net, acc, 1, root)
      acc
    end)
    {error_sum, acc}
  end
  def get_backprop(net, acc, time, {:previous, vec}), do: get_backprop(net, acc, time - 1, vec)
  def get_backprop(net, acc, time, {:next, vec}), do: get_backprop(net, acc, time + 1, vec)
  def get_backprop(net, acc, time, vec) when time >= length(acc) or time == 0 do
    {backprops, values} = Enum.reduce(Map.fetch!(net.vec_defs, vec), {%{}, %{}}, fn component, {backprops, values} ->
      {
        Map.put(backprops, component, 0),
        Map.put(values, component, 0)
      }
    end)
    vec_acc_data = %{backprops: backprops, values: values}
    {vec_acc_data.backprops, update_acc(acc, time, vec, vec_acc_data)}
  end
  def get_backprop(net, acc, time, vec) do
    vec_acc_data = Map.fetch!(Enum.at(acc, time), vec)
    if Map.has_key? vec_acc_data, :backprops do
      {vec_acc_data.backprops, acc}
    else
      affects = Map.get(net.affects, vec)
      {backprop_error, acc} = Enum.reduce(Map.fetch!(net.vec_defs, vec), {%{}, acc}, fn component, {backprop_error, acc} ->
        {sum, acc} = if affects != nil do
          Enum.reduce(affects, {0, acc}, fn affected, {sum, acc} ->
            dec_affected = NeuralNet.deconstruct(affected)
            {affected_specs, _} = Map.fetch!(Map.merge(net.operations, net.net_layers), dec_affected)
            {affected_backprops, acc} = get_backprop(net, acc, time, affected)
            addon = case affected_specs do
              {:net_layer, _, _} ->
                Enum.reduce(Map.fetch!(net.vec_defs, dec_affected), 0, fn affected_component, sum ->
                  con_vec = case affected do
                    {:next, aff} -> {:previous, vec}
                    _ -> vec
                  end
                  sum + Map.fetch!(affected_backprops, affected_component) * Map.fetch!(Map.fetch!(net.weight_map, dec_affected), {{con_vec, component}, affected_component})
                end)
              {:tanh_given_weights, weight_vec, actual_inputs} ->
                if weight_vec == vec do
                  {_source, affected_component} = component
                  Map.fetch!(Map.fetch!(affected_backprops, affected_component), {vec, component})
                else
                  Enum.reduce(Map.fetch!(net.vec_defs, dec_affected), 0, fn affected_component, sum ->
                    sum + Map.fetch!(Map.fetch!(affected_backprops, affected_component), {vec, component})
                  end)
                end
              _ ->
                backprop_component = Map.fetch!(affected_backprops, component)
                if is_number(backprop_component) do
                  backprop_component
                else
                  Map.fetch!(backprop_component, vec)
                end
            end
            {sum + addon, acc}
          end)
        else
          {0, acc}
        end
        {Map.put(backprop_error, component, sum), acc}
      end)
      vec_acc_data = apply_backprop(vec_acc_data, backprop_error)
      {vec_acc_data.backprops, update_acc(acc, time, vec, vec_acc_data)}
    end
  end

  @doc "Multiplies in the transmitted backprop with the partial derivatives of this node."
  def apply_backprop(vec_acc_data, backprop_error) do
    Map.put(vec_acc_data, :backprops,
      Enum.reduce(vec_acc_data.partial_derivs, %{}, fn {component, partial_deriv}, backprops ->
        backprop_component = if is_number(partial_deriv) do
          partial_deriv * Map.fetch!(backprop_error, component)
        else
          Enum.reduce(partial_deriv, %{}, fn {source, sub_partial_deriv}, backprop_component_map ->
            Map.put(backprop_component_map, source, sub_partial_deriv * Map.fetch!(backprop_error, component))
          end)
        end
        Map.put(backprops, component, backprop_component)
      end)
    )
  end

  def update_acc(acc, time, vec, value) do
    List.update_at acc, time, fn time_frame ->
      Map.put(time_frame, vec, value)
    end
  end

  def fetch!(acc, time, {:previous, vec}), do: fetch!(acc, time - 1, vec)
  def fetch!(acc, time, vec) when time >= 0 do
    Map.fetch!(Enum.at(acc, time), vec)
  end
end
