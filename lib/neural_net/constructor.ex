defmodule NeuralNet.Constructor do
  @moduledoc "Contains the nuts and bolts for constructing the network in the process dictionary. These are use by the functions in NeuralNetHelpers."

  defp key, do: :network_defs
  def put_neural_net(data), do: Process.put(key, data)
  def get_neural_net(), do: Process.get(key, %NeuralNet{})
  def update!(fun), do: put_neural_net(fun.(get_neural_net))
  def update!(key, fun), do: update!(fn net -> Map.update!(net, key, fun) end)
  def add_operation(id, data), do: add_component(:operations, id, data)
  def add_net_layer(id, data), do: add_component(:net_layers, id, data)
  def add_component(type, id, data) do
    update! type, fn map ->
      Map.put(map, id, data)
    end
  end
  def link(inputs, output) do
    append_deps(output, inputs)
    Enum.each(inputs, fn input -> append_affect(input, output) end)
  end
  def append_deps(id, new_deps) do
    update! :deps, fn deps ->
      its_deps = Map.get(deps, id, [])
      Map.put(deps, id, new_deps ++ its_deps)
    end
  end
  def append_affect(id, new_affect) do
    update! :affects, fn affects ->
      its_affects = Map.get(affects, id, [])
      Map.put(affects, id, [new_affect | its_affects])
    end
  end

  @doc "Returns the network with the randomly generated weight map."
  def gen_random_weight_map(net, weight_gen_fun \\ &gen_random_weight/0) do
    Map.update! net, :weight_map, fn weight_map ->
      Enum.reduce(net.net_layers, %{}, fn {output, {{_, _}, inputs}}, weight_map -> #net layers are named by their output
        Map.put(weight_map, output,
          Enum.reduce(
            Enum.flat_map(inputs, fn input ->
              Enum.map(NeuralNet.get_vec_def(net, input), fn id -> {input, id} end)
            end),
          %{}, fn id, map ->
            Map.put(map, id, weight_gen_fun.())
          end)
        )
      end)
    end
  end

  def gen_random_weight do
    0.1 * (2*:rand.uniform - 1)
  end
end
