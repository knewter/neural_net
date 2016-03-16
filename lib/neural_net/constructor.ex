defmodule NeuralNet.Constructor do
  @moduledoc "Contains the nuts and bolts for constructing the network in the process dictionary. These are use by the functions in NeuralNetHelpers."

  defp key, do: :network_defs
  def put_neural_net(net), do: Process.put(key, net)
  def get_neural_net(), do: Process.get(key, %NeuralNet{})
  def update!(fun), do: put_neural_net(fun.(get_neural_net))
  def update!(key, fun), do: update!(fn net -> Map.update!(net, key, fun) end)

  def add_operation(id, data={_, inputs}) do
    add_vec_grouping([id | inputs])
    add_component(:operations, id, data)
  end
  def add_net_layer(id, data={_, inputs}) do
    Enum.each [id | inputs], fn vec -> add_vec_grouping([vec]) end
    add_component(:net_layers, id, data)
  end
  def add_component(type, id, data) do
    update! type, fn map ->
      Map.put(map, id, data)
    end
    id
  end

  def add_vec_grouping(new_group_list) do
    new_group = Enum.reduce new_group_list, MapSet.new, fn vec, set -> MapSet.put(set, NeuralNet.deconstruct(vec)) end
    {matching_groups, remaining_groups} = Enum.partition get_neural_net().vec_groupings, fn group ->
      !MapSet.disjoint?(group, new_group)
    end
    super_group = Enum.reduce [new_group | matching_groups], MapSet.new, fn group, super_group ->
      MapSet.union(super_group, group)
    end
    update! :vec_groupings, fn _ ->
      [super_group | remaining_groups]
    end
  end

  def link(inputs, output) do
    Enum.each(inputs, fn input ->
      case input do
        {:previous, vec} ->
          append_root(input)
          append_affect(vec, {:next, output})
        :input ->
          append_root(input)
          append_affect(input, output)
        _ ->
          append_affect(input, output)
      end
    end)
  end
  def append_affect(id, new_affect) do
    update! :affects, fn affects ->
      its_affects = Map.get(affects, id, [])
      Map.put(affects, id, [new_affect | its_affects])
    end
  end
  def append_root(root) do
    update! :roots, fn roots ->
      MapSet.put(roots, root)
    end
  end

  def confirm_groupings_defined(net) do
    vec_defs = net.vec_defs
    Enum.each net.vec_groupings, fn group ->
      if !Map.has_key?(vec_defs, Enum.at(group, 0)) do
        raise "The following group of vectors lacks a definition for its components: #{inspect(group)}"
      end
    end
    net
  end

  @doc "Returns the network with the randomly generated weight map."
  def gen_random_weight_map(net, weight_gen_fun \\ &gen_random_weight/0) do
    Map.put(net, :weight_map,
      Enum.reduce(net.net_layers, %{}, fn {output, {{:net_layer, _, _}, inputs}}, weight_map -> #net layers are named by their output
        Map.put(weight_map, output,
          Enum.reduce(NeuralNet.Constructor.get_weight_ids(net, output, inputs), %{}, fn id, map ->
            Map.put(map, id, weight_gen_fun.())
          end)
        )
      end)
    )
  end

  def gen_random_weight do
    0.2 * (:rand.uniform - 0.5)
  end

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
end
