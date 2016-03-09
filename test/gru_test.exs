defmodule GRUTest do
  use ExUnit.Case

  setup do
    {:ok, [gru: GRU.new(%{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d]})]}
  end

  test "GRU breakdown", context do
    IO.puts "\ndeps: " <> inspect(context.gru.deps)
    IO.puts "\naffects: " <> inspect(context.gru.affects)
    IO.puts "\noperations: " <> inspect(context.gru.operations)
    IO.puts "\nnet_layers: " <> inspect(context.gru.net_layers)
    IO.puts "\nvec_defs: " <> inspect(context.gru.vec_defs)
    IO.puts "\nweight_map: " <> inspect(context.gru.weight_map)
  end
end
