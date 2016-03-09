defmodule GRUTest do
  use ExUnit.Case

  setup do
    {:ok, [gru: GRU.new(%{input_size: 5, output_size: 10})]}
  end

  test "size_defs", context do
    assert context.gru.size_defs == %{forgetting_gate: 10, gated_prev_out: 5,
     gated_update: 10, input: 5, input_gate: 5, output: 10, purged_output: 10,
     update_candidate: 10, update_gate: 10}
  end

  test "components", context do
    IO.puts "\ndeps: " <> inspect(context.gru.deps)
    IO.puts "\naffects: " <> inspect(context.gru.affects)
    IO.puts "\noperations: " <> inspect(context.gru.operations)
    IO.puts "\nnet_layers: " <> inspect(context.gru.net_layers)
  end
end
