defmodule GRUTest do
  use ExUnit.Case

  setup do
    {:ok, [gru: gen_gru]}
  end

  def gen_gru, do: GRU.new(%{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d]})

  # test "GRU breakdown", context do
  #   IO.puts "\naffects: " <> inspect(context.gru.affects)
  #   IO.puts "\noperations: " <> inspect(context.gru.operations)
  #   IO.puts "\nnet_layers: " <> inspect(context.gru.net_layers)
  #   IO.puts "\nvec_defs: " <> inspect(context.gru.vec_defs)
  #   IO.puts "\nweight_map: " <> inspect(context.gru.weight_map)
  # end

  # test "GRU eval", context do
  #   input = [
  #     %{x: 1.0, y: 1.0},
  #     %{x: 1.1, y: 0.9},
  #     %{x: 1.2, y: 0.8}
  #   ]
  #
  #   IO.puts "\n\n" <> inspect(NeuralNet.eval(context.gru, input))
  # end

  # test "GRU backprop", context do
  #   input = [
  #     %{x: 1.0, y: 1.0},
  #     %{x: 1.1, y: 0.9},
  #     %{x: 1.2, y: 0.8}
  #   ]
  #   exp_output = %{a: 0.1, b: 0.2, c: 0.3, d: 0.4}
  #
  #   IO.puts "\n\n" <> inspect(NeuralNet.get_backprop(context.gru, input, exp_output))
  # end

  test "GRU training", context do
    IO.puts "Training GRU"
    NeuralNet.Tester.test_training(context.gru, gen_gru, 100, 10, 0.5, 2, 0.2)
  end
end
