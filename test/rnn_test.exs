defmodule RNNTest do
  use ExUnit.Case

  setup do
    {:ok, [rnn: gen_rnn]}
  end

  def gen_rnn, do: RNN.new(%{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d]})

  test "RNN training", context do
    IO.puts "Training RNN"
    NeuralNet.Tester.test_training(context.rnn, gen_rnn, 100, 4, 1.5, 2, 0.2)
  end
end
