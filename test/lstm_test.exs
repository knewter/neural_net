defmodule LSTMTest do
  use ExUnit.Case

  setup do
    {:ok, [lstm: gen_lstm]}
  end

  def gen_lstm, do: LSTM.new(%{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d]})

  test "LSTM training", context do
    IO.puts "Training LSTM"
    NeuralNet.Tester.test_training(context.lstm, gen_lstm, 100, 10, 1.5, 2, 0.2)
  end
end
