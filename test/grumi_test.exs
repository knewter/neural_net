defmodule GRUMITest do
  use ExUnit.Case

  setup do
    {:ok, [grumi: gen_grumi]}
  end

  def gen_grumi, do: GRUMI.new(%{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d], memory_size: 10, sub_memory_size: 10})

  test "GRUMI training", context do
    IO.puts "Training GRUMI"
    NeuralNet.Tester.test_training(context.grumi, gen_grumi, 100, 10, 1.5, 2, 0.2)
  end
end
