defmodule GRUMTest do
  use ExUnit.Case

  setup do
    {:ok, [grum: gen_grum]}
  end

  def gen_grum, do: GRUM.new(%{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d], memory_size: 10})

  test "GRUM training", context do
    IO.puts "Training GRUM"
    TrainingTester.test_training(context.grum, gen_grum, 100, 10, 1.5, 2, 0.2)
  end
end
