defmodule GRUITest do
  use ExUnit.Case

  setup do
    {:ok, [grui: gen_grui]}
  end

  def gen_grui, do: GRUI.new(%{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d], memory_size: 10})

  test "GRUI training", context do
    IO.puts "Training GRUI"
    GRUTest.test_training(context.grui, gen_grui, 100, 10, 5.5, 2, 0.2)
  end
end
