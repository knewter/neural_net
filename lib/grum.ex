defmodule GRUM do
  @moduledoc "GRU (Gated Recurrent Unit) is a variation on LSTM (Long short term :memory). It is, for the most part, equally effective but computationally cheaper."
  use NeuralNet

  def template(inp, out) do
    GRU.template(inp, :memory)
    tanh [:memory, inp], out
  end

  defp define(args) do
    template(input, output)

    def_vec(input, args.input_ids)
    def_vec_by_size(:memory, args.memory_size)
    def_vec(output, args.output_ids)
  end
end
