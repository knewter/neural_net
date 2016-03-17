defmodule RNN do
  @moduledoc "The classic simple recurrent neural network."
  use NeuralNet

  def template(inp, out \\ uid) do
    tanh [inp, previous(out)], out
  end

  defp define(args) do
    template(input, output)

    def_vec(input, args.input_ids)
    def_vec(output, args.output_ids)
  end
end
