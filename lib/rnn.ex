defmodule RNN do
  @moduledoc "The classic simple recurrent neural network."

  @doc "Takes an argument map with the keys :input_ids and :output_ids. Both values should be a list of component/id names for the input and output vectors (respectively)."
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
