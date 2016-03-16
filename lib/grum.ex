defmodule GRUM do
  @moduledoc "GRUM is an experimental modification to GRUs. It has a dedicated memory that can be set to any arbitrary size using `memory_size`."
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
