defmodule GRUM do
  @moduledoc "GRUM is an experimental modification to GRUs. It has a dedicated memory that can be set to any arbitrary size using `memory_size`."
  use NeuralNet

  def template(inp, out \\ uid()) do
    memory = GRU.template(inp)
    tanh [memory, inp], out
    {memory, out}
  end

  defp define(args) do
    {memory, _} = template(input, output)

    def_vec(input, args.input_ids)
    def_vec_by_size(memory, args.memory_size)
    def_vec(output, args.output_ids)
  end
end
