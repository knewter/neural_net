defmodule GRUM do
  @moduledoc "GRUM is home-made to GRU. It has a dedicated memory that can be set to any arbitrary size using `memory_size`."

  @doc "Takes an argument map with the keys :input_ids, :output_ids, and :memory_size. The values for :input_ids and :output_ids should be a list of component/id names for the input and output vectors (respectively). The value for :memory_size should be an integer for the number of components in the memory vector."
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
