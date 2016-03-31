defmodule GRUMI do
  @moduledoc false
  # @moduledoc "GRUMI is an experimental neural network design in development."

  use NeuralNet

  def template(inp, out \\ uid) do
    grumi_memory = GRU.template(inp)
    {grum_memory, weight_vec} = GRUM.template(inp)
    tanh_given_weights [grumi_memory, inp], weight_vec, out
    {grum_memory, grumi_memory, out}
  end

  defp define(args) do
    {grum_memory, grumi_memory, _} = template(input, output)

    def_vec(input, args.input_ids)
    def_vec_by_size(grumi_memory, args.memory_size)
    def_vec_by_size(grum_memory, args.sub_memory_size)
    def_vec(output, args.output_ids)
  end
end
