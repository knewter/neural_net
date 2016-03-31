defmodule LSTM do
  @moduledoc "LSTM stands for Long Short Term Memory. An article explaining it can be found here: http://colah.github.io/posts/2015-08-Understanding-LSTMs/."

  @doc "Takes an argument map with the keys :input_ids and :output_ids. Both values should be a list of component/id names for the input and output vectors (respectively)."
  use NeuralNet

  def template(inp, out \\ uid()) do
    memory = uid()
    forgetting_gate = sigmoid [previous(out), inp]
    purged_memory = mult [forgetting_gate, previous(memory)]
    content_filter = sigmoid [previous(out), inp]
    content_candidate = tanh [previous(out), inp]
    purged_content = mult [content_candidate, content_filter]
    add [purged_memory, purged_content], memory
    compressed_memory = pointwise_tanh memory
    output_filter = sigmoid [previous(out), inp]
    mult [compressed_memory, output_filter], out
  end

  defp define(args) do
    template(input, output)

    def_vec(input, args.input_ids)
    def_vec(output, args.output_ids)
  end
end
