defmodule GRU do
  @moduledoc "GRU (Gated Recurrent Unit) is a variation on LSTM (Long short term memory). It is, for the most part, equally effective but computationally cheaper. An article explaining LSTMs and GRUs can be found here: http://colah.github.io/posts/2015-08-Understanding-LSTMs/."

  @doc "Takes an argument map with the keys :input_ids and :output_ids. Both values should be a list of component/id names for the input and output vectors (respectively)."
  use NeuralNet

  def template(inp, out \\ uid) do
    update_gate = sigmoid [inp, previous(out)]
    negated_update_gate = mult_const update_gate, -1
    forgetting_gate = add_const negated_update_gate, 1

    prev_out_gate = sigmoid [inp, previous(out)]
    gated_prev_out = mult [prev_out_gate, previous(out)]
    update_candidate = tanh [inp, gated_prev_out]

    gated_update = mult [update_candidate, update_gate]

    purged_output = mult [previous(out), forgetting_gate]
    add [purged_output, gated_update], out
  end

  defp define(args) do
    template(input, output)

    def_vec(input, args.input_ids)
    def_vec(output, args.output_ids)
  end
end
