defmodule GRU do
  @moduledoc "GRU (Gated Recurrent Unit) is a variation on LSTM (Long short term memory). It is, for the most part, equally effective but computationally cheaper."
  use NeuralNet

  def template(inp, out) do
    sigmoid [inp, previous(out)], :update_gate
    mult_const :update_gate, -1, :negated_update_gate
    add_const :negated_update_gate, 1, :forgetting_gate

    sigmoid [inp, previous(out)], :prev_out_gate
    mult [:prev_out_gate, previous(out)], :gated_prev_out
    tanh [inp, :gated_prev_out], :update_candidate

    mult [:update_candidate, :update_gate], :gated_update

    mult [previous(out), :forgetting_gate], :purged_output
    add [:purged_output, :gated_update], out
  end

  defp define(args) do
    template(input, output)

    def_vec(input, args.input_ids)
    def_vec(output, args.output_ids)
  end
end
