defmodule GRU do
  @moduledoc "GRU (Gated Recurrent Unit) is a variation on LSTM (Long short term memory). It is, for the most part, equally effective but computationally cheaper."
  use NeuralNet

  defp define(args) do
    def_vec args.input_ids, [
      input
    ]
    def_vec args.output_ids, [
      :update_gate, :negated_update_gate, :forgetting_gate,
      :prev_out_gate, :gated_prev_out, :update_candidate,
      :gated_update,
      :purged_output, output
    ]

    sigmoid [input, previous(output)], :update_gate
    mult_const :update_gate, -1, :negated_update_gate
    add_const :negated_update_gate, 1, :forgetting_gate

    sigmoid [input, previous(output)], :prev_out_gate
    mult [:prev_out_gate, previous(output)], :gated_prev_out
    tanh [input, :gated_prev_out], :update_candidate

    mult [:update_candidate, :update_gate], :gated_update

    mult [previous(output), :forgetting_gate], :purged_output
    add [:purged_output, :gated_update], output
  end
end
