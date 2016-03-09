defmodule GRU do
  @moduledoc "GRU (Gated Recurrent Unit) is a variation on LSTM (Long short term memory). It is, for the most part, equally effective but computationally cheaper."
  use NeuralNet

  defp define(args) do
    set_size args.input_size, [
      input, :input_gate, :gated_prev_out
    ]
    set_size args.output_size, [
      output, :update_gate, :forgetting_gate, :update_candidate, :gated_update, :purged_output
    ]

    sigmoid [input, previous(output)], :update_gate
    mult_const [:update_gate], -1, :forgetting_gate

    sigmoid [input, previous(output)], :input_gate
    mult [:input_gate, previous(output)], :gated_prev_out
    tanh [input, :gated_prev_out], :update_candidate

    mult [:update_candidate, :update_gate], :gated_update

    mult [previous(output), :forgetting_gate], :purged_output
    add [:purged_output, :gated_update], output
  end
end
