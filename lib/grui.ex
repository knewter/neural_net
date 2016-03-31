defmodule GRUI do
  @moduledoc false
  # @moduledoc "GRUI is an experimental neural network design in development."

  use NeuralNet

  def template(inp, out \\ uid) do
    update_gate = sigmoid [inp, previous(out)]
    negated_update_gate = mult_const update_gate, -1
    forgetting_gate = add_const negated_update_gate, 1

    content_weights = GRU.template(inp)

    prev_out_gate = sigmoid [inp, previous(out)]
    gated_prev_out = mult [prev_out_gate, previous(out)]
    update_candidate = tanh_given_weights [inp, gated_prev_out], content_weights

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
