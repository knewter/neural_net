defmodule NeuralNet do
  @moduledoc "This module allows you to define and train neural networks with complex architectures. See lib/gru.ex for an example implementation of the Gated Recurrent Unit architecture."

  defstruct vec_defs: %{}, deps: %{}, affects: %{}, operations: %{}, net_layers: %{}, weight_map: %{}

  defmacro __using__(_) do
    quote location: :keep do
      import NeuralNet.Helpers

      def new(args) do
        define(args)
        NeuralNet.Constructor.get_neural_net #retrieves constructed network from the Process dictionary
        |> NeuralNet.Constructor.gen_random_weight_map()
      end
    end
  end

  def get_vec_def(net, {:previous, vec}), do: get_vec_def(net, vec)
  def get_vec_def(net, vec), do: Map.fetch!(net.vec_defs, vec)
end
