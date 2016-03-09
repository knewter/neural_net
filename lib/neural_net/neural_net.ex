defmodule NeuralNet do
  defstruct size_defs: %{}, deps: %{}, affects: %{}, operations: %{}, net_layers: %{}

  defmacro __using__(_) do
    quote location: :keep do
      import NeuralNet.Helpers

      def new(args) do
        define(args)
        NeuralNet.Constructor.get_neural_net #retrieves constructed network from the Process dictionary
      end
    end
  end


end
