# NeuralNet

NeuralNet allows for the construction and training of complex recurrent neural networks. Architectures such as LSTM or GRU can be specified in under 20 lines of code. Any neural network that can be built with the NeuralNet DSL can be trainined with automatically implemented BPTT (back-propagation through time).

## Installation

Add neural_nets to your list of dependencies in `mix.exs`:

    def deps do
      [{:neural_net, "~> 1.0"}]
    end

## Simple RNN

```elixir
defmodule SimpleRNN do
  use NeuralNet

  defp define(args) do
    # This simple RNN is just a single neural network layer, using tanh as its
    # activation function, that connects the current input and previous output
    # to produce the current output.
    tanh [input(), previous(output())], output()

    def_vec(input(), args.input_ids)
    def_vec(output(), args.output_ids)
  end
end

net = SimpleRNN.new(%{input_ids: [:x, :y], output_ids: [:a, :b]})

#evaluate for 2 times frames with the two example input vectors.
inputs = [%{x: 1.0, y: -0.5}, %{x: -0.6, y: 0.4}]
{output, _} = NeuralNet.eval(net, inputs)
IO.puts inspect(output)

training_data = [
  {inputs, %{a: -0.25, b: 0.75}}
]

#Train with a learn_val of 1.5, and bath_size of 1. It will call our checking function every 0.05 seconds.
NeuralNet.train(net, training_data, 1.5, 1, fn info ->
  info.error <= 0.000000001
end, 0.05)
IO.puts "Training complete."
```

## GRU

Below is an example implmentation of the GRU architecture (a more proper version can be found in lib/gru.ex)

```elixir
defmodule GRU do
  use NeuralNet

  def define(args) do
    update_gate = sigmoid [input(), previous(output())]
    negated_update_gate = mult_const update_gate, -1
    forgetting_gate = add_const negated_update_gate, 1

    prev_out_gate = sigmoid [input(), previous(output())]
    gated_prev_out = mult [prev_out_gate, previous(output())]
    update_candidate = tanh [input(), gated_prev_out]

    gated_update = mult [update_candidate, update_gate]

    purged_output = mult [previous(output()), forgetting_gate]
    add [purged_output, gated_update], output()

    def_vec(input(), args.input_ids)
    def_vec(output(), args.output_ids)
  end
end

```
