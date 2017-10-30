defmodule SampleProjects.Or do
  @moduledoc false
  def run do
    training_data = gen_training_data
    blank_vector = NeuralNet.get_blank_vector([:left, :right])

    IO.puts "Generating neural network."
    net = NN.new(%{input_ids: [:left, :right], output_ids: [:out]})

    IO.puts "Beginning training."
    IO.inspect net
    {net, _} = NeuralNet.train(net, training_data, 0.1, 1, fn info ->
      {input, exp_output} = Enum.random(training_data)

      {o, acc_plain} = NeuralNet.eval(info.net, input) #Get its expected letters given the whole word.

      actual = stringify [hd(input) | exp_output]
      letters = stringify [hd(input) | get_values(acc_plain)]
      info.error < 0.0001
    end, 2)
    IO.inspect net
    IO.puts "done training, moving on"
    IO.puts "00"
    IO.inspect NeuralNet.eval(net, [%{left: 0, right: 0}])
    IO.puts "01"
    IO.inspect NeuralNet.eval(net, [%{left: 0, right: 1}])
    IO.puts "10"
    IO.inspect NeuralNet.eval(net, [%{left: 1, right: 1}])
    IO.puts "11"
    IO.inspect NeuralNet.eval(net, [%{left: 1, right: 0}])
  end

  def get_values(acc) do
    Enum.map(Enum.slice(acc, 1..(length(acc) - 1)), fn time_frame ->
      time_frame.output.values
    end)
  end
  def stringify(vectors) do
    Enum.map(vectors, fn vec ->
      r=NeuralNet.get_max_component(vec)
      r
    end)
  end

  def gen_training_data do
    [
      {[%{ left: 0, right: 0}], [%{ out: 0 }]},
      {[%{ left: 0, right: 1}], [%{ out: 1 }]},
      {[%{ left: 1, right: 0}], [%{ out: 1 }]},
      {[%{ left: 1, right: 1}], [%{ out: 1 }]},
    ]
  end
end
