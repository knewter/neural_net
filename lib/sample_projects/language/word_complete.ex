defmodule SampleProjects.Language.WordComplete do
  @moduledoc false
  def run do
    {training_data, letters} = gen_training_data
    blank_vector = NeuralNet.get_blank_vector(letters)

    IO.puts "Generating neural network."
    net = GRUM.new(%{input_ids: letters, output_ids: letters, memory_size: 100})

    IO.puts "Beginning training."
    NeuralNet.train(net, training_data, 1.5, 2, fn info ->
      IO.puts "#{info.error}, iteration ##{info.iterations}"
      {input, exp_output} = Enum.random(training_data)

      {_, acc_plain} = NeuralNet.eval(info.net, input) #Get its expected letters given the whole word.
      {_, acc_feedback} = Enum.reduce 1..10, {hd(input), [%{}]}, fn _, {letter, acc} -> #generates with feedback
        {vec, acc} = NeuralNet.eval(info.net, [letter], acc)
        {Map.put(blank_vector, NeuralNet.get_max_component(vec), 1), acc}
      end

      actual = stringify [hd(input) | exp_output]
      letters = stringify [hd(input) | get_values(acc_plain)]
      feedbacked = stringify [hd(input) | get_values(acc_feedback)]
      IO.puts "#{actual} / #{letters}  | #{feedbacked}"
      info.error < 0.0001
    end, 2)
  end

  def get_values(acc) do
    Enum.map(Enum.slice(acc, 1..(length(acc) - 1)), fn time_frame ->
      time_frame.output.values
    end)
  end
  def stringify(vectors) do
    Enum.map(vectors, fn vec ->
      NeuralNet.get_max_component(vec)
    end)
  end

  def gen_training_data do
    {_, words} = SampleProjects.Language.Parse.parse("lib/sample_projects/language/common_sense.txt")
    IO.puts "Sample data contains #{MapSet.size(words)} words."
    words = words
      |> Enum.to_list()
      |> Enum.map(&String.to_char_list/1)
      |> Enum.filter(fn word -> length(word) > 1 end)
    letters = Enum.to_list(hd('a')..hd('z'))
    blank_vector = NeuralNet.get_blank_vector(letters)
    training_data = Enum.map words, fn word ->
      word = Enum.map(word, fn letter ->
        if !Enum.member?(letters, letter), do: raise "Weird word #{word}"
        Map.put(blank_vector, letter, 1)
      end)
      last = length(word) - 1
      {Enum.slice(word, 0..(last - 1)), Enum.slice(word, 1..last)}
    end
    {training_data, letters}
  end
end
