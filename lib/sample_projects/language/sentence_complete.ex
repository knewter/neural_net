defmodule SampleProjects.Language.SentenceComplete do
  @moduledoc false
  def run do
    {training_data, words} = gen_training_data
    blank_vector = NeuralNet.get_blank_vector(words)

    IO.puts "Generating neural network."
    net = GRU.new(%{input_ids: words, output_ids: words})

    IO.puts "Beginning training."
    NeuralNet.train(net, training_data, 1.5, 2, fn info ->
      IO.puts "#{info.error}, iteration ##{info.iterations}"
      {input, _} = Enum.random(training_data)

      # {_, acc} = NeuralNet.eval(info.net, input) #Get its expected word given a whole sentence.
      {_, acc} = Enum.reduce 1..10, {hd(input), [%{}]}, fn _, {word, acc} -> #Generates with feedback
        {vec, acc} = NeuralNet.eval(info.net, [word], acc)
        {Map.put(blank_vector, NeuralNet.get_max_component(vec), 1), acc}
      end

      vectors = [hd(input) | Enum.map(Enum.slice(acc, 1..(length(acc) - 1)), fn time_frame ->
        time_frame.output.values
      end)]
      words = Enum.map(vectors, fn vec ->
        Atom.to_string(NeuralNet.get_max_component(vec))
      end)
      IO.puts Enum.join(words, " ")
      info.error < 0.0001
    end, 0.2)
  end

  def gen_training_data do
    {sentences, words} = SampleProjects.Language.Parse.parse("lib/sample_projects/language/common_sense_small.txt")
    IO.puts "Sample data contains #{length(sentences)} sentences, and #{MapSet.size(words)} words."
    sentences = Enum.map sentences, fn sentence ->
      Enum.map(sentence, fn word ->
        String.to_atom(word)
      end)
    end
    words = Enum.map Enum.to_list(words), &String.to_atom/1
    blank_vector = NeuralNet.get_blank_vector(words)
    training_data = Enum.map sentences, fn sentence ->
      sentence = Enum.map(sentence, fn word ->
        Map.put(blank_vector, word, 1)
      end)
      last = length(sentence) - 1
      {Enum.slice(sentence, 0..(last - 1)), Enum.slice(sentence, 1..last)}
    end
    {training_data, words}
  end
end
