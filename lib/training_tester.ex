defmodule TrainingTester do

  def test_training(net, desired_net, number_of_samples, time_frames, learn_val, batch_size, freq, verbose \\ true) do
    training_data = gen_training_data(desired_net, number_of_samples, time_frames)
    NeuralNet.train(net, training_data, learn_val, batch_size, fn info ->
      if verbose, do: IO.puts info.error
      info.eval_time > 59 or info.error < 0.0001
    end, freq)
  end

  def gen_inputs(net, time_frames \\ 3) do
    Enum.map 1..time_frames, fn _ ->
      Enum.reduce(NeuralNet.get_vec_def(net, :input), %{}, fn component, acc ->
        Map.put(acc, component, :rand.uniform)
      end)
    end
  end

  def gen_training_data(net, number_of_samples \\ 100, time_frames \\ 3) do
    Enum.map 1..number_of_samples, fn _ ->
      inputs = gen_inputs(net, time_frames)
      {output, _} = NeuralNet.eval(net, inputs)
      {inputs, output}
    end
  end
end
