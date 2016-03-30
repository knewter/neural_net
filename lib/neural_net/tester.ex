defmodule NeuralNet.Tester do

  def test_training(net, desired_net, number_of_samples, time_frames, learn_val, batch_size, freq, verbose \\ true, test_only_final_output \\ true) do
    training_data = gen_training_data(desired_net, number_of_samples, time_frames, test_only_final_output)
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

  def gen_training_data(net, number_of_samples \\ 100, time_frames \\ 3, test_only_final_output \\ true) do
    Enum.map 1..number_of_samples, fn _ ->
      inputs = gen_inputs(net, time_frames)
      output = if !test_only_final_output do
        {_, acc} = NeuralNet.Backprop.get_feedforward(net, inputs)
        Enum.map(tl(acc), fn time_frame ->
          time_frame.output.values
        end)
      else
        {output, acc} = NeuralNet.eval(net, inputs)
        output
      end
      {inputs, output}
    end
  end
end
