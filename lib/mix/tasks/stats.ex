defmodule Mix.Tasks.Stats do
  use Mix.Task

  def default_args(), do: %{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d], memory_size: 3}

  def get_data(module, learn_val, args) do
    to_train = module.new(args)
    role_model = module.new(args)
    {_, info} = NeuralNet.Tester.test_training(to_train, role_model, 100, 10, learn_val, 2, 0, false)
    {info.eval_time, info.iterations}
  end

  def perform_statistics(str, data) do
    n = length(data)
    mean = Enum.reduce data, 0, fn diff, mean ->
      mean + (diff / n)
    end
    variance = Enum.reduce data, 0, fn datum, var ->
      var + (:math.pow(datum - mean, 2) / (n - 1))
    end
    sd = :math.sqrt(variance)

    digits = 5
    IO.puts "#{str} | Mean: #{Float.round(mean, digits)}, SD: #{Float.round(sd, digits)}, N: #{n}"
  end

  def compare(module1, module2, args1 \\ default_args(), args2 \\ default_args(), sample_size \\ 40) do
    {time_data, its_data} = Enum.reduce 1..sample_size, {[], []}, fn num, {time_data, its_data} ->
      learn_val = 0.5 + :random.uniform*2.5
      IO.write "\rCollecting data sample #{num}/#{sample_size} with learn_val #{learn_val}"
      {time1, iterations1} = get_data(module1, learn_val, args1)
      {time2, iterations2} = get_data(module2, learn_val, args2)
      {[time1 - time2 | time_data], [iterations1 - iterations2 | its_data]}
    end
    IO.puts ""
    perform_statistics("Time      ", time_data)
    perform_statistics("Iterations", its_data)
  end

  def run(_) do
    compare(GRU, GRUM)
  end
end
