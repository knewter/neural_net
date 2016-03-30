defmodule Mix.Tasks.Stats do
  use Mix.Task

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

  def compare(module1, module2, argfun1, argfun2, sample_size \\ 200) do
    {time_data, its_data} = Enum.reduce 1..sample_size, {[], []}, fn num, {time_data, its_data} ->
      learn_val = 0.5 + :random.uniform*2.5
      IO.write "\rCollecting data sample #{num}/#{sample_size} with learn_val #{Float.round(learn_val, 3)}"
      {time1, iterations1} = get_data(module1, learn_val, argfun1.())
      {time2, iterations2} = get_data(module2, learn_val, argfun2.())
      {[time1 - time2 | time_data], [iterations1 - iterations2 | its_data]}
    end
    IO.puts ""
    perform_statistics("Time      ", time_data)
    perform_statistics("Iterations", its_data)
  end

  def run(_) do
    argfun = fn ->
      in_size = :random.uniform(10)
      out_size = :random.uniform(10)
      %{input_ids: Enum.to_list(1..in_size), output_ids: Enum.to_list(1..out_size), memory_size: out_size}
    end

    compare(GRU, GRUM, argfun, argfun)
  end
end
