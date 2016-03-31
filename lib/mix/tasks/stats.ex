defmodule Mix.Tasks.Stats do
  @moduledoc false
  use Mix.Task

  def get_datum(module, learn_val, args) do
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
      {time1, iterations1} = get_datum(module1, learn_val, argfun1.())
      {time2, iterations2} = get_datum(module2, learn_val, argfun2.())
      {[time1 - time2 | time_data], [iterations1 - iterations2 | its_data]}
    end
    IO.puts ""
    perform_statistics("Time      ", time_data)
    perform_statistics("Iterations", its_data)
  end

  def get_data(module, argfun, sample_size \\ 200, record \\ false) do
    time_file = if record, do: File.open!("#{inspect(module)}_time.dat", [:write])
    iterations_file = if record, do: File.open!("#{inspect(module)}_iterations.dat", [:write])
    {time_data, its_data} = Enum.reduce 1..sample_size, {[], []}, fn num, {time_data, its_data} ->
      learn_val = 0.5 + :random.uniform*2.5
      IO.write "\rCollecting #{inspect(module)} data sample #{num}/#{sample_size} with learn_val #{Float.round(learn_val, 3)}"
      {time, iterations} = get_datum(module, learn_val, argfun.())
      if record do
        IO.puts(time_file, time)
        IO.puts(iterations_file, iterations)
      end
      {[time | time_data], [iterations | its_data]}
    end
    if record do
      File.close(time_file)
      File.close(iterations_file)
    end
    IO.puts ""
    perform_statistics("#{inspect(module)} Time      ", time_data)
    perform_statistics("#{inspect(module)} Iterations", its_data)
  end

  def run(_) do
    argfun = fn ->
      in_size = 1+:random.uniform(9)
      out_size = 1+:random.uniform(9)
      mem_size = 1+:random.uniform(9)
      %{input_ids: Enum.to_list(1..in_size), output_ids: Enum.to_list(1..out_size), memory_size: mem_size}
    end

    # compare(GRU, GRUM, argfun, argfun)
    get_data(GRU, argfun, 200, true)
    get_data(GRUM, argfun, 200, true)
  end
end
