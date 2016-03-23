defmodule Mix.Tasks.Stats do
  use Mix.Task

  def args, do: %{input_ids: [:x, :y], output_ids: [:a, :b, :c, :d]}
  def gen(module), do: module.new(args())

  def get_iterations(module, learn_val) do
    to_train = gen(module)
    role_model = gen(module)
    {_, info} = TrainingTester.test_training(to_train, role_model, 100, 100, learn_val, 2, 0, false)
    info.eval_time
  end

  def run(_) do
    data = Enum.map 1..40, fn num ->
      IO.puts "Collecting data sample ##{num}."
      learn_val = 0.1 + :random.uniform*1.9
      get_iterations(GRU, learn_val) - get_iterations(LSTM, learn_val)
    end
    n = length(data)
    mean = Enum.reduce data, 0, fn diff, mean ->
      mean + (diff / n)
    end
    variance = Enum.reduce data, 0, fn datum, var ->
      var + (:math.pow(datum - mean, 2) / (n - 1))
    end
    sd = :math.sqrt(variance)

    IO.puts "Mean: #{mean}, SD: #{sd}, N: #{n}"
  end
end
