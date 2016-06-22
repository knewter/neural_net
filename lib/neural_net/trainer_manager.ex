defmodule TrainerManager do
  @moduledoc "Simple synchronous training can be done using NeuralNet.train. TrainerManager lets you start, manage, pause, resume, and save multiple AsyncTrainers."

  use GenServer

  def start_link(args \\ %{}, name) do
    GenServer.start_link(__MODULE__, args, name: name)
  end

  def init(args) do
    path = Map.get(args, :path, "training_save.trs")
    state = if File.exists?(path) do
      IO.puts "Resuming session saved at #{path}"
      {trainers, id_counter} = :erlang.binary_to_term(File.read!(path))
      trainers = Enum.reduce(trainers, %{}, fn {id, {trainer_state, date}}, acc ->
        Map.put(acc, id, {AsyncTrainer.start_link(trainer_state), date})
      end)
      %{trainers: trainers, id_counter: id_counter}
    else
      IO.puts "Initializing new session at #{path}"
      %{trainers: %{}, id_counter: 0}
    end
    {:ok, Map.put(state, :args, args)}
  end

  def handle_call(:ping, _from, state) do
    {:reply, :pong, state}
  end
  def handle_cast(:save, state) do
    trainers = Enum.reduce(state.trainers, %{}, fn {id, {trainer_pid, date}}, acc ->
      Map.put(acc, id, {AsyncTrainer.get_state(trainer_pid), date})
    end)
    IO.puts "Saving to #{state.path}"
    File.write!(state.path, :erlang.term_to_binary({trainers, state.id_counter}))
    {:noreply, state}
  end

  def handle_cast({:new, train_state}, state) do
    trainer = {AsyncTrainer.start_link(train_state), DateTime.utc_now}
    id = state.id_counter
    {:noreply, state |> Map.update!(:trainers, &Map.put(&1, id, trainer)) |> Map.put(:id_counter, id+1)}
  end

  defmodule Sup do
    use Supervisor

    def start_link do
      Supervisor.start_link(__MODULE__, :ok)
    end

    def init(:ok) do
      children = [
        worker(TrainerManager, [%{}, TrainerManager])
      ]

      supervise(children, strategy: :one_for_one)
    end
  end

  defmodule App do
    use Application

    def start(_type, _args) do
      TrainerManager.Sup.start_link
    end
  end
end
