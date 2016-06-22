defmodule AsyncTrainer do
  @moduledoc "Simple synchronous training can be done using NeuralNet.train. For longer training session, Trainer acts as a GenServer trainer that can be paused and resumed."

  use GenServer

  def start_link(state \\ %{}, opts \\ []), do: GenServer.start_link(__MODULE__, state, opts)

  def init(state) do
    state
      |> Map.put(:is_training, false)
      |> Map.put(:should_train, false)
  end

  defp init_training(state) do
    GenServer.cast(self, :train_step) #queue up training again (doesn't start until I finish)
    state
      |> Map.put(:is_training, true)
      |> Map.put(:should_train, true)
  end

  def handle_cast(:train_step, state) do
    state = state.should_train do
      GenServer.cast(self, :train_step) #queue up training again (doesn't start until I finish)
      Map.put(state, :train_state, NeuralNet.train_step(state.train_state))
    else
      Map.put(state, :is_training, false)
    end
    {:noreply, state}
  end

  def handle_cast(:pause, state) do
    {:noreply, Map.put(state, :should_train, false)}
  end

  def handle_cast(:resume, state) do
    {:noreply, init_training(state)}
  end

  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  def pause(pid), do: GenServer.cast(pid, :pause)
  def resume(pid), do: GenServer.cast(pid, :resume)
  def get_state(pid), do: GenServer.call(pid, :get_state)
end
