defmodule Mix.Tasks.Or do
  @moduledoc false
  use Mix.Task

  def run(_) do
    SampleProjects.Or.run()
  end
end
