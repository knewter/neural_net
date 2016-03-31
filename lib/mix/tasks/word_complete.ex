defmodule Mix.Tasks.WordComplete do
  @moduledoc false
  use Mix.Task

  def run(_) do
    SampleProjects.Language.WordComplete.run()
  end
end
