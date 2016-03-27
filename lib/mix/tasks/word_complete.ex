defmodule Mix.Tasks.WordComplete do
  use Mix.Task

  def run(_) do
    SampleProjects.Language.WordComplete.run()
  end
end
