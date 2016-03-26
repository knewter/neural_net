defmodule Mix.Tasks.SentenceComplete do
  use Mix.Task

  def run(_) do
    SampleProjects.Language.SentenceComplete.run()
  end
end
