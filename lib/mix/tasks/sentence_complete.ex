defmodule Mix.Tasks.SentenceComplete do
  @moduledoc false
  use Mix.Task

  def run(_) do
    SampleProjects.Language.SentenceComplete.run()
  end
end
