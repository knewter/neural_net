defmodule NeuralNet.Mixfile do
  use Mix.Project

  def project do
    [app: :neural_net,
     version: "1.0.0",
     elixir: "~> 1.2",
     build_embedded: Mix.env == :prod,
     start_permanent: Mix.env == :prod,
     description: description,
     package: package,
     deps: deps,
     name: "NeuralNet",
     source_url: "https://gitlab.com/onnoowl/neural_net",
     docs: [main: NeuralNet]]
  end

  def deps do
    [
      {:earmark, "~> 0.1", only: :dev},
      {:ex_doc, "~> 0.11", only: :dev}
    ]
  end

  defp description do
    """
    NeuralNet is an A.I. library that allows for the construction and training of complex recurrent neural networks. Architectures such as LSTM or GRU can be specified in under 20 lines of code. Any neural network that can be built with the NeuralNet DSL can be trainined with automatically implemented BPTT (back-propagation through time).
    """
  end

  defp package do
    [
      maintainers: ["Stuart Hunt"],
      licenses: ["Apache 2.0"],
      links: %{"GitLab" => "https://gitlab.com/onnoowl/neural_net"}
    ]
  end
end
