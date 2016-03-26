defmodule SampleProjects.Language.Parse do
  def parse(path) do
    {:ok, text} = File.read path
    # matches = Regex.scan(~r/(?:^|\s{2,})[A-Z][a-z]+\s[A-Za-z\s\-\$;:,]+\./, text)
    text = Regex.replace(~r/\s+/, text, " ")
    # IO.puts text
    matches = Regex.scan(~r/(?<=^|\. )[A-Z][a-z]+(?: [A-Za-z;:,"]+)+(?=\.|!)/, text)
    sentences = Enum.map(matches, fn [m] ->
      s = Regex.replace(~r/(?!\a)\s+/, m, " ")
      s = Regex.replace(~r/(?:^ |[;:,\.])/, s, "")
      Regex.split(~r/ /, String.downcase(s))
    end)
    {
      sentences,
      Enum.reduce(sentences, MapSet.new, fn sentence, acc ->
        Enum.reduce(sentence, acc, fn word, acc ->
          MapSet.put(acc, word)
        end)
      end)
    }
  end
end
