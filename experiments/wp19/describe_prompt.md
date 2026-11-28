You are shown one meme image posted to Reddit. Describe what it is, as data for a research
project. Reply with a single JSON object and nothing else, with these fields:

- "text": the text in the image, transcribed as written, panels in reading order separated
  by " / ". Empty string if there is none.
- "template": the conventional name of the base image or format if you are confident of it
  (for example "Drake Hotline Bling", "Surprised Pikachu"). Otherwise five to ten words
  describing the base image. Use "none" for plain text, screenshots, or a one-off image.
- "referents": a list of the specific things the joke is about or depends on: named events,
  trends, catchphrases, releases, people, in-jokes. Use the most common short name for each.
  Empty list if the joke is self-contained. A broad subject such as school, gaming or
  relationships is not a referent.
- "premise": one plain sentence stating the joke.

Describe only what the meme says and refers to. Do not say or imply anything about how
popular, famous, viral, successful, old, new, overused or well received the meme, its
template or its subject is or was.
