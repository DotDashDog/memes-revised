You are describing meme images for a research project's reference labels. Each image is
described on its own; you are not grouping anything.

Your images: open `/home/linus/memes-revised/experiments/WP-04/sample-001/stage1/batches.json`
and take the lists named below (key and index given at the end of this message). Each entry
is an id; the image is at
`/home/linus/memes-revised/experiments/WP-04/sample-001/label_images/<id>.jpg`. View every
image with the Read tool. A few at a time is fine.

Blindness matters here, because these labels will be used to audit a model and must not be
contaminated by it. Look at the images and nothing else: do not open any other file in the
repository (no docs, no other folders in the sample directory, no other agents' outputs),
and do not search the web. Describe each image from what is in it plus your own general
knowledge of meme formats and of what was going on at the time (the posts are from r/memes,
calendar year 2019).

For each image write one JSON object:

- `id`: the id.
- `text`: the text in the image, transcribed as written, panels in reading order separated
  by " / ". Empty string if none.
- `template`: the conventional name of the base image or format if it has one ("Drake
  Hotline Bling", "Lisa Simpson presentation", "Surprised Pikachu", "Me and the boys").
  If it has no name you know, five to ten words describing the base image, specific enough
  that another person describing a different post built on the same base image would write
  something recognisably the same ("man in red jacket gesturing at whiteboard"). Use `none`
  for plain text, screenshots of tweets or posts, and one-off original images with no reused
  base.
- `referents`: a list of the specific things the joke is about or depends on: named events,
  trends, catchphrases, releases, in-jokes ("Area 51 raid", "No Nut November", "Reddit cake
  day", "OK boomer", "Minecraft versus Fortnite"). Use the most common short name for each.
  Leave empty for a self-contained joke. A broad subject such as school, gaming or
  relationships is not a referent.
- `premise`: one plain sentence stating the joke.
- `notes`: anything that makes the image hard to classify (blank image, unreadable text,
  several memes in one image, not a meme). Empty string otherwise.

Write accurately and do not guess a template name you are not confident of; a description is
better than a wrong name. Transcribe text even when it is crude; these are research records.

Output: write a JSON array of these objects, one per image, in the order given, to the
output path named below. Write the file as you go (after every ten to twenty images) so an
interruption does not lose the work. When finished, check the file parses as JSON and has
exactly one object per id, then reply with one line: the number of images described and any
ids you could not view.
