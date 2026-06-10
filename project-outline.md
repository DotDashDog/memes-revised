# Project Outline

This project's goal is to ascertain whether, and with what accuracy, information about internet memes' content can predict their spread online. The end goal is to have a machine learning model, that when given meaning-embedding information about a meme, can predict the spread parameters (defined later) for that meme.

Note that when I say meme in this document, I'm referring to any viral internet phenomenon, including but not limited to the text-over-images memes.

## Context

I attempted this in a previous iteration of the project. I began the project looking at interest-over-time curves from Google Trends. I observed that the interest over time curves for many internet memes (often those that arise 'organically' on the internet) can be fit very well by the $I(t)$ curve of the SIR model, a simple epidemiological model. It places people into bins, whose dynamics are governed by the following system of differential equations.

$$ 

\frac{dS}{dt} = -\beta I S

\frac{dI}{dt} = \beta I S - \gamma I

\frac{dR}{dt} = \gamma I 

$$

Where S is the fraction of the population that is susceptible to the disease, I is the fraction currently infected, R is the fraction that have recovered, $\beta$ is the infectiousness, and $\gamma$ is the recovery rate

Assuming memes are well-modeled by the SIR model boils down to a few main assumptions:
- The 'interest' in a meme (or the number of people searching on it) is proportional to the number of people 'infected' (actively spreading it)
- Once someone stops being 'infected,' they can't be re-infected
- The meme is spreading in a single, homogeneous (same infectiousness and recovery rate throughout), well-mixing population.

The goal of this project (at least as of now) is to predict the spread parameters: 
- $\beta$
- $\gamma$
- $s$: a scaling parameter that captures the proportionality (including how often people search/post/etc and the size of the initial susceptible population) 
from a meaning-embedding of a given meme.

## Previous Work

In my previous work on this (back in 2023), I scraped text from knowyourmeme.com and wikipedia to build a dataset of ~1500 discrete memes and their embedded descriptions (as well as an extra entry in the vector encoding the time that the meme originated). For each of these memes, I then scraped interest-over-time data from Google Trends and fit the SIR model to it, extracting the three fit parameters (plus $I(t=0)$, the number of infected people at the start of the time window).

I trained a neural network on this data, and it achieved an accuracy of better than random chance: an average of ~250% error in the logs of the spread parameters, as compared to ~350% when it was given no information about a meme. Obviously, this isn't very good, but it's not nothing either.

There were quite a few problems with the past iteration of this project:
1. Not enough/poor quality Trends data. Many of the memes didn't have robust interest over time curves available on Google Trends. We threw out fits where this was obviously true, which reduced our meme count to around 700.
2. Not enough memes, and poor quality descriptions. Many of the descriptions were single-sentence, and didn't provide that much detail.
3. Possible data contamination with knowledge of the memes' spread. We tried to vet the meme descriptions to cut out information about how much they actually spread online, but it's hard to completely get rid of all implicit knowledge there. Also, the fact that they're even descibed on these sites signifies some level of virality.
4. Possible embedding model contamination with knowledge of the memes' spread. The embedding model we used was trained after most of these memes went viral. It's possible (even likely) that the embedding essentially contained context about how well the meme spread from the model's training data. To some extent, this is unavoidable, and the main thing that would rule this out is testing a finished meme prediction model on memes that went viral after the embedding model was trained.
5. Poor embeddings. I'm sure that embedding models have improved a ton in the past few years.

## Current Project
My plan for a new iteration of this project should solve many of these past problems. My current idea is as follows:

### 1. Downloading and Embedding The Data
Download massive amounts of data from a social media site. Right now, this is Reddit. Currently, I've got an archive of all the posts until before some time in the 2020s from r/memes. Over time, I plan to incorporate more, but I figure a single large subreddit is a good starting point. I have access to similar data for all of the top 40000 subreddits.

Once I have the data, I want to embed each individual post.

One specific challenge with r/memes (and possibly other subreddits) is that much of the meaning is carried in the text on the images. It's my impression that current image embedding models aren't good enough to full embed that text, so I currenly plan to run OCR to extract the text, then embed that, the post text, and the image. The problem is that OCR on memes is hard. They have uneven backgrounds, text with varying colors and nonstandard fonts, and wildly varying styles. I've been working on it, but I haven't achieved the reliability I want from the OCR.

Also, I think there is additional information contained in post comments and upvotes, but I haven't decided what the best way to include them is yet.

#### The Embedding Model

I'm not sure what the right model is. A few years ago, CLIP seemed like a fairly good choice, but I'm sure better models have since come out. I need to balance price (I'm planning on embedding millions of memes) with quality. I have access to a 3090 GPU to run local models on, if need be.

### 2. Clustering in The Embedding Space
After embedding all these posts, I'll have a set of millions of time-stamped points in this high-dimensional space. I can run clustering on them, and hopefully, if I do clustering correctly and the embedding space is well-behaved, each cluster will roughly correspond to a meme, with the cluster center giving a rough characterization (description) of it. The distribution of the timestamps in this cluster (combined with upvotes and comments in some way) can give me an interest over time curve, to which I can fit the SIR model I(t) curve. This allows me to extract the spread parameters.

### 3. Training the Prediction Model
Train a model to take in cluster center (meme embeddings) and predict the spread parameters.

## Specific Improvement Areas
### Better fitting for the SIR model
Since then, I've gained a lot of experience in Bayesian inference, especially advanced gradient-informed MCMC methods. It probably wouldn't be that useful for the actual dataset generation, but it would be interesting to see if there are strong degeneracies between any fit parameters and what the full posteriors look like. This is lower priority, however.

On a similar topic, as part of the SIR fitting, I previously generated the I(t) curves with one of scipy's integration functions. However, this was relatively slow, especially considering I needed to do so many times to fit a curve. I do wonder if there are better ways to generate an I(t) curve than integration (such as solutions to the SIR model using special functions). Whether or not this is possible, it might also be useful to be able to take gradients through the I(t) curve (both for curve fitting and model training, if I decide to fit an SIR I(t) curve to the interest over time curve directly, instead of predicting the parameters).

### More Complex Spread Models
It's also possible that the SIR model is too simple, or that (for example) there's a strong degeneracy between the proportionality factor and the other two parameters. It's worth exploring different models for meme spread.

### LLM/New Architectures
I have some experience with neural networks, but I think it's likely there are smart architectures that would be well suited to this specific problem.


## Speculative Improvement Areas
### Holistic View
Instead of looking at individual memes as clusters, maybe we can train something to watch/understand the dynamics of posts/memes through the whole space over time. I'm not exactly sure how this would work. Foundation model??? It's a little above my pay grade.

### Long-term: generating memes
Once I can predict how a meme spreads (which is itself a very hard problem), is it possible to generate memes with desired traits?

## The Current State
It's been a while since I worked on this codebase, but I've currently hit a roadblock with the OCR, and trying to achieve the accuracy I want.