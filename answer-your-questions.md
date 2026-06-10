# Foundational/Framing Questions

## 1a

My hypothesis is that a large amount of variation in a memes' success can be explained by its content, but I don't really know what's achievable. I think finding that out is essentially the goal of this project. It does make it harder, so for now let's set a preliminary 'goal of $R^2=0.3$ on the logs of each of the spread parameters. 

## 1b

I think that this and the embedding contamination issue (1c) are the biggest likely issues with the project. We will need a substantial set of non-viral memes, and especially after clustering, these just won't exist. I think both your ideas: looking at low-upvote posts on r/memes and sweeping Reddit content more broadly will likely be needed. 

## 1c

I think training a model from scratch is probably too difficult to do well, and generating a contamination-controlled corpus that is also large enough to be useful is likely impossible. I think the hard barrier created by time is probably our best bet. This would mean using a smaller/older model. Choosing the right balance between model's quality (which presumably increases with time) and contamination will be hard. I think ideally, I'd like at least ~30% of the training corpus to be from after the embedding model's training cutoff.

It's also possible that there already exist more advanced modern models trained on mostly contamination-free data. E.g. one trained only on Wikipedia. It's worth looking into.

I'd be interested to hear more about your ideas for constrastive setups.

## 1d

I want to push back on this a bit. I think as long as the relationship between people transmitting the meme and people posting it is linear, the proportinality factor should absorb all that. It's okay if all the factors (number of people posting per infected, etc) change from meme to meme or year to year. We won't know the contribution from any one factor, but the modeling should work overall. It does lose some of its real-world meaning as the population size, but that's alright. That being said, you make a good point about possible inherent nonlinearity in that relationship. I think that's probably just a hit we'd have to take. Fundamentally, what we're trying to do is predict the posts (or interest) over time curve, which is directly observable, and this doesn't change that. It does affect how well we can interpret the SIR parameters, however.

# The SIR Model

## 2a

I started out thinking of the SIR model as a mechanical model, and I think there is some truth to that. I think it may be better (at least with my current idea for interest-over-time fitting) to treat it as an empirical model. I've noted that it does fit most meme curves well, and maybe all I need is a way to compress the most of the information about a meme's spread into a few parameters. With that in mind, I do think SIR is a better empirical model than SIRS and SIS, at least. With SIRS and SIS, you eventually approach a non-zero equilibrium in I(t), which isn't something I typically observe with viral memes. They almost invariably die off to zero.

## 2b

I agree that we need to quantify the parameter degeneracy. In previous fits, I made the choice to always fix $S(t=0) = 1-I(t=0)$ and $R(t=0)=0$. I think this is a good choice, since there's likely a strong (if not complete) degeneracy between R(t=0), $\beta$, and $s$. But I agree that we need to quantify any degeneracy in our choice of parameter set before settling on it.

## 2c

I think this is a big issue with the SIR fitting. In my previous google trends dataset, this sort of behavior showed up for (very roughly) ~15% of memes. I'm tempted to just take the largest hump, since I think a full treatment of this behavior (at least with the SIR assumption) would involve introducing the appearance of other populations with different spread parameters. I this could be done, but it would likely require some kind of complex Bayesian approach to do without overfitting.

## 2d

I think as long as the meme isn't the event itself, external triggers may not be a huge issue. If the world event just primes the environment for and then 'spawns' the meme, from which point the meme spreads organically, I think that's okay. However, I think if there is more complex external driving, we run into the same problems as 2c.

# Clustering

## 3a

I like the idea of hierarchical clustering to solve this problem. I would definitely test different clustering algorithms and hyperparameters as well. One of my main checks would be looking at the actual text and images for random memes within a cluster. If they all seem to belong to the same viral meme, that's a good sign for my clustering.

## 3b

That's a good point. I think the first step for verifying clustering is probably doing what I talked about for 3a and manually checking clustering results. We could also look at the post-over-time curve and see if it looks coherent. Selecting for this does introduce its own biases, but I think there is definitely a shape that interest curves for memes tend towards.

## 3c

That's a good point. I'm not sure what the best way to represent variance in the cluster is, especially because we're working in very high-dimensional spaces, but this sort of second-order information is worth considering.

# OCR/embedding Strategy

## 4a

I hadn't realized that modern VLMs were that good. As long as they're cost-effective for the scales I'm looking at, I'm in favor of trying and possibly using them.

## 4b

I was thinking of concatenating (as a string) the text for the meme image and the post text (possibly with text markers for each), and then embedding that text and the image separately. Then, I could concatenate the embedding vectors. This seems like the best way to do things without losing information, but I'm not sure if there are more well-motivated ways to do it.

## 4c

My initial idea of how to incorporate upvotes is to weight the posts in by their upvotes when generating the I(t) curve. I'm not sure what the best way to incorporate comments is, and I'd be interested to hear any ideas you have.

# Scope / methodology

## 5a

With other social media platforms, I run into a data issue. I'd need enormous volumes of data that I don't think they provide. (If you have sources, I'd be very interested). For now, I'd like to focus on modeling meme spread within a single platform, and Reddit seems like a good place to start.

## 5b

I haven't considered this deeply. Let's make a note to come back to this once we have robust clustering working.

## 5c

I also haven't thought too deeply about baselines. All of these are good suggestions for baselines. The one thing I'd be worried about with a frontier LLM is making sure the LLM's training cutoff is before the meme came onto the scene.

# The Smaller Things

- The SIR model I used in the outline has S, I, and R as fractions of the population. That way, $s$ just scales the whole population. If there's an issue with my framing there, I definitely want to hear it.
- I agree, for measuring future predictions, error in the log of the spread parameters isn't the best. I like $R^2$ in log space. I haven't heard of Spearman rank correlation before, but I'd be willing to try.

# Other Related Thoughts

## Parallels to Physics-Informed Neural Nets

As I've thought about this project, my perspective on why the SIR model is useful (if it is at all) has evolved. Since neural nets are universal approximators, there's not a strict reason I'd need it to predict interest-over-time curves. However, it's a useful way of distilling messy real-world data about a meme's spread into a few key parameters, that in many cases capture almost all the information. 

I think the idea is somewhat similar to physics-informed neural nets, where the governing differential equations are encoded into the loss function. I think there's some amount of insight into the dynamics of the spread of viral memes that is encoded by the SIR model. Obviously, however, it is an oversimplification in a way that enforcing physical constraints (say conservation of energy) just isn't. 

I do wonder if there are things we can learn from PINNs in this realm, however. It could give insight into different ways to utilize this sort of information.

## Long-Term Idea: Learning the Holistic Posting Landscape

I've thought a little more about how best to learn the data in a holistic, non-cluster-based way. Drawing parallels to low-dimensional spaces, I'm thinking we could frame posts as a Poissonian-like process on an underlying rate field over the embedding space. The goal would then be to predict the rate field for future times given past information. However, this draws on a lot of physics-based intuition that might not apply for very high dimensional spaces where volumes are wonky. Learning a set of differential equations for this field on this space would be very interesting though. People probably have smart techniques here.

We could draw from physics-informed neural nets (see above for my thoughts) and incorporate an a loss penalty that enforces SIR-like dynamics with a certain strength.