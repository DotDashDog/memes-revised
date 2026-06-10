NOT FINSIHED. DO NOT REFERENCE. This is my doc with running thoughts about the more speculative ways to model/predict meme spread over embedding space

## Modes of Prediction/Modeling

### Neural Spatio-Temporal Hawkes
As I see it, this has some benefits and downsides
#### Benefits
It's fundamentally closer to what's happening. Posts are discete events and that's the way the dataset is organized, so we can calculate likelihoods in a simple way. 

#### Downsides
The end goal is prediction. It seems to me like prediction would be pretty hard.

### Fluid-Limit Neural ODE
#### Benefits


#### Downsides

### Commonalities
Both should be able to encode any relationship, since the neural Hawkes relation changing is effectively the same as changing the differential equation for the neural ODE