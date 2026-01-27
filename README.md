# GroovePal
**Masterthesis Project**:

Humanising Drum Synthesis: An xLSTM Approach for Stylistically Coherent Drum Continuations

## Abstract
Despite advances in music modelling, generating realistic human-like drumming remains
challenging. This thesis addresses this challenge with a novel approach that employs the
Extended Long Short-Term Memory (xLSTM) architecture, combined with MusicalBits’
proprietary reduced drum data representation, ”DNA”. Accurately modelling expressive
qualities such as microtiming and dynamics, and producing stylistically fitting drum
patterns, were crucial for capturing attributes associated with human performance. To
explore the viability of the xLSTM architecture, this work proposes a sequence-continuation
setup combined with empirical assessment of inputs and extrapolations. Furthermore,
a multitude of architectural configurations were evaluated, including comparisons of
multitask against sequential tokenisation schemes, absolute vs. relative onset encodings,
and the impact of beat positional encoding. Results demonstrated that a multitask
modelling approach performed favourably over the REMI-like sequential architecture.
While challenges remain in sustained coherence over longer sequences the findings validate
the xLSTM’s capability to reproduce key humanising factors to an extent.

## Scientific Question
Can an xLSTM-based model with targeted optimisations effectively extrapolate rock and metal drum sequences while preserving their unique style? 

## Credits
- **MusicalBits GmbH**: High-Quality Drum Performances and basis of Tokenisation
