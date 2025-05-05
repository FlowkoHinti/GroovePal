# loss function for masked language modelling
# insert mask tokens into sequence and let model predict sequence without mask tokens I_mask -> I
# teacher forcing can be a great optimisation as well
# add loss / training function for the various setups