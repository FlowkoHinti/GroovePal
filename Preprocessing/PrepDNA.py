# order data into directories
# clean up
# prepare for json extraction
# train val test splits?
from DNAPrepper import GigaMidiPrepper

preppers = [GigaMidiPrepper(train_size=200000, val_size=20000, test_size=1000)]

for prepper in preppers:
    prepper.prepare()