# order data into directories
# clean up
# prepare for json extraction
# train val test splits?
from Preprocessing.Prepper.DrumLearningMidiPrepper import DrumLearningMidiPrepper
from Preprocessing.Prepper.GigaMidiPrepper import GigaMidiPrepper
from Preprocessing.Prepper.GrooveMidiPrepper import GrooveMidiPrepper

preppers = [
    #GigaMidiPrepper(train_size=200000, val_size=20000, test_size=1000)
    #GrooveMidiPrepper(),
    DrumLearningMidiPrepper(train_ratio=0.8, val_ratio=0.15, test_ratio=0.05),
    ]

for prepper in preppers:
    prepper.prepare()
