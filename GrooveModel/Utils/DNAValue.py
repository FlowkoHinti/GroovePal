from enum import IntEnum


class InstrumentValues(IntEnum):
    Rest = 0
    Kick = 1 << 0
    Snare = 1 << 1
    Toms = 1 << 2
    HiHat = 1 << 3
    Ride = 1 << 4
    Crash = 1 << 5

class RemappedInstrumentValues(IntEnum):
    Rest = 0
    Kick = 1
    Snare = 2
    Toms = 3
    HiHat = 4
    Ride = 5
    Crash = 6

DNA_VALUE_SIZE = sum([value.value for value in InstrumentValues]) + 1

def dna_to_dna_instruments_list(value) -> list[InstrumentValues]:
    """
    Get the instruments values from the dna value.
    :param value: composite value
    :return: list of instrument values
    """
    instruments = []
    for instrument in InstrumentValues:
        if value & instrument.value:
            instruments.append(instrument.value)
    return instruments

def dna_instrument_to_remapped_value(value) -> RemappedInstrumentValues:
    """
    Get the remapped instruments values from the dna value.
    :param value: value of a single instrument
    :return: list of remapped instrument values
    """
    dna_instrument = InstrumentValues(value).name
    return RemappedInstrumentValues[dna_instrument]

def dna_to_instruments(value) -> list[str]:
    """
    Get the instruments from the dna value.
    :param value: The value to get the instruments from.
    :return: The instruments.
    """
    instruments = []
    for instrument in InstrumentValues:
        if value & instrument.value:
            instruments.append(instrument.name)
    return instruments


def instruments_to_dna(instruments: list[str]) -> int:
    """
    Get the dna value from the instruments.
    :param instruments: The instruments to get the dna value from.
    :return: The dna value.
    """
    value = 0
    for instrument in instruments:
        value |= InstrumentValues[instrument].value
    return value
