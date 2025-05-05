from enum import Enum


class InstrumentValues(Enum):
    Rest = 0
    Kick = 1 << 0
    Snare = 1 << 1
    Toms = 1 << 2
    HHClose = 1 << 3
    HHOpen = 1 << 4
    Ride = 1 << 5
    Cymbal = 1 << 6


DNA_VALUE_SIZE = sum([value.value for value in InstrumentValues]) + 1


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
