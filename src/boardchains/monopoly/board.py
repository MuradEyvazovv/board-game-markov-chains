"""The standard US Monopoly board: squares, color groups, card decks and rent tables.

Square numbers run 0 (Go) to 39 (Boardwalk) in the direction of travel. Square 10 is
Jail; a token there is either Just Visiting or actually in jail, and I keep those apart
everywhere. In the 41-way aggregate used for results, index 40 means "In Jail" and
index 10 means "Just Visiting".
"""

from __future__ import annotations

from dataclasses import dataclass

N_SQUARES = 40
GO = 0
JUST_VISITING = 10
GO_TO_JAIL = 30
IN_JAIL = 40  # extra category index for a token that is actually in jail
N_CATEGORIES = 41

SQUARE_NAMES: tuple[str, ...] = (
    "Go",
    "Mediterranean Avenue",
    "Community Chest 1",
    "Baltic Avenue",
    "Income Tax",
    "Reading Railroad",
    "Oriental Avenue",
    "Chance 1",
    "Vermont Avenue",
    "Connecticut Avenue",
    "Just Visiting",
    "St. Charles Place",
    "Electric Company",
    "States Avenue",
    "Virginia Avenue",
    "Pennsylvania Railroad",
    "St. James Place",
    "Community Chest 2",
    "Tennessee Avenue",
    "New York Avenue",
    "Free Parking",
    "Kentucky Avenue",
    "Chance 2",
    "Indiana Avenue",
    "Illinois Avenue",
    "B&O Railroad",
    "Atlantic Avenue",
    "Ventnor Avenue",
    "Water Works",
    "Marvin Gardens",
    "Go To Jail",
    "Pacific Avenue",
    "North Carolina Avenue",
    "Community Chest 3",
    "Pennsylvania Avenue",
    "Short Line",
    "Chance 3",
    "Park Place",
    "Luxury Tax",
    "Boardwalk",
)
CATEGORY_NAMES: tuple[str, ...] = SQUARE_NAMES + ("In Jail",)

CHANCE_SQUARES = (7, 22, 36)
COMMUNITY_CHEST_SQUARES = (2, 17, 33)
RAILROADS = (5, 15, 25, 35)
UTILITIES = (12, 28)
TAXES = (4, 38)

COLOR_GROUPS: tuple[str, ...] = (
    "brown",
    "light blue",
    "pink",
    "orange",
    "red",
    "yellow",
    "green",
    "dark blue",
)


@dataclass(frozen=True)
class Property:
    square: int
    name: str
    group: str
    price: int
    house_cost: int
    # rent with: no houses (single lot), 1, 2, 3, 4 houses, hotel
    rents: tuple[int, int, int, int, int, int]


def _p(square: int, group: str, price: int, house: int, rents: tuple[int, ...]) -> Property:
    return Property(square, SQUARE_NAMES[square], group, price, house, tuple(rents))  # type: ignore[arg-type]


# Official US title deeds (the values printed on the standard Hasbro edition).
PROPERTIES: tuple[Property, ...] = (
    _p(1, "brown", 60, 50, (2, 10, 30, 90, 160, 250)),
    _p(3, "brown", 60, 50, (4, 20, 60, 180, 320, 450)),
    _p(6, "light blue", 100, 50, (6, 30, 90, 270, 400, 550)),
    _p(8, "light blue", 100, 50, (6, 30, 90, 270, 400, 550)),
    _p(9, "light blue", 120, 50, (8, 40, 100, 300, 450, 600)),
    _p(11, "pink", 140, 100, (10, 50, 150, 450, 625, 750)),
    _p(13, "pink", 140, 100, (10, 50, 150, 450, 625, 750)),
    _p(14, "pink", 160, 100, (12, 60, 180, 500, 700, 900)),
    _p(16, "orange", 180, 100, (14, 70, 200, 550, 750, 950)),
    _p(18, "orange", 180, 100, (14, 70, 200, 550, 750, 950)),
    _p(19, "orange", 200, 100, (16, 80, 220, 600, 800, 1000)),
    _p(21, "red", 220, 150, (18, 90, 250, 700, 875, 1050)),
    _p(23, "red", 220, 150, (18, 90, 250, 700, 875, 1050)),
    _p(24, "red", 240, 150, (20, 100, 300, 750, 925, 1100)),
    _p(26, "yellow", 260, 150, (22, 110, 330, 800, 975, 1150)),
    _p(27, "yellow", 260, 150, (22, 110, 330, 800, 975, 1150)),
    _p(29, "yellow", 280, 150, (24, 120, 360, 850, 1025, 1200)),
    _p(31, "green", 300, 200, (26, 130, 390, 900, 1100, 1275)),
    _p(32, "green", 300, 200, (26, 130, 390, 900, 1100, 1275)),
    _p(34, "green", 320, 200, (28, 150, 450, 1000, 1200, 1400)),
    _p(37, "dark blue", 350, 200, (35, 175, 500, 1100, 1300, 1500)),
    _p(39, "dark blue", 400, 200, (50, 200, 600, 1400, 1700, 2000)),
)


def square_kind(square: int) -> str:
    """Color group for a property, otherwise the kind of square ("railroad", "chance", ...)."""
    if square == IN_JAIL:
        return "jail"
    for prop in PROPERTIES:
        if prop.square == square:
            return prop.group
    if square in RAILROADS:
        return "railroad"
    if square in UTILITIES:
        return "utility"
    if square in CHANCE_SQUARES:
        return "chance"
    if square in COMMUNITY_CHEST_SQUARES:
        return "community chest"
    if square in TAXES:
        return "tax"
    return "corner"


# Card decks. Each card is (text, effect). Effects:
#   ("square", n)        advance to square n
#   ("nearest", squares) advance to the next square in `squares`, moving forward
#   ("back", k)          go back k squares (and resolve whatever is there)
#   ("jail",)            go directly to jail
#   ("none",)            no movement
CHANCE_CARDS: tuple[tuple[str, tuple], ...] = (
    ("Advance to Go", ("square", GO)),
    ("Advance to Illinois Avenue", ("square", 24)),
    ("Advance to St. Charles Place", ("square", 11)),
    ("Advance to the nearest Utility", ("nearest", UTILITIES)),
    ("Advance to the nearest Railroad", ("nearest", RAILROADS)),
    ("Advance to the nearest Railroad", ("nearest", RAILROADS)),
    ("Advance to Boardwalk", ("square", 39)),
    ("Take a trip to Reading Railroad", ("square", 5)),
    ("Go back 3 spaces", ("back", 3)),
    ("Go to Jail", ("jail",)),
    ("Bank pays you dividend of $50", ("none",)),
    ("Get Out of Jail Free", ("none",)),
    ("Make general repairs on all your property", ("none",)),
    ("Speeding fine $15", ("none",)),
    ("You have been elected Chairman of the Board", ("none",)),
    ("Your building loan matures", ("none",)),
)

COMMUNITY_CHEST_CARDS: tuple[tuple[str, tuple], ...] = (
    ("Advance to Go", ("square", GO)),
    ("Go to Jail", ("jail",)),
    ("Bank error in your favor", ("none",)),
    ("Doctor's fee", ("none",)),
    ("From sale of stock you get $50", ("none",)),
    ("Get Out of Jail Free", ("none",)),
    ("Holiday fund matures", ("none",)),
    ("Income tax refund", ("none",)),
    ("It is your birthday", ("none",)),
    ("Life insurance matures", ("none",)),
    ("Pay hospital fees", ("none",)),
    ("Pay school fees", ("none",)),
    ("Receive $25 consultancy fee", ("none",)),
    ("You are assessed for street repair", ("none",)),
    ("You have won second prize in a beauty contest", ("none",)),
    ("You inherit $100", ("none",)),
)


def next_square_in(position: int, squares: tuple[int, ...]) -> int:
    """First square in ``squares`` strictly ahead of ``position`` (wrapping past Go)."""
    return min(squares, key=lambda s: (s - position - 1) % N_SQUARES)
