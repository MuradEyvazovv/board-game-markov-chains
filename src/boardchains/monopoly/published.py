"""Published long-run landing percentages, used as an outside check on the chain.

Source: Truman Collins, "Probabilities in the Game of Monopoly",
http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml (first table, values in
percent, 4 decimals). Rows are in this package's category order: squares 0..39 and
then In Jail.

* ``SHORT_JAIL``: the player pays to leave jail right away. This matches my
  ``"leave_early"`` policy.
* ``LONG_JAIL``: the player stays in jail as long as allowed. This one only matches if
  a player who rolls doubles to get out of jail also gets to roll again, which the
  rulebook doesn't allow. ``chain.solve("stay", jail_doubles_reroll=True)`` builds
  that variant.
"""

SHORT_JAIL = (
    3.0961, 2.1314, 1.8849, 2.1624, 2.3285, 2.9631, 2.2621, 0.8650, 2.3210, 2.3003, 2.2695,
    2.7017, 2.6040, 2.3721, 2.4649, 2.9200, 2.7924, 2.5945, 2.9356, 3.0852, 2.8836, 2.8358,
    1.0480, 2.7357, 3.1858, 3.0659, 2.7072, 2.6789, 2.8074, 2.5861, 0.0000, 2.6774, 2.6252,
    2.3661, 2.5006, 2.4326, 0.8669, 2.1864, 2.1799, 2.6260, 3.9499,
)

LONG_JAIL = (
    2.9143, 2.0073, 1.7750, 2.0369, 2.1934, 2.8010, 2.1317, 0.8152, 2.1874, 2.1680, 2.1392,
    2.5560, 2.6140, 2.1741, 2.4255, 2.6354, 2.6802, 2.2957, 2.8210, 2.8118, 2.8253, 2.6143,
    1.0448, 2.5671, 2.9929, 2.8930, 2.5370, 2.5191, 2.6507, 2.4381, 0.0000, 2.5236, 2.4721,
    2.2276, 2.3531, 2.2906, 0.8158, 2.0595, 2.0521, 2.4832, 9.4569,
)
