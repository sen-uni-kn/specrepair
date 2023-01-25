from typing import List, Tuple, Callable, Dict

from deep_opt.models.property import Property, OutputConstraint, BoxConstraint, ExtremumConstraint, \
    MultiOutputExtremumConstraint, ConstraintAnd

pi = 3.141592


class ACASXuProperty(Property):
    """
    Properties for ACASXu and HCAS networks.
    Additionally to the attributes of Property these objects
    also have a network_ranges attribute that is used to determine
    if a property should be checked for an ACASXu network.
    """

    def __init__(self,
                 lower_bounds: Dict[int, float],
                 upper_bounds: Dict[int, float],
                 output_constraint: OutputConstraint,
                 network_ranges: Tuple[range, range, Callable[[int, int], bool]],
                 property_name: str):
        super().__init__(lower_bounds, upper_bounds, output_constraint, property_name=property_name)
        self.network_ranges = network_ranges


def property_1() -> ACASXuProperty:
    """
    Property φ1.

    If the intruder is distant and is significantly slower than the ownship,
    the score of a COC advisory will always be below a certain fixed threshold.

    - Tested on: all 45 networks.
    – Input constraints: ρ ≥ 55947.691 and v_own ≥ 1145 and v_int ≤ 60
    - Desired output property: COC ≤ 1500

    :return: φ1 property.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 55947.691,
            3: 1145,
        },
        upper_bounds={
            4: 60
        },
        output_constraint=BoxConstraint(0, '<=', 1500),
        network_ranges=(tuple(range(1, 6)), tuple(range(1, 10)), lambda x, y: False),
        property_name='ACASXu φ1'
    )


def property_2() -> ACASXuProperty:
    """
    Property φ2.

    If the intruder is distant and is significantly slower than the ownship,
    the score of a COC advisory will never be maximal.

    – Tested on: N_{x,y} for all x ≥ 2 and for all y.
    – Input constraints: ρ ≥ 55947.691, v_own ≥ 1145, v_int ≤ 60
    – Desired output property: the score for COC is not the maximal score.

    :return: φ2 property.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 55947.691,
            3: 1145,
        },
        upper_bounds={
            4: 60
        },
        output_constraint=ExtremumConstraint(0, '!=', 'max'),
        network_ranges=(tuple(range(2, 6)), tuple(range(1, 10)), lambda x, y: False),
        property_name='ACASXu φ2'
    )


def property_3() -> ACASXuProperty:
    """
    Property φ3.

    If the intruder is directly ahead and is moving towards the ownship,
    the score for COC will not be minimal.

    – Tested on: all networks except N1,7, N1,8, and N1,9.
    – Input constraints: 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, ψ ≥ 3.10, v_own ≥ 980, v_int ≥ 960.
    – Desired output property: the score for COC is not the minimal score.

    [1652,07287 1,54867127 3,51549386 1178,49389 1088,93263
    4.58583379e+00 4.73706629e+00 1.93628379e+01 4.97560180e+00 1.92426759e+01]
    :return: φ3 property.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 1500,
            1: -0.06,
            2: 3.10,
            3: 980,
            4: 960
        },
        upper_bounds={
            0: 1800,
            1: 0.06
        },
        output_constraint=ExtremumConstraint(0, '!=', 'min'),
        network_ranges=(tuple(range(1, 6)), tuple(range(1, 10)), lambda x, y: x == 1 and (y == 7 or y == 8 or y == 9)),
        property_name='ACASXu φ3'
    )


def property_4() -> ACASXuProperty:
    """
    Property φ4.

    – Description: If the intruder is directly ahead and is moving away from the ownship
    but at a lower speed than that of the ownship,
    the score for COC will not be minimal.

    – Tested on: all networks except N1,7, N1,8, and N1,9.
    – Input constraints: 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, ψ = 0, vown ≥ 1000, 700 ≤ vint ≤ 800.
    – Desired output property: the score for COC is not the minimal score.

    :return: Property φ4.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 1500,
            1: -0.06,
            2: 0,
            3: 1000,
            4: 700
        },
        upper_bounds={
            0: 1800,
            1: 0.06,
            2: 0,
            4: 800
        },
        output_constraint=ExtremumConstraint(0, '!=', 'min'),
        network_ranges=(tuple(range(1, 6)), tuple(range(1, 10)), lambda x, y: x == 1 and (y == 7 or y == 8 or y == 9)),
        property_name='ACASXu φ4'
    )


def property_5() -> ACASXuProperty:
    """
    Property φ5.

    – Description: If the intruder is near and approaching from the left, the network advises “strong right”.
    – Tested on: N1,1.
    – Input constraints: 250 ≤ ρ ≤ 400, 0.2 ≤ θ ≤ 0.4, −3.141592 ≤ ψ ≤ −3.141592 + 0.005, 100 ≤ vown ≤ 400,
      0 ≤ vint ≤ 400.
    – Desired output property: the score for “strong right” is the minimal score.

    :return: Property φ5.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 250,
            1: 0.2,
            2: -pi,
            3: 100,
            4: 0
        },
        upper_bounds={
            0: 400,
            1: 0.4,
            2: -pi + 0.005,
            3: 400,
            4: 400
        },
        output_constraint=ExtremumConstraint(4, '==', 'strict_min'),
        network_ranges=(tuple(range(1, 2)), tuple(range(1, 2)), lambda x, y: False),
        property_name='ACASXu φ5'
    )


def property_6_a() -> ACASXuProperty:
    """
    Property φ6 (a).

    Description: If the intruder is sufficiently far away, the network advises COC.
    – Tested on: N1,1.
    – Input constraints: 12000 ≤ ρ ≤ 62000, (0.7 ≤ θ ≤ 3.141592) ∨ (−3.141592 ≤ θ ≤ −0.7),
      −3.141592 ≤ ψ ≤ −3.141592 + 0.005, 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200.
    – Desired output property: the score for COC is the minimal score.

    :return: Property φ6 (a).
    """
    return ACASXuProperty(
        lower_bounds={
            0: 12000,
            1: 0.7,
            2: -pi,
            3: 100,
            4: 0
        },
        upper_bounds={
            0: 62000,
            1: pi,
            2: -pi + 0.005,
            3: 1200,
            4: 1200
        },
        output_constraint=ExtremumConstraint(0, '==', 'strict_min'),
        network_ranges=(tuple(range(1, 2)), tuple(range(1, 2)), lambda x, y: False),
        property_name='ACASXu φ6 (a)'
    )


def property_6_b() -> ACASXuProperty:
    """
    Property φ6 (b).
    :return: Property φ6 (b).
    """
    return ACASXuProperty(
        lower_bounds={
            0: 12000,
            1: -pi,
            2: -pi,
            3: 100,
            4: 0
        },
        upper_bounds={
            0: 62000,
            1: -0.7,
            2: -pi + 0.005,
            3: 1200,
            4: 1200
        },
        output_constraint=ExtremumConstraint(0, '==', 'strict_min'),
        network_ranges=(tuple(range(1, 2)), tuple(range(1, 2)), lambda x, y: False),
        property_name='ACASXu φ6 (b)'
    )


def property_7() -> ACASXuProperty:
    """
    Property φ7.

    – Description: If vertical separation is large, the network will never advise a strong turn.
    – Tested on: N1,9.
    – Input constraints: 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ 3.141592, −3.141592 ≤
      ψ ≤ 3.141592, 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200.
    – Desired output property: the scores for “strong right” and “strong left” are
    never the minimal scores.
    :return: Property φ7.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 0,
            1: -pi,
            2: -pi,
            3: 100,
            4: 0
        },
        upper_bounds={
            0: 60760,
            1: pi,
            2: pi,
            3: 1200,
            4: 1200
        },
        output_constraint=ConstraintAnd(
            ExtremumConstraint(3, '!=', 'min'),
            ExtremumConstraint(4, '!=', 'min')
        ),
        network_ranges=(tuple(range(1, 2)), tuple(range(9, 10)), lambda x, y: False),
        property_name='ACASXu φ7'
    )


# def property_7_a() -> ACASXuProperty:
#     """
#     Property φ7 (a).
#
#     – Description: If vertical separation is large, the network will never advise a strong turn.
#     – Tested on: N1,9.
#     – Input constraints: 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ 3.141592, −3.141592 ≤
#       ψ ≤ 3.141592, 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200.
#     – Desired output property: the scores for “strong right” and “strong left” are
#     never the minimal scores.
#     :return: Property φ7 (a).
#     """
#     return ACASXuProperty(
#         lower_bounds={
#             0: 0,
#             1: -pi,
#             2: -pi,
#             3: 100,
#             4: 0
#         },
#         upper_bounds={
#             0: 60760,
#             1: pi,
#             2: pi,
#             3: 1200,
#             4: 1200
#         },
#         output_constraint=ExtremumConstraint(3, '!=', 'min'),
#         network_ranges=(tuple(range(1,2)), tuple(range(9,10)), lambda x, y: False),
#         property_name='ACASXu φ7 (a)'
#     )


# def property_7_b() -> ACASXuProperty:
#     """
#     Property φ7 (b).
#
#     :return: Property φ7 (b).
#     """
#     return ACASXuProperty(
#         lower_bounds={
#             0: 0,
#             1: -pi,
#             2: -pi,
#             3: 100,
#             4: 0
#         },
#         upper_bounds={
#             0: 60760,
#             1: pi,
#             2: pi,
#             3: 1200,
#             4: 1200
#         },
#         output_constraint=ExtremumConstraint(4, '!=', 'min'),
#         network_ranges=(tuple(range(1,2)), tuple(range(9,10)), lambda x, y: False),
#         property_name='ACASXu φ7 (b)'
#     )


def property_8() -> ACASXuProperty:
    """
    Property φ8.

    For a large vertical separation and a previous “weak left” advisory,
    the network will either output COC or continue advising “weak left”.

    – Tested on: N2,9.
    – Input constraints: 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ −0.75 * 3.141592, −0.1 ≤ ψ ≤ 0.1, 600 ≤ vown ≤ 1200,
      600 ≤ vint ≤ 1200.
    – Desired output property: COC minimal or WL minimal.
    :return: φ8 property.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 0,
            1: -pi,
            2: -0.1,
            3: 600,
            4: 600
        },
        upper_bounds={
            0: 60760,
            1: -0.75 * pi,
            2: 0.1,
            3: 1200,
            4: 1200
        },
        output_constraint=MultiOutputExtremumConstraint('strict_min', 'in', [0, 1]),
        network_ranges=(tuple(range(2, 3)), tuple(range(9, 10)), lambda x, y: False),
        property_name='ACASXu φ8'
    )


# def property_8_a() -> ACASXuProperty:
#     """
#     Property φ8 (a).
#
#     For a large vertical separation and a previous “weak left” advisory,
#     the network will either output COC or continue advising “weak left”.
#
#     – Tested on: N2,9.
#     – Input constraints: 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ −0.75 * 3.141592, −0.1 ≤ ψ ≤ 0.1, 600 ≤ vown ≤ 1200,
#       600 ≤ vint ≤ 1200.
#     – Desired output property: COC minimal or WL minimal.
#
#     :param network: ACAS Xu network bounds.
#     :return: φ8 (a) property.
#     """
#     return ACASXuProperty(
#         lower_bounds={
#             0: 0,
#             1: -pi,
#             2: -0.1,
#             3: 600,
#             4: 600
#         },
#         upper_bounds={
#             0: 60760,
#             1: -0.75 * pi,
#             2: 0.1,
#             3: 1200,
#             4: 1200
#         },
#         output_constraint=ExtremumConstraint(0, '==', 'strict_min'),
#         network_ranges=(tuple(range(2,3)), tuple(range(9,10)), lambda x, y: False),
#         property_name='ACASXu φ8 (a)'
#     )


# def property_8_b() -> ACASXuProperty:
#     """
#     Property φ8 (b).
#     :return: φ8 (b) property.
#     """
#     return ACASXuProperty(
#         lower_bounds={
#             0: 0,
#             1: -pi,
#             2: -0.1,
#             3: 600,
#             4: 600
#         },
#         upper_bounds={
#             0: 60760,
#             1: -0.75 * pi,
#             2: 0.1,
#             3: 1200,
#             4: 1200
#         },
#         output_constraint=ExtremumConstraint(1, '==', 'strict_min'),
#         network_ranges=(tuple(range(2,3)), tuple(range(9,10)), lambda x, y: False),
#         property_name='ACASXu φ8 (b)'
#     )


def property_9() -> ACASXuProperty:
    """
    Property φ9.

    – Description: Even if the previous advisory was “weak right”,
      the presence of a nearby intruder will cause the network to output a “strong left” advisory instead.
    – Tested on: N3,3.
    – Input constraints: 2000 ≤ ρ ≤ 7000, −0.4 ≤ θ ≤ −0.14, −3.141592 ≤ ψ ≤ −3.141592 + 0.01,
      100 ≤ vown ≤ 150, 0 ≤ vint ≤ 150.
    – Desired output property: the score for “strong left” is minimal.

    [ 6938,04914  0,313750300 -1.88723883e+00  1.13128887e+02  1.97777276e+00
    3.07808963e+01  4.76403567e+01  1.04381764e+01 4.91446320e+01  9.53591646e+00]
    :return: Property φ9.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 2000,
            1: -0.4,
            2: -pi,
            3: 100,
            4: 0
        },
        upper_bounds={
            0: 7000,
            1: -0.14,
            2: -pi + 0.01,
            3: 150,
            4: 150
        },
        output_constraint=ExtremumConstraint(3, '==', 'strict_min'),
        network_ranges=(tuple(range(3, 4)), tuple(range(3, 4)), lambda x, y: False),
        property_name='ACASXu φ9'
    )


def property_10() -> ACASXuProperty:
    """
    Property φ10.

    – Description: For a far away intruder, the network advises COC.
    – Tested on: N4,5.
    – Input constraints: 36000 ≤ ρ ≤ 60760, 0.7 ≤ θ ≤ 3.141592, −3.141592 ≤ ψ ≤ −3.141592 + 0.01,
      900 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200.
    – Desired output property: the score for COC is minimal.
    :return: Property φ10.
    """
    return ACASXuProperty(
        lower_bounds={
            0: 36000,
            1: 0.7,
            2: -pi,
            3: 900,
            4: 600
        },
        upper_bounds={
            0: 60760,
            1: pi,
            2: -pi + 0.01,
            3: 1200,
            4: 1200
        },
        output_constraint=ExtremumConstraint(0, '==', 'strict_min'),
        network_ranges=(tuple(range(4, 5)), tuple(range(5, 6)), lambda x, y: False),
        property_name='ACASXu φ10'
    )


def get_properties() -> List[Tuple[Property, str]]:
    return [(property_1(), "1"),
            (property_2(), "2"),
            (property_3(), "3"),
            (property_4(), "4"),
            (property_5(), "5"),
            (property_6_a(), "6a"), (property_6_b(), "6b"),
            (property_7(), "7"),
            # (property_7_a(), "7a"), (property_7_b(), "7b"),
            (property_8(), "8"),
            # (property_8_a(), "8a"), (property_8_b(), "8b"),
            (property_9(), "9"),
            (property_10(), "10")]
