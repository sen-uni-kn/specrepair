# expose some important classes and functions on top level
from deep_opt.models import NeuralNetwork
from deep_opt.models import Property, RobustnessPropertyFactory, dump_specification, load_specification
from deep_opt.models import BoxConstraint, ExtremumConstraint, MultiOutputExtremumConstraint

from deep_opt.numerics.optimization import optimize_by_property
from deep_opt.numerics.optimization import CounterexampleAmount
