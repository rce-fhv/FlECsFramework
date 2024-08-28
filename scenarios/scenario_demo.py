import mosaik
import mosaik.util
import sys


# Sim config. and other parameters
SIM_CONFIG = {
    'SimpleAdderSimulator': {
        'python': 'simulators.simple_adder_simulator:SimpleAdderSimulator',
    },
    'Collector': {
        'cmd': '%(python)s FlECs_Frameworks/framework/mosaik/simulator/collector.py %(addr)s',
    },
}
END = 10  # 10 seconds


# Create World
world = mosaik.World(SIM_CONFIG)

# Start simulators
examplesim = world.start('SimpleAdderSimulator', eid_prefix='Model_')
collector = world.start('Collector')

# Instantiate models
model = examplesim.simple_add_model(init_val=2)
monitor = collector.Monitor()

# Connect entities
world.connect(model, monitor, 'val', 'delta')

# Create more entities
more_models = examplesim.simple_add_model.create(2, init_val=3)
mosaik.util.connect_many_to_one(world, more_models, monitor, 'val', 'delta')

# Run simulation
world.run(until=END)

