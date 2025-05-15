#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from basf2 import set_log_level, register_module, process, LogLevel, \
#    set_random_seed, print_params, create_path, statistics, conditions
import basf2 as b2
import simulation as si
import reconstruction as re
import glob as glob
import sys
import time
import os

b2.set_log_level(b2.LogLevel.ERROR)

particlegun = b2.register_module('ParticleGun')

# background (collision) files
bg_local = glob.glob('/group/belle2/dataprod/BGOverlay/early_phase3/release-08-00-03/overlay/BGx1/set0/*.root')

b2.set_random_seed(round(time.time() * 1000) + os.getpid())

particlegun.param('pdgCodes', [22]) # code 22 is photon (I think)
particlegun.param('nTracks', 1)
particlegun.param('varyNTracks', False)

particlegun.param('momentumGeneration', 'uniform')
particlegun.param('momentumParams', [0.5, 5]) # GeV/c
particlegun.param('thetaGeneration', 'uniform')
particlegun.param('thetaParams', [15, 170]) #degrees
particlegun.param('phiGeneration', 'uniform')
particlegun.param('phiParams', [0, 360]) #degrees

b2.print_params(particlegun)

eventinfosetter = b2.register_module('EventInfoSetter')
eventinfosetter.param({'evtNumList': [1000], 'expList': [1003]})

progress = b2.register_module('Progress')
gearbox = b2.register_module('Gearbox')
geometry = b2.register_module('Geometry')

main = b2.create_path()

main.add_module(eventinfosetter)
main.add_module(progress)
main.add_module(particlegun)

si.add_simulation(main) # no background files
#si.add_simulation(main, bkgfiles=bg_local)

re.add_reconstruction(main)

main.add_module("RootOutput", outputFileName=sys.argv[1])

b2.process(main)

print(b2.statistics)
