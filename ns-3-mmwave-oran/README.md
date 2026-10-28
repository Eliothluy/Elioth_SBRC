# ns3-mmwave fork for ns-O-RAN

================================

This repository is a fork of the [ns3-mmwave project](https://github.com/nyuwireless-unipd/ns3-mmwave) with updates to make it work with the [ns3-o-ran-e2](https://github.com/o-ran-sc/sim-ns3-o-ran-e2) ns-3 module.

This module enables the support for running multiple terminations of an O-RAN-compliant E2 interface inside the simulation process. It has been developed as part of a collaborative effort between the [Institute for the Wireless Internet of Things (WIoT)](https://wiot.northeastern.edu) at Northeastern University, Sapienza University of Rome, Mavenir, and the University of Padova.

## How to use

Please refer to this [quick start guide](https://openrangym.com/tutorials/ns-o-ran) that presents a tutorial to bridge ns-O-RAN and Colosseum RIC (i.e., OSC RIC bronze reduced) ns-O-RAN.

Additional material:

- Framework presentation https://openrangym.com/ran-frameworks/ns-o-ran 
- Tutorial OSC RIC version E ns-O-RAN connection  https://www.nsnam.org/tutorials/consortium23/oran-tutorial-slides-wns3-2023.pdf 
- Recording of the tutorial OSC RIC version E done at the WNS3 2023 https://vimeo.com/867704832 
- xApp repositories working with ns-O-RAN:
  - https://github.com/wineslab/ns-o-ran-scp-ric-app-kpimon 
  - https://github.com/wineslab/ns-o-ran-xapp-rc 
- Gymnasium Environment wrapper for ns-O-RAN https://github.com/wineslab/ns-o-ran-gym-environment

## Scenario three: paper energy-saving setup

`scratch/scenario-three.cc` implements the energy-saving scenario used in the paper. In
addition to the original options it exposes the following knobs:

- `--configuration=3`: paper radio configuration (850 MHz, 20 MHz bandwidth, ISD 1700 m,
  1 antenna per device, 20 Mbps transport data rate).
- `--trafficModel=4`: paper traffic mix, applied per UE:
  - 25% TCP full-buffer traffic at 20 Mbps,
  - 25% UDP bursty traffic averaging around 20 Mbps,
  - 25% TCP bursty traffic averaging around 750 kbps,
  - 25% TCP bursty traffic averaging around 150 kbps.
- `--positionAllocator=2 --nBsNoUesAlloc=N`: UEs are placed uniformly in a disc of radius
  equal to the ISD around the gNBs. `N` gNBs (chosen with the ns-3 RNG, so the choice is
  reproducible through `--RngRun`) are left without UEs and the UEs are split as evenly as
  possible over the remaining gNBs. The paper uses `--ues=9` (7 gNBs, 63 UEs); the scenario
  default is `--ues=7` (49 UEs).
- `--heuristicType=3`: paper random policy. Every second a number of cells uniformly
  distributed in `[0, 7]` is turned off, and all the other cells are turned on. The policy
  uses the ns-3 RNG, so it is reproducible through `--RngRun`.

The paper tuples `(bsOn, bsIdle, bsSleep, bsOff)` used by the static/dynamic heuristics
(`--heuristicType=1` / `2`) are:

| `--bsOn` | `--bsIdle` | `--bsSleep` | `--bsOff` |
|----------|------------|-------------|-----------|
| 4        | 2          | 0           | 1         |
| 3        | 2          | 0           | 2         |
| 2        | 2          | 0           | 3         |

For example:

```bash
./ns3 run "scratch/scenario-three --configuration=3 --trafficModel=4 --positionAllocator=2 \
  --nBsNoUesAlloc=1 --ues=9 --simTime=10 --indicationPeriodicity=0.1 --heuristicType=3"
```

### External energy-saving control

The external controller (for instance `ns-o-ran-gym`) writes a CSV file named
`es_actions_for_ns3.csv` with one line per action:

```
timestamp_ms,cellId,hoAllowed
0,2,0
0,3,0
```

`hoAllowed` is `1` to keep the cell on and `0` to turn it off (the paper maps ON/IDLE to
1 and SLEEP/OFF to 0). Point the simulation at the file with
`--controlFileName=es_actions_for_ns3.csv`. The scenario uses the default
`ScheduleControlMessages=false` mode, so the actions are applied as soon as the file is
read. In that mode the file is truncated after being parsed, which lets an external
controller rewrite it at every indication period.

The action is forwarded from `LteEnbRrc` through a registered callback to the corresponding
`MmWaveEnbNetDevice::TurnOn()`/`TurnOff()`, so the mmWave RF state really changes.
`bsState.txt` reports the actual state of every mmWave cell (`Id State`, with `State` 1
for ON and 0 for OFF) and therefore reflects the applied actions.

### Handover robustness when cells are turned off

With aggressive sleeping policies (for instance `--heuristicType=3`, which can turn all
seven gNBs off in the same second) a UE may lose its serving cell while every remaining
cell is off, i.e. there is no valid handover target. The fork originally scheduled an
eviction handover towards the invalid target cell `0` in that case, crashing the
simulation in `EpcX2` (`Missing infos for targetCellId = 0`). Two guards were added in
`src/lte/model/lte-enb-rrc.cc`:

- `EvictUsersFromSecondaryCell()` skips UEs that have no allowed target cell (the
  periodic outage detection then moves the UE back to the LTE stack);
- `UeManager::PrepareHandover()` ignores handover requests addressed to cell `0`.

With the guards, all-off seconds leave the affected UEs on LTE until cells are turned
back on.

A third guard is needed when the xApp flips a cell while a secondary-cell handover is
already pending. `LteEnbRrc::PerformHandover()` used to assert that the handover event
still referred to the cell the UE was attached to
(`handoverInfo.sourceCellId == m_lastMmWaveCell[imsi]`). That invariant does not hold
when the external controller turns cells on and off every E2 indication: the UE may
have moved to another secondary cell in the meantime, and the abort
(`NS_ASSERT failed ... The secondary cell to which the UE is attached has changed`)
killed the simulation. The stale event is now dropped instead, and the periodic
handover logic re-evaluates the UE afterwards.

### Output files

- `cu-cp-cell-<id>.txt`, `cu-up-cell-<id>.txt`: CU-CP/CU-UP E2 reports (cell 1 is the LTE
  eNB, cells 2-8 are the mmWave gNBs).
- `du-cell-<id>.txt`: DU E2 reports for the mmWave cells. These are produced only when NR
  E2 reporting is enabled, i.e. add `--e2nrEnabled=1`; the default scenario value is
  `false`, so the exact smoke command without that flag does not create `du-*.txt`.
- `bsState.txt`: actual mmWave cell states over time.

### TCP stack fixes needed by the paper traffic mix

The paper traffic mix (20 Mbps TCP full buffer + UDP bursty) exercises the TCP
selective-acknowledgement path much harder than the default mix, and two latent
bugs of ns-3.38 killed the simulation:

- `TcpOptionSack::Deserialize()` computed `sackCount = (size - 2) / 8` in `uint8_t`
  arithmetic. A malformed option with `size < 2` therefore underflowed to 254 and the
  loop read up to 248 bytes past the end of the option, corrupting the scoreboard of
  the TCP socket. The option is now discarded when `size < 2 || size > 34`.
- `TcpTxBuffer::Remove()` and `TcpTxBuffer::ResetLastSegmentSent()` erased items of
  `m_sentList` without invalidating `m_highestSack.first`, leaving a dangling
  iterator that crashed the simulation in `TcpTxBuffer::Update()` (SIGSEGV).

Both are in the upstream ns-3 code base and were not specific to the energy-saving
use case, but they make the difference between a campaign that runs to completion
and one that is truncated after a few seconds.

A third one lives in the LTE PDCP: `McEnbPdcp::DoReceivePdu()` (and
`LtePdcp::DoReceivePdu()`) removed the PDCP header without checking that the PDU
was long enough to contain it. A truncated PDU — delivered after a handover, for
instance — made `LtePdcpHeader::Deserialize()` read past the end of the packet and
abort the simulation (`read beyond the bounds of the available buffer space`). The
user-plane counterpart (`McUePdcp::DoReceivePdu()`) already performed that check;
both sides do it now, and `LtePdcpHeader::Deserialize()` refuses to read a
truncated header.

## How to contribute

See the [contributing](./CONTRIBUTING.md) page.

## References

More information can be found in the technical paper:

> A. Lacava, M. Bordin, M. Polese, R. Sivaraj, T. Zugno, F. Cuomo, and T. Melodia. "ns-O-RAN: Simulating O-RAN 5G Systems in ns-3", Proceedings of the 2023 Workshop on ns-3 (2023), [DOI:10.1145/3592149.3592161](https://dl.acm.org/doi/abs/10.1145/3592149.3592161) 

If you use the scenario-one.cc or the traffic steering implementation please cite:

>A. Lacava, M. Polese, R. Sivaraj, R. Soundrarajan, B. Bhati, T. Singh, T. Zugno, F. Cuomo, and T. Melodia. "Programmable and Customized Intelligence for Traffic Steering in 5G Networks Using Open RAN Architectures", IEEE Transactions on Mobile Computing (2024), [DOI:10.1109/TMC.2023.3266642](https://doi.org/10.1109/TMC.2023.3266642) [pdf](https://ieeexplore.ieee.org/document/10102369) [bibtex](https://ece.northeastern.edu/wineslab/wines_bibtex/andrea/LacavaAMC22.txt)


# mmWave ns-3 module #

This is an [ns-3](https://www.nsnam.org "ns-3 Website") module for the simulation
of 5G cellular networks operating at mmWaves. A description of this module can be found in [this paper](https://ieeexplore.ieee.org/document/8344116/ "mmwave paper").

Main features:

* Support of a wide range of channel models, including the model based on 3GPP TR 38.901 for frequencies between 0.5 and 100 GHz. Ray tracing and measured traces can also be used.

* Custom PHY and MAC classes supporting the 3GPP NR frame structure and numerologies.

* Custom schedulers for supporting dynamic TDD formats

* Carrier Aggregation at the MAC layer

* Enhancements to the RLC layer with re-segmentation of packets for retransmissions

* Dual Connectivity with LTE base stations, with fast secondary cell handover and channel tracking

* Simulation of core network elements (with also the MME as a real node)

## Installation
This repository contains a complete ns-3 installation with the addition of the mmwave module. 

Use these commands to download and build `ns3-mmwave`:
```
git clone https://github.com/nyuwireless-unipd/ns3-mmwave.git
cd ns3-mmwave
./ns3 configure --disable-python --enable-examples && ./ns3 build
```

## Usage example
You can use the following command to run the `mmwave-simple-epc` example. 
```
./ns3 --run mmwave-simple-epc
```
Other examples are included in `src/mmwave/examples/`

## Documentation
The documentation of this module is available at [this link](./src/mmwave/doc/mmwave-doc.md).

## Related modules
- MilliCar is an ns-3 module for the simulation of mmWave NR V2X networks. Check [this repo](https://github.com/signetlabdei/millicar) for further details.
- A seperate module is being developed for [mmWave UE Energy Consumption](https://github.com/arghasen10/mmwave-energy "mmwave-energy"). You can use this module for analyzing 
Energy Consumption behaviour of mmwave UE. Check this repository for further details.
- `ns3-mmwave-iab` is an extended version of `ns3-mmWave` adding wireless relaying capabilities to an ns-3 NetDevice, and the possibility of simulating in-band relaying at mmWave frequencies. Check [this repo](https://github.com/signetlabdei/ns3-mmwave-iab) for further details.

## References 
The following papers describe in detail the features implemented in the mmWave
module:
- [End-to-End Simulation of 5G mmWave Networks](https://ieeexplore.ieee.org/document/8344116/ "comst paper") is a comprehensive tutorial with a detailed description of the whole module. We advise the researchers interested in this module to start reading from this paper;
- [Integration of Carrier Aggregation and Dual Connectivity for the ns-3 mmWave Module](https://arxiv.org/abs/1802.06706 "wns3 2018") describes the Carrier Aggregation implementation;
- [Implementation of A Spatial Channel Model for ns-3](https://arxiv.org/abs/2002.09341 "wns3 2020") describes the integration of the spatial channel model based on the 3GPP specifications TR 38.901 V15.0.0;
- [Performance Comparison of Dual Connectivity and Hard Handover for LTE-5G Tight Integration](https://arxiv.org/abs/1607.05425 "simutools paper") describes the Dual Connectivity feature.

These other papers describe features that were implemented in older releases: 
- [ns-3 Implementation of the 3GPP MIMO Channel Model for Frequency Spectrum above 6 GHz](https://dl.acm.org/citation.cfm?id=3067678 "wns3 2017") describes the implementation of the 3GPP channel model based on TR 38.900;
- [Multi-Sector and Multi-Panel Performance in 5G mmWave Cellular Networks](https://arxiv.org/abs/1808.04905 "globecom2018") describes the multi-sector addition to the 3GPP channel model;

If you use this module in your research, please cite:

M. Mezzavilla, M. Zhang, M. Polese, R. Ford, S. Dutta, S. Rangan, M. Zorzi, _"End-to-End Simulation of 5G mmWave Networks,"_ in IEEE Communications Surveys & Tutorials, vol. 20, no. 3, pp. 2237-2263, thirdquarter 2018. [bibtex available here](https://ieeexplore.ieee.org/document/8344116/)

## Future work
We are actively developing new features for the mmWave module, including:
- 3GPP NR beam tracking
- 3GPP NR Integrated Access and Backhaul feature (see [this repo](https://github.com/signetlabdei/ns3-mmwave-iab) for more details)

## About
This module is being developed by [NYU Wireless](http://wireless.engineering.nyu.edu/) and the [University of Padova](http://mmwave.dei.unipd.it/).
This  work  was  supported  in  part by  the  U.S.  Department  of  Commerce  National  Institute  of  Standards  and Technology through the Project “An End-to-End Research Platform for Public Safety  Communications  above  6  GHz”  under  Award  70NANB17H16.



<!-- The new-handover branch offers integration between LTE and mmWave and dual connectivity features.
 -->

## Authors ##

The ns-3 mmWave module is the result of the development effort carried out by different people. The main contributors are: 
- Tommaso Zugno, University of Padova
- Michele Polese, University of Padova
- Matteo Pagin, University of Padova
- Mattia Lecci, University of Padova
- Matteo Drago, University of Padova
- Mattia Rebato, University of Padova
- Menglei Zhang, NYU Wireless
- Marco Giordani, University of Padova
- Marco Mezzavilla, NYU Wireless
- Sourjya Dutta, NYU Wireless
- Russell Ford, NYU Wireless
- Gabriel Arrobo, Intel

## License ##

This software is licensed under the terms of the GNU GPLv2, as like as ns-3. See the LICENSE file for more details.
