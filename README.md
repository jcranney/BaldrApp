# BaldrApp

Simulating Baldr - the Zernike Wavefront Sensor (ZWFS) for VLTI/Asgard

Various modules and examples for testing end-to-end a ZWFS. Optionally includes, and extends the machinary of pyZelda to deal with specific details of Baldr, including its unique optics, coldstops, DMs and phasemasks. 
```
python
import baldrapp
```
## Installation with `pip`
```
pip install baldrapp
```
This has a dependancy on a forked version of the pyZELDA package (https://github.com/courtney-barrer/pyZELDA) which must be installed seperately
```
pip install pyzelda@git+https://github.com/courtney-barrer/pyZELDA.git@b42aaea5c8a47026783a15391df5e058360ea15e
```    
Alternatvely the project can be cloned or forked from the Github:
```bash
git clone https://github.com/courtney-barrer/BaldrApp
```
The pip installation was tested on only on python 3.12.7. 

## Installation with Nix `flake`s
If you are using Nix for package management, you can initialise a `devShell` with all the required dependencies, and installing the python package and GUIs using:
```bash
nix develop
```

This will also give you a shell alias to allow you to start and stop the simulator, e.g.:
```bash
heimbal-sim start  # to start the simulator
```

## VLTI/Baldr Simulator Architecture

The BaldrApp comes with a module designed to replicate the full 4 beam control data flow of the Baldr instrument on the VLTI system. It operates via a shared memory loop:
- Input: The module monitors and reads the DM shared memory for updated mirror commands.
- Optical Model: It calculates the optical propagation through the ZWFS based on the current state of the DM, phase mask and optics.
- Output: The resulting simulated camera frames are written directly to the camera shared memory.

Because this replicates Baldrs software interfaces, any RTC that consumes camera frames and writes DM commands via shared memory can interface with the simulator without modification. This allows for agnostic testing of the control loop against the simulated instrument.

To run VLTI Asgard/Baldr simulator (with shared memory - tested on Ubuntu 20.04). Install requirements in virtual enviornment. Once activated e.g.
```
source venv/bin/activate
```
then 
```
./baldrapp/apps/paranal_simulator/heimbal_simulation_servers.sh start 
```
This will create the camera and DM shared memory objects, begin a simulated camera and DM server, and run the Baldr simulator. The Paranal DM gui and camera shared memory viewer should open automatically and show updating frames. To stop all the server processes and close the simulator nicely, simply run 
```
./baldrapp/apps/paranal_simulator/heimbal_simulation_servers.sh stop
```
You can also view the status of the servers processes:
```
./baldrapp/apps/paranal_simulator/heimbal_simulation_servers.sh status 
```
If you are running this as another user on your linux you might need to give yourself permission to read/write to shared memory which can be done by 
```
sudo chown <user>:<user> /dev/shm/*some_shms* 
```
obviously using your username and whatever shared memory addresses (SHMs) you need to use
